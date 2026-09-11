"""Block geometry and sidecar containers for pseudo-labelling the unlabeled lmd volumes.

Pure functions plus two writers, kept apart from `pseudolabel.py` so they can be tested without a
model, a GPU or a store under /groups.

**Blocks are disjoint cells of a fixed partition, chosen by a per-volume permutation.** A round
labels the first K cells of that permutation, so round 2's blocks contain round 1's: each round
relabels what it had with the better model and adds new data, which is what the Segment Anything
data engine did across its retrains. Disjointness is a requirement rather than tidiness -- each
block becomes its own `volumes:` entry with its own `bounding_box`, so a crop never spans two
blocks and instance ids only ever need to be unique within one; two overlapping blocks would write
two numberings into the same voxels, and a crop from one would read ids belonging to the other.

**Labels are written on the source's own level-0 grid**, same shape and axis order as `raw/s0`, so
miao co-registers them with the image by construction -- no scale or translation of ours to get
wrong, and the corpus's per-level origin conventions never enter. The cost is up to 8x the voxels
of the 8 nm lattice for a 4 nm volume, which is cheap: a labelling has no information below the
mask stride (4 lattice voxels = 32 nm), it compresses ~20x, and only the blocks actually labelled
occupy chunks. Voxels never labelled read as `IGNORE` (-1) from the array's fill value; voxels
inside a labelled block that no confident mask claimed are 0. The promptable strategy treats both
as "not an object" -- ids are only ever drawn from values above 0 -- so the distinction is for
whoever reads these labels next, not for this training.

**The container is a sidecar.** The published stores are read-only, so each labelled volume gets
its own OME-Zarr directory whose `raw` is a symlink to the real one and whose `labels/<name>` is
ours, in the source's own zarr format (85 of the 87 are zarr3, 2 are zarr2, and a volume's
`zarr_version` opens both its image and its labels). Rounds share the container under different
label names.
"""

from __future__ import annotations

import itertools
import json
import shutil
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import zarr

#: The value `promptable_seg`'s target sampler clips to 0 and `affinity_seg` excludes from its
#: loss. Signed on purpose -- see banis/mia_pseudolabel.py for the uint16 trap.
IGNORE = -1
#: Chunking of the label array. A 128-cube of int32 is 8 MB before compression.
CHUNKS = (128, 128, 128)

Box = list[list[int]]


def volume_seed(name: str) -> int:
    """A stable per-volume seed: `hash()` of a str is salted per process."""
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def block_edges(
    lattice_block: int,
    lattice_voxel_nm: Sequence[float],
    native_voxel_nm: Sequence[float],
    full_extent: Sequence[int],
) -> list[int]:
    """Native-voxel edge of a block spanning `lattice_block` voxels of the training lattice.

    Per storage axis, clamped to the volume: on an axis thinner than a block the block spans the
    whole axis, which is what makes thin serial-section volumes labellable at all.
    """
    if lattice_block < 1:
        raise ValueError(f"lattice_block must be positive, got {lattice_block}")
    edges = []
    for lattice_nm, native_nm, extent in zip(
        lattice_voxel_nm, native_voxel_nm, full_extent, strict=True
    ):
        edge = int(round(lattice_block * float(lattice_nm) / float(native_nm)))
        edges.append(max(1, min(edge, int(extent))))
    return edges


def partition(full_box: Sequence[Sequence[int]], edges: Sequence[int]) -> list[Box]:
    """Disjoint cells of `edges` inside `full_box` (`[lo, hi)` per axis), storage order.

    The largest whole number of cells that fits on each axis, centred so the remainder is split
    between the two faces rather than all left at the far end. An axis shorter than its edge holds
    exactly one cell spanning it.
    """
    axes: list[list[list[int]]] = []
    for (lo, hi), edge in zip(full_box, edges, strict=True):
        extent = int(hi) - int(lo)
        if extent < 1:
            raise ValueError(f"empty axis in box {full_box}")
        step = min(int(edge), extent)
        count = extent // step
        offset = int(lo) + (extent - count * step) // 2
        axes.append([[offset + k * step, offset + (k + 1) * step] for k in range(count)])
    return [list(map(list, cell)) for cell in itertools.product(*axes)]


def nested_choice(cells: Sequence[Box], seed: int, count: int) -> list[tuple[int, Box]]:
    """The first `count` cells of a seeded permutation -> `(cell index, box)` pairs.

    A prefix of one fixed permutation, so a later round asking for more cells gets a superset of an
    earlier round's, and the cell index names the block stably across rounds.
    """
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    order = np.random.default_rng(seed).permutation(len(cells))
    return [(int(index), cells[int(index)]) for index in order[: min(count, len(cells))]]


def permute_box(box: Sequence[Sequence[int]], from_axes: str, to_axes: str) -> Box:
    """Reorder a per-axis box from one spatial axis order to another, e.g. storage zyx -> xyz."""
    if sorted(from_axes) != sorted(to_axes) or len(from_axes) != len(box):
        raise ValueError(f"cannot permute a {len(box)}-axis box from {from_axes!r} to {to_axes!r}")
    return [list(map(int, box[from_axes.index(axis)])) for axis in to_axes]


# ------------------------------------------------------------------------------- the sidecar


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def create_sidecar(source: Path, out: Path, image_key: str, zarr_version: str) -> None:
    """An OME-Zarr container that borrows `image_key` from `source` and will own its labels.

    The root metadata is copied verbatim (zarr3: `zarr.json`; zarr2: `.zgroup` and `.zattrs`),
    which is right precisely because the image group behind the symlink is the same tree the
    metadata already describes. Idempotent: a second call on an existing sidecar refreshes the
    link and leaves any labels already written alone.
    """
    if zarr_version not in ("zarr2", "zarr3"):
        raise ValueError(f"zarr_version must be 'zarr2' or 'zarr3', got {zarr_version!r}")
    out.mkdir(parents=True, exist_ok=True)
    names = ("zarr.json",) if zarr_version == "zarr3" else (".zgroup", ".zattrs")
    for name in names:
        if (source / name).exists():
            shutil.copyfile(source / name, out / name)
    link = out / image_key
    if link.is_symlink() or link.exists():
        link.unlink()
    # `.resolve()`: symlink_to stores the target verbatim, and a relative source would otherwise
    # resolve relative to `out/` and dangle.
    link.symlink_to((source / image_key).resolve())


def create_label_array(
    out: Path,
    label_name: str,
    axes: Sequence[dict[str, Any]],
    shape: Sequence[int],
    scale: Sequence[float],
    translation: Sequence[float],
    zarr_version: str,
    chunks: Sequence[int] = CHUNKS,
) -> zarr.Array:
    """`labels/<label_name>/s0` in the sidecar: int32, `IGNORE` everywhere until written.

    One level only, advertised with the source image's own level-0 scale and translation, in the
    source's spatial axis order: miao resolves a label pyramid independently of the image's, and a
    single-level pyramid beside a deep image pyramid is well-formed as long as the level it has is
    the one asked for. A translation is written only when the source declares a non-zero one, so a
    corner-aligned store gets the same metadata shape it has for its own image.

    Idempotent per label name -- rebuilding a round's labels replaces the array -- and additive
    across names: the OME `labels` list accumulates, so an earlier round stays readable after a
    later one is written.
    """
    if zarr_version not in ("zarr2", "zarr3"):
        raise ValueError(f"zarr_version must be 'zarr2' or 'zarr3', got {zarr_version!r}")
    if not (len(axes) == len(shape) == len(scale) == len(translation)):
        raise ValueError(
            f"axes ({len(axes)}), shape ({len(shape)}), scale ({len(scale)}) and translation "
            f"({len(translation)}) must agree on the rank"
        )
    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    group = labels_dir / label_name
    group.mkdir(exist_ok=True)

    transforms: list[dict[str, Any]] = [{"type": "scale", "scale": [float(s) for s in scale]}]
    if any(abs(float(t)) > 0 for t in translation):
        transforms.append({"type": "translation", "translation": [float(t) for t in translation]})
    multiscales = {
        "axes": [dict(axis) for axis in axes],
        "datasets": [{"path": "s0", "coordinateTransformations": transforms}],
    }

    if zarr_version == "zarr3":
        listing = _read_json(labels_dir / "zarr.json")
        known = list(listing.get("attributes", {}).get("ome", {}).get("labels", []))
        if label_name not in known:
            known.append(label_name)
        _write_json(labels_dir / "zarr.json", {
            "zarr_format": 3, "node_type": "group",
            "attributes": {"ome": {"version": "0.5", "labels": known}},
        })
        _write_json(group / "zarr.json", {
            "zarr_format": 3, "node_type": "group",
            "attributes": {"ome": {"version": "0.5", "multiscales": [multiscales]}},
        })
        zarr_format = 3
    else:
        listing = _read_json(labels_dir / ".zattrs")
        known = list(listing.get("labels", []))
        if label_name not in known:
            known.append(label_name)
        _write_json(labels_dir / ".zgroup", {"zarr_format": 2})
        _write_json(labels_dir / ".zattrs", {"labels": known})
        _write_json(group / ".zgroup", {"zarr_format": 2})
        _write_json(group / ".zattrs", {"multiscales": [{"version": "0.4", **multiscales}]})
        zarr_format = 2

    return zarr.create_array(
        store=str(group / "s0"), shape=tuple(int(s) for s in shape), dtype="int32",
        chunks=tuple(min(int(c), int(s)) for c, s in zip(chunks, shape, strict=True)),
        fill_value=IGNORE, overwrite=True, zarr_format=zarr_format,
    )


def spatial_axes_metadata(axes: Sequence[dict[str, Any]], spatial_indices: Sequence[int]
                          ) -> list[dict[str, Any]]:
    """The OME axis entries of the spatial dimensions only, in storage order.

    An image with a channel axis (`cxyz`) gets labels without one: miao accepts a channel axis on
    one side and not the other, and a labelling has no channels.
    """
    return [dict(axes[i]) for i in spatial_indices]
