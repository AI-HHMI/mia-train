"""Crop one arm's prediction artifacts to another arm's extents, so both are scored over the same region.

The leaderboard scores a record over its artifact's extent clipped to the annotation, and it refuses
to rank records scored over different regions together. A model labelled with 128-voxel windows
(arm 9) covers more of each volume than the 256-voxel lattice of every other arm, so its row lands
in a table of its own unless its artifacts (prediction and co-registered truth alike) are cut down
to the other lattice's region.

That is only possible when the two grids line up. `VolumeGrid` centres each window-dependent lattice
in the box, so the 128 lattice starts 7-45 native voxels before the 256 one, and the two lattices'
output voxels differ by up to 2.4%. Each artifact records the native box it covers and its output
voxel; this crops at the offset those give and refuses when no crop can align the grids -- in which
case predict the source onto the other's box instead. (Keeping the leading voxels, as this script
used to, scored arm 9 on regions overlapping the common one by 79.5-96.7%. Its `origin` check could
not catch that: predict.py writes origin 0 on every artifact.)

    python experiments/sam_lmd_v1/crop_artifacts.py --src eval/arm9_r0/test --like eval/arm8_r0/test --dst eval/arm9_r0_lattice256/test
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr

SLAB = 64


def _offset(name: str, s_attrs: dict[str, Any], l_attrs: dict[str, Any]) -> list[int]:
    """Where `like`'s grid starts inside `src`'s, in src output voxels per spatial axis -- or refuse."""
    for key in ("native_box", "effective_voxel_nm", "scale"):
        if key not in s_attrs or key not in l_attrs:
            raise SystemExit(f"{name}: no {key!r} recorded to align the two grids by")
    s_voxel = np.asarray(s_attrs["effective_voxel_nm"], dtype=float)
    l_voxel = np.asarray(l_attrs["effective_voxel_nm"], dtype=float)
    if not np.allclose(s_voxel, l_voxel, rtol=1e-6):
        raise SystemExit(
            f"{name}: the output voxels differ ({s_voxel.tolist()} vs {l_voxel.tolist()} nm), so "
            "no crop lines the two grids up; predict the source onto the other's box instead"
        )
    s_low = np.asarray([low for low, _ in s_attrs["native_box"]], dtype=float)
    l_low = np.asarray([low for low, _ in l_attrs["native_box"]], dtype=float)
    shift = (l_low - s_low) * np.asarray(s_attrs["scale"], dtype=float)
    whole = np.round(shift)
    if (np.abs(shift - whole) > 1e-6).any() or (whole < 0).any():
        raise SystemExit(
            f"{name}: the other grid starts {shift.tolist()} output voxels into this one, and a "
            "crop needs a whole, non-negative offset; predict the source onto the other's box"
        )
    return [int(v) for v in whole]


def _shift_translation(ome: dict[str, Any], offset: list[int], voxel: list[float]) -> dict[str, Any]:
    """The OME translation of a crop starting `offset` output voxels into the source."""
    ome = copy.deepcopy(ome)
    for transform in ome["multiscales"][0]["datasets"][0]["coordinateTransformations"]:
        if transform["type"] == "translation":
            shift = transform["translation"]
            lead = len(shift) - len(offset)  # a leading channel axis carries no translation
            for axis, (k, v) in enumerate(zip(offset, voxel, strict=True)):
                shift[lead + axis] += k * v
    return ome


def crop_one(src: Path, like: Path, dst: Path) -> None:
    s_group = zarr.open_group(str(src), mode="r")
    l_group = zarr.open_group(str(like), mode="r")
    s, l = s_group["s0"], l_group["s0"]
    s_attrs, l_attrs = dict(s_group.attrs.asdict()), dict(l_group.attrs.asdict())
    target = tuple(int(v) for v in l.shape)
    offset = _offset(src.name, s_attrs, l_attrs)
    lead = s.ndim - len(offset)  # a leading channel axis is taken whole
    start = [0] * lead + offset
    if s.ndim != len(target) or any(
        a + t > u for a, t, u in zip(start, target, s.shape, strict=True)
    ):
        raise SystemExit(
            f"{src.name}: like-shape {target} at offset {start} does not fit the source "
            f"{tuple(s.shape)}"
        )
    started = time.perf_counter()
    d_group = zarr.open_group(str(dst), mode="w", zarr_format=3)
    d = d_group.create_array(name="s0", shape=target, dtype=s.dtype, chunks=tuple(int(c) for c in s.chunks))
    tail = tuple(slice(a, a + t) for a, t in zip(start[1:], target[1:], strict=True))
    for a in range(0, target[0], SLAB):
        b = min(a + SLAB, target[0])
        d[a:b] = s[(slice(start[0] + a, start[0] + b), *tail)]
    # Where the crop now sits, so the position a guard compares is the real one.
    attrs = dict(s_attrs)
    attrs["native_box"] = l_attrs["native_box"]
    attrs["ome"] = _shift_translation(s_attrs["ome"], offset, s_attrs["effective_voxel_nm"])
    attrs["cropped_from_shape"] = [int(v) for v in s.shape]
    attrs["cropped_offset"] = start
    attrs["cropped_like"] = str(like)
    d_group.attrs.update(attrs)
    level_attrs = dict(s.attrs.asdict())
    level_attrs.update(native_box=l_attrs["native_box"])
    d.attrs.update(level_attrs)
    check = min(SLAB, target[0])
    if not np.array_equal(d[:check], s[(slice(start[0], start[0] + check), *tail)]):
        raise SystemExit(f"{src.name}: verification slab differs")
    print(f"  {src.name}: {tuple(s.shape)} -> {target} from offset {start} in "
          f"{time.perf_counter() - started:.0f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--like", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()
    stores = sorted(p for p in args.src.glob("*.zarr"))
    if not stores:
        raise SystemExit(f"no *.zarr under {args.src}")
    for p in stores:
        like = args.like / p.name
        if not like.exists():
            raise SystemExit(f"no counterpart {like}")
    args.dst.mkdir(parents=True, exist_ok=True)
    for p in stores:
        crop_one(p, args.like / p.name, args.dst / p.name)
    print(f"cropped {len(stores)} stores from {args.src} to {args.dst}, on the grid of {args.like}")


if __name__ == "__main__":
    main()
