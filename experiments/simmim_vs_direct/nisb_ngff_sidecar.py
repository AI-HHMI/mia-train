"""Give a natively-published NISB variant the OME-NGFF metadata `miao` needs, without copying it.

NISB publishes each cube as a plain zarr v2 group -- `data.zarr` holding `img` and `seg` as flat
arrays -- which is exactly what the benchmark's own BANIS code reads. `miao` instead expects
OME-NGFF: a multiscale group whose `.zattrs` names the axes and gives each level's voxel size. The
pixel data is identical and perfectly usable; the only thing absent is a few hundred bytes of JSON
describing it.

So this writes that JSON, and points it at the arrays that already exist:

    train_100-ngff/train/seed0.zarr/
        img/.zattrs          axes [x, y, z, c], one level at 9 x 9 x 20 nm
        img/s0       ------> .../train_100/train/seed0/data.zarr/img
        labels/seg/.zattrs   axes [x, y, z], same voxel size
        labels/seg/s0 -----> .../train_100/train/seed0/data.zarr/seg
        skeleton.pkl -------> .../train_100/train/seed0/skeleton.pkl

3.5 KB per cube rather than 15 GB. Nothing under the source tree is read, moved, or modified, so
the published data stays exactly as downloaded and a sidecar can be deleted and regenerated freely.

The resulting layout deliberately mirrors the `base` variant a colleague converted earlier --
`<seed>.zarr` with `img`, `labels/seg` and `skeleton.pkl` -- so one miao config shape serves both.

**Voxel size is asserted, not discovered.** The published cubes carry no scale metadata at all, so
`--voxel-size` supplies it. The default of 9 x 9 x 20 nm is not a guess: `base` declares it, and
`base/train/seed0` was verified byte-identical to `train_100/train/seed0` over a sampled block,
which also fixes the axis order the arrays are stored in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# NISB's published cubes: `img` carries a trailing channel axis, `seg` does not.
IMAGE_AXES = "xyzc"
LABEL_AXES = "xyz"
DEFAULT_VOXEL_SIZE = (9.0, 9.0, 20.0)  # nanometres, x y z -- see the module docstring
SOURCE_GROUP = "data.zarr"
SKELETON = "skeleton.pkl"

README = """\
# OME-NGFF metadata sidecar -- NOT a copy of the data

Every `s0` in this tree is a **symlink** into the sibling `{source}/` directory. This tree holds
only `.zattrs`/`.zgroup` metadata: a few kilobytes per cube, against ~15 GB of real data each.

*Deleting `{source}/` destroys the data.* This directory cannot stand in for it.

It exists because NISB publishes its cubes as plain zarr v2 (`data.zarr` with flat `img`/`seg`
arrays -- what the benchmark's BANIS code reads), while `miao` needs OME-NGFF multiscale metadata.
The pixel data was always fine; only the description was missing.

Regenerate with:

    python experiments/simmim_vs_direct/nisb_ngff_sidecar.py {source} --out {out}

Note this differs from `liconn-ngff`, which *is* a full data copy.
"""


def multiscale_attrs(axes: str, voxel_size: tuple[float, ...]) -> dict:
    """OME-NGFF `multiscales` for a single full-resolution level named `s0`.

    One level, because that is all NISB publishes. A coarser `resolutions` request in a miao
    config will therefore fail rather than silently downsample -- which is the right failure: the
    pyramid genuinely is not there.

    The channel axis gets scale 1.0. It is not a spatial dimension, but OME requires one entry per
    axis, and miao reads the spatial ones by name.
    """
    scale = [1.0 if name == "c" else voxel_size["xyz".index(name)] for name in axes]
    return {
        "multiscales": [
            {
                "version": "0.4",
                "name": "sidecar",
                "axes": [
                    {"name": name, "type": "channel"}
                    if name == "c"
                    else {"name": name, "type": "space", "unit": "nanometer"}
                    for name in axes
                ],
                "datasets": [
                    {"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": scale}]}
                ],
            }
        ]
    }


def _write_group(path: Path, attrs: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".zgroup").write_text(json.dumps({"zarr_format": 2}, indent=1))
    if attrs is not None:
        (path / ".zattrs").write_text(json.dumps(attrs, indent=1))


def _link(link: Path, target: Path) -> None:
    """Point `link` at `target`, replacing any previous link so re-running is safe.

    Absolute targets: a relative one would break the moment the sidecar tree is moved, and it
    would break by silently resolving to nothing rather than by failing loudly.
    """
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.resolve())


def build_cube(source: Path, destination: Path, voxel_size: tuple[float, ...]) -> None:
    """One published cube directory -> one OME-NGFF `<seed>.zarr` sidecar."""
    data = source / SOURCE_GROUP
    if not (data / "img").is_dir():
        raise FileNotFoundError(f"{data} has no 'img' array; is this a NISB cube?")

    _write_group(destination)
    _write_group(destination / "img", multiscale_attrs(IMAGE_AXES, voxel_size))
    _link(destination / "img" / "s0", data / "img")

    _write_group(destination / "labels")
    _write_group(destination / "labels" / "seg", multiscale_attrs(LABEL_AXES, voxel_size))
    _link(destination / "labels" / "seg" / "s0", data / "seg")

    if (source / SKELETON).exists():
        # Kept beside the arrays exactly as `base` does, so the eval tooling finds it in the
        # same place regardless of which variant it was handed.
        _link(destination / SKELETON, source / SKELETON)


def build_variant(source_root: Path, out_root: Path, voxel_size: tuple[float, ...]) -> int:
    """Every cube under every split of one variant. Returns how many were written."""
    written = 0
    for split in sorted(p for p in source_root.iterdir() if p.is_dir()):
        for cube in sorted(p for p in split.iterdir() if (p / SOURCE_GROUP).is_dir()):
            build_cube(cube, out_root / split.name / f"{cube.name}.zarr", voxel_size)
            written += 1
    if written:
        (out_root / "README.md").write_text(
            README.format(source=source_root.name, out=out_root)
        )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="a published variant, e.g. .../nisb/train_100")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output tree; defaults to a '<source>-ngff' sibling, matching the existing naming",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        default=list(DEFAULT_VOXEL_SIZE),
        metavar=("X", "Y", "Z"),
        help="nanometres per voxel in x y z order (default: 9 9 20, as declared by `base`)",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_dir():
        raise SystemExit(f"no such variant: {source}")
    out = (args.out or source.with_name(f"{source.name}-ngff")).resolve()
    if out == source:
        raise SystemExit("--out must differ from the source; this never writes into the source")

    written = build_variant(source, out, tuple(args.voxel_size))
    if not written:
        raise SystemExit(f"no cubes found under {source} (expected <split>/<seed>/{SOURCE_GROUP})")
    print(f"{written} cube(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
