"""Crop one arm's prediction artifacts to another arm's extents, so both are scored over the same region.

The leaderboard scores a record over its artifact's extent clipped to the annotation, and it refuses
to rank records scored over different regions together. A model labelled with 128-voxel windows
(arm 9) covers more of each volume than the 256-voxel lattice of every other arm -- the tail that
does not complete a 256 stride -- so its row lands in a table of its own. Cropping its artifacts
(prediction and co-registered truth alike) to the 256 lattice's shapes puts it on the common region;
every artifact starts at origin 0, so the OME translation stays valid and only the tail is dropped.

    python experiments/sam_lmd_v1/crop_artifacts.py --src eval/arm9_r0/test --like eval/arm8_r0/test --dst eval/arm9_r0_lattice256/test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import zarr

SLAB = 64


def crop_one(src: Path, like: Path, dst: Path) -> None:
    s_group = zarr.open_group(str(src), mode="r")
    l_group = zarr.open_group(str(like), mode="r")
    s, l = s_group["s0"], l_group["s0"]
    target = tuple(int(v) for v in l.shape)
    if any(t > u for t, u in zip(target, s.shape, strict=True)):
        raise SystemExit(f"{src.name}: like-shape {target} exceeds the source {tuple(s.shape)} on some axis")
    origin_s = list(s_group.attrs.get("origin", [0] * len(target)))
    origin_l = list(l_group.attrs.get("origin", [0] * len(target)))
    if origin_s != origin_l:
        raise SystemExit(f"{src.name}: origins differ ({origin_s} vs {origin_l}); cropping the tail alone is not enough")
    started = time.perf_counter()
    d_group = zarr.open_group(str(dst), mode="w", zarr_format=3)
    d = d_group.create_array(name="s0", shape=target, dtype=s.dtype, chunks=tuple(int(c) for c in s.chunks))
    tail = tuple(slice(0, t) for t in target[1:])
    for a in range(0, target[0], SLAB):
        b = min(a + SLAB, target[0])
        d[a:b] = s[(slice(a, b), *tail)]
    attrs = dict(s_group.attrs.asdict())
    attrs["cropped_from_shape"] = [int(v) for v in s.shape]
    attrs["cropped_like"] = str(like)
    d_group.attrs.update(attrs)
    d.attrs.update(dict(s.attrs.asdict()))
    check = min(SLAB, target[0])
    if not np.array_equal(d[:check], s[(slice(0, check), *tail)]):
        raise SystemExit(f"{src.name}: verification slab differs")
    print(f"  {src.name}: {tuple(s.shape)} -> {target} in {time.perf_counter() - started:.0f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--like", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    stores = sorted(p for p in args.src.glob("*.zarr"))
    if not stores:
        raise SystemExit(f"no *.zarr under {args.src}")
    for p in stores:
        like = args.like / p.name
        if not like.exists():
            raise SystemExit(f"no counterpart {like}")
        crop_one(p, like, args.dst / p.name)
    print(f"cropped {len(stores)} stores from {args.src} to {args.dst}, shapes of {args.like}")


if __name__ == "__main__":
    main()
