"""Writing a prediction (or its ground truth) as a single-level OME-Zarr artifact.

The array goes in `<name>.zarr/s0` and the group carries OME-NGFF 0.5 `multiscales` metadata with
one dataset: the axes in storage order with nanometre units, the lattice voxel as the `scale`, and
the physical position of the first voxel as the `translation`. That is the layout the corpus's own
stores use, so a viewer that opens the raw volume and this artifact places both in the same
physical frame without being told anything -- which a bare array cannot do, because it has no way
to say where in the volume it sits or how big its voxels are.

The translation follows the stores' convention (a level's translation is the centre of its first
voxel, with the native level's first voxel at the store's own level-0 translation): lattice voxel
0 spans native voxels `[low, low + v/e)` of size `v`, so its centre is `low * v + (e - v) / 2`.
For a lattice at the native voxel size this is just `low * v`.

Everything a scorer or a reader needs (`kind`, `background_id`, provenance) is written on the
group AND repeated on the `s0` array, so a tool that opens the array directly still finds it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr

from prediction.grid import VolumeGrid

LEVEL = "s0"


def ome_geometry(grid: VolumeGrid) -> dict[str, Any]:
    """`axes`, `voxel_nm`, `translation_nm` of the grid's output lattice, storage order."""
    info = grid.info
    level_translation = info.image_meta.scales[grid.image_level].translation_or_zeros()
    origin = [float(level_translation[i]) for i in info.img_spatial_idx]
    lows = [low for low, _ in grid.native_box()]
    translation = [
        low * native + (lattice - native) / 2 + shift
        for low, native, lattice, shift in zip(
            lows, grid.image_voxel, grid.effective_voxel, origin, strict=True
        )
    ]
    return {
        "axes": grid.axes,
        "voxel_nm": [float(v) for v in grid.effective_voxel],
        "translation_nm": [float(t) for t in translation],
    }


def multiscales(axes: str, voxel_nm: list[float], translation_nm: list[float],
                channels: bool, name: str) -> dict[str, Any]:
    """OME-NGFF 0.5 `multiscales` for one level, with a leading channel axis when asked."""
    axis_list: list[dict[str, str]] = [
        {"name": a, "type": "space", "unit": "nanometer"} for a in axes
    ]
    scale, shift = list(voxel_nm), list(translation_nm)
    if channels:
        axis_list.insert(0, {"name": "c", "type": "channel"})
        scale.insert(0, 1.0)
        shift.insert(0, 0.0)
    return {
        "version": "0.5",
        "multiscales": [{
            "name": name,
            "axes": axis_list,
            "datasets": [{
                "path": LEVEL,
                "coordinateTransformations": [
                    {"type": "scale", "scale": scale},
                    {"type": "translation", "translation": shift},
                ],
            }],
        }],
    }


def write_ome_artifact(
    path: Path,
    array: np.ndarray,
    *,
    axes: str,
    voxel_nm: list[float],
    translation_nm: list[float],
    attrs: dict[str, Any],
    chunks: tuple[int, ...] | None = None,
) -> Path:
    """Write `array` as `<path>/s0` inside an OME-Zarr 0.5 group carrying `attrs`."""
    if array.ndim not in (len(axes), len(axes) + 1):
        raise ValueError(
            f"array of rank {array.ndim} does not fit axes {axes!r} (with or without a leading "
            "channel axis)"
        )
    channels = array.ndim == len(axes) + 1
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    level = group.create_array(
        name=LEVEL, shape=array.shape, dtype=array.dtype,
        chunks=chunks or tuple(min(256, s) for s in array.shape),
    )
    level[:] = array
    group.attrs.update(
        ome=multiscales(axes, voxel_nm, translation_nm, channels, Path(path).name), **attrs
    )
    level.attrs.update(**attrs)
    return Path(path)
