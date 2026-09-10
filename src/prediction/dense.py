"""The dense default: weighted-average tiling for any strategy whose output is a fixed field.

`predict_volume` is the path every scored run in this repo was produced with -- tile, run
`logits`, `squash`, blend overlapping tiles by weighted average -- and `DensePredictor` is that same
path behind the `VolumePredictor` interface, so that the entrypoint holds every strategy to one
contract. `select_predictor` is the dispatch: a strategy's own predictor if it declares one, else
this. `tests/unit/test_predict.py` pins the wrapper as byte-identical to the bare function.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .grid import VolumeGrid
from .types import VolumePrediction, VolumePredictor


def blend_weight(shape: tuple[int, ...]) -> np.ndarray:
    """Confidence of a tile's own prediction: low at its faces, high at its centre.

    A tile sees no context beyond its border, so its edge voxels are its worst; weighting
    overlapping predictions this way hides the seams that would otherwise cut objects at tile
    boundaries -- which connected components then reports as split errors.

    The chessboard distance to the outside, which for a box of ones padded by one voxel is the
    per-axis distance minimised over axes. Numerically identical to BANIS'
    `distance_transform_cdt` of that input, verified for cubic sizes 8 through 512, and generalised
    here to a per-axis shape so a non-cubic patch works.
    """
    rank = len(shape)
    weight: np.ndarray | None = None
    for axis, extent in enumerate(shape):
        ramp = (
            np.minimum(np.arange(extent), extent - 1 - np.arange(extent)) + 1
        ).astype(np.float32)
        view = [1] * rank
        view[axis] = -1
        broadcast = ramp.reshape(view)
        weight = broadcast if weight is None else np.minimum(weight, broadcast)
    assert weight is not None                                # rank >= 1 by construction
    return np.broadcast_to(weight, shape).astype(np.float32)


@torch.no_grad()
def predict_volume(algorithm: Any, grid: VolumeGrid, device: torch.device) -> np.ndarray:
    """Blended predictions over the aligned region -> (channels, *output) float16.

    Channels, the squashing applied before blending, and the forward pass all come from the
    algorithm, so this serves any dense-output algorithm without knowing which one it has.

    Squashed *before* blending, not after: overlapping tiles are averaged in the stored
    representation. A weighted mean of logits is not the logit of a weighted mean of probabilities,
    so the order is part of the convention and is recorded with the data.
    """
    handle = grid.image_handle()
    channels = int(algorithm.prediction_channels)
    total = np.zeros((channels, *grid.output_shape), dtype=np.float32)
    weight = np.zeros((1, *grid.output_shape), dtype=np.float32)
    single = blend_weight(tuple(grid.patch))[None]

    tiles = grid.tiles
    print(f"{len(tiles)} tiles of {grid.patch} -> output {grid.output_shape} at "
          f"{[round(v, 3) for v in grid.effective_voxel]} nm/voxel ({grid.axes}); "
          f"{channels} channels of {algorithm.prediction_kind}; "
          f"lattice covers {100 * grid.box_coverage:.1f}% of the annotated box, centred",
          flush=True)
    for index, (native, out) in enumerate(tiles):
        volumes = torch.from_numpy(grid.read_image(handle, native)[None, None]).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = algorithm.logits(volumes)
        stored = algorithm.squash(logits.float())[0].cpu().numpy()

        window = tuple(slice(o, o + p) for o, p in zip(out, grid.patch, strict=True))
        total[(slice(None), *window)] += stored * single
        weight[(slice(None), *window)] += single
        if (index + 1) % 25 == 0 or index + 1 == len(tiles):
            print(f"  {index + 1}/{len(tiles)}", flush=True)

    # Divided a slab at a time. `(total / weight).astype(f16)` materialises a full float32 quotient
    # before downcasting, which at 7 gigavoxels over 6 channels is an extra 170 GB on top of the
    # accumulator. Slabbing costs nothing numerically and bounds the temporary at one slab.
    blended = np.empty(total.shape, dtype=np.float16)
    slab = max(1, grid.output_shape[0] // 16)
    for start in range(0, total.shape[1], slab):
        stop = start + slab
        blended[:, start:stop] = total[:, start:stop] / np.maximum(weight[:, start:stop], 1e-8)
    return blended


DENSE_PROTOCOL = ("logits", "squash", "squash_convention", "prediction_kind", "prediction_channels")


class DensePredictor:
    """The whole-volume prediction every dense-output strategy gets for free.

    A thin wrapper -- `run` is `predict_volume` plus the attrs the artifact needs -- so that the
    entrypoint can hold every strategy to one interface (`VolumePredictor`) without the dense path
    itself moving. That path is what every scored run in this repo was produced with, and
    `tests/unit/test_predict.py` pins this wrapper's output as byte-identical to calling it
    directly.
    """

    def __init__(self, algorithm: Any) -> None:
        missing = [name for name in DENSE_PROTOCOL if not hasattr(algorithm, name)]
        if missing:
            raise SystemExit(
                f"{type(algorithm).__name__} lacks {missing}. Prediction over a whole volume needs "
                "an algorithm that declares what its output means and how to produce it; see "
                "`affinity_seg` for the members required, or have the strategy return its own "
                "`volume_predictor()`."
            )
        self.algorithm = algorithm

    def run(self, grid: VolumeGrid, device: torch.device) -> VolumePrediction:
        return VolumePrediction(
            array=predict_volume(self.algorithm, grid, device),
            kind=str(self.algorithm.prediction_kind),
            attrs={
                "convention": f"{self.algorithm.squash_convention}, blended in that space",
                "channels": int(self.algorithm.prediction_channels),
            },
        )


def select_predictor(algorithm: Any) -> VolumePredictor:
    """The strategy's own predictor if it declares one, else the dense default.

    One line, but it is the whole point: `predict.py` never asks *which* strategy it holds. A
    strategy with a non-dense output answers `volume_predictor()`, and one without inherits the
    path that was already here. Neither side needs to know the other exists.
    """
    return algorithm.volume_predictor() or DensePredictor(algorithm)
