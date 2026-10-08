"""The dense default: weighted-average tiling for any strategy whose output is a fixed field.

`predict_volume` is the path every scored run in this repo was produced with -- tile, run
`logits`, `squash`, blend overlapping tiles by weighted average -- and `DensePredictor` is that same
path behind the `VolumePredictor` interface, so that the entrypoint holds every strategy to one
contract. `select_predictor` is the dispatch: a strategy's own predictor if it declares one, else
this. `tests/unit/test_predict.py` pins the wrapper as byte-identical to the bare function.

`accumulate` and `blend` are the path's two halves -- sum the weighted tiles over a box, divide --
exposed so that `prediction.blockwise` runs the very same code one block at a time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .grid import AxisOrder, VolumeGrid
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


def overlapping(
    tiles: list[tuple[tuple[int, ...], tuple[int, ...]]],
    patch: list[int],
    low: tuple[int, ...],
    shape: tuple[int, ...],
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """The tiles whose output window meets the box [low, low + shape), in lattice order."""
    return [
        (native, out) for native, out in tiles
        if all(o < lo + s and o + p > lo
               for o, p, lo, s in zip(out, patch, low, shape, strict=True))
    ]


@torch.no_grad()
def accumulate(
    algorithm: Any,
    grid: VolumeGrid,
    device: torch.device,
    low: tuple[int, ...],
    shape: tuple[int, ...],
    *,
    handle: Any = None,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Every tile's weighted prediction summed over the output box [low, low + shape).

    Returns (weighted sum, weight sum, tiles run): float32, (channels, *shape) and (1, *shape).
    Only the tiles that meet the box are run, in lattice order, and each adds only its part inside
    the box. So every voxel receives the same tiles' values, added in the same order, as it does
    when the box is the whole lattice: a box's sums equal that part of the whole region's sums bit
    for bit, which is what lets `prediction.blockwise` cut a region into blocks.
    """
    handle = grid.image_handle() if handle is None else handle
    channels = int(algorithm.prediction_channels)
    total = np.zeros((channels, *shape), dtype=np.float32)
    weight = np.zeros((1, *shape), dtype=np.float32)
    single = blend_weight(tuple(grid.patch))[None]
    # Tiles reach the model in the axis order it was trained on, and its output returns to the
    # store's; an affinity channel is re-indexed with its axis. A no-op when the two orders agree.
    order = AxisOrder.of(grid, algorithm)
    remap = (
        order.channel_order(algorithm.offsets[:channels])
        if algorithm.prediction_kind == "affinity" and not order.identity else None
    )

    tiles = overlapping(grid.tiles, grid.patch, low, shape)
    for index, (native, out) in enumerate(tiles):
        tile = np.ascontiguousarray(order.to_model(grid.read_image(handle, native)))
        volumes = torch.from_numpy(tile[None, None]).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = algorithm.logits(volumes)
        stored = order.to_storage(algorithm.squash(logits.float())[0], leading=1)
        if remap is not None:
            stored = stored[remap]
        stored = stored.cpu().numpy()

        inside = tuple(
            slice(max(o, lo) - o, min(o + p, lo + s) - o)
            for o, p, lo, s in zip(out, grid.patch, low, shape, strict=True)
        )
        window = tuple(
            slice(max(o, lo) - lo, min(o + p, lo + s) - lo)
            for o, p, lo, s in zip(out, grid.patch, low, shape, strict=True)
        )
        part = (slice(None), *inside)
        total[(slice(None), *window)] += stored[part] * single[part]
        weight[(slice(None), *window)] += single[part]
        if progress and ((index + 1) % 25 == 0 or index + 1 == len(tiles)):
            print(f"  {index + 1}/{len(tiles)}", flush=True)
    return total, weight, len(tiles)


def blend(total: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """The weighted mean of `accumulate`'s sums, as float16.

    Divided a slab at a time. `(total / weight).astype(f16)` materialises a full float32 quotient
    before downcasting, which at 7 gigavoxels over 6 channels is an extra 170 GB on top of the
    accumulator. Slabbing costs nothing numerically and bounds the temporary at one slab.
    """
    blended = np.empty(total.shape, dtype=np.float16)
    slab = max(1, total.shape[1] // 16)
    for start in range(0, total.shape[1], slab):
        stop = start + slab
        blended[:, start:stop] = total[:, start:stop] / np.maximum(weight[:, start:stop], 1e-8)
    return blended


def argmax_classes(total: np.ndarray) -> np.ndarray:
    """The channel each voxel's summed scores rank first, as the narrowest unsigned type.

    The argmax of `accumulate`'s weighted sums is that of their weighted mean -- a voxel's weight is
    one positive number shared by all its channels -- so the float16 mean `blend` would write is
    never made, and its rounding cannot move a near-tie. Taken a slab at a time, as `blend` divides,
    so argmax's int64 indices never span the whole region.
    """
    labels = np.empty(total.shape[1:], dtype=np.min_scalar_type(total.shape[0] - 1))
    slab = max(1, total.shape[1] // 16)
    for start in range(0, total.shape[1], slab):
        labels[start:start + slab] = total[:, start:start + slab].argmax(axis=0)
    return labels


@torch.no_grad()
def predict_volume(
    algorithm: Any, grid: VolumeGrid, device: torch.device, *, argmax: bool = False
) -> np.ndarray:
    """Blended predictions over the aligned region -> (channels, *output) float16.

    Channels, the squashing applied before blending, and the forward pass all come from the
    algorithm, so this serves any dense-output algorithm without knowing which one it has.

    Squashed *before* blending, not after: overlapping tiles are averaged in the stored
    representation. A weighted mean of logits is not the logit of a weighted mean of probabilities,
    so the order is part of the convention and is recorded with the data.

    With `argmax`, the channel each voxel's blended scores rank first instead, (*output) unsigned
    (`argmax_classes`): for class scores, the labelling they imply, at a fraction of the size.
    """
    print(f"{len(grid.tiles)} tiles of {grid.patch} -> output {grid.output_shape} at "
          f"{[round(v, 3) for v in grid.effective_voxel]} nm/voxel ({grid.axes}); "
          f"{int(algorithm.prediction_channels)} channels of {algorithm.prediction_kind}; "
          f"lattice covers {100 * grid.box_coverage:.1f}% of the annotated box, centred",
          flush=True)
    total, weight, _ = accumulate(
        algorithm, grid, device, (0,) * len(grid.output_shape), tuple(grid.output_shape)
    )
    return argmax_classes(total) if argmax else blend(total, weight)


#: What a dense strategy declares. `input_axes` is the axis order it was trained on (tiles are
#: handed over in that order, `AxisOrder`); an affinity strategy also has `offsets`, one a channel.
DENSE_PROTOCOL = ("logits", "squash", "squash_convention", "prediction_kind", "prediction_channels",
                  "input_axes")


class DensePredictor:
    """The whole-volume prediction every dense-output strategy gets for free.

    A thin wrapper -- `run` is `predict_volume` plus the attrs the artifact needs -- so that the
    entrypoint can hold every strategy to one interface (`VolumePredictor`) without the dense path
    itself moving. That path is what every scored run in this repo was produced with, and
    `tests/unit/test_predict.py` pins this wrapper's output as byte-identical to calling it
    directly.

    With `argmax`, a class-score strategy's prediction is written as the labelling it implies,
    `kind = "class_labels"` (`predict_volume(argmax=True)`).
    """

    def __init__(self, algorithm: Any, argmax: bool = False) -> None:
        missing = [name for name in DENSE_PROTOCOL if not hasattr(algorithm, name)]
        if getattr(algorithm, "prediction_kind", None) == "affinity" and not hasattr(
                algorithm, "offsets"):
            missing.append("offsets")
        if missing:
            raise SystemExit(
                f"{type(algorithm).__name__} lacks {missing}. Prediction over a whole volume needs "
                "an algorithm that declares what its output means and how to produce it; see "
                "`affinity_seg` for the members required, or have the strategy return its own "
                "`volume_predictor()`."
            )
        if argmax and algorithm.prediction_kind != "class_scores":
            raise SystemExit(
                f"--argmax ranks class scores, but {type(algorithm).__name__} predicts "
                f"{algorithm.prediction_kind!r}, whose channels are not classes"
            )
        self.algorithm = algorithm
        self.argmax = argmax

    @property
    def kind(self) -> str:
        return "class_labels" if self.argmax else str(self.algorithm.prediction_kind)

    @property
    def attrs(self) -> dict[str, Any]:
        """What a reader needs beside the data to interpret it, whichever path wrote the data.

        `model_axes` is the axis order the model was handed its tiles in; the artifact itself is
        in the store's. An affinity artifact also states its `offsets`, one per channel along the
        store's axes, so a consumer can check the layout it assumes rather than trust it. A class
        labelling (`argmax`) states how many `classes` it was ranked from, and `background_id`,
        class 0, which mia-evals requires of a labelling.
        """
        channels = int(self.algorithm.prediction_channels)
        model_axes = "".join(a for a in self.algorithm.input_axes if a in "xyz")
        if self.argmax:
            return {
                "convention": f"argmax of the {self.algorithm.squash_convention}, blended in "
                              "that space",
                "classes": channels,
                "background_id": 0,
                "model_axes": model_axes,
            }
        attrs: dict[str, Any] = {
            "convention": f"{self.algorithm.squash_convention}, blended in that space",
            "channels": channels,
            "model_axes": model_axes,
        }
        if self.algorithm.prediction_kind == "affinity":
            attrs["offsets"] = [[int(v) for v in o] for o in self.algorithm.offsets[:channels]]
        return attrs

    def run(self, grid: VolumeGrid, device: torch.device) -> VolumePrediction:
        return VolumePrediction(
            array=predict_volume(self.algorithm, grid, device, argmax=self.argmax),
            kind=self.kind, attrs=self.attrs,
        )


def select_predictor(algorithm: Any) -> VolumePredictor:
    """The strategy's own predictor if it declares one, else the dense default.

    One line, but it is the whole point: `predict.py` never asks *which* strategy it holds. A
    strategy with a non-dense output answers `volume_predictor()`, and one without inherits the
    path that was already here. Neither side needs to know the other exists.
    """
    return algorithm.volume_predictor() or DensePredictor(algorithm)
