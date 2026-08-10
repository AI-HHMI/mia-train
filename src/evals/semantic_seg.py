"""Semantic segmentation evaluation: tiled inference, orthoplane prediction, IoU.

Three things a training loop's validation step cannot do, and which decide whether a model is
actually useful:

**Whole volumes, not crops.** Training samples a window that fits in memory. A real evaluation
has to answer for every voxel of a crop, which means tiling with overlap and blending, so that
what is measured is the model rather than the crop size it was trained at. The alternative --
resizing a volume to fit -- changes the physical scale of every structure and fabricates label
values between class indices, so it is not offered here.

**A 2D model on a 3D volume.** `orthoplane` runs the model over every plane normal to x, then to
y, then to z, and averages the three sets of class scores per voxel. Each pass sees a different
2D cross-section of the same 3D structure, and averaging turns three partial views into one
consistent answer -- the standard way to use a 2D encoder on volumetric EM, and the reason
`cellmap_2d` keeps all three slice orientations in training.

**Metrics that survive class imbalance.** Everything is accumulated into one confusion matrix
over the whole evaluation set, so IoU and Dice are computed on totals rather than averaged over
batches. Per-batch IoU is biased by which classes happen to appear in a batch; the totals are not.
Background dominates these volumes, so pixel accuracy alone is close to meaningless -- it is
reported, but mean IoU over present classes is the number to read.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, cast

import torch
import torch.nn as nn
import torch.utils.data as data

from .base import BaseEvalTask
from .registry import EvalRegistry

AXES = ("x", "y", "z")


class SegmentationAlgorithm(Protocol):
    """What this task needs of the thing it evaluates.

    Spelled out because `nn.Module.__getattr__` types every attribute access as `Tensor | Module`,
    so a bare `model.logits(...)` is neither callable nor checkable. Naming the contract lets the
    hasattr guard below narrow to something with real types.
    """

    spatial_rank: int

    def logits(self, x: torch.Tensor) -> torch.Tensor: ...


def tile_starts(extent: int, tile: int, overlap: int) -> list[int]:
    """Window origins covering `extent`, the last pulled back to land inside.

    Pulling the final window back rather than padding means every position is predicted from real
    context; a padded edge would be predicted from invented data and then scored.
    """
    if tile >= extent:
        return [0]
    stride = max(tile - overlap, 1)
    starts = list(range(0, extent - tile + 1, stride))
    if starts[-1] != extent - tile:
        starts.append(extent - tile)
    return starts


def blend_weight(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    """Confidence of a tile's own prediction: low at its faces, high at its centre.

    A tile sees no context beyond its border, so its edge voxels are its worst. Weighting
    overlapping predictions this way removes the seams that would otherwise appear as stripes of
    misclassification along tile boundaries.
    """
    weight = torch.ones(tuple(shape), device=device)
    for axis, extent in enumerate(shape):
        ramp = torch.minimum(
            torch.arange(extent, device=device), torch.arange(extent - 1, -1, -1, device=device)
        ).float() + 1.0
        weight = weight * ramp.reshape([-1 if i == axis else 1 for i in range(len(shape))])
    return weight


@torch.no_grad()
def tiled_logits(
    predict: Callable[[torch.Tensor], torch.Tensor],
    images: torch.Tensor,
    tile: Sequence[int],
    overlap: int,
    num_classes: int,
    device: torch.device,
    batch_size: int = 1,
) -> torch.Tensor:
    """(N, *spatial) images -> (N, num_classes, *spatial) blended class scores, on CPU.

    `images` is a batch of independent inputs of the same rank: one volume (N=1) for a 3D model,
    or many planes for a 2D one. `predict` maps (B, 1, *tile) to (B, num_classes, *tile).

    The accumulator stays on the host and only tiles go to the device, so GPU memory is bounded by
    the tile rather than by the volume -- which matters because scores are `num_classes` deep, and
    64 classes over a 500-cube is 32 GiB in float32 before anything else.
    """
    spatial = tuple(images.shape[1:])
    tile = tuple(min(t, s) for t, s in zip(tile, spatial, strict=True))
    total = torch.zeros((images.shape[0], num_classes, *spatial), dtype=torch.float32)
    weight = torch.zeros((1, 1, *spatial), dtype=torch.float32)
    single = blend_weight(tile, torch.device("cpu"))[None, None]

    starts = [tile_starts(s, t, overlap) for s, t in zip(spatial, tile, strict=True)]
    corners: list[tuple[int, ...]] = [()]
    for axis_starts in starts:
        corners = [c + (s,) for c in corners for s in axis_starts]

    # The tiling is identical for every image, so the weight map is accumulated once over the
    # corners rather than once per (image, corner). Doing it inside the job loop would divide by
    # N times too much -- invisible for a single volume, and exactly wrong for a stack of planes.
    for c in corners:
        window = tuple(slice(o, o + t) for o, t in zip(c, tile, strict=True))
        weight[0, 0][window] += single[0, 0]

    jobs = [(n, c) for n in range(images.shape[0]) for c in corners]
    for begin in range(0, len(jobs), batch_size):
        chunk = jobs[begin : begin + batch_size]
        windows = torch.stack([
            images[n][tuple(slice(o, o + t) for o, t in zip(c, tile, strict=True))]
            for n, c in chunk
        ])
        scores = predict(windows[:, None].to(device)).float().cpu()
        for (n, c), score in zip(chunk, scores, strict=True):
            window = (slice(None), *(slice(o, o + t) for o, t in zip(c, tile, strict=True)))
            total[n][window] += score * single[0]
    return total / weight.clamp_min(1e-8)


@torch.no_grad()
def orthoplane_logits(
    predict: Callable[[torch.Tensor], torch.Tensor],
    volume: torch.Tensor,
    tile: Sequence[int],
    overlap: int,
    num_classes: int,
    device: torch.device,
    batch_size: int = 8,
    axes: Sequence[str] = AXES,
) -> torch.Tensor:
    """(X, Y, Z) volume + a 2D predictor -> (num_classes, X, Y, Z) averaged class scores.

    For each named axis the volume is sliced into planes normal to it, every plane is predicted
    (tiled if it is larger than one tile), and the results are put back. The three stacks are then
    averaged in *score* space rather than after argmax: averaging labels would be a vote between
    three hard answers, discarding how confident each pass was, and ties would need arbitrary
    resolution.
    """
    accumulated = torch.zeros((num_classes, *volume.shape), dtype=torch.float32)
    for axis in axes:
        index = AXES.index(axis)
        planes = volume.movedim(index, 0)                      # (n_planes, H, W)
        scores = tiled_logits(
            predict, planes, tile, overlap, num_classes, device, batch_size
        )                                                       # (n_planes, K, H, W)
        # (n_planes, K, d1, d2) -> (K, n_planes, d1, d2) -> put the plane axis back where it
        # came from, so every pass lands in the same (K, X, Y, Z) frame before averaging.
        accumulated += scores.movedim(1, 0).movedim(1, index + 1)
    return accumulated / len(axes)


class ConfusionMatrix:
    """Totals over the whole evaluation set, from which every metric is derived.

    Kept as counts rather than running means because IoU is a ratio of sums: averaging per-volume
    IoUs would weight a crop containing three voxels of a class the same as one containing three
    million, and would have to invent a value for volumes where a class is absent.
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.counts = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, predicted: torch.Tensor, target: torch.Tensor, ignore_index: int) -> None:
        valid = target != ignore_index
        p, t = predicted[valid].flatten(), target[valid].flatten()
        indices = t * self.num_classes + p
        self.counts += torch.bincount(
            indices, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes).cpu()

    def metrics(self) -> dict[str, float]:
        counts = self.counts.double()
        true_positive = counts.diag()
        actual, predicted = counts.sum(1), counts.sum(0)
        union = actual + predicted - true_positive
        present = actual > 0                     # classes the ground truth actually contains

        iou = torch.where(union > 0, true_positive / union.clamp_min(1), torch.zeros_like(union))
        dice = torch.where(
            (actual + predicted) > 0,
            2 * true_positive / (actual + predicted).clamp_min(1),
            torch.zeros_like(union),
        )
        result = {
            "pixel_accuracy": float(true_positive.sum() / counts.sum().clamp_min(1)),
            "mean_iou": float(iou[present].mean()) if present.any() else 0.0,
            "mean_dice": float(dice[present].mean()) if present.any() else 0.0,
            "classes_present": float(present.sum()),
        }
        for klass in torch.nonzero(present).flatten().tolist():
            result[f"iou/class_{klass}"] = float(iou[klass])
        return result


@EvalRegistry.register("semantic_seg")
class SemanticSegmentationEval(BaseEvalTask):
    """Score a trained semantic segmentation algorithm over whole volumes.

    `mode` selects how the model is applied:

      "native"     - tile the model over the data at its own rank (2D on planes, 3D on volumes).
      "orthoplane" - a 2D model over a 3D volume, averaging the x, y and z passes.

    "orthoplane" is only meaningful for a 2D model given 3D data, and is rejected otherwise
    rather than silently doing something else: applying it to a 3D model would feed it planes it
    cannot consume, and applying it to 2D data has no third axis to average over.
    """

    def __init__(
        self,
        num_classes: int = 64,
        tile: Sequence[int] = (256, 256, 256),
        overlap: int = 32,
        mode: str = "native",
        batch_size: int = 1,
        ignore_index: int = -1,
        axes: Sequence[str] = AXES,
        device: str | None = None,
    ) -> None:
        super().__init__()
        if mode not in ("native", "orthoplane"):
            raise ValueError(f"mode must be 'native' or 'orthoplane', got {mode!r}")
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        unknown = sorted(set(axes) - set(AXES))
        if unknown:
            raise ValueError(f"axes must be drawn from x, y, z; got {unknown}")
        self.num_classes = num_classes
        self.tile = tuple(int(t) for t in tile)
        self.overlap = overlap
        self.mode = mode
        self.batch_size = batch_size
        self.ignore_index = ignore_index
        self.axes = tuple(axes)
        self.device = device

    def evaluate(self, model: nn.Module, dataloader: data.DataLoader) -> dict[str, float]:
        """`model` is the trained algorithm; it must expose `logits` and `spatial_rank`."""
        if not hasattr(model, "logits"):
            raise TypeError(
                f"{type(model).__name__} has no `logits` method; this task evaluates a "
                "`semantic_seg` algorithm, not a bare encoder"
            )
        algorithm = cast(SegmentationAlgorithm, model)
        rank = int(algorithm.spatial_rank)
        if self.mode == "orthoplane" and rank != 2:
            raise ValueError(
                f"mode='orthoplane' feeds the model 2D planes, but this algorithm is {rank}D. "
                "Use mode='native' for a 3D model."
            )
        if len(self.tile) != rank:
            raise ValueError(
                f"tile {self.tile} has {len(self.tile)} entries but the model is {rank}D"
            )

        device = torch.device(
            self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model.eval().to(device)
        confusion = ConfusionMatrix(self.num_classes)

        def predict(windows: torch.Tensor) -> torch.Tensor:
            return algorithm.logits(windows)

        for batch in dataloader:
            # (B, L, C, *spatial) images and (B, L, *spatial) labels, as the datasets emit them.
            images = batch["img"][:, 0, 0]
            labels = batch["label"][:, 0].long()
            for image, label in zip(images, labels, strict=True):
                if self.mode == "orthoplane":
                    scores = orthoplane_logits(
                        predict, image, self.tile, self.overlap, self.num_classes,
                        device, self.batch_size, self.axes,
                    )
                else:
                    scores = tiled_logits(
                        predict, image[None], self.tile, self.overlap, self.num_classes,
                        device, self.batch_size,
                    )[0]
                confusion.update(scores.argmax(0), label, self.ignore_index)

        return confusion.metrics()
