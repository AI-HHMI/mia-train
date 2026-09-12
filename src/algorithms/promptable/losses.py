"""The mask losses, and the reduction that makes a model ambiguity-aware.

All four functions take `(P, K, *spatial)` logits against `(P, K, *spatial)` targets -- `P` prompts
each producing `K` candidate masks -- and reduce over space only, so a caller keeps one number per
candidate. That shape is what `best_of` needs, and it is the whole reason the reduction is not
folded into the losses.

Targets are *soft*, in [0, 1], because they arrive area-pooled from voxel resolution
(`targets.pooled_masks`). Focal and dice both accept that reading without modification; the IoU the
prediction head is trained against does not, so it thresholds.

Every function takes an optional `weight` of the targets' shape (broadcastable), the labelled
fraction of each cell from `targets.pooled_known`: a cell of unknown voxels weighs nothing, so it
is neither penalised nor rewarded and does not count in an IoU. `None` is all ones, and every
function reproduces its unweighted value exactly under all-ones weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _weights(targets: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    return torch.ones_like(targets) if weight is None else weight.expand_as(targets)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted mean focal loss per candidate mask -> `(P, K)`.

    The reference's weights. `gamma` down-weights voxels the model already gets right, which in a
    3D crop is nearly all of them: an object occupying 1% of the volume leaves 99% easy background,
    and plain cross-entropy would spend its gradient there.
    """
    probabilities = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    agreement = probabilities * targets + (1 - probabilities) * (1 - targets)
    loss = cross_entropy * (1 - agreement).pow(gamma)
    if alpha >= 0:
        loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))
    w = _weights(targets, weight).flatten(2)
    return (loss.flatten(2) * w).sum(-1) / w.sum(-1).clamp_min(1e-6)


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1.0,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft dice loss per candidate mask -> `(P, K)`, over the weighted cells.

    Scale-free where focal is not: it is a ratio of overlaps, so a 200-voxel object and a
    200,000-voxel one contribute comparably. That is what keeps the small objects in a microscopy
    crop from being optimised away, and it is why the reference combines the two rather than
    picking one.
    """
    w = _weights(targets, weight).flatten(2)
    probabilities = logits.sigmoid().flatten(2) * w
    flat = targets.flatten(2) * w
    intersection = (probabilities * flat).sum(-1)
    denominator = probabilities.sum(-1) + flat.sum(-1)
    return 1 - (2 * intersection + eps) / (denominator + eps)


def mask_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_threshold: float = 0.5,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """IoU of the thresholded prediction against the thresholded target -> `(P, K)`.

    Both the quantity the IoU head is trained to predict and the quantity a run is judged by, so
    they cannot disagree about what a "0.8 mask" is. Thresholded rather than soft because that is
    what a consumer of the mask will do with it.

    An empty target scores 0 against an empty prediction rather than 1. That is deliberate: the
    only way to get an empty target here is a padding draw, which is excluded from the loss anyway,
    and rewarding "predict nothing" would otherwise be the easiest way to a perfect score.
    """
    predicted = logits > 0
    actual = targets > target_threshold
    w = _weights(targets, weight).flatten(2)
    intersection = ((predicted & actual).flatten(2) * w).sum(-1)
    union = ((predicted | actual).flatten(2) * w).sum(-1)
    return intersection / union.clamp_min(1e-6)


def voxel_mask_iou(
    logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """IoU at VOXEL resolution, computed on the coarse grid -> `(P, K)`.

    `mask_iou` scores the prediction against the *pooled* target, so both live on the mask grid and
    the number it reports is blind to everything the pooling threw away. That flatters the model,
    and worse, it is not comparable between strides: a finer head is scored against a harder target
    and can report a lower number while being strictly better. Any experiment that changes
    `mask_upscale` needs a metric that does not move with it.

    This is that metric, and it costs nothing. The prediction is constant on each block, so
    upsampling it with nearest-neighbour and scoring against the true binary mask has a closed
    form: a block the model calls foreground contributes `stride^3 * target` intersecting voxels
    and `stride^3` predicted ones, and the true mask has `stride^3 * sum(target)` voxels in total.
    Every `stride^3` cancels in the ratio, so the soft target -- which is exactly the fraction of
    each block belonging to the object -- carries all the information needed:

        IoU = sum(p * t) / (sum(p) + sum(t) - sum(p * t))

    with `p` the thresholded prediction and `t` the soft target. **Exact**, not an approximation,
    and pinned against an explicit upsample-and-score in `tests/unit/test_promptable_seg.py`.

    The gap between this and `mask_iou` is the quantisation ceiling. Measured on this corpus
    (`sam3d/stride_ceiling.py`): a *perfect* model at stride 4 caps at 0.755 overall and 0.555 on
    objects under 4k voxels, against 0.912 at stride 2. So the two numbers are far apart, and the
    coarse one is the misleading one.
    """
    w = _weights(targets, weight).flatten(2)
    # With `w` the labelled fraction of a block and `t` the object's share of the labelled voxels,
    # a predicted block holds `w * t` true and `w` counted voxels (over `stride^3`), so the same
    # closed form scores the labelled voxels only.
    predicted = (logits > 0).to(targets.dtype).flatten(2) * w
    flat_targets = targets.flatten(2)
    intersection = (predicted * flat_targets).sum(-1)
    union = predicted.sum(-1) + (w * flat_targets).sum(-1) - intersection
    return intersection / union.clamp_min(1e-6)


def best_of(losses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(P, K)` per-candidate losses -> the lowest of each row, and which candidate it was.

    This one line is what makes the model ambiguity-aware. A single point inside a mitochondrion
    could mean the cristae, the organelle, or the cell containing it, and all three are correct
    answers to an under-specified question. Averaging their losses would train the model to predict
    the *union* of the three -- a blurred object that is none of them. Backpropagating only the
    best candidate lets the `K` outputs specialise instead, which is how the reference gets whole,
    part and subpart out of one click.

    The other candidates still receive gradient through the IoU head, which is scored against all
    of them -- otherwise the two that lost would never learn that they lost, and ranking them at
    inference would be guesswork.
    """
    values, indices = losses.min(dim=1)
    return values, indices
