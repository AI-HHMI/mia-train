"""The mask losses, and the reduction that makes a model ambiguity-aware.

All four functions take `(P, K, *spatial)` logits against `(P, K, *spatial)` targets -- `P` prompts
each producing `K` candidate masks -- and reduce over space only, so a caller keeps one number per
candidate. That shape is what `best_of` needs, and it is the whole reason the reduction is not
folded into the losses.

Targets are *soft*, in [0, 1], because they arrive area-pooled from voxel resolution
(`targets.pooled_masks`). Focal and dice both accept that reading without modification; the IoU the
prediction head is trained against does not, so it thresholds.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """Mean focal loss per candidate mask -> `(P, K)`.

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
    return loss.flatten(2).mean(-1)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft dice loss per candidate mask -> `(P, K)`.

    Scale-free where focal is not: it is a ratio of overlaps, so a 200-voxel object and a
    200,000-voxel one contribute comparably. That is what keeps the small objects in a microscopy
    crop from being optimised away, and it is why the reference combines the two rather than
    picking one.
    """
    probabilities = logits.sigmoid().flatten(2)
    flat = targets.flatten(2)
    intersection = (probabilities * flat).sum(-1)
    denominator = probabilities.sum(-1) + flat.sum(-1)
    return 1 - (2 * intersection + eps) / (denominator + eps)


def mask_iou(
    logits: torch.Tensor, targets: torch.Tensor, target_threshold: float = 0.5
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
    intersection = (predicted & actual).flatten(2).sum(-1)
    union = (predicted | actual).flatten(2).sum(-1)
    return intersection / union.clamp_min(1)


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
