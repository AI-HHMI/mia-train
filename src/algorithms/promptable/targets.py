"""Choosing which objects to prompt for, and where to click.

Promptable segmentation trains on one object at a time, but a microscopy crop holds many: measured
over the 24 instance-labelled volumes in `lmd-v0.0.1`, a 256-cube contains between 8 and 5,533
distinct objects, with a median around 70. Every object cannot be decoded in a step, so a subset is
drawn -- which is exactly what the reference does, at up to 64 masks per GPU.

**Where the work happens is the point of this module.** Counting an object's voxels, finding its
bounding box, and picking a voxel inside it are all passes over the whole crop, and all depend only
on that crop. Done on the training device they sit on the critical path between the batch arriving
and the loss; done in a dataloader worker they cost no GPU time and run `num_workers`-way parallel.
So `PromptTargets` is a `BaseAlgorithm.sample_transform`, following `SplitDisconnectedLabels`.

**What crosses the worker boundary is small, and deliberately so.** A binary mask per object would
be `masks_per_sample` volumes -- a gigabyte per sample at 64 masks on a 256-cube -- for information
already present in the label crop the batch carries anyway. So only ids, points and boxes are
returned, a few hundred numbers, and the binary target is derived on the device from
`labels == id`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import cc3d
import numpy as np
import torch

# Reused rather than reimplemented: this is the same connected-components pass the affinity
# strategy runs, in the same place (a worker), for a reason that applies here too -- a crop can cut
# an object into pieces that no longer touch, and a click on one piece should not be asked to
# produce the other. It also leaves ids dense in 1..N, which is what makes `cc3d.statistics`
# affordable: the corpus stores 64-bit segment ids, and a statistics pass over that id space would
# try to allocate one entry per possible id.
from ..affinity.targets import relabel_connected_cc3d

#: Keys `PromptTargets` adds to a sample. Named here so the algorithm and the transform cannot
#: drift apart about them.
IDS_KEY = "object_ids"
POINTS_KEY = "object_points"
BOXES_KEY = "object_boxes"
VALID_KEY = "object_valid"
SPLIT_LABEL_KEY = "split_label"


def eligible_objects(
    labels: np.ndarray, min_voxels: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense-id label volume -> the ids worth prompting for, their voxel counts and their boxes.

    `boxes` is `(n, 2, rank)`: the inclusive lower corner and the exclusive upper one, in voxels.

    `min_voxels` is not a tidiness filter. Instance annotation in this corpus is full of
    single-figure fragments -- a neurite clipped by the crop face, a supervoxel boundary artefact --
    and an object of three voxels has no interior to click in and no shape to segment. Training on
    them teaches the decoder that a click means "the speck under the cursor".
    """
    stats = cc3d.statistics(labels)
    counts = stats["voxel_counts"]
    # Index 0 is background, and cc3d reports `None` for ids that do not occur.
    ids = np.flatnonzero(counts >= min_voxels)
    ids = ids[ids > 0]
    spans = [stats["bounding_boxes"][i] for i in ids]
    boxes = np.array(
        [[[axis.start for axis in span], [axis.stop for axis in span]] for span in spans],
        dtype=np.int64,
    ).reshape(len(ids), 2, labels.ndim)
    return ids, counts[ids], boxes


def area_stratified_choice(counts: np.ndarray, draws: int) -> np.ndarray:
    """Pick `draws` indices, spreading them over object *sizes* rather than over objects.

    Uniform sampling would follow the size distribution, and that distribution is extremely skewed:
    a crop holding twenty cell bodies and two thousand clipped neurite fragments would spend
    99% of its prompts on fragments. Weighting each object by the reciprocal of how many objects
    share its size octave makes the draw uniform over octaves instead, so large and small objects
    are seen in comparable numbers.

    Without replacement whenever the crop has that many objects to offer, so every slot is a
    distinct object and the step's gradient is `draws` different examples. A crop with fewer falls
    back to replacement, which keeps the batch rectangular rather than making the batch size depend
    on the data. Those repeats share a first click -- one interior voxel is drawn per *object*,
    because drawing one per slot would cost a pass over the crop per slot -- but their interactive
    rounds still diverge, since each round samples its next point from its own error region.
    """
    octave = np.floor(np.log2(np.maximum(counts, 1))).astype(np.int64)
    _, inverse, occurrences = np.unique(octave, return_inverse=True, return_counts=True)
    weights = torch.from_numpy(1.0 / occurrences[inverse])
    replacement = len(counts) < draws
    return torch.multinomial(weights, draws, replacement=replacement).numpy()


def random_voxel_per_object(
    labels: torch.Tensor, slot_of_id: torch.Tensor, slots: int
) -> torch.Tensor:
    """One uniformly random voxel of each selected object -> `(slots, rank)` voxel indices.

    `slot_of_id` maps a dense label id to its slot in the output, or -1 for an object that was not
    selected. One pass over the crop for every object at once, rather than a `labels == id` scan
    per object: at a thousand objects the per-object form is a thousand passes over the volume.

    The pick is the arg-min of a fresh uniform key over each object's voxels, which is a uniform
    draw from that object. Two passes over the crop, and a `rand` the size of it -- 67 MB at a
    256-cube -- which is the memory cost of doing every object at once.
    """
    flat = labels.reshape(-1)
    slot = slot_of_id[flat.clamp_min(0)]
    selected = slot >= 0
    key = torch.where(selected, torch.rand(flat.shape), torch.full(flat.shape, float("inf")))

    best = torch.full((slots,), float("inf")).scatter_reduce(
        0, slot.clamp_min(0), key, reduce="amin", include_self=True
    )
    winner = selected & (key == best[slot.clamp_min(0)])
    positions = winner.nonzero(as_tuple=True)[0]

    voxels = torch.zeros(slots, labels.dim(), dtype=torch.long)
    # `stack` of the unravelled index, rather than `unravel_index`, so this reads the same on the
    # torch versions that predate it.
    coordinates = []
    remaining = positions
    for extent in reversed(labels.shape):
        coordinates.append(remaining % extent)
        remaining = remaining // extent
    voxels[slot[positions]] = torch.stack(list(reversed(coordinates)), dim=-1)
    return voxels


class PromptTargets:
    """Draw prompt-able objects from a sample's label crop, in a worker process.

    A callable class rather than a closure, for the reason `SplitDisconnectedLabels` is one: a
    worker is forked from a pickled copy of the dataset, and a closure over an algorithm would drag
    the model into every worker. This holds keys and integers.

    Replaces the sample's label with the connected-components-split version under
    `SPLIT_LABEL_KEY`, keeping the original: the split volume is what the drawn ids index into, and
    deriving the binary target on the device from an unsplit volume would silently union an
    object's separated pieces back together.

    A crop with no eligible object at all is a real state, not an error -- four of the corpus's 24
    instance-labelled volumes have their annotation somewhere other than the volume centre, so a
    crop drawn from one can be entirely background. Those samples come back with `object_valid` all
    false and are excluded from the loss rather than dropped, because dropping them would make the
    batch size depend on the data.
    """

    def __init__(
        self,
        label_key: str = "label",
        masks_per_sample: int = 16,
        min_object_voxels: int = 64,
        level: int = 0,
    ) -> None:
        if masks_per_sample < 1:
            raise ValueError(f"masks_per_sample must be at least 1, got {masks_per_sample}")
        if min_object_voxels < 1:
            raise ValueError(f"min_object_voxels must be at least 1, got {min_object_voxels}")
        self.label_key = label_key
        self.masks_per_sample = masks_per_sample
        self.min_object_voxels = min_object_voxels
        self.level = level

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.label_key not in sample:
            raise KeyError(
                f"no {self.label_key!r} in the sample, which carries {sorted(sample)}; the "
                "promptable segmentation strategy draws the objects it prompts for from the "
                "instance labels, and has nothing to train on without them"
            )
        labels = sample[self.label_key]
        if labels.dim() < 2:
            raise ValueError(
                f"labels have shape {tuple(labels.shape)}; a sample's label crop carries its "
                "scale-level axis and at least one spatial axis"
            )

        # The finest level, for the reason `affinity_seg._prepare_labels` gives: a multi-scale
        # encoder's dense prediction is co-registered with level 0 and with nothing else.
        split = relabel_connected_cc3d(labels[self.level])
        # `.cpu()` because this may be driven on a batch that already reached the device --
        # `PromptableSegmentation._objects` falls back to running this transform itself when the
        # engine has not attached it, and by then the batch is on the training device. cc3d is a
        # host library either way, so the copy is the one this code already implies.
        dense = split.detach().cpu().numpy().clip(min=0).astype(np.uint32)
        ids, counts, boxes = eligible_objects(dense, self.min_object_voxels)

        masks = self.masks_per_sample
        sample = dict(sample)
        sample[SPLIT_LABEL_KEY] = split
        if len(ids) == 0:
            rank = split.dim()
            sample[IDS_KEY] = torch.zeros(masks, dtype=torch.long)
            sample[POINTS_KEY] = torch.zeros(masks, rank, dtype=torch.long)
            sample[BOXES_KEY] = torch.zeros(masks, 2, rank, dtype=torch.long)
            sample[VALID_KEY] = torch.zeros(masks, dtype=torch.bool)
            return sample

        picked = area_stratified_choice(counts, masks)
        chosen_ids = torch.from_numpy(ids[picked]).long()

        # One slot per *distinct* id, so a repeated draw costs one pass, not two; the slots are
        # then fanned back out to the drawn order.
        unique_ids, inverse = torch.unique(chosen_ids, return_inverse=True)
        slot_of_id = torch.full((int(split.max()) + 1,), -1, dtype=torch.long)
        slot_of_id[unique_ids] = torch.arange(len(unique_ids))
        points = random_voxel_per_object(split, slot_of_id, len(unique_ids))[inverse]

        sample[IDS_KEY] = chosen_ids
        sample[POINTS_KEY] = points
        sample[BOXES_KEY] = torch.from_numpy(boxes[picked]).long()
        sample[VALID_KEY] = torch.ones(masks, dtype=torch.bool)
        return sample


def pooled_masks(
    labels: torch.Tensor, object_ids: torch.Tensor, stride: int | Sequence[int]
) -> torch.Tensor:
    """`(B, *spatial)` label ids and `(B, M)` object ids -> `(B, M, *spatial // stride)` targets.

    The binary mask of each drawn object, area-pooled to the resolution the decoder emits. Soft, in
    [0, 1]: at stride 4 a max-pool would fatten every structure by up to three voxels, and EM
    membranes are one or two voxels thick, so the fattening would be most of the object.

    **One pass over the crop for every object at once.** The direct form -- `labels == id` per
    object, then `avg_pool3d` -- costs `M` passes and materialises `M` float volumes; at 16 objects
    on a 192-cube that is 450 MB before any pooling, and it grows with the crop cubed. Here each
    voxel is scattered once into the `(object, block)` cell it belongs to, so the largest tensor is
    the *pooled* result. This is the difference between "many masks" being affordable in 3D and not.

    **Every shape here is static and nothing reads a tensor's value on the host.** Which slot a
    voxel belongs to is resolved with a batched `searchsorted` over the drawn ids rather than
    through a lookup table sized by `labels.max()`, and undrawn voxels are scattered into a discard
    row rather than filtered out by a boolean mask of data-dependent length. Both of those would be
    device-to-host synchronizations on the critical path of every step -- and under `torch.compile`
    they are graph breaks and shape recompilations, which is the difference between a Blackwell
    device being worth using and not (measured elsewhere in this repo: eager B300 is *slower* than
    eager H200 because it is launch-bound, and compilation is the entire reason to move).

    A repeated draw -- common, since objects are drawn with replacement once a crop has fewer
    objects than slots -- resolves to the same accumulator slot and is copied out to both. A padding
    draw (`id == 0`) comes back empty rather than picking up the background.
    """
    batch, masks = object_ids.shape
    spatial = tuple(labels.shape[1:])
    strides = (stride,) * len(spatial) if isinstance(stride, int) else tuple(stride)
    if labels.shape[0] != batch:
        raise ValueError(f"labels carry {labels.shape[0]} samples but object_ids carry {batch}")
    if len(strides) != len(spatial) or any(step < 1 for step in strides):
        raise ValueError(
            f"stride {stride} does not give a positive step for each of the {len(spatial)} "
            "spatial axes"
        )
    if any(extent % step for extent, step in zip(spatial, strides, strict=True)):
        raise ValueError(
            f"crop {spatial} is not divisible by the mask stride {strides}; the decoder emits "
            "whole blocks, and a partial one has no target to score against"
        )

    device = labels.device
    blocks = tuple(extent // step for extent, step in zip(spatial, strides, strict=True))
    n_blocks = math.prod(blocks)

    # Which pooling block each voxel falls in. Shared by every sample and every object, so it is
    # built once from the shape rather than per mask.
    block = torch.zeros(spatial, dtype=torch.long, device=device)
    for axis, (extent, step) in enumerate(zip(spatial, strides, strict=True)):
        shape = [1] * len(spatial)
        shape[axis] = -1
        along = (torch.arange(extent, device=device) // step).reshape(shape)
        block = block * (extent // step) + along
    block = block.reshape(1, -1).expand(batch, -1)

    # Padding draws become -1, which no label can equal (labels are clamped at 0 below), so
    # background never resolves to a padding slot.
    ids = torch.where(object_ids > 0, object_ids, torch.full_like(object_ids, -1))
    sorted_ids, order = ids.sort(dim=1)
    flat = labels.reshape(batch, -1).clamp_min(0)

    # `searchsorted` with the default left bias lands on the FIRST of a run of equal ids, so two
    # slots holding the same object share one accumulator row by construction.
    position = torch.searchsorted(sorted_ids.contiguous(), flat).clamp_max(masks - 1)
    matched = sorted_ids.gather(1, position) == flat
    slot = torch.where(matched, order.gather(1, position), torch.full_like(position, masks))

    # Row `masks` is the discard bin for voxels belonging to no drawn object. Scattering into it
    # rather than filtering them out is what keeps every shape static.
    accumulator = torch.zeros(batch, masks + 1, n_blocks, device=device)
    accumulator.reshape(batch, -1).scatter_add_(
        1,
        slot * n_blocks + block,
        torch.ones((), device=device).expand(batch, flat.shape[1]),
    )
    accumulator = accumulator[:, :masks] / float(math.prod(strides))

    drawn = torch.searchsorted(sorted_ids.contiguous(), ids).clamp_max(masks - 1)
    canonical = order.gather(1, drawn)
    rows = torch.arange(batch, device=device).unsqueeze(1).expand(batch, masks)
    pooled = accumulator[rows, canonical]
    return (pooled * (object_ids > 0).unsqueeze(-1)).reshape(batch, masks, *blocks)
