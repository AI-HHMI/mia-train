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
import torch.nn.functional as F

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
#: Per slot: this prompt is a click on a voxel that belongs to NO object (label 0). The mask
#: losses skip it; the IoU head is trained to answer 0 for every candidate it produces.
OFFOBJECT_KEY = "object_offobject"
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


def boundary_shell(labels: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Voxels within `radius` (Chebyshev) of one carrying a different label -> `(*spatial)` bool.

    Both sides of every boundary: the rim of each object AND the background voxels touching it,
    since a max- and a min-pool over the label ids both differ from the centre wherever the
    neighbourhood is not uniform. The crop's faces are not boundaries -- pooling pads with -inf, so
    a voxel at a face is compared only with what is inside.
    """
    if radius < 1:
        raise ValueError(f"radius must be at least 1, got {radius}")
    pool = {2: F.max_pool2d, 3: F.max_pool3d}.get(labels.dim())
    if pool is None:
        raise ValueError(f"labels must have 2 or 3 spatial axes, got shape {tuple(labels.shape)}")
    # float32 is exact for the dense ids `relabel_connected_cc3d` leaves (far below 2**24).
    values = labels.to(torch.float32)[None, None]
    size = 2 * radius + 1
    highest = pool(values, size, stride=1, padding=radius)
    lowest = -pool(-values, size, stride=1, padding=radius)
    return ((highest != values) | (lowest != values))[0, 0]


def random_background_voxels(labels: torch.Tensor, count: int) -> torch.Tensor:
    """Up to `count` distinct uniformly random voxels with label exactly 0 -> `(k, rank)`.

    Label 0 only. Negative labels mean "unknown" -- the corpus's ignore value, and what the
    pseudo-label sidecars write wherever a teacher kept no mask -- and a click there is not a
    click on nothing; it is a click on something nobody has annotated. Fewer than `count` rows
    when the crop has fewer background voxels, none when it has none.
    """
    flat = labels.reshape(-1)
    background = flat == 0
    available = int(background.sum())
    if count < 1 or available == 0:
        return torch.zeros(0, labels.dim(), dtype=torch.long)
    key = torch.where(background, torch.rand(flat.shape), torch.full(flat.shape, float("inf")))
    positions = key.topk(min(count, available), largest=False).indices
    coordinates = []
    remaining = positions
    for extent in reversed(labels.shape):
        coordinates.append(remaining % extent)
        remaining = remaining // extent
    return torch.stack(list(reversed(coordinates)), dim=-1)


class PromptTargets:
    """Draw prompt-able objects from a sample's label crop, in a worker process.

    Three kinds of round-0 prompt come out of here, and the mix is the recipe:

      * an **interior click** -- a voxel drawn uniformly from the object (the reference's prompt);
      * a **boundary click** -- with probability `boundary_prob`, the voxel is drawn from the
        object's rim instead (`boundary_shell`, `boundary_radius` wide): the hardest, most
        ambiguous positive click, which a grid of prompts issues constantly against thin
        neurites and which uniform sampling almost never produces for a thick object;
      * an **off-object click** -- with probability `offobject_prob` per slot, a click on a voxel
        with label 0, asked with the same foreground-click prompt the grid issues everywhere. It
        carries no object: the mask losses skip it, and the IoU head is trained to answer 0 for
        every candidate. This is the prompt segment-everything issues most and training never
        showed the model; without it the head's confidence on membranes and unannotated space is
        untrained, and gating on it filters nothing (measured: pseudo-masks passing 0.7 predicted
        IoU had 0.24 true IoU).

    Boxes are drawn for object slots only, and the strategy never turns an off-object slot into a
    box.

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
        offobject_prob: float = 0.0,
        boundary_prob: float = 0.0,
        boundary_radius: int = 1,
    ) -> None:
        if masks_per_sample < 1:
            raise ValueError(f"masks_per_sample must be at least 1, got {masks_per_sample}")
        if min_object_voxels < 1:
            raise ValueError(f"min_object_voxels must be at least 1, got {min_object_voxels}")
        if not 0.0 <= offobject_prob <= 1.0:
            raise ValueError(f"offobject_prob must be a probability, got {offobject_prob}")
        if not 0.0 <= boundary_prob <= 1.0:
            raise ValueError(f"boundary_prob must be a probability, got {boundary_prob}")
        if boundary_radius < 1:
            raise ValueError(f"boundary_radius must be at least 1, got {boundary_radius}")
        self.label_key = label_key
        self.masks_per_sample = masks_per_sample
        self.min_object_voxels = min_object_voxels
        self.level = level
        self.offobject_prob = offobject_prob
        self.boundary_prob = boundary_prob
        self.boundary_radius = boundary_radius

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
        rank = split.dim()
        sample = dict(sample)
        sample[SPLIT_LABEL_KEY] = split

        out_ids = torch.zeros(masks, dtype=torch.long)
        out_points = torch.zeros(masks, rank, dtype=torch.long)
        out_boxes = torch.zeros(masks, 2, rank, dtype=torch.long)
        valid = torch.zeros(masks, dtype=torch.bool)
        offobject = torch.zeros(masks, dtype=torch.bool)

        # Off-object slots take the tail; only as many as the crop has background voxels to
        # supply, so a densely annotated crop simply yields fewer of them.
        if self.offobject_prob > 0:
            wanted = int((torch.rand(masks) < self.offobject_prob).sum())
            background = random_background_voxels(split, wanted)
            if background.shape[0]:
                out_points[masks - background.shape[0]:] = background
                offobject[masks - background.shape[0]:] = True
        objects = masks - int(offobject.sum())

        if len(ids) and objects:
            picked = area_stratified_choice(counts, objects)
            chosen_ids = torch.from_numpy(ids[picked]).long()

            # One slot per *distinct* id, so a repeated draw costs one pass, not two; the slots
            # are then fanned back out to the drawn order.
            unique_ids, inverse = torch.unique(chosen_ids, return_inverse=True)
            slot_of_id = torch.full((int(split.max()) + 1,), -1, dtype=torch.long)
            slot_of_id[unique_ids] = torch.arange(len(unique_ids))
            points = random_voxel_per_object(split, slot_of_id, len(unique_ids))[inverse]
            if self.boundary_prob > 0:
                points = self._boundary_points(split, slot_of_id, unique_ids, inverse, points)

            out_ids[:objects] = chosen_ids
            out_points[:objects] = points
            out_boxes[:objects] = torch.from_numpy(boxes[picked]).long()
            valid[:objects] = True

        sample[IDS_KEY] = out_ids
        sample[POINTS_KEY] = out_points
        sample[BOXES_KEY] = out_boxes
        sample[VALID_KEY] = valid
        sample[OFFOBJECT_KEY] = offobject
        return sample

    def _boundary_points(
        self,
        split: torch.Tensor,
        slot_of_id: torch.Tensor,
        unique_ids: torch.Tensor,
        inverse: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """Swap a `boundary_prob` share of the interior clicks for clicks in the object's rim.

        One `boundary_shell` pass over the crop for every object at once, then the same one-pass
        draw as the interior click on the rim-restricted label volume. An object without a rim
        voxel in the crop -- one that fills it -- keeps its interior click.
        """
        shell = boundary_shell(split, self.boundary_radius)
        rim = torch.where(shell, split, torch.zeros_like(split))
        rim_points = random_voxel_per_object(rim, slot_of_id, len(unique_ids))
        has_rim = torch.bincount(rim[rim > 0], minlength=int(split.max()) + 1)[unique_ids] > 0
        use = (torch.rand(points.shape[0]) < self.boundary_prob) & has_rim[inverse]
        return torch.where(use.unsqueeze(1), rim_points[inverse], points)

def pooled_known(labels: torch.Tensor, stride: int | Sequence[int]) -> torch.Tensor:
    """`(B, *spatial)` labels -> `(B, 1, *spatial // stride)`: the labelled fraction of each block.

    Labelled means `>= 0`: background and every object are decisions, `-1` is the absence of one --
    a pseudo-label's unclaimed space, a crop's unannotated margin. The mask losses, the IoU the head
    is trained to predict, and the correction clicks all weight cells by this, so an unknown voxel
    is neither foreground nor background but silence. Without it every unlabelled voxel trains as
    "not this object", and a model trained on truncated pseudo-masks learns a boundary wherever the
    labeller's window ended (measured: the sam_lmd_v1 data engine's round-1 targets).
    """
    spatial = tuple(labels.shape[1:])
    strides = (stride,) * len(spatial) if isinstance(stride, int) else tuple(stride)
    if len(strides) != 3:
        raise ValueError(f"pooled_known expects 3 spatial axes, got {len(strides)}")
    known = (labels >= 0).to(torch.float32).unsqueeze(1)
    return F.avg_pool3d(known, kernel_size=strides, stride=strides)


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
