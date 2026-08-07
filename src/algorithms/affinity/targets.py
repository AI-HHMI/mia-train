"""Turning an instance-segmentation crop into affinity targets.

Affinities are the representation the NISB baseline predicts: for each voxel and each of a small
set of offsets, "does the voxel at this offset belong to the same object as I do?". A network
predicting them learns object *boundaries* rather than object identities, which is what makes the
task independent of how many neurons happen to be in a crop and of what ids they were given.

Kept out of the algorithm module because both halves of this file are plain tensor functions with
their own failure modes, and they are far easier to test directly than through a training step.
"""

from __future__ import annotations

import torch

# The NISB baseline's six offsets, in the order it emits them: three short-range (nearest
# neighbour along each axis) then three long-range. They are expressed in *spatial axis order as
# the batch arrives*, so with `output_axes = "lcxyz"` channel 0 is +1 in x. Nothing here checks
# that -- the algorithm does, once, against the dataset's declared axes.
SHORT_RANGE = 1
LONG_RANGE = 10


def affinity_offsets(
    spatial_rank: int, long_range: int = LONG_RANGE
) -> tuple[tuple[int, ...], ...]:
    """The 2 * rank offsets, short-range block first, matching the reference channel order."""
    offsets = []
    for distance in (SHORT_RANGE, long_range):
        for axis in range(spatial_rank):
            offset = [0] * spatial_rank
            offset[axis] = distance
            offsets.append(tuple(offset))
    return tuple(offsets)


def relabel_connected(labels: torch.Tensor) -> torch.Tensor:
    """Give each *spatially connected* run of an id its own id -> same shape, int64.

    A neuron that leaves the crop and re-enters keeps one id on disk, but inside this crop it is
    two disconnected pieces. Left alone, the long-range affinity between them is a positive target
    the network cannot satisfy from local evidence. The reference splits them with
    `cc3d.connected_components`; this does the same with 6-connectivity and no dependency.

    The algorithm is the classic parallel connected-components pair, alternating until stable:

      *hooking* -- for every 6-connected pair sharing an id, the higher of the two component
      *roots* is pointed at the lower, and

      *pointer jumping* -- `parent = parent[parent]`, repeated, which halves the depth of every
      pointer chain each time.

    Both halves matter, and getting either wrong is slow rather than loud. Hooking voxels instead
    of roots advances the frontier one voxel per round, so a component needs as many rounds as its
    *geodesic* length -- and a neuron winding through a crop is far longer than the crop is wide.
    Measured on a real 128-voxel NISB crop: voxel-hooking needed 224 rounds and took 22 s, and
    capping it at the crop width silently reported 170 components where there were 40. Hooking
    roots and compressing brings the same crop to ~1 s and a handful of rounds.

    Background (id 0) and ignore (negative ids) pass through untouched: they are not objects, so
    splitting them is meaningless and merging them would be wrong.
    """
    if labels.ndim < 1:
        raise ValueError(
            f"labels must have at least one spatial axis, got shape {tuple(labels.shape)}"
        )

    spatial_shape = labels.shape
    foreground = labels > 0
    # Every voxel starts as its own component, identified by flat index. Background participates
    # as a singleton rather than being forced to a sentinel, which keeps the pointer array a valid
    # permutation and avoids colliding with whatever voxel happens to sit at index 0.
    parent = torch.arange(labels.numel(), device=labels.device, dtype=torch.int64)
    source, neighbour = _adjacency(labels, foreground)
    if source.numel() == 0:  # nothing touches anything; every voxel is already its own component
        _, dense = torch.unique(parent.view(spatial_shape), return_inverse=True)
        return torch.where(foreground, dense.reshape(spatial_shape) + 1, labels)

    while True:
        before = parent.clone()

        # Hook *roots*, not voxels. Pointing each voxel at its neighbour's minimum would advance
        # the frontier one voxel per round, so a component would need as many rounds as its
        # geodesic length. Merging the two roots instead joins whole trees at once, which is what
        # makes the round count logarithmic rather than proportional to the winding path.
        source_root = parent[source]
        neighbour_root = parent[neighbour]
        merged = torch.minimum(source_root, neighbour_root)
        parent.scatter_reduce_(0, source_root, merged, reduce="amin")
        parent.scatter_reduce_(0, neighbour_root, merged, reduce="amin")

        while True:  # pointer jumping: halves every chain's depth per pass
            jumped = parent[parent]
            if torch.equal(jumped, parent):
                break
            parent = jumped

        # Cloned above, not aliased: `parent` is mutated in place by the scatters, so comparing
        # against a plain reference would compare it with itself and stop after one round.
        if torch.equal(parent, before):
            break

    # Dense 1..K ids, so downstream code can assume small numbers, with non-objects restored.
    _, dense = torch.unique(parent.view(spatial_shape), return_inverse=True)
    return torch.where(foreground, dense.reshape(spatial_shape) + 1, labels)


def _adjacency(
    labels: torch.Tensor, foreground: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat indices of every 6-connected voxel pair that shares a positive id.

    Built once and reused every round, since the edge set does not change as the pointers do.
    Slicing rather than rolling, because a circular shift would make the far face neighbour the
    near one and merge components that never touch.
    """
    flat_index = torch.arange(labels.numel(), device=labels.device, dtype=torch.int64)
    flat_index = flat_index.view(labels.shape)

    sources, neighbours = [], []
    for axis in range(labels.ndim):
        if labels.shape[axis] < 2:
            continue
        lo = tuple(slice(0, -1) if i == axis else slice(None) for i in range(labels.ndim))
        hi = tuple(slice(1, None) if i == axis else slice(None) for i in range(labels.ndim))
        connected = (labels[lo] == labels[hi]) & foreground[lo]
        sources.append(flat_index[lo][connected])
        neighbours.append(flat_index[hi][connected])

    if not sources:
        empty = torch.empty(0, dtype=torch.int64, device=labels.device)
        return empty, empty
    return torch.cat(sources), torch.cat(neighbours)


def affinities_from_labels(
    labels: torch.Tensor,
    offsets: tuple[tuple[int, ...], ...],
    ignore_index: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Instance ids -> (affinity target, loss mask), both (B, len(offsets), *spatial).

    Two voxels are affine when they carry the same *positive* id. Background (0) is affine to
    nothing, following the reference, which zeroes every affinity whose source voxel is
    background.

    The mask says where the target is knowable. It excludes pairs where either voxel is
    `ignore_index`, and the border slab each offset shifts in from -- there is no neighbour there,
    so any value would be invented. Scoring those would teach the network to predict the padding.
    """
    if labels.ndim < 2:
        raise ValueError(
            "labels must be (B, *spatial) with at least one spatial axis, got "
            f"{tuple(labels.shape)}"
        )
    spatial_rank = labels.ndim - 1
    for offset in offsets:
        if len(offset) != spatial_rank:
            raise ValueError(
                f"offset {offset} has {len(offset)} components but labels carry {spatial_rank} "
                "spatial axes"
            )

    labelled = labels != ignore_index
    affinity = torch.zeros(
        (labels.shape[0], len(offsets), *labels.shape[1:]),
        dtype=torch.bool,
        device=labels.device,
    )
    mask = torch.zeros_like(affinity)

    for channel, offset in enumerate(offsets):
        source, target = _offset_slices(offset, labels.shape[1:])
        a = labels[(slice(None), *source)]
        b = labels[(slice(None), *target)]
        # Same object, and an object at all: two background voxels are not affine.
        window: tuple[slice | int, ...] = (slice(None), channel, *source)
        affinity[window] = (a == b) & (a > 0)
        mask[window] = (
            labelled[(slice(None), *source)] & labelled[(slice(None), *target)]
        )

    return affinity, mask


def _offset_slices(
    offset: tuple[int, ...], spatial_shape: torch.Size
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """The overlapping regions a shift by `offset` relates, as (source, target) slice tuples.

    Both cover the same number of voxels; `target` is `source` translated by `offset`. Anything
    outside is dropped rather than wrapped or padded, and the caller masks it out of the loss.
    """
    source: list[slice] = []
    target: list[slice] = []
    for axis_offset, extent in zip(offset, spatial_shape, strict=True):
        overlap = max(extent - abs(axis_offset), 0)
        if axis_offset >= 0:
            source.append(slice(0, overlap))
            target.append(slice(axis_offset, axis_offset + overlap))
        else:
            source.append(slice(-axis_offset, -axis_offset + overlap))
            target.append(slice(0, overlap))
    return tuple(source), tuple(target)
