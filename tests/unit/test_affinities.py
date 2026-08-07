"""Unit tests for affinity target construction and the connected-components relabel."""

from __future__ import annotations

import pytest
import torch

from algorithms.affinity.targets import (
    affinities_from_labels,
    affinity_offsets,
    relabel_connected,
)


def _naive_components(labels: torch.Tensor) -> int:
    """Count 6-connected components by flood fill, written the obvious slow way.

    An independent oracle for `relabel_connected`: same definition, no shared machinery, so a bug
    in the parallel version cannot hide in both.
    """
    seen = torch.zeros_like(labels, dtype=torch.bool)
    count = 0
    for start in torch.nonzero(labels > 0):
        start_tuple = tuple(start.tolist())
        if seen[start_tuple]:
            continue
        count += 1
        stack = [start_tuple]
        seen[start_tuple] = True
        while stack:
            here = stack.pop()
            for axis in range(labels.ndim):
                for step in (-1, 1):
                    there = list(here)
                    there[axis] += step
                    if not (0 <= there[axis] < labels.shape[axis]):
                        continue
                    neighbour = tuple(there)
                    if seen[neighbour] or labels[neighbour] != labels[here]:
                        continue
                    seen[neighbour] = True
                    stack.append(neighbour)
    return count


# ---------------------------------------------------------------- offsets


@pytest.mark.unit
def test_offsets_match_the_benchmark_channel_order():
    # Three short-range then three long-range, one per axis, in x,y,z order -- the order the
    # benchmark's 6 affinity channels are defined in.
    assert affinity_offsets(3, long_range=10) == (
        (1, 0, 0), (0, 1, 0), (0, 0, 1), (10, 0, 0), (0, 10, 0), (0, 0, 10),
    )


# ---------------------------------------------------------------- relabel


@pytest.mark.unit
def test_relabel_splits_a_disconnected_instance():
    # One id, two separated blobs -- exactly the case this exists for: a neuron that leaves the
    # crop and comes back keeps one id on disk but is two objects inside the crop.
    labels = torch.zeros(1, 1, 7, dtype=torch.long)
    labels[0, 0, 0:2] = 5
    labels[0, 0, 5:7] = 5

    out = relabel_connected(labels)
    assert torch.unique(out[labels > 0]).numel() == 2
    assert out[0, 0, 0] == out[0, 0, 1], "a connected run keeps one id"
    assert out[0, 0, 0] != out[0, 0, 5], "the separated run gets a different one"


@pytest.mark.unit
def test_relabel_keeps_a_connected_instance_whole():
    labels = torch.zeros(4, 4, 4, dtype=torch.long)
    labels[1:3, 1:3, 1:3] = 9
    out = relabel_connected(labels)
    assert torch.unique(out[labels > 0]).numel() == 1


@pytest.mark.unit
def test_relabel_does_not_merge_touching_different_instances():
    labels = torch.zeros(1, 1, 4, dtype=torch.long)
    labels[0, 0, 0:2] = 1
    labels[0, 0, 2:4] = 2  # adjacent, different id
    out = relabel_connected(labels)
    assert out[0, 0, 1] != out[0, 0, 2]


@pytest.mark.unit
def test_relabel_leaves_background_and_ignore_untouched():
    labels = torch.tensor([[[0, -1, 3, 3]]], dtype=torch.long)
    out = relabel_connected(labels)
    assert out[0, 0, 0] == 0, "background stays background"
    assert out[0, 0, 1] == -1, "ignore stays ignore"
    assert out[0, 0, 2] == out[0, 0, 3] > 0


@pytest.mark.unit
def test_relabel_does_not_wrap_around_the_volume():
    # The two runs touch only if the far face is treated as adjacent to the near one, which a
    # circular shift would do.
    labels = torch.zeros(1, 1, 6, dtype=torch.long)
    labels[0, 0, 0] = 7
    labels[0, 0, 5] = 7
    assert torch.unique(relabel_connected(labels)[labels > 0]).numel() == 2


@pytest.mark.unit
def test_relabel_handles_an_empty_volume():
    labels = torch.zeros(3, 3, 3, dtype=torch.long)
    assert torch.equal(relabel_connected(labels), labels)


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(6))
def test_relabel_matches_a_flood_fill_on_random_volumes(seed):
    """The property that matters, against an independent implementation.

    Random blobby volumes produce winding, interleaved components -- which is exactly where a
    connected-components pass that stops iterating too early goes wrong.
    """
    torch.manual_seed(seed)
    labels = (torch.rand(8, 8, 8) < 0.45).long() * torch.randint(1, 4, (8, 8, 8))

    out = relabel_connected(labels)
    assert torch.unique(out[labels > 0]).numel() == _naive_components(labels)

    # Same partition, not merely the same count: every pair sharing an output id must share an
    # input id, and connected voxels must stay together.
    for axis in range(labels.ndim):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        adjacent_same_input = (labels[tuple(lo)] == labels[tuple(hi)]) & (labels[tuple(lo)] > 0)
        assert torch.equal(
            out[tuple(lo)][adjacent_same_input], out[tuple(hi)][adjacent_same_input]
        ), "adjacent voxels of one instance must land in one component"


# ---------------------------------------------------------------- affinity targets


@pytest.mark.unit
def test_affinity_is_one_within_an_instance_and_zero_across_a_boundary():
    labels = torch.tensor([[[[1, 1, 2, 2]]]], dtype=torch.long)  # (B=1, 1, 1, 4)
    target, mask = affinities_from_labels(labels, ((0, 0, 1),))

    assert target[0, 0, 0, 0, 0], "same instance -> affine"
    assert not target[0, 0, 0, 0, 1], "1 | 2 boundary -> not affine"
    assert target[0, 0, 0, 0, 2], "same instance -> affine"
    assert not mask[0, 0, 0, 0, 3], "the last voxel has no +1 neighbour"


@pytest.mark.unit
def test_background_is_affine_to_nothing():
    # Following the reference, which zeroes every affinity whose source voxel is background --
    # background is not an object, so two background voxels are not "the same object".
    labels = torch.tensor([[[[0, 0, 1, 1]]]], dtype=torch.long)
    target, mask = affinities_from_labels(labels, ((0, 0, 1),))

    assert not target[0, 0, 0, 0, 0], "background to background is not affine"
    assert not target[0, 0, 0, 0, 1], "background to foreground is not affine"
    assert target[0, 0, 0, 0, 2]
    assert mask[0, 0, 0, 0, 0], "but it is still a known target, so it is scored"


@pytest.mark.unit
def test_ignore_index_is_excluded_from_the_mask_on_either_side():
    labels = torch.tensor([[[[1, -1, 1, 1]]]], dtype=torch.long)
    _, mask = affinities_from_labels(labels, ((0, 0, 1),), ignore_index=-1)

    assert not mask[0, 0, 0, 0, 0], "source known, neighbour ignored -> unknowable"
    assert not mask[0, 0, 0, 0, 1], "source ignored -> unknowable"
    assert mask[0, 0, 0, 0, 2], "both known -> scored"


@pytest.mark.unit
def test_border_slab_is_masked_out_for_every_offset():
    labels = torch.ones(1, 6, 6, 6, dtype=torch.long)
    offsets = affinity_offsets(3, long_range=2)
    _, mask = affinities_from_labels(labels, offsets)

    for channel, offset in enumerate(offsets):
        axis = [i for i, value in enumerate(offset) if value != 0][0]
        shift = offset[axis]
        # Exactly the last `shift` planes along that axis have no neighbour to compare against.
        kept = mask[0, channel].sum().item()
        assert kept == (6 - shift) * 36, f"channel {channel} offset {offset}"


@pytest.mark.unit
def test_long_range_offset_reaches_past_a_thin_gap():
    # 1 1 0 0 1 1 with offset 4: voxel 0 sees voxel 4, both instance 1 -> affine, even though the
    # short-range affinity across the gap is 0. This is what the long-range channels are for.
    labels = torch.tensor([[[[1, 1, 0, 0, 1, 1]]]], dtype=torch.long)
    short, _ = affinities_from_labels(labels, ((0, 0, 1),))
    long_range, _ = affinities_from_labels(labels, ((0, 0, 4),))

    assert not short[0, 0, 0, 0, 1], "adjacent across the gap: not affine"
    assert long_range[0, 0, 0, 0, 0], "four apart, same instance: affine"


@pytest.mark.unit
def test_rejects_an_offset_of_the_wrong_rank():
    labels = torch.ones(1, 4, 4, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="spatial axes"):
        affinities_from_labels(labels, ((1, 0),))


@pytest.mark.unit
def test_offset_larger_than_the_volume_masks_everything():
    labels = torch.ones(1, 3, 3, 3, dtype=torch.long)
    _, mask = affinities_from_labels(labels, ((10, 0, 0),))
    assert mask.sum() == 0, "nothing is knowable when the offset leaves the volume entirely"
