"""Choosing objects to prompt for, and turning them into targets.

Everything here runs before the model sees anything, and everything here fails silently if it is
wrong: a click outside its object, a target pooled from the wrong volume or a repeated draw sharing
one accumulator slot all produce a run that trains, converges, and has learned the wrong task.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from algorithms.promptable.targets import (
    BOXES_KEY,
    IDS_KEY,
    POINTS_KEY,
    SPLIT_LABEL_KEY,
    VALID_KEY,
    PromptTargets,
    area_stratified_choice,
    eligible_objects,
    pooled_masks,
)


def _blobs() -> torch.Tensor:
    """`(1, 32, 32, 32)` labels: five cubes of 512 voxels and one two-voxel speck."""
    labels = torch.zeros(1, 32, 32, 32, dtype=torch.int64)
    corners = [(2, 2, 2), (14, 2, 2), (2, 14, 2), (2, 2, 14), (18, 18, 18)]
    for index, (x, y, z) in enumerate(corners):
        labels[0, x : x + 8, y : y + 8, z : z + 8] = 900_000_000_000 + index + 1
    labels[0, 30:32, 31:32, 31:32] = 999_999_999_999
    return labels


@pytest.mark.unit
def test_every_drawn_point_lies_inside_the_object_it_names():
    torch.manual_seed(0)
    sample = PromptTargets(masks_per_sample=4, min_object_voxels=64)({"label": _blobs()})
    split = sample[SPLIT_LABEL_KEY]
    for point, identifier in zip(sample[POINTS_KEY], sample[IDS_KEY], strict=True):
        assert split[tuple(point.tolist())] == identifier


@pytest.mark.unit
def test_objects_too_small_to_click_in_are_never_drawn():
    torch.manual_seed(0)
    transform = PromptTargets(masks_per_sample=8, min_object_voxels=64)
    sample = transform({"label": _blobs()})
    split = sample[SPLIT_LABEL_KEY]
    for identifier in sample[IDS_KEY].tolist():
        assert int((split == identifier).sum()) >= 64


@pytest.mark.unit
def test_draws_are_distinct_while_the_crop_has_objects_to_offer():
    torch.manual_seed(0)
    sample = PromptTargets(masks_per_sample=5, min_object_voxels=64)({"label": _blobs()})
    assert len(set(sample[IDS_KEY].tolist())) == 5, "five eligible objects, five slots"
    # More slots than objects falls back to repetition rather than a ragged batch.
    crowded = PromptTargets(masks_per_sample=9, min_object_voxels=64)({"label": _blobs()})
    assert crowded[IDS_KEY].shape == (9,)
    assert torch.all(crowded[VALID_KEY])


@pytest.mark.unit
def test_a_crop_with_no_eligible_object_is_a_state_not_an_error():
    # Four of the corpus's 24 instance-labelled volumes have their annotation somewhere other than
    # the volume centre, so an all-background crop is routine. Dropping the sample would make the
    # batch size depend on the data.
    sample = PromptTargets(masks_per_sample=3)({"label": torch.zeros(1, 8, 8, 8, dtype=torch.long)})
    assert sample[IDS_KEY].shape == (3,)
    assert not sample[VALID_KEY].any()
    assert sample[POINTS_KEY].shape == (3, 3)
    assert sample[BOXES_KEY].shape == (3, 2, 3)


@pytest.mark.unit
def test_disconnected_pieces_of_one_id_become_separate_objects():
    # A crop cuts objects. Two pieces of one neurite that no longer touch are two things a click
    # can point at, and prompting for their union would ask the model to segment across a gap it
    # cannot see.
    labels = torch.zeros(1, 16, 16, 16, dtype=torch.int64)
    labels[0, 1:5, 1:5, 1:5] = 77
    labels[0, 10:14, 10:14, 10:14] = 77
    sample = PromptTargets(masks_per_sample=2, min_object_voxels=8)({"label": labels})
    split = sample[SPLIT_LABEL_KEY]
    assert len(torch.unique(split[split > 0])) == 2
    assert sample[IDS_KEY].tolist() != [77, 77]


@pytest.mark.unit
def test_bounding_boxes_are_the_tight_ones():
    labels = torch.zeros(1, 16, 16, 16, dtype=torch.int64)
    labels[0, 3:9, 4:11, 5:6] = 5
    sample = PromptTargets(masks_per_sample=1, min_object_voxels=1)({"label": labels})
    torch.testing.assert_close(sample[BOXES_KEY][0], torch.tensor([[3, 4, 5], [9, 11, 6]]))


@pytest.mark.unit
def test_eligible_objects_filters_by_size_and_skips_background():
    dense = np.zeros((8, 8, 8), dtype=np.uint32)
    dense[0:4, 0:4, 0:4] = 1  # 64 voxels
    dense[7, 7, 7] = 2  # 1 voxel
    ids, counts, boxes = eligible_objects(dense, min_voxels=64)
    assert ids.tolist() == [1]
    assert counts.tolist() == [64]
    assert boxes.shape == (1, 2, 3)


@pytest.mark.unit
def test_size_stratified_sampling_does_not_follow_the_size_distribution():
    # A crop of twenty cell bodies and two thousand clipped fragments: uniform sampling would spend
    # 99% of its prompts on fragments, and the model would learn that a click means "the speck
    # under the cursor".
    torch.manual_seed(0)
    counts = np.concatenate([np.full(2000, 100), np.full(20, 100_000)])
    picked = area_stratified_choice(counts, 4000)
    large_fraction = (picked >= 2000).mean()
    assert large_fraction > 0.3, f"large objects got {large_fraction:.3f} of the draws"


@pytest.mark.unit
def test_pooled_masks_is_exactly_average_pooling_of_the_binary_mask():
    labels = torch.zeros(2, 8, 8, 8, dtype=torch.long)
    labels[0, 0:4, 0:4, 0:4] = 3
    labels[0, 5:8, 5:8, 5:8] = 7
    labels[1, 2:6] = 2
    ids = torch.tensor([[3, 7, 3, 0], [2, 2, 0, 0]])
    pooled = pooled_masks(labels, ids, stride=2)

    assert pooled.shape == (2, 4, 4, 4, 4)
    for sample, slot, identifier in ((0, 0, 3), (0, 1, 7), (1, 0, 2)):
        reference = F.avg_pool3d((labels[sample] == identifier).float()[None, None], 2)[0, 0]
        torch.testing.assert_close(pooled[sample, slot], reference)


@pytest.mark.unit
def test_a_repeated_draw_gets_the_same_mask_and_a_padding_draw_gets_none():
    labels = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    labels[0, 0:2] = 9
    pooled = pooled_masks(labels, torch.tensor([[9, 9, 0]]), stride=2)
    torch.testing.assert_close(pooled[0, 0], pooled[0, 1])
    assert pooled[0, 2].abs().max() == 0, "padding must not pick up the background"


@pytest.mark.unit
def test_pooling_accepts_a_stride_per_axis_and_rejects_a_partial_block():
    labels = torch.zeros(1, 8, 8, 4, dtype=torch.long)
    labels[0, 1:6, 2:7, 1:3] = 5
    pooled = pooled_masks(labels, torch.tensor([[5]]), stride=(2, 2, 1))
    reference = F.avg_pool3d((labels[0] == 5).float()[None, None], (2, 2, 1))[0, 0]
    torch.testing.assert_close(pooled[0, 0], reference)

    with pytest.raises(ValueError, match="not divisible"):
        pooled_masks(labels, torch.tensor([[5]]), stride=3)


@pytest.mark.unit
def test_pooling_survives_labels_larger_than_any_drawn_id():
    # The drawn ids are a subset, and the lookup is indexed by *every* voxel's label -- so sizing
    # it by the largest drawn id reads out of bounds on the first crop where an undrawn object
    # happens to have a higher id, which is most of them.
    labels = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    labels[0, 0:2] = 1
    labels[0, 2:4] = 900
    pooled = pooled_masks(labels, torch.tensor([[1]]), stride=2)
    assert pooled.shape == (1, 1, 2, 2, 2)
    assert pooled[0, 0].sum() == 4


@pytest.mark.unit
def test_the_transform_survives_labels_that_already_reached_a_device():
    # The dataloader hands this CPU tensors, but `PromptableSegmentation._objects` falls back to
    # running the same transform itself when nothing attached it -- and by then the batch is on the
    # training device. cc3d is a host library, so the transform has to make that copy rather than
    # failing on `.numpy()`. Checked on CPU by construction (`.detach()` on a graph-attached
    # tensor is the same code path that `.cpu()` is), since the test tier has no GPU.
    labels = _blobs().requires_grad_(False).clone()
    tracked = labels.to(torch.float32).requires_grad_(True).detach().to(torch.int64)
    sample = PromptTargets(masks_per_sample=2, min_object_voxels=64)({"label": tracked})
    assert sample[VALID_KEY].all()
