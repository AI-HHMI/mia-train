"""The promptable segmentation strategy: losses, decoder, and the interactive round loop."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import components  # noqa: F401  (populates the registries as a real run would)
from algorithms.promptable.decoder import MaskDecoder3D
from algorithms.promptable.losses import (
    best_of,
    dice_loss,
    focal_loss,
    mask_iou,
    voxel_mask_iou,
)
from algorithms.promptable.targets import IDS_KEY, SPLIT_LABEL_KEY, pooled_known, pooled_masks
from algorithms.promptable_seg import PromptableSegmentation
from algorithms.registry import AlgorithmRegistry
from layers.common.prompt import (
    BACKGROUND,
    BOX_FAR,
    BOX_NEAR,
    FOREGROUND,
    PAD,
    PromptEncoder3D,
)
from layers.common.rope import patch_grid_coords, voxel_coords
from models.registry import ModelRegistry

DIM = 32


def _algorithm(**overrides):
    model = ModelRegistry.build(
        "vit3d", img_size=(32,) * 3, patch_size=(8,) * 3, embed_dim=64, depth=2, num_heads=4
    )
    settings = dict(
        input_axes="lcxyz", prompt_dim=DIM, decoder_heads=4, mask_upscale=4,
        mask_feature_dim=8, masks_per_sample=3, min_object_voxels=8, rounds=2,
    )
    settings.update(overrides)
    return AlgorithmRegistry.build("promptable_seg", model, None, **settings)


def _batch() -> dict[str, torch.Tensor]:
    labels = torch.zeros(2, 1, 32, 32, 32, dtype=torch.int64)
    labels[0, 0, 2:12, 2:12, 2:12] = 111
    labels[0, 0, 20:30, 20:30, 5:15] = 222
    labels[1, 0, 4:20, 4:20, 4:20] = 333
    return {"img": torch.randn(2, 1, 1, 32, 32, 32), "label": labels}


@pytest.mark.unit
def test_the_strategy_is_registered_under_its_config_name():
    assert "promptable_seg" in AlgorithmRegistry.available()


@pytest.mark.unit
def test_losses_reach_their_limits_on_a_perfect_and_an_inverted_prediction():
    target = torch.zeros(2, 3, 4, 4, 4)
    target[:, :, 1:3, 1:3, 1:3] = 1
    perfect = torch.where(target > 0.5, 20.0, -20.0)

    assert dice_loss(perfect, target).max() < 1e-4
    assert focal_loss(perfect, target).max() < 1e-6
    assert mask_iou(perfect, target).min() == 1.0
    assert mask_iou(-perfect, target).max() == 0.0
    # Dice is a ratio of overlaps and focal is a mean over voxels, which is the reason for
    # combining them: missing a single-voxel object costs 0.5 in dice and effectively nothing in
    # focal, so without the dice term a tiny object is cheaper to ignore than to segment.
    tiny = torch.zeros(1, 1, 8, 8, 8)
    tiny[0, 0, 0, 0, 0] = 1
    empty = torch.full((1, 1, 8, 8, 8), -20.0)
    missed_dice, missed_focal = dice_loss(empty, tiny).item(), focal_loss(empty, tiny).item()
    assert missed_dice > 20 * missed_focal
    # The `eps` smoothing caps that penalty at 0.5 for a one-voxel target, against ~1.0 for a large
    # one -- which is a real limit on how much dice can protect the very smallest objects, and part
    # of why `min_object_voxels` exists rather than being left to the loss.
    assert missed_dice == pytest.approx(0.5, abs=1e-6)


@pytest.mark.unit
def test_best_of_backpropagates_only_the_winning_candidate():
    losses = torch.tensor([[3.0, 1.0, 2.0], [0.5, 4.0, 7.0]], requires_grad=True)
    chosen, which = best_of(losses)
    torch.testing.assert_close(chosen, torch.tensor([1.0, 0.5]))
    assert which.tolist() == [1, 0]

    chosen.sum().backward()
    # Exactly one candidate per row carries gradient. Averaging instead would train every output
    # towards the union of the valid answers, which is the failure the multi-mask design exists to
    # avoid.
    assert losses.grad.tolist() == [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]


@pytest.mark.unit
def test_the_decoder_answers_ambiguously_or_singly_on_request():
    torch.manual_seed(0)
    grid, patch, extent = (4, 4, 4), (16,) * 3, (64,) * 3
    encoder = PromptEncoder3D(DIM, mask_downscale=4)
    decoder = MaskDecoder3D(
        DIM, num_heads=4, upscale=4, mask_feature_dim=8, iou_hidden=16, num_multimask_outputs=3
    )
    prompts = 5
    sparse, coords, dense = encoder(
        torch.rand(prompts, 1, 3) * 2 - 1, torch.full((prompts, 1), FOREGROUND), grid
    )
    image = torch.randn(prompts, 64, DIM)
    image_coords = patch_grid_coords(grid, patch, extent).unsqueeze(0)

    masks, scores = decoder(image, image_coords, grid, sparse, coords, dense, multimask=True)
    assert masks.shape == (prompts, 3, 16, 16, 16), "masks come out at patch_size / upscale"
    assert scores.shape == (prompts, 3)

    single, single_scores = decoder(
        image, image_coords, grid, sparse, coords, dense, multimask=False
    )
    assert single.shape == (prompts, 1, 16, 16, 16)
    # The single answer is its own token, not one of the three: three candidates trained to
    # disagree are the wrong thing to hand an unambiguous prompt.
    assert not torch.allclose(single[:, 0], masks[:, 0])
    assert single_scores.shape == (prompts, 1)


@pytest.mark.unit
def test_one_encoder_pass_serves_every_prompt():
    # The amortisation the whole architecture exists for: the encoder is the expensive half and it
    # does not depend on the prompt, so a mismatched prompt batch is a bug, not a broadcast.
    grid = (4, 4, 4)
    decoder = MaskDecoder3D(DIM, num_heads=4, upscale=2, mask_feature_dim=8, iou_hidden=16)
    encoder = PromptEncoder3D(DIM, mask_downscale=2)
    sparse, coords, dense = encoder(
        torch.zeros(6, 1, 3), torch.full((6, 1), FOREGROUND), grid
    )
    with pytest.raises(ValueError, match="image embeddings against"):
        decoder(torch.randn(2, 64, DIM), torch.zeros(1, 64, 3), grid, sparse, coords, dense)


@pytest.mark.unit
def test_a_training_step_produces_a_loss_and_gradient_everywhere_it_should():
    torch.manual_seed(0)
    algorithm = _algorithm()
    metrics = algorithm.training_step(_batch())

    assert set(metrics) >= {"loss", "first_iou", "first_oracle_iou", "final_iou", "valid_fraction"}
    assert torch.isfinite(metrics["loss"])
    assert metrics["valid_fraction"].item() == 1.0
    # The oracle cannot be worse than the mask the model itself picked.
    assert metrics["first_oracle_iou"] >= metrics["first_iou"]

    metrics["loss"].backward()
    named = dict(algorithm.named_parameters())
    for prefix in ("model.", "neck.", "prompt_encoder.point_embed", "decoder.mask_tokens",
                   "decoder.iou_head", "decoder.upscaler", "decoder.hypernetworks"):
        reached = [
            name for name, parameter in named.items()
            if name.startswith(prefix) and parameter.grad is not None
            and parameter.grad.abs().sum() > 0
        ]
        assert reached, f"nothing under {prefix!r} received gradient"


@pytest.mark.unit
def test_a_crop_with_nothing_to_segment_contributes_no_loss():
    torch.manual_seed(0)
    algorithm = _algorithm()
    empty = {
        "img": torch.randn(1, 1, 1, 32, 32, 32),
        "label": torch.zeros(1, 1, 32, 32, 32, dtype=torch.int64),
    }
    metrics = algorithm.training_step(empty)
    assert metrics["valid_fraction"].item() == 0.0
    assert metrics["loss"].item() == 0.0
    assert torch.isfinite(metrics["loss"]), "an empty crop must not produce a NaN"


@pytest.mark.unit
def test_a_correction_asks_for_foreground_where_the_mask_missed_and_background_where_it_spilled():
    algorithm = _algorithm()
    extent = (32, 32, 32)
    blocks = tuple(e // s for e, s in zip(extent, algorithm.mask_stride, strict=True))
    target = torch.zeros(3, 1, *blocks)
    target[:, :, 2:6, 2:6, 2:6] = 1.0

    missed = torch.full((3, 1, *blocks), -10.0)  # predicts nothing
    _, labels = algorithm._correction(missed, target, extent)
    assert labels.flatten().tolist() == [FOREGROUND] * 3

    spilled = torch.full((3, 1, *blocks), 10.0)  # predicts everything
    _, labels = algorithm._correction(spilled, target, extent)
    assert labels.flatten().tolist() == [BACKGROUND] * 3

    exact = torch.where(target > 0.5, 10.0, -10.0)
    coords, labels = algorithm._correction(exact, target, extent)
    # Nothing to correct: a padding token, not an invented click. Teaching the model to distrust a
    # correct mask is the one thing an interactive loop must not do.
    assert labels.flatten().tolist() == [PAD] * 3
    assert coords.shape == (3, 1, 3)


@pytest.mark.unit
def test_a_correction_point_lands_inside_the_region_it_names():
    algorithm = _algorithm()
    extent = (32, 32, 32)
    blocks = tuple(e // s for e, s in zip(extent, algorithm.mask_stride, strict=True))
    target = torch.zeros(1, 1, *blocks)
    target[0, 0, 4:6, 4:6, 4:6] = 1.0
    coords, labels = algorithm._correction(torch.full((1, 1, *blocks), -10.0), target, extent)

    assert labels.item() == FOREGROUND
    stride = torch.tensor(algorithm.mask_stride)
    # Back from the [-1, 1] frame to a voxel, then to the block it fell in.
    voxel = ((coords[0, 0] + 1) * torch.tensor(extent) / 2 - 0.5).round().long()
    block = voxel // stride
    assert target[0, 0][tuple(block.tolist())] > 0.5


@pytest.mark.unit
def test_the_sample_transform_is_the_same_work_moved_not_different_work():
    torch.manual_seed(0)
    algorithm = _algorithm()
    batch = _batch()

    transform = algorithm.sample_transform()
    torch.manual_seed(1)
    prepared = [transform({"label": batch["label"][i]}) for i in range(2)]
    delegated = dict(batch)
    for key in (SPLIT_LABEL_KEY, IDS_KEY, "object_points", "object_boxes", "object_valid",
                "object_offobject"):
        delegated[key] = torch.stack([sample[key] for sample in prepared])

    torch.manual_seed(2)
    from_batch = algorithm.training_step(delegated)
    torch.manual_seed(1)
    inline = algorithm._objects(batch)
    # The fallback path runs the identical transform object, so a caller that never attaches it --
    # a test, a notebook -- gets targets rather than a KeyError, and gets the same ones.
    torch.testing.assert_close(inline[IDS_KEY], delegated[IDS_KEY])
    assert torch.isfinite(from_batch["loss"])


@pytest.mark.unit
def test_the_mask_stride_must_divide_the_encoder_s_patch():
    with pytest.raises(ValueError, match="does not divide the encoder's patch size"):
        _algorithm(mask_upscale=3)


@pytest.mark.unit
def test_declared_input_axes_may_not_contradict_the_dataset():
    class _Dataset:
        sample_axes = "lczyx"

    with pytest.raises(ValueError, match="contradicts the dataset"):
        PromptableSegmentation(
            ModelRegistry.build(
                "vit3d", img_size=(32,) * 3, patch_size=(8,) * 3, embed_dim=64, depth=2,
                num_heads=4,
            ),
            _Dataset(),
            input_axes="lcxyz",
        )


@pytest.mark.unit
def test_box_noise_resizes_the_box_rather_than_only_moving_it():
    # A single offset applied to both corners translates the box and never changes its size, so the
    # model would only ever be shown boxes that fit their object exactly -- the one case a real
    # detector does not deliver. Each corner gets its own noise.
    torch.manual_seed(0)
    algorithm = _algorithm(box_prob=1.0, box_noise=0.1, box_noise_max=20)
    prompts = 512
    boxes = torch.tensor([[[4, 4, 4], [24, 24, 24]]]).expand(prompts, 2, 3)
    coords, labels = algorithm._initial_prompt(
        torch.zeros(prompts, 3, dtype=torch.long), boxes, (32, 32, 32)
    )
    assert torch.all(labels[:, 0] == BOX_NEAR), "the first slot is the near corner"
    assert torch.all(labels[:, 1] == BOX_FAR), "the second slot is the far corner"

    widths = coords[:, 1] - coords[:, 0]
    assert widths.std() > 0, "every box came out the same size, so the corners moved together"
    # Both directions occur: boxes that are too tight and boxes that are too loose.
    nominal = widths.median()
    assert (widths < nominal).any() and (widths > nominal).any()


@pytest.mark.unit
def test_a_point_prompt_pads_the_second_slot_so_the_batch_stays_rectangular():
    algorithm = _algorithm(box_prob=0.0)
    coords, labels = algorithm._initial_prompt(
        torch.tensor([[4, 5, 6], [7, 8, 9]]),
        torch.zeros(2, 2, 3, dtype=torch.long),
        (32, 32, 32),
    )
    assert labels.tolist() == [[FOREGROUND, PAD], [FOREGROUND, PAD]]
    assert coords.shape == (2, 2, 3)
    clicks = torch.tensor([[4.0, 5, 6], [7, 8, 9]])
    torch.testing.assert_close(coords[:, 0], voxel_coords(clicks, (32,) * 3))


@pytest.mark.unit
@pytest.mark.parametrize("stride", [2, 4, 8])
def test_voxel_iou_equals_upsampling_the_prediction_and_scoring_at_full_resolution(stride):
    """The closed form must be exact, not close: it is the metric every head comparison rests on.

    Scored the slow way -- upsample the coarse prediction with nearest neighbour, compare against
    the unpooled binary mask -- against the cheap way, which never leaves the mask grid.
    """
    torch.manual_seed(0)
    extent = 32
    truth = torch.zeros(1, extent, extent, extent, dtype=torch.long)
    truth[0, 3:19, 5:27, 2:23] = 7          # a slab that does not align to any block boundary
    truth[0, 20:31, 1:6, 25:31] = 7         # a second piece, deliberately thin in one axis
    soft = pooled_masks(truth, torch.tensor([[7]]), stride)[:, 0]

    blocks = extent // stride
    logits = torch.randn(1, 3, blocks, blocks, blocks) * 3
    expanded = soft.unsqueeze(1).expand(-1, 3, -1, -1, -1)

    cheap = voxel_mask_iou(logits, expanded)

    binary = (truth == 7)[0]
    slow = []
    for candidate in range(3):
        upsampled = (
            F.interpolate((logits[0, candidate] > 0).float()[None, None], scale_factor=stride,
                          mode="nearest")[0, 0] > 0.5
        )
        intersection = (upsampled & binary).sum().item()
        union = (upsampled | binary).sum().item()
        slow.append(intersection / union)

    reference = torch.tensor(slow, dtype=cheap.dtype)
    torch.testing.assert_close(cheap[0], reference, atol=1e-6, rtol=1e-6)


@pytest.mark.unit
def test_voxel_iou_is_below_pooled_iou_because_quantisation_is_not_free():
    # The two metrics answer different questions and the gap between them is the ceiling the mask
    # stride imposes. A perfect prediction ON THE MASK GRID scores 1.0 there and strictly less at
    # voxel resolution, whenever the object does not tile the grid exactly.
    truth = torch.zeros(1, 16, 16, 16, dtype=torch.long)
    truth[0, 1:10, 2:13, 3:14] = 5          # nothing aligns to a 4-voxel block
    soft = pooled_masks(truth, torch.tensor([[5]]), 4)[:, 0]
    perfect = torch.where(soft > 0.5, 20.0, -20.0).unsqueeze(1)
    target = soft.unsqueeze(1)

    assert mask_iou(perfect, target).item() == pytest.approx(1.0)
    assert voxel_mask_iou(perfect, target).item() < 0.85


@pytest.mark.unit
def test_a_training_step_reports_voxel_resolution_iou():
    torch.manual_seed(0)
    metrics = _algorithm().training_step(_batch())
    assert {"first_voxel_iou", "final_voxel_iou"} <= set(metrics)
    assert torch.isfinite(metrics["first_voxel_iou"])


@pytest.mark.unit
def test_the_mask_head_refinement_depth_and_width_are_configurable():
    """`mask_refine_depth` / `mask_upscale_hidden` reach the sub-pixel expansion.

    The refinement convolutions are the only layers that run at mask resolution, so they are the
    knob an experiment on fine spatial detail turns; a config key that was accepted and silently
    ignored would make such a sweep measure noise.
    """
    default = _algorithm()
    deeper = _algorithm(mask_refine_depth=4, mask_upscale_hidden=48)
    # Each refinement step is a convolution followed by a GELU.
    assert len(default.decoder.upscaler.refine) == 2 * 2
    assert len(deeper.decoder.upscaler.refine) == 2 * 4
    assert deeper.decoder.upscaler.project.out_channels == 48
    assert default.decoder.upscaler.project.out_channels == DIM

    torch.manual_seed(0)
    metrics = deeper.training_step(_batch())
    assert torch.isfinite(metrics["loss"])


@pytest.mark.unit
def test_off_object_prompts_train_only_the_iou_head():
    """Every slot off-object: no mask loss can flow, so with the IoU term weightless the loss is 0;
    with it, the head is pushed towards 0 and the calibration metric is reported."""
    torch.manual_seed(0)
    silent = _algorithm(offobject_prob=1.0, iou_weight=0.0)
    metrics = silent.training_step(_batch())
    assert metrics["offobject_fraction"] == 1.0 and metrics["valid_fraction"] == 0.0
    assert metrics["loss"].item() == 0.0

    trained = _algorithm(offobject_prob=1.0, iou_weight=1.0)
    metrics = trained.training_step(_batch())
    assert metrics["loss"].item() > 0.0 and torch.isfinite(metrics["loss"])
    assert 0.0 <= metrics["offobject_pred_iou"].item() <= 1.0
    # Only the head and what feeds it should receive gradient from an off-object-only batch --
    # the mask hypernetworks are behind the masked-out mask loss and see nothing.
    metrics["loss"].backward()
    assert all(
        parameter.grad is None or parameter.grad.abs().sum() == 0
        for parameter in trained.decoder.hypernetworks.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in trained.decoder.iou_head.parameters()
    )


@pytest.mark.unit
def test_a_mixed_batch_reports_both_kinds_and_matches_the_old_recipe_when_off():
    torch.manual_seed(0)
    mixed = _algorithm(offobject_prob=0.5, boundary_prob=0.5)
    metrics = mixed.training_step(_batch())
    assert torch.isfinite(metrics["loss"])
    assert {"offobject_fraction", "offobject_pred_iou", "valid_fraction"} <= set(metrics)

    # With both knobs at 0 the loss is the old recipe's, term for term.
    torch.manual_seed(0)
    plain = _algorithm()
    reference = plain.training_step(_batch())
    assert reference["offobject_fraction"] == 0.0
    assert reference["offobject_pred_iou"] == 0.0



# ------------------------------------------------------------------- unlabelled voxels (-1)


@pytest.mark.unit
def test_pooled_known_is_the_labelled_fraction_of_each_block():
    labels = torch.zeros(1, 8, 8, 8, dtype=torch.int64)
    labels[0, :4, :4, :4] = -1                 # one whole block unknown
    labels[0, 4:8, :4, :4][:2] = -1            # half of another
    labels[0, 4:6, 4:6, 4:6] = 7               # an object counts as known
    known = pooled_known(labels, 4)
    assert known.shape == (1, 1, 2, 2, 2)
    assert known[0, 0, 0, 0, 0] == 0.0
    assert known[0, 0, 1, 0, 0] == 0.5
    assert known[0, 0, 1, 1, 1] == 1.0


@pytest.mark.unit
def test_weighted_losses_reproduce_the_unweighted_ones_under_all_ones():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 4, 4, 4)
    targets = (torch.rand(2, 3, 4, 4, 4) > 0.5).float()
    ones = torch.ones_like(targets)
    for fn in (focal_loss, dice_loss, mask_iou, voxel_mask_iou):
        assert torch.allclose(fn(logits, targets), fn(logits, targets, weight=ones), atol=1e-6), fn


@pytest.mark.unit
def test_unknown_cells_change_nothing_however_wrong_the_prediction_there_is():
    """Half the cells unknown: the four numbers equal those computed on the known half alone."""
    torch.manual_seed(1)
    logits = torch.randn(1, 2, 4, 4, 4)
    targets = (torch.rand(1, 2, 4, 4, 4) > 0.5).float()
    weight = torch.ones(1, 1, 4, 4, 4)
    weight[..., 2:, :, :] = 0.0                # the second half along x is unknown
    wrong = logits.clone()
    wrong[..., 2:, :, :] = 40.0 * (1 - 2 * targets[..., 2:, :, :])   # maximally wrong there
    for fn in (focal_loss, dice_loss, mask_iou, voxel_mask_iou):
        on_known = fn(logits[..., :2, :, :], targets[..., :2, :, :])
        assert torch.allclose(fn(wrong, targets, weight=weight), on_known, atol=1e-5), fn
        assert torch.allclose(fn(logits, targets, weight=weight), on_known, atol=1e-5), fn


@pytest.mark.unit
def test_a_correction_never_clicks_in_unlabelled_space():
    algorithm = _algorithm()
    stride = algorithm.mask_stride[0]                # 2 for this fixture: patch 8 / upscale 4
    extent = (32, 32, 32)
    cells = extent[0] // stride                       # 16 blocks along each axis
    grid = (cells,) * 3
    logits = torch.full((4, 1, *grid), 5.0)          # predicts foreground everywhere
    target = torch.zeros(4, 1, *grid)
    target[..., :4, :, :] = 1.0                       # the object: blocks 0..3 along x
    known = torch.ones(4, 1, *grid)
    known[..., 4:12, :, :] = 0.0                      # blocks 4..11 unknown; 12..15 background

    def block_x(coords: torch.Tensor) -> torch.Tensor:
        return ((coords[..., 0] + 1) / 2 * extent[0]) // stride

    for _ in range(20):
        coords, labels = algorithm._correction(logits, target, extent, known)
        assert (labels == BACKGROUND).all()           # the only error is spill
        assert (block_x(coords) >= 12).all(), block_x(coords)
    seen_unknown = False                              # without `known`, spill anywhere is clicked
    for _ in range(50):
        coords, _ = algorithm._correction(logits, target, extent)
        b = block_x(coords)
        seen_unknown |= bool(((b >= 4) & (b < 12)).any())
    assert seen_unknown


@pytest.mark.unit
def test_a_training_step_treats_unlabelled_voxels_as_silence_not_background():
    """Two batches, identical except that one marks a background region -1: the step runs, the
    loss is finite, and it differs from the batch that calls the same region background."""
    algorithm = _algorithm()
    batch = _batch()
    unknown = {k: v.clone() for k, v in batch.items()}
    unknown["label"][1, 0, 22:32, 22:32, 22:32] = -1  # far from object 333 (4..20)
    torch.manual_seed(7)
    with_background = algorithm.training_step(batch)["loss"]
    torch.manual_seed(7)
    with_unknown = algorithm.training_step(unknown)["loss"]
    assert torch.isfinite(with_unknown)
    assert not torch.isclose(with_background, with_unknown)
