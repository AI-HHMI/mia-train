"""Unit tests for DINOv3 self-supervised pretraining: losses, view generation, and the algorithm."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from algorithms.dinov3.losses import DINOLoss, IBOTPatchLoss, KoLeoLoss, sinkhorn_knopp
from algorithms.dinov3.multicrop import (
    AugmentationConfig,
    block_mask,
    photometric,
    random_resized_crop,
)
from algorithms.dinov3_ssl import DINOv3
from algorithms.registry import AlgorithmRegistry
from layers.dinov3.head import DINOHead
from models.dinov3_vit import DinoVisionTransformer
from models.dinov3_vit3d import DinoVisionTransformer3D
from models.vit import ViT3D

PROTOTYPES = 64


def _encoder(rank: int):
    # Annotated because a heterogeneous dict literal infers dict[str, object], which cannot be
    # unpacked into the models' typed signatures.
    common: dict[str, Any] = dict(patch_size=8, in_chans=1, embed_dim=96, depth=2, num_heads=4,
                                  pos_embed_rope_dtype="fp32")
    if rank == 2:
        return DinoVisionTransformer(img_size=32, **common)
    return DinoVisionTransformer3D(img_size=16, **common)


def _algorithm(rank: int = 3, **overrides: Any) -> DINOv3:
    kwargs: dict[str, Any] = dict(
        input_axes="lcxy" if rank == 2 else "lcxyz",
        total_steps=100,
        global_crop_size=32 if rank == 2 else 16,
        local_crop_size=16 if rank == 2 else 8,
        n_local_crops=2,
        head_n_prototypes=PROTOTYPES,
        head_hidden_dim=64,
        head_bottleneck_dim=32,
        warmup_teacher_temp_steps=10,
    )
    kwargs.update(overrides)
    return DINOv3(_encoder(rank), **kwargs)


def _batch(rank: int, batch_size: int = 2) -> dict[str, torch.Tensor]:
    spatial = (64, 64) if rank == 2 else (32, 32, 32)
    return {"img": torch.rand(batch_size, 1, 1, *spatial)}


BOTH_RANKS = pytest.mark.parametrize("rank", [2, 3])


# ---------------------------------------------------------------- Sinkhorn


@pytest.mark.unit
def test_sinkhorn_rows_are_distributions():
    torch.manual_seed(0)
    out = sinkhorn_knopp(torch.randn(8, PROTOTYPES), 0.07)
    assert torch.allclose(out.sum(dim=-1), torch.ones(8), atol=1e-5)
    assert (out >= 0).all()


@pytest.mark.unit
def test_sinkhorn_balances_prototype_usage():
    """The point of it: prevent collapse by making every prototype carry similar mass.

    A raw softmax over logits that happen to favour a few prototypes leaves the rest unused, which
    is exactly the degenerate solution the teacher must not hand the student.
    """
    torch.manual_seed(0)
    # A realistic logit scale: the head L2-normalises its bottleneck, so logits are bounded by the
    # prototype row norms rather than free to be arbitrarily large.
    logits = torch.randn(64, PROTOTYPES) * 0.5
    plain = torch.softmax(logits / 0.07, dim=-1)
    balanced = sinkhorn_knopp(logits, 0.07)

    # Coefficient of variation across prototypes: lower means more evenly used.
    spread = lambda p: (p.sum(0).std() / p.sum(0).mean()).item()  # noqa: E731
    assert spread(balanced) < spread(plain)
    assert (balanced.sum(0) > 1e-6).all(), "no prototype is left completely unused"


@pytest.mark.unit
def test_sinkhorn_overflows_on_extreme_logits_like_the_reference():
    """Documenting a real sharp edge rather than papering over it.

    `exp(logits / temperature)` has no clamp, matching upstream, so a logit far outside what an
    L2-normalised bottleneck can produce overflows to inf and poisons the loss. Real training does
    not reach here, but a future change to the head's scale could.
    """
    extreme = sinkhorn_knopp(torch.randn(8, PROTOTYPES) * 3, 0.07)
    assert not torch.isfinite(extreme).all()


# ---------------------------------------------------------------- DINO loss


@pytest.mark.unit
def test_dino_loss_of_uniform_predictions_is_log_k():
    """Cross-entropy against a uniform target is ln(K) when the student is also uniform."""
    student = torch.zeros(2, 4, PROTOTYPES)
    teacher = torch.full((2, 4, PROTOTYPES), 1.0 / PROTOTYPES)
    loss = DINOLoss(PROTOTYPES)(student, teacher)
    assert loss.item() == pytest.approx(math.log(PROTOTYPES), abs=1e-5)


@pytest.mark.unit
def test_dino_loss_rewards_agreeing_with_the_teacher():
    torch.manual_seed(0)
    teacher = torch.zeros(1, 3, PROTOTYPES)
    teacher[0, torch.arange(3), torch.tensor([1, 2, 3])] = 1.0  # confident, one-hot

    agreeing = teacher * 20.0  # student logits that put mass in the same places
    disagreeing = torch.roll(teacher, shifts=5, dims=-1) * 20.0

    loss = DINOLoss(PROTOTYPES)
    assert loss(agreeing, teacher) < loss(disagreeing, teacher)


@pytest.mark.unit
def test_ignore_diagonal_drops_the_self_pairing():
    """With 2 views, ignoring the diagonal must leave exactly the 2 cross terms."""
    torch.manual_seed(0)
    student = torch.randn(2, 4, PROTOTYPES)
    teacher = torch.softmax(torch.randn(2, 4, PROTOTYPES), dim=-1)
    loss = DINOLoss(PROTOTYPES)

    log_probs = torch.log_softmax(student / loss.student_temp, dim=-1)
    cross = -sum(
        torch.einsum("bk,bk->", log_probs[s], teacher[t]) for s, t in [(0, 1), (1, 0)]
    ) / (4 * 2)
    assert loss(student, teacher, ignore_diagonal=True).item() == pytest.approx(
        cross.item(), abs=1e-5
    )


# ---------------------------------------------------------------- iBOT and KoLeo


@pytest.mark.unit
def test_ibot_masked_loss_only_counts_masked_patches():
    torch.manual_seed(0)
    masks = torch.zeros(4, 16, dtype=torch.bool)
    masks[:2, :5] = True
    n_masked = int(masks.sum())
    student = torch.randn(n_masked, PROTOTYPES)
    teacher = torch.softmax(torch.randn(n_masked, PROTOTYPES), dim=-1)

    loss = IBOTPatchLoss(PROTOTYPES).forward_masked(
        student, teacher, masks, n_masked_patches=n_masked
    )
    assert torch.isfinite(loss) and loss > 0


@pytest.mark.unit
def test_koleo_penalises_a_collapsed_batch():
    """Its whole job: embeddings that crowd together should score worse than spread ones."""
    torch.manual_seed(0)
    spread = torch.randn(16, 8)
    collapsed = torch.randn(1, 8).expand(16, 8) + torch.randn(16, 8) * 1e-3

    loss = KoLeoLoss()
    assert loss(collapsed) > loss(spread)


# ---------------------------------------------------------------- head


@pytest.mark.unit
def test_head_l2_normalises_its_bottleneck():
    """The prototype layer sees a unit vector, so its logits are cosine-like."""
    head = DINOHead(32, PROTOTYPES, hidden_dim=64, bottleneck_dim=16)
    head.init_weights()
    bottleneck = torch.nn.functional.normalize(head.mlp(torch.randn(5, 32)), dim=-1)
    assert torch.allclose(bottleneck.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert head(torch.randn(5, 32)).shape == (5, PROTOTYPES)


@pytest.mark.unit
def test_head_prototype_layer_has_no_bias():
    # Unlike DINOv2 there is no weight norm here either -- a plain bias-free Linear.
    head = DINOHead(32, PROTOTYPES)
    assert head.last_layer.bias is None
    assert not hasattr(head.last_layer, "weight_g")


# ---------------------------------------------------------------- views


@pytest.mark.unit
@BOTH_RANKS
def test_two_global_views_differ(rank):
    """DINO's objective is trivial if the views match: the constant function minimises it."""
    torch.manual_seed(0)
    spatial = (64,) * rank
    volumes = torch.rand(4, 1, *spatial)
    out = (16,) * rank
    first = random_resized_crop(volumes, out, (0.32, 1.0))
    second = random_resized_crop(volumes, out, (0.32, 1.0))

    assert first.shape == (4, 1, *out)
    assert not torch.allclose(first, second)


@pytest.mark.unit
@BOTH_RANKS
def test_local_crops_cover_less_than_global_ones(rank):
    """Local crops must be a narrower field of view, or the local term teaches nothing extra."""
    torch.manual_seed(0)
    # A linear ramp, so the range of values a crop contains *is* the extent it covers. (Variance
    # would say the opposite of what you expect: a tight crop is upsampled, which preserves the
    # source's contrast, while a wide one is averaged down.)
    ramp = torch.linspace(0, 1, 32)
    for _ in range(rank - 1):
        ramp = ramp.unsqueeze(-1)
    ramp = ramp.expand(*((32,) * rank)).contiguous()[None, None].repeat(64, 1, *([1] * rank))

    wide = random_resized_crop(ramp, (16,) * rank, (0.9, 1.0), flip_prob=0, rotate_prob=0)
    tight = random_resized_crop(ramp, (16,) * rank, (0.05, 0.1), flip_prob=0, rotate_prob=0)
    spanned = lambda t: (  # noqa: E731
        t.amax(dim=tuple(range(1, t.ndim))) - t.amin(dim=tuple(range(1, t.ndim)))
    ).mean()
    assert spanned(tight) < spanned(wide) / 2


@pytest.mark.unit
@BOTH_RANKS
def test_photometric_leaves_shape_and_finiteness_alone(rank):
    torch.manual_seed(0)
    views = torch.rand(4, 1, *((16,) * rank))
    out = photometric(views.clone(), AugmentationConfig())
    assert out.shape == views.shape
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, views)


@pytest.mark.unit
def test_blur_preserves_the_mean():
    """A normalised kernel with reflected borders must not shift overall intensity; zero padding
    would darken every edge, a systematic artefact the model could key on."""
    from algorithms.dinov3.multicrop import _gaussian_blur

    torch.manual_seed(0)
    views = torch.rand(4, 1, 16, 16, 16)
    blurred = _gaussian_blur(views, torch.full((4,), 1.5))
    assert blurred.mean().item() == pytest.approx(views.mean().item(), abs=1e-3)
    assert blurred.var() < views.var()


# ---------------------------------------------------------------- masking


@pytest.mark.unit
@pytest.mark.parametrize("grid", [(8, 8), (4, 4, 4)])
def test_mask_respects_its_ratio_bound_and_leaves_some_samples_whole(grid):
    torch.manual_seed(0)
    masks = block_mask(grid, batch=8, ratio=(0.1, 0.5), sample_probability=0.5)

    fractions = masks.float().mean(dim=1)
    assert masks.shape == (8, math.prod(grid))
    assert fractions.max() <= 0.5 + 1e-6, "the upper bound is a cap, not a suggestion"
    assert (fractions == 0).sum() == 4, "half the batch is deliberately left unmasked"


@pytest.mark.unit
def test_mask_blocks_are_contiguous_not_scattered():
    """Block masking is the point: scattered patches can be filled in from immediate neighbours."""
    torch.manual_seed(0)
    grid = (12, 12)
    masks = block_mask(grid, batch=2, ratio=(0.4, 0.4), sample_probability=1.0).reshape(2, *grid)

    # A contiguous block has far fewer masked/unmasked boundaries than a scattered set of the
    # same size would.
    masked = masks[0]
    boundaries = (masked[:, :-1] != masked[:, 1:]).sum() + (masked[:-1] != masked[1:]).sum()
    scattered = torch.zeros_like(masked).reshape(-1)
    scattered[torch.randperm(masked.numel())[: int(masked.sum())]] = True
    scattered = scattered.reshape(grid)
    scattered_boundaries = (scattered[:, :-1] != scattered[:, 1:]).sum() + (
        scattered[:-1] != scattered[1:]
    ).sum()
    assert boundaries < scattered_boundaries


# ---------------------------------------------------------------- the algorithm


@pytest.mark.unit
def test_registered_under_its_config_name():
    assert AlgorithmRegistry.get("dinov3") is DINOv3


@pytest.mark.unit
@BOTH_RANKS
def test_one_algorithm_serves_both_ranks(rank):
    algorithm = _algorithm(rank)
    assert algorithm.spatial_rank == rank
    out = algorithm.training_step(_batch(rank))
    assert torch.isfinite(out["loss"])
    assert {"dino_global", "dino_local", "koleo", "loss"} <= set(out)


@pytest.mark.unit
def test_rejects_an_encoder_without_the_masked_forward():
    encoder = ViT3D(img_size=(16, 16, 16), patch_size=(8, 8, 8), in_channels=1, embed_dim=32,
                    depth=1, num_heads=4)
    with pytest.raises(TypeError, match="forward_features"):
        DINOv3(encoder, input_axes="lcxyz")


@pytest.mark.unit
def test_loss_starts_near_log_prototypes():
    """At initialisation every prototype is equally likely, so the cross-entropy is ln(K).

    A useful smoke signal: a loss far from this at step 0 means the head or the centering is
    wrong, not that training is going badly.
    """
    torch.manual_seed(0)
    out = _algorithm(3).training_step(_batch(3))
    assert out["dino_global"].item() == pytest.approx(math.log(PROTOTYPES), rel=0.05)


@pytest.mark.unit
def test_teacher_starts_identical_to_the_student():
    """It has to: distilling from a differently-initialised teacher is distilling noise."""
    algorithm = _algorithm(3)
    student = dict(algorithm.student.named_parameters())
    for name, teacher_param in algorithm.teacher.named_parameters():
        assert torch.equal(teacher_param, student[name]), name


@pytest.mark.unit
def test_teacher_takes_no_gradient_and_moves_only_by_ema():
    algorithm = _algorithm(3)
    before = algorithm.teacher["backbone"].blocks[0].attn.qkv.weight.detach().clone()

    algorithm.training_step(_batch(3))["loss"].backward()
    assert all(p.grad is None for p in algorithm.teacher.parameters())
    assert torch.equal(algorithm.teacher["backbone"].blocks[0].attn.qkv.weight, before)

    with torch.no_grad():  # pretend the optimizer moved the student
        algorithm.student["backbone"].blocks[0].attn.qkv.weight.add_(1.0)
    algorithm.update_teacher()
    assert not torch.equal(algorithm.teacher["backbone"].blocks[0].attn.qkv.weight, before)


@pytest.mark.unit
def test_teacher_stays_in_eval_mode():
    algorithm = _algorithm(3)
    algorithm.train()
    assert algorithm.student.training and not algorithm.teacher.training


@pytest.mark.unit
def test_teacher_temperature_warms_up_then_holds():
    algorithm = _algorithm(3, warmup_teacher_temp=0.04, teacher_temp=0.07,
                           warmup_teacher_temp_steps=10)
    assert algorithm.current_teacher_temp() == pytest.approx(0.04)
    algorithm._step.fill_(5)
    assert algorithm.current_teacher_temp() == pytest.approx(0.055)
    algorithm._step.fill_(50)
    assert algorithm.current_teacher_temp() == pytest.approx(0.07), "held after warmup"


@pytest.mark.unit
def test_teacher_momentum_rises_to_one_over_the_run():
    """Ending at 1.0 freezes the teacher, so the target stops moving as the student converges."""
    algorithm = _algorithm(3, total_steps=100, momentum_teacher=0.992,
                           final_momentum_teacher=1.0)
    assert algorithm.current_momentum() == pytest.approx(0.992)
    algorithm._step.fill_(100)
    assert algorithm.current_momentum() == pytest.approx(1.0)
    algorithm._step.fill_(50)
    assert 0.992 < algorithm.current_momentum() < 1.0


@pytest.mark.unit
def test_step_counter_survives_a_state_dict_round_trip():
    """The schedules depend on it, and the trainer restores model state without telling an
    algorithm what step it resumed at."""
    algorithm = _algorithm(3)
    algorithm._step.fill_(1234)

    restored = _algorithm(3)
    restored.load_state_dict(algorithm.state_dict())
    assert restored.step == 1234


@pytest.mark.unit
def test_only_training_steps_advance_the_schedule():
    algorithm = _algorithm(3)
    algorithm.validation_step(_batch(3))
    assert algorithm.step == 0
    algorithm.training_step(_batch(3))
    assert algorithm.step == 1
