"""Tensor parallelism over the DINOv3 stack must compute what one process computes.

The failure this tier exists to catch is silent. A fused q/k/v projection sharded the obvious way
still reshapes, still runs attention, and still returns a tensor of the right shape -- it is simply
attending with half of `k` where `v` belongs. Nothing about a loss curve says so. So every check
here is against an unparallelized copy of the same weights rather than against a shape or a
finite-value assertion.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import _StridedShard

from distributed.grad_norm import clip_grad_norm_
from distributed.parallel_dims import ParallelDims
from distributed.parallelize import apply_tensor_parallel, parallelize_model, shard_algorithm
from engine.activation_checkpoint import apply_activation_checkpointing
from models.dinov3_vit3d import DinoVisionTransformer3D

# Small enough to run on four CPU processes, but structurally the 7B model: SwiGLU feed-forward,
# LayerScale on both residual branches, storage tokens in front of the patch tokens, and a head
# count divisible by every tp size tested.
MODEL = dict(
    img_size=32,
    patch_size=16,
    in_chans=1,
    embed_dim=64,
    depth=4,
    num_heads=4,
    ffn_ratio=3.0,
    ffn_layer="swiglu",
    layerscale_init=1.0e-5,
    n_storage_tokens=2,
    qkv_bias=True,
    mask_k_bias=True,
    pos_embed_rope_type="vanilla",
    pos_embed_rope_dtype="fp32",
)


def _build(seed: int = 0, **overrides) -> DinoVisionTransformer3D:
    torch.manual_seed(seed)
    return DinoVisionTransformer3D(**{**MODEL, **overrides})


def _volume(seed: int = 1, size: int = 32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(2, 1, size, size, size, generator=generator)


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def _forward_worker(rank: int, world_size: int, size: int) -> tuple[float, int]:
    """Sharded patch tokens against the same weights run unsharded, and the token count."""
    dims = ParallelDims(tp=world_size)
    mesh = dims.build_mesh("cpu")

    reference = _build()
    sharded = _build()  # same seed, so the same weights before anything is distributed
    apply_tensor_parallel(sharded, mesh, dims)
    assert isinstance(sharded.blocks[0].attn.qkv.weight, DTensor)

    volume = _volume(size=size)
    with torch.no_grad():
        expected, _ = reference.patch_features(volume)
        actual, _ = sharded.patch_features(volume)
    return _max_abs_diff(expected, actual), expected.shape[1]


def _gradient_worker(rank: int, world_size: int) -> float:
    dims = ParallelDims(tp=world_size)
    mesh = dims.build_mesh("cpu")

    reference = _build()
    sharded = _build()
    apply_tensor_parallel(sharded, mesh, dims)

    volume = _volume()
    reference.patch_features(volume)[0].square().mean().backward()
    sharded.patch_features(volume)[0].square().mean().backward()

    worst = 0.0
    reference_grads = dict(reference.named_parameters())
    for name, parameter in sharded.named_parameters():
        grad = parameter.grad
        assert grad is not None, f"{name} received no gradient under tp={world_size}"
        full = grad.full_tensor() if isinstance(grad, DTensor) else grad
        worst = max(worst, _max_abs_diff(reference_grads[name].grad, full))
    return worst


def _qkv_placement_worker(rank: int, world_size: int) -> bool:
    """The fused projection's local rows are this rank's heads' q, k and v -- not a slice of q."""
    dims = ParallelDims(tp=world_size)
    mesh = dims.build_mesh("cpu")

    reference = _build()
    sharded = _build()
    apply_tensor_parallel(sharded, mesh, dims)

    weight = sharded.blocks[0].attn.qkv.weight
    assert weight.placements == (_StridedShard(0, split_factor=3),)

    dim = MODEL["embed_dim"]
    per_rank = dim // world_size
    whole = reference.blocks[0].attn.qkv.weight
    expected = torch.cat(
        [
            whole[part * dim + rank * per_rank : part * dim + (rank + 1) * per_rank]
            for part in range(3)
        ]
    )
    local_matches = torch.equal(weight.to_local(), expected)
    # And the global view is unchanged, so a checkpoint written under tp is laid out like any other
    reassembles = torch.equal(weight.full_tensor(), whole)
    return local_matches and reassembles


def _bias_mask_worker(rank: int, world_size: int) -> bool:
    """`LinearKMaskedBias`'s mask must be cut where the bias is, or it masks the wrong third."""
    dims = ParallelDims(tp=world_size)
    mesh = dims.build_mesh("cpu")
    sharded = _build()
    apply_tensor_parallel(sharded, mesh, dims)

    qkv = sharded.blocks[0].attn.qkv
    assert isinstance(qkv.bias_mask, DTensor)
    local = qkv.bias_mask.to_local()
    per_rank = MODEL["embed_dim"] // world_size
    # q enabled, k masked out, v enabled -- the same pattern the unsharded mask has, per rank.
    expected = torch.cat(
        [torch.ones(per_rank), torch.zeros(per_rank), torch.ones(per_rank)]
    )
    return torch.equal(local, expected)


class _Algorithm(nn.Module):
    """A stand-in for a strategy that owns a head beside the encoder, as `affinity_seg` does."""

    def __init__(self, model: DinoVisionTransformer3D) -> None:
        super().__init__()
        self.model = model
        self.head = nn.Linear(model.embed_dim, 3)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.model.patch_features(volume)
        return self.head(tokens)

    def validation_step(self, volume: torch.Tensor) -> torch.Tensor:
        # `shard_algorithm` registers this as an FSDP entry point, as it does for a real
        # `BaseAlgorithm`; without it the stand-in would diverge from what is being tested.
        return self(volume)


def _tp_fsdp_grad_clip_worker(rank: int, world_size: int) -> tuple[float, float]:
    """2D (dp_shard, tp) with a head outside the plan, through the clip the trainer runs.

    The clip is the point, and it is where tensor parallelism used to stop working: the plan turns
    the projections into DTensors on the 2D mesh while the patch embedding, the learned tokens, the
    output norm and the whole head stay on the 1D data-parallel one, and torch's `clip_grad_norm_`
    stacks the per-tensor norms and rejects the mix. Returned beside a single-process norm over the
    same weights, because "it no longer raises" is not the same claim as "it clips by the right
    amount".
    """
    dims = ParallelDims(dp_shard=world_size // 2, tp=2)
    mesh = dims.build_mesh("cpu")

    reference = _Algorithm(_build())
    reference(_volume()).square().mean().backward()
    expected = float(torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0))

    algorithm = _Algorithm(_build())
    apply_tensor_parallel(algorithm.model, mesh, dims)
    shard_algorithm(algorithm, mesh, dims)  # type: ignore[arg-type]

    algorithm(_volume()).square().mean().backward()
    norm = clip_grad_norm_(algorithm.parameters(), 1.0)
    # The multi-mesh path returns a plain, already-reduced norm rather than a `_NormPartial`.
    assert not isinstance(norm, DTensor)
    return float(norm), expected


def _tp_then_checkpointing_worker(rank: int, world_size: int) -> bool:
    """Applied in the trainer's order, the plan still reaches the projections.

    Reversed -- checkpointing first -- `parallelize_module` would resolve `blocks.0.attn.qkv`
    through `named_children()`, find `_checkpoint_wrapped_module` where it expected `attn`, and
    match nothing *without raising*, leaving a run that trains fully replicated at tp times the
    memory the setting was chosen to avoid. `apply_tensor_parallel` refuses instead, which is the
    other half of what this pins.
    """
    dims = ParallelDims(tp=world_size)
    mesh = dims.build_mesh("cpu")

    algorithm = _Algorithm(_build())
    apply_tensor_parallel(algorithm.model, mesh, dims)
    apply_activation_checkpointing(algorithm.model)
    sharded_after = isinstance(algorithm.model.blocks[0]._checkpoint_wrapped_module.attn.qkv.weight,
                               DTensor)

    reversed_order = _Algorithm(_build())
    apply_activation_checkpointing(reversed_order.model)
    try:
        apply_tensor_parallel(reversed_order.model, mesh, dims)
        refused = False
    except ValueError as error:
        refused = "already wrapped for activation checkpointing" in str(error)

    out = algorithm(_volume())
    return sharded_after and refused and out.isfinite().all().item()


def _stochastic_depth_refused_worker(rank: int, world_size: int) -> bool:
    dims = ParallelDims(tp=world_size)
    mesh = dims.build_mesh("cpu")
    model = _build(drop_path_rate=0.1)
    try:
        apply_tensor_parallel(model, mesh, dims)
    except ValueError as error:
        return "stochastic depth" in str(error)
    return False


@pytest.mark.cpu_dist
@pytest.mark.parametrize("world_size", [2, 4])
def test_tp_forward_matches_one_process(run_distributed, world_size):
    results = run_distributed(_forward_worker, world_size=world_size, args=(32,))
    for difference, _ in results:
        assert difference < 1e-5, f"tp={world_size} forward diverged by {difference}"


@pytest.mark.cpu_dist
def test_tp_forward_matches_one_process_with_an_indivisible_sequence(run_distributed):
    # 2^3 patches + CLS + 2 storage = 11 tokens over 4 ranks. Sequence parallelism has to shard
    # that unevenly, gather it for attention, and scatter it back.
    results = run_distributed(_forward_worker, world_size=4, args=(32,))
    difference, tokens = results[0]
    assert tokens == 8, "the sequence this case is about is set by img_size / patch_size"
    assert difference < 1e-5


@pytest.mark.cpu_dist
def test_tp_gradients_match_one_process(run_distributed):
    for worst in run_distributed(_gradient_worker, world_size=2):
        assert worst < 1e-5, f"gradients diverged by {worst}"


@pytest.mark.cpu_dist
def test_fused_qkv_shards_by_head_and_reassembles(run_distributed):
    assert all(run_distributed(_qkv_placement_worker, world_size=4))


@pytest.mark.cpu_dist
def test_masked_key_bias_follows_its_bias(run_distributed):
    assert all(run_distributed(_bias_mask_worker, world_size=2))


@pytest.mark.cpu_dist
def test_tp_composes_with_fsdp_through_grad_clipping(run_distributed):
    results = run_distributed(_tp_fsdp_grad_clip_worker, world_size=4)
    for norm, expected in results:
        assert norm == pytest.approx(expected, rel=1e-4), (
            f"clipped on a norm of {norm} where one process measures {expected}"
        )
    assert all(norm == pytest.approx(results[0][0]) for norm, _ in results)


@pytest.mark.cpu_dist
def test_tensor_parallel_must_precede_activation_checkpointing(run_distributed):
    assert all(run_distributed(_tp_then_checkpointing_worker, world_size=2))


@pytest.mark.cpu_dist
def test_tp_refuses_stochastic_depth(run_distributed):
    assert all(run_distributed(_stochastic_depth_refused_worker, world_size=2))


def _fsdp_units_worker(rank: int, world_size: int) -> bool:
    """Each block is its own FSDP unit, so a forward gathers one block's weights at a time."""
    from torch.distributed.fsdp import FSDPModule

    dims = ParallelDims(dp_shard=world_size)
    mesh = dims.build_mesh("cpu")
    model = _build()
    parallelize_model(model, mesh, dims)
    every_block_is_a_unit = all(isinstance(block, FSDPModule) for block in model.blocks)
    model.patch_features(_volume())[0].square().mean().backward()
    return every_block_is_a_unit and model.blocks[0].attn.qkv.weight.grad is not None


@pytest.mark.cpu_dist
def test_transformer_blocks_become_their_own_fsdp_units(run_distributed):
    assert all(run_distributed(_fsdp_units_worker, world_size=2))
