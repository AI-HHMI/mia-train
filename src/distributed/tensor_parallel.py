"""Tensor-parallel styles for this repo's transformer stacks, beyond torch's stock four.

`ColwiseParallel`, `RowwiseParallel`, `SequenceParallel` and `PrepareModuleInput` describe a
Megatron block whose q, k and v are separate Linears and whose stack is driven through a single
tensor. The DINOv3 port is neither, so two gaps have to be filled here.

**The fused q/k/v projection.** `SelfAttention.qkv` is one `Linear(dim -> 3 * dim)` whose output is
`[q | k | v]`, and a plain column-parallel `Shard(0)` cuts that concatenation into contiguous
thirds of the *fused* extent rather than into whole heads: at `tp = 4` and `dim = 8`, rank 2 gets
global rows 12-17, which is the tail of `k` and the head of `v`. Attention cannot be computed from
that, and nothing about the shapes says so -- the reshape to `(B, N, 3, heads, head_dim)` still
succeeds and produces numbers.

`_StridedShard(0, split_factor=3)` is the placement that says "this extent was already cut into 3
pieces before you sharded it", which is exactly true of a fused qkv. It hands rank *r* rows
`[q_r | k_r | v_r]` -- its own heads' share of each of the three -- so the existing reshape is
correct on the local tensor with the local head count, and `full_tensor()` still reassembles the
ordinary `[q | k | v]` global weight, so a checkpoint written under tensor parallelism is
byte-identical in layout to one written without it. It is also the placement FSDP2 itself composes
with a TP shard, so the 2D (dp, tp) case works rather than being a special case to write.

**Entering and leaving sequence parallelism.** Sequence parallelism is what makes tensor
parallelism worth anything for large volumetric crops: plain TP shards the projections but leaves
the residual stream replicated, and the residual stream is what activation checkpointing stores --
40 blocks of `(B, N, embed_dim)`, which at a 1024-cube is 86 GB and is the term that decides
whether a crop fits. Sharding the token axis divides it by `tp` at *identical* communication
volume, since the all-reduce a plain TP block does is already an all-gather plus a reduce-scatter.

Torch's `PrepareModuleInput` would do the entry, but it indexes `inputs[0]` expecting a tensor and
`SelfAttentionBlock.forward` takes a *list* of per-crop token tensors; `ScatterSequence` below does
the same job through that signature, applied to the first block so the stream is sharded from there
on.

The matching **exit is not here** -- it is in the models, which gather the stream themselves before
their output norms. It was a `GatherSequence` forward hook until `torch.compile` was tried over
`tp > 1`: Dynamo does not run a forward hook that replaces a module's output, so under compile the
stream stayed sharded, met the replicated final norm, and raised `aten.native_layer_norm.default
got mixed torch.Tensor and DTensor` -- with nothing to say a hook had been skipped. Forward
*pre*-hooks are honoured, which is why the entry can stay one.

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import (
    DTensor,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor.placement_types import _StridedShard

#: How many projections the fused qkv Linear holds, in output order.
QKV_FUSION = 3

#: Token axis of a `(batch, tokens, features)` activation.
SEQUENCE_DIM = 1


class FusedQKVParallel(ColwiseParallel):
    """Column-parallel for one Linear whose output holds q, k and v concatenated.

    Differs from `ColwiseParallel` in the placement alone -- `_StridedShard(0, split_factor=3)`
    rather than `Shard(0)` -- for the reason in this module's docstring. Everything else, including
    the replicated input and the local output the attention math consumes, is inherited.

    Buffers are distributed alongside the parameters, which the base style does not do because no
    stock Linear has one. `LinearKMaskedBias` does: a 0/1 mask over the fused output extent that is
    multiplied into the bias, so it has to be cut exactly where the bias is cut or the two stop
    lining up and the key bias stops being the part that is masked.
    """

    def _placements(self) -> list[Any]:
        return [_StridedShard(0, split_factor=QKV_FUSION)]

    def _partition_linear_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh) -> None:
        placements = self._placements()
        for param_name, param in list(module.named_parameters(recurse=False)):
            module.register_parameter(
                param_name,
                nn.Parameter(
                    distribute_tensor(
                        param, device_mesh, placements, src_data_rank=self.src_data_rank
                    ),
                    requires_grad=param.requires_grad,
                ),
            )
        for buffer_name, buffer in list(module.named_buffers(recurse=False)):
            module.register_buffer(
                buffer_name,
                distribute_tensor(
                    buffer, device_mesh, placements, src_data_rank=self.src_data_rank
                ),
                persistent=buffer_name not in module._non_persistent_buffers_set,
            )

    @staticmethod
    def _prepare_output_fn(
        output_layouts: Any, use_local_output: bool, mod: nn.Module, outputs: Any, device_mesh: Any
    ) -> torch.Tensor:
        """Hand back the local `[q_r | k_r | v_r]` block, never a redistribution of it.

        The base implementation redistributes to `output_layouts` first, and the placement
        propagation infers for this matmul is the strided one rather than the plain `Shard(-1)`
        that `ColwiseParallel.__init__` defaults to -- so inheriting it would insert an all-gather
        per block to reach a layout the attention math does not want anyway.
        """
        return outputs.to_local()


def _map_tokens(value: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """Apply `fn` to a token tensor, or to each of a list of them.

    `SelfAttentionBlock` accepts either: `forward_features_list` passes a list, one entry per crop,
    while `get_intermediate_layers` passes a bare tensor.
    """
    if isinstance(value, list):
        return [fn(item) for item in value]
    return fn(value)


class ScatterSequence(ParallelStyle):
    """Enter sequence parallelism at this module: replicated tokens in, this rank's slice out.

    Applied to the *first* block of a stack. The input is replicated by construction -- every
    tensor-parallel peer reads the same batch and runs the same patch embedding -- so taking a
    slice needs no collective; the gradient path back out is the all-gather that pairs with it.
    """

    def __init__(self, *, sequence_dim: int = SEQUENCE_DIM) -> None:
        super().__init__()
        self.sequence_dim = sequence_dim

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        def hook(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
            scattered = _map_tokens(
                args[0],
                lambda t: DTensor.from_local(
                    t, device_mesh, [Replicate()], run_check=False
                ).redistribute(placements=[Shard(self.sequence_dim)]),
            )
            return (scattered, *args[1:])

        module.register_forward_pre_hook(hook)
        return module


def feedforward_projection_names(ffn: nn.Module) -> tuple[tuple[str, ...], str]:
    """A feed-forward block's Linears split into the column-parallel ones and the row-parallel one.

    By position rather than by name, because the two variants disagree about names -- `Mlp` has
    `fc1`/`fc2` and `SwiGLUFFN` `w1`/`w2`/`w3` -- while both register the projections that widen
    to the hidden dimension before the one that narrows back. Naming them here would mean this
    file tracked `layers/` by hand, and a plan that missed a projection would leave it replicated
    with nothing to say so.
    """
    linears = [
        (name, child) for name, child in ffn.named_children() if isinstance(child, nn.Linear)
    ]
    if len(linears) < 2:
        raise ValueError(
            f"{type(ffn).__name__} has {len(linears)} Linear child(ren); a feed-forward block "
            "column-parallel on the way up and row-parallel on the way back needs at least two"
        )
    *up, (down_name, down) = linears
    widths = {module.out_features for _, module in up}
    if widths != {down.in_features}:
        raise ValueError(
            f"{type(ffn).__name__}'s last Linear {down_name!r} takes {down.in_features} features "
            f"but the ones before it produce {sorted(widths)}; the projections do not meet at a "
            "single hidden width, so which of them is the row-parallel one is not decidable here"
        )
    return tuple(name for name, _ in up), down_name


def self_attention_block_plan(
    prefix: str,
    *,
    ffn_inputs: Sequence[str],
    ffn_output: str,
    layer_scales: Sequence[str] = ("ls1", "ls2"),
    first: bool = False,
) -> dict[str, ParallelStyle]:
    """The Megatron TP + sequence-parallel plan for one `layers.dinov3.block.SelfAttentionBlock`.

    Written here rather than in either model because the 2D and 3D DINOv3 encoders are the same
    block stack and a plan restated per model is a plan that drifts per model. The FFN's projection
    names are arguments because the two feed-forward variants disagree about them -- `Mlp` has
    `fc1`/`fc2`, `SwiGLUFFN` has `w1`/`w2`/`w3` -- and a plan naming only one of them would shard
    nothing on a model built with the other, silently, since a module absent from a plan is simply
    left replicated.

    The token axis stays sharded from `first` to `last`; the normalizations, the LayerScale gammas
    and the residual adds between them all operate on that sharded stream, which is where the
    activation memory goes. Attention and the FFN gather it back because each needs the whole
    sequence -- attention by definition, the FFN because its input is what the column-parallel
    projections expect replicated.
    """
    plan: dict[str, ParallelStyle] = {}
    if first:
        plan[prefix] = ScatterSequence()

    # Elementwise in the token axis, so they run directly on the sharded stream. `SequenceParallel`
    # replicates their parameters and keeps the DTensor wrapper on the way out, which is what lets
    # the residual add see two operands with the same placement.
    for name in ("norm1", "norm2", *layer_scales):
        plan[f"{prefix}.{name}"] = SequenceParallel()

    # Gather once at the module boundary rather than per projection: the FFN feeds the same input
    # to two column-parallel Linears, and letting each gather it would double the collective.
    plan[f"{prefix}.attn"] = PrepareModuleInput(
        input_layouts=(Shard(SEQUENCE_DIM),),
        desired_input_layouts=(Replicate(),),
        use_local_output=True,
    )
    plan[f"{prefix}.attn.qkv"] = FusedQKVParallel()
    plan[f"{prefix}.attn.proj"] = RowwiseParallel(
        output_layouts=Shard(SEQUENCE_DIM), use_local_output=False
    )

    plan[f"{prefix}.mlp"] = PrepareModuleInput(
        input_layouts=(Shard(SEQUENCE_DIM),),
        desired_input_layouts=(Replicate(),),
        use_local_output=True,
    )
    for name in ffn_inputs:
        plan[f"{prefix}.mlp.{name}"] = ColwiseParallel()
    plan[f"{prefix}.mlp.{ffn_output}"] = RowwiseParallel(
        output_layouts=Shard(SEQUENCE_DIM), use_local_output=False
    )
    return plan


def self_attention_stack_plan(
    blocks: Sequence[nn.Module], *, prefix: str = "blocks"
) -> dict[str, ParallelStyle]:
    """The plan for a whole `SelfAttentionBlock` stack, sequence entry and exit included.

    Written here rather than in either DINOv3 model because the 2D and 3D encoders are the same
    stack, and a plan restated per model is a plan that drifts per model.

    The stack's *exit* from sequence parallelism is the model's business, not this plan's; see the
    module docstring.

    Stochastic depth is refused rather than silently mis-sharded: with `drop_path > 0` the block
    runs its residual branches on a random *subset* of the batch through `SelfAttention.
    forward_list` and `Mlp.forward_list`, which are called as methods rather than through
    `__call__` and so bypass every hook `parallelize_module` installs. The projections would still
    be sharded, the forward would still run, and each rank would compute attention over its own
    quarter of the heads as though it were the whole thing.
    """
    if len(blocks) < 2:
        raise ValueError(
            "sequence parallelism enters at the first block and leaves at the last, so a stack of "
            f"{len(blocks)} block(s) cannot be planned; use tp = 1"
        )
    drop_path = {getattr(block, "sample_drop_ratio", 0.0) for block in blocks}
    if drop_path != {0.0}:
        raise ValueError(
            f"tensor parallelism does not support stochastic depth (drop_path_rate={max(drop_path)}"
            "), whose subset path drives attention and the FFN through `forward_list`, which the "
            "parallel styles' hooks do not see. Set drop_path_rate = 0."
        )

    # `cast` because `nn.Module.__getattr__` is typed `Tensor | Module`; the blocks this plans
    # for are `SelfAttentionBlock`s, whose `mlp` is always a module.
    ffn_inputs, ffn_output = feedforward_projection_names(cast(nn.Module, blocks[0].mlp))
    plan: dict[str, ParallelStyle] = {}
    for index, block in enumerate(blocks):
        layer_scales = tuple(
            name for name in ("ls1", "ls2") if not isinstance(getattr(block, name), nn.Identity)
        )
        plan.update(
            self_attention_block_plan(
                f"{prefix}.{index}",
                ffn_inputs=ffn_inputs,
                ffn_output=ffn_output,
                layer_scales=layer_scales,
                first=index == 0,
            )
        )
    return plan
