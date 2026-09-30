from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.tensor.parallel import ParallelStyle


class BaseModel(nn.Module, abc.ABC):
    """Pure architecture definition; exposes parameter counts and FLOP calculators."""

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        ...

    @abc.abstractmethod
    def flops(self, input_shape: tuple[int, ...]) -> int:
        """Estimated forward-pass FLOPs for a single input of the given shape (no batch dim).

        `input_shape` is the shape the caller intends to run, and an implementation must either
        answer for that shape or raise -- never quietly answer for the one it was configured with.
        The domain is the architecture's own: the DINOv3 encoders accept any shape they could run,
        because multi-crop SSL costs several resolutions within one step, while `ViT3D` and
        `MuViT3D` accept only their configured geometry, because `embed` admits nothing else.
        Both honour the contract; they differ in how wide it is, so a caller holding a `BaseModel`
        should be prepared for a `ValueError` on a shape that model could not run.

        This is an estimate of the *model's* arithmetic, not of a training step's: it excludes the
        backward pass and knows nothing about masking, multi-crop, or an algorithm's decoder or
        head. `engine.mfu` measures a real step instead of scaling this, and the module docstring
        there records how far apart the two land.
        """

    def prepare_input(self, batch: torch.Tensor, axes: str) -> torch.Tensor:
        """Turn a dataset-shaped batch into whatever this architecture consumes.

        `axes` is the dataset's per-sample axis order, e.g. miao's "lcxyz" — batch dimension
        excluded. How many scale levels an architecture accepts, and where it expects the
        channel axis, are properties of the architecture, so each one states and enforces its
        own contract here: a single-resolution encoder rejects a multi-scale batch, while a
        multi-scale one consumes the level axis directly.

        Only algorithms that hand raw dataset batches to a model need this; the base class
        cannot guess a correct answer, so it declines rather than inventing one.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement prepare_input, so it cannot be driven "
            "by an algorithm that passes dataset-shaped batches through the model"
        )

    def patch_features(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Encode one input into per-patch features -> (B, N, C) tokens and their patch grid.

        The entry point dense downstream tasks need. Encoders here disagree about how to produce
        tokens -- `ViT3D` splits `embed`/`encode` so masked autoencoding can drop tokens in
        between, while the DINOv3 models expose `forward_features` and return a dict -- and a
        segmentation head should not have to know which it was handed. Tokens come back in grid
        (row-major) order together with the grid they fill, because a head has to fold them back
        into a volume and the token count alone does not say what shape that was.

        Not every architecture has a single answer: a multi-scale encoder's sequence spans several
        grids at once, so it declines here rather than inventing one.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement patch_features, so it cannot drive a "
            "dense prediction head"
        )

    def layer_patch_features(
        self, x: torch.Tensor, layers: Sequence[int]
    ) -> tuple[list[torch.Tensor], tuple[int, ...]]:
        """Encode one input -> the patch tokens after each block in `layers`, and their patch grid.

        The multi-depth sibling of `patch_features`, for a head that reads intermediate layers as well
        as the last (UNETR's skip connections): one `(B, N, C)` tensor per entry of `layers` -- 0-based
        block indices, strictly increasing -- in that order, on the same grid `patch_features`
        returns. Declines by default, as `patch_features` does, so a head that needs it fails at
        construction on an architecture that has not said which tokens those are.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement layer_patch_features, so it cannot drive a "
            "head that reads intermediate encoder layers"
        )

    def pyramid_features(self, x: torch.Tensor) -> tuple[list[torch.Tensor], tuple[int, ...]]:
        """Encode one input -> a feature map per stride, finest first, and the last map's grid.

        The hierarchical sibling of `patch_features`, for a decoder that takes an encoder map at
        every resolution it climbs through (a U-Net's skip connections). One `(B, C_i, *grid_i)` map
        per stage, each at twice the previous stride. The last map carries the features
        `patch_features` hands a head, on the grid it returns. An encoder that implements this also
        names the maps' widths and strides as `pyramid_dims` and `pyramid_strides`, so a head can be
        built before anything runs. Declines by default, since a ViT has a single grid; for it,
        `layer_patch_features` and the UNETR head are the equivalent.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement pyramid_features, so it cannot drive a "
            "head that reads a feature map at each stride"
        )

    def extra_forward_methods(self) -> tuple[str, ...]:
        """Methods besides `forward` through which this model's parameters get used.

        FSDP2 all-gathers sharded parameters around `nn.Module.forward` and nothing else, so a
        model an algorithm drives directly — MAE runs `ViT3D.embed`, masks the tokens, then runs
        `ViT3D.encode` — must name those methods for `parallelize_model` to wrap them. Left empty
        by architectures that are only ever called through `forward`.
        """
        return ()

    def checkpointable_modules(self) -> tuple[nn.Module, ...]:
        """Submodules worth recomputing in backward when activation checkpointing is on.

        A transformer's answer is its blocks: they are repeated, each holds activations
        proportional to the sequence length, and each is cheap to rerun relative to what it
        stores. The engine decides *whether* to checkpoint; what constitutes a worthwhile region
        is a property of the architecture, so it is answered here.

        Empty by default, which makes `[trainer].activation_checkpointing` an error rather than a
        silent no-op on an architecture that has not declared one.
        """
        return ()

    def fsdp_units(self) -> tuple[nn.Module, ...]:
        """Submodules to shard as FSDP units of their own, inside the unit the whole model forms.

        FSDP2 all-gathers a unit's parameters for the duration of that unit's forward. With the
        model as the only unit, that is *every* parameter for the whole forward pass, so a sharded
        7B model still materializes 27 GB of fp32 weights on each rank and 27 GB of gradients
        beside them — the optimizer state is sharded and nothing else is. Naming the transformer
        blocks makes each one its own unit, so a block's weights are gathered when it runs and
        resharded when it finishes, and the resident cost falls to roughly one block's worth.

        The same division as `checkpointable_modules`, and for a related reason: a repeated,
        self-contained region is what both mechanisms want. They are separate methods because they
        answer different questions — one is about recomputing activations, the other about when
        parameters exist — and an architecture may have a good answer to one and not the other.

        Empty by default, which leaves the current behaviour (the model as a single unit) in place
        for architectures that have not declared a split.
        """
        return ()

    def lora_target_groups(self) -> dict[str, tuple[nn.Linear, ...]]:
        """Named groups of `nn.Linear` layers a low-rank adapter may be attached to.

        The same division of labour as `checkpointable_modules`: the architecture says *what* can be
        adapted and under which name, `[lora].targets` says which of those to use, and
        `engine.lora` does it. Names rather than a flat list because which projections to adapt is
        the main thing a LoRA run varies -- attention only, attention plus the FFN -- and a config
        naming a group this model does not offer should fail against the menu declared here rather
        than silently adapting nothing.

        Empty by default, which makes `[lora]` an error rather than a silent no-op on an
        architecture that has not declared any targets.
        """
        return {}

    def lora_required_trainable(self) -> tuple[str, ...]:
        """Parameter names that must keep training under LoRA whatever the config says.

        For invariants the engine cannot see. `DinoVisionTransformer3D` with superposition RoPE
        holds its entire use of the depth axis in one zero-initialised scalar, so freezing it as
        "part of the backbone" would leave a 3D model whose positional encoding cannot distinguish
        one z-slice from another -- a run that trains, converges, and is quietly solving a different
        problem. Which parameters carry a load-bearing initial value is architecture knowledge, so
        it is answered here rather than pattern-matched in a config.

        Matched as exact `named_parameters()` names, relative to the model.
        """
        return ()

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)

    def tensor_parallel_plan(self) -> dict[str, ParallelStyle] | None:
        """Optional module-path -> ParallelStyle plan for TP; None means unsupported."""
        return None


def single_scale_volumes(model: nn.Module, batch: torch.Tensor, axes: str) -> torch.Tensor:
    """(B, *axes) -> (B, C, D, H, W): the `prepare_input` of every single-scale 3D encoder.

    An encoder with one input grid at one resolution is single-scale by construction. miao's scale
    levels share a centre but cover different physical extents, which makes them neither
    pixel-aligned (so they cannot be channels) nor interchangeable with independent samples (so
    folding them into the batch would quietly redefine `batch_size`). Rather than pick one of those
    for the caller, this requires a single level and says how to configure it. A multi-scale encoder
    overrides `prepare_input` and consumes the level axis itself.
    """
    if "l" not in axes:
        raise ValueError(f"axis order must contain 'l' (scale level), got {axes!r}")

    # After the level axis is dropped, what is left has to be readable as (C, D, H, W) or
    # (D, H, W). A trailing channel such as "lzyxc" would otherwise put a spatial axis where
    # the channel belongs, and no downstream shape check would catch it.
    remainder = axes.replace("l", "", 1)
    if not (len(remainder) == 3 or (len(remainder) == 4 and remainder[0] == "c")):
        raise ValueError(
            f"axis order {axes!r} is not usable by a 3D encoder: after the level axis it "
            'must be three spatial axes, optionally preceded by \'c\' (e.g. "lzyx" or '
            f'"lcxyz"), got {remainder!r}'
        )

    expected_dims = len(axes) + 1
    if batch.dim() != expected_dims:
        raise ValueError(
            f"axis order {axes!r} implies a {expected_dims}-D batch (batch + {len(axes)} "
            f"axes), got {tuple(batch.shape)}; it must match the dataset's output_axes"
        )

    level_dim = axes.index("l") + 1
    levels = batch.shape[level_dim]
    if levels != 1:
        raise ValueError(
            f"{type(model).__name__} is single-scale, but this batch carries {levels} scale "
            f"levels on axis 'l' (shape {tuple(batch.shape)}). Configure the dataset for one "
            "level per sample: give `resolutions` a single entry, or use "
            "`resolution_sampling` with `n_scales = 1`, which still varies the resolution but "
            "draws it independently per sample. A multi-scale encoder should override "
            "prepare_input and consume the level axis itself."
        )

    volumes = batch.squeeze(level_dim)
    if volumes.dim() == 4:  # axis order declared no channel; add a singleton
        volumes = volumes.unsqueeze(1)
    return volumes
