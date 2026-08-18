from __future__ import annotations

import abc
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
