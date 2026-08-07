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
        """Estimated forward-pass FLOPs for a single input of the given shape (no batch dim)."""

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

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)

    def tensor_parallel_plan(self) -> dict[str, ParallelStyle] | None:
        """Optional module-path -> ParallelStyle plan for TP; None means unsupported."""
        return None
