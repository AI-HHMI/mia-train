"""`[trainer].compile` — that it reaches the step, and that it does not reach anything else.

The interesting content here is not that compilation happens. It is the invariant that makes it
safe: `torch.compile` returns a wrapper whose parameters are renamed `_orig_mod.<name>`, and
`engine.optimizer` reads parameter names to decide what a tensor *is*. Binding that wrapper to
`Trainer.algorithm` would silently repartition the network. These tests pin the separation so a
later refactor cannot undo it quietly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.utils.data as data

from algorithms.base import BaseAlgorithm
from data.base import BaseDataset
from engine.config import TrainerConfig
from engine.optimizer import is_stem
from engine.trainer import Trainer
from models.base import BaseModel


class _TinyModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        # Named to match what the real encoders expose: `engine.optimizer` recognises a stem by
        # these substrings, so a test model without one cannot exercise the disagreement below.
        self.patch_proj = nn.Linear(8, 8)
        self.blocks = nn.ModuleList([nn.Linear(8, 8)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks[0](self.patch_proj(x))

    def flops(self, input_shape: tuple[int, ...]) -> int:
        return 0


class _ReconstructAlgorithm(BaseAlgorithm):
    def training_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"loss": (self.model(batch) - batch).pow(2).mean()}

    def validation_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"loss": (self.model(batch) - batch).pow(2).mean()}


class _Items(data.Dataset):
    def __init__(self, n: int = 4) -> None:
        generator = torch.Generator().manual_seed(0)
        self._items = [torch.randn(8, generator=generator) for _ in range(n)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._items[index]


class _SyntheticDataset(BaseDataset):
    def build_dataset(self) -> data.Dataset:
        return _Items()


def _trainer(tmp_path: Path, *, compile: bool) -> Trainer:
    return Trainer(
        algorithm=_ReconstructAlgorithm(_TinyModel()),
        train_dataset=_SyntheticDataset(),
        config=TrainerConfig(max_steps=2, batch_size=2, lr=1e-3, compile=compile),
        output_dir=tmp_path,
    )


@pytest.mark.unit
def test_compile_defaults_off():
    assert TrainerConfig(max_steps=2, batch_size=1, lr=1e-3).compile is False


@pytest.mark.unit
def test_forward_is_the_algorithm_itself_when_compile_is_off(tmp_path: Path):
    """No wrapper, no indirection, and nothing to go stale when the flag is not set."""
    trainer = _trainer(tmp_path, compile=False)
    assert trainer._forward is trainer.algorithm


@pytest.mark.unit
def test_compile_wraps_the_step_but_not_the_algorithm(tmp_path: Path):
    """The wrapper drives the step; `Trainer.algorithm` stays the real module.

    Everything that reads parameter names -- the optimizer's layerwise decay, the frozen-backbone
    warm-up, DCP -- goes through `self.algorithm`, so it must never become the wrapper.
    """
    trainer = _trainer(tmp_path, compile=True)

    assert trainer._forward is not trainer.algorithm
    assert isinstance(trainer.algorithm, _ReconstructAlgorithm)
    assert type(trainer._forward).__name__ == "OptimizedModule"

    # The same Parameters, so there is one set of weights rather than two to keep in sync.
    assert all(
        a is b
        for a, b in zip(trainer.algorithm.parameters(), trainer._forward.parameters(), strict=True)
    )
    # And the names the rest of the engine reads are unprefixed.
    assert not any(n.startswith("_orig_mod.") for n, _ in trainer.algorithm.named_parameters())


@pytest.mark.unit
def test_compiled_wrapper_would_misclassify_every_backbone_parameter():
    """Why the separation above exists, stated as a fact about torch rather than a convention.

    `engine.optimizer._depth` asks `name.startswith("model.")` to decide whether a parameter sits
    inside the backbone at all. Under the wrapper nothing does, so every encoder tensor would be
    treated as a head and `layerwise_lr_decay` would quietly stop applying. `is_stem` matches by
    substring and keeps working, so the two would disagree about where the backbone begins -- the
    exact failure `is_stem`'s docstring calls the worst of both.
    """
    algorithm = _ReconstructAlgorithm(_TinyModel())
    compiled = torch.compile(algorithm)

    names = [name for name, _ in compiled.named_parameters()]
    assert names, "a compiled module still exposes its parameters"
    assert all(name.startswith("_orig_mod.") for name in names)
    assert not any(name.startswith("model.") for name in names)

    # Both halves of the disagreement, on one renamed parameter.
    renamed = "_orig_mod.model.patch_proj.weight"
    assert is_stem(renamed), "is_stem matches by substring, so it survives the rename"
    assert not renamed.startswith("model."), "the backbone test does not"
