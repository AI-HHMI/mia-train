"""Pins the frozen-backbone warm-up: what it holds fixed, when it lets go, and what the LR does.

The failure modes worth guarding are all silent. A freeze applied before the optimizer is built
leaves the backbone out of it permanently; a freeze re-derived from the config rather than the
restored step retrains a resumed run frozen; and a warm-up that never releases produces a run that
looks fine and learns nothing past the stem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.utils.data as data

from algorithms.base import BaseAlgorithm
from data.base import BaseDataset
from engine.config import TrainerConfig
from engine.optimizer import is_stem, lr_multiplier, unfreeze_ramp
from engine.trainer import Trainer
from models.base import BaseModel

CPU = torch.device("cpu")


class _StemAndBlocks(BaseModel):
    """A model shaped like the real ones: a recognised stem, then a `blocks` stack."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(8, 8)
        self.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])
        self.norm = nn.LayerNorm(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def flops(self, input_shape: tuple[int, ...]) -> int:
        return 0


class _HeadAlgorithm(BaseAlgorithm):
    """Wraps the model with its own head, as SimMIM does -- the head must stay trainable."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        self.head = nn.Linear(8, 8)

    def training_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"loss": self.head(self.model(batch)).pow(2).mean()}

    def validation_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.training_step(batch)


class _Fixed(data.Dataset):
    def __init__(self, n: int = 16) -> None:
        g = torch.Generator().manual_seed(0)
        self._items = [torch.randn(8, generator=g) for _ in range(n)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self._items[i]


class _Data(BaseDataset):
    def build_dataset(self) -> data.Dataset:
        return _Fixed()


def _param(module: nn.Module, name: str) -> torch.nn.Parameter:
    """A parameter by name.

    Attribute access (`model.blocks[0].weight`) is untyped under mypy -- `nn.Module.__getattr__`
    returns `Tensor | Module` and `nn.ModuleList` indexing likewise -- so the tests read through
    `named_parameters()`, as the rest of the suite does.
    """
    return dict(module.named_parameters())[name]


def _trainer(tmp: Path, **overrides: Any) -> Trainer:
    config = TrainerConfig(
        max_steps=overrides.pop("max_steps", 8),
        batch_size=4,
        lr=1e-3,
        warmup_steps=1,
        log_every=1000,
        seed=0,
        measure_mfu=False,
        **overrides,
    )
    return Trainer(
        algorithm=_HeadAlgorithm(_StemAndBlocks()),
        train_dataset=_Data(),
        config=config,
        output_dir=tmp,
        device=CPU,
    )


# ---------------------------------------------------------------- what counts as the backbone


@pytest.mark.unit
def test_stem_is_excluded_from_the_backbone() -> None:
    assert is_stem("patch_embed.proj.weight")
    assert is_stem("model.patch_proj.0.weight")
    assert not is_stem("blocks.0.attn.qkv.weight")
    assert not is_stem("norm.weight")


@pytest.mark.unit
def test_freeze_covers_blocks_but_not_the_stem_or_the_head(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, freeze_backbone_steps=4)
    trainer._set_backbone_frozen(True)

    model = trainer.algorithm.model
    assert _param(model, "patch_embed.weight").requires_grad, "the stem must keep training"
    assert not _param(model, "blocks.0.weight").requires_grad
    assert not _param(model, "norm.weight").requires_grad
    # The head lives on the algorithm, outside the model, and is the point of the warm-up.
    assert _param(trainer.algorithm, "head.weight").requires_grad


# ---------------------------------------------------------------- the learning-rate ramp


@pytest.mark.unit
def test_no_ramp_when_the_feature_is_off() -> None:
    config = TrainerConfig(max_steps=100, batch_size=1, lr=1e-3, warmup_steps=0)
    assert all(unfreeze_ramp(s, config) == 1.0 for s in (0, 1, 50, 99))


@pytest.mark.unit
def test_ramp_is_linear_across_the_boundary_and_one_after() -> None:
    config = TrainerConfig(
        max_steps=100, batch_size=1, lr=1e-3, warmup_steps=0,
        freeze_backbone_steps=10, unfreeze_warmup_steps=4,
    )
    assert unfreeze_ramp(9, config) == 1.0  # still frozen: ordinary schedule
    assert unfreeze_ramp(10, config) == pytest.approx(0.25)
    assert unfreeze_ramp(11, config) == pytest.approx(0.50)
    assert unfreeze_ramp(13, config) == pytest.approx(1.00)
    assert unfreeze_ramp(14, config) == 1.0  # ramp complete, back to the schedule


@pytest.mark.unit
def test_the_ramp_only_scales_the_ordinary_schedule() -> None:
    """The decay curve must be the same function of step whether or not a freeze is configured."""
    common: dict[str, Any] = dict(
        max_steps=100, batch_size=1, lr=1e-3, warmup_steps=0, lr_schedule="linear"
    )
    plain = TrainerConfig(**common)
    frozen = TrainerConfig(**common, freeze_backbone_steps=10, unfreeze_warmup_steps=4)

    for step in (0, 5, 9, 20, 50, 99):
        expected = lr_multiplier(step, plain) * unfreeze_ramp(step, frozen)
        assert lr_multiplier(step, frozen) == pytest.approx(expected)


# ---------------------------------------------------------------- the boundary, end to end


@pytest.mark.unit
def test_backbone_is_frozen_during_warmup_and_released_after(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, max_steps=6, freeze_backbone_steps=3)
    before = _param(trainer.algorithm.model, "blocks.0.weight").detach().clone()

    trainer.train()

    # Released by the end of the run...
    assert _param(trainer.algorithm.model, "blocks.0.weight").requires_grad
    # ...and actually moved once it was, so the release is real rather than nominal.
    assert not torch.equal(before, _param(trainer.algorithm.model, "blocks.0.weight"))


@pytest.mark.unit
def test_frozen_backbone_does_not_move_while_frozen(tmp_path: Path) -> None:
    """Steps taken while frozen must leave the backbone bit-identical.

    Driven directly rather than through `train()`, because the config deliberately refuses a
    freeze that outlasts the run -- so a run long enough to observe is also long enough to
    unfreeze. This isolates the frozen steps themselves.
    """
    trainer = _trainer(tmp_path, max_steps=8, freeze_backbone_steps=4, weight_decay=0.5)
    trainer._set_backbone_frozen(True)
    before = {n: p.detach().clone() for n, p in trainer.algorithm.model.named_parameters()}

    batch = torch.randn(4, 8)
    for _ in range(5):
        trainer.algorithm(batch)["loss"].backward()
        trainer.optimizer.step()
        trainer.scheduler.step()
        trainer.optimizer.zero_grad(set_to_none=True)

    for name, parameter in trainer.algorithm.model.named_parameters():
        if is_stem(name):
            assert not torch.equal(before[name], parameter), f"stem {name} should have trained"
        else:
            # Weight decay is deliberately non-zero here: decoupled decay is scaled by the group
            # learning rate, but a parameter with no grad gets no optimizer state at all, so it
            # must not drift even so.
            assert torch.equal(before[name], parameter), f"frozen {name} moved"


@pytest.mark.unit
def test_optimizer_keeps_every_group_across_the_boundary(tmp_path: Path) -> None:
    """Freezing after the optimizer is built is what keeps checkpoints loadable across a resume."""
    trainer = _trainer(tmp_path, max_steps=6, freeze_backbone_steps=3)
    groups_before = [len(g["params"]) for g in trainer.optimizer.param_groups]
    trainer.train()
    groups_after = [len(g["params"]) for g in trainer.optimizer.param_groups]
    assert groups_before == groups_after


@pytest.mark.unit
def test_resuming_past_the_boundary_comes_back_unfrozen(tmp_path: Path) -> None:
    """The freeze is derived from the restored step; deriving it from config alone is the bug."""
    first = _trainer(tmp_path, max_steps=4, freeze_backbone_steps=2, checkpoint_every=4)
    assert first.train() == 4

    resumed = _trainer(tmp_path, max_steps=8, freeze_backbone_steps=2, checkpoint_every=4)
    resumed.train()
    assert _param(resumed.algorithm.model, "blocks.0.weight").requires_grad


# ---------------------------------------------------------------- configuration


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"freeze_backbone_steps": -1}, "freeze_backbone_steps must be >= 0"),
        ({"freeze_backbone_steps": 10}, "must be < max_steps"),
        ({"unfreeze_warmup_steps": -5}, "unfreeze_warmup_steps must be >= 0"),
    ],
)
def test_invalid_settings_are_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TrainerConfig(max_steps=10, batch_size=1, lr=1e-3, warmup_steps=0, **kwargs)


@pytest.mark.unit
def test_disabled_by_default(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, max_steps=2)
    assert trainer.config.freeze_backbone_steps == 0
    trainer.train()
    assert _param(trainer.algorithm.model, "blocks.0.weight").requires_grad
