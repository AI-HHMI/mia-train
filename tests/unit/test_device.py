from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.utils.data as data

import engine.trainer
from algorithms.base import BaseAlgorithm
from data.base import BaseDataset
from engine.config import TrainerConfig
from engine.trainer import Trainer
from models.base import BaseModel
from utils.device import move_to_device

CPU = torch.device("cpu")


@pytest.mark.unit
def test_moves_a_bare_tensor():
    moved = move_to_device(torch.randn(2, 2), CPU)
    assert moved.device == CPU


@pytest.mark.unit
def test_walks_nested_mappings_and_sequences():
    batch = {
        "img": torch.randn(2, 3),
        "meta": {"coordinate": [torch.tensor([1]), torch.tensor([2])]},
        "pair": (torch.randn(1), torch.randn(1)),
    }
    moved = move_to_device(batch, CPU)

    assert moved["img"].device == CPU
    assert moved["meta"]["coordinate"][0].device == CPU
    assert moved["pair"][0].device == CPU
    assert isinstance(moved["pair"], tuple)
    assert isinstance(moved["meta"]["coordinate"], list)


@pytest.mark.unit
def test_passes_non_tensors_through_untouched():
    batch = {"name": "volume_a", "step": 3, "ratio": 0.5, "nothing": None}
    assert move_to_device(batch, CPU) == batch


@pytest.mark.unit
def test_preserves_miao_batch_structure():
    # Mirrors what miao's VolumeDataset yields, including the empty label tensor it uses
    # when a volume has no labels.
    batch = {
        "img": torch.randn(2, 3, 8, 8, 8),
        "label": torch.empty(0, dtype=torch.int64),
        "bbox": torch.randn(2, 3, 2, 3),
        "pixel_size": torch.randn(2, 3, 3),
        "meta": {"volume": ["a", "b"], "source_levels": [torch.tensor([0, 1])]},
    }
    moved = move_to_device(batch, CPU)

    assert sorted(moved) == ["bbox", "img", "label", "meta", "pixel_size"]
    assert moved["img"].shape == (2, 3, 8, 8, 8)
    assert moved["label"].numel() == 0
    assert moved["meta"]["volume"] == ["a", "b"]


class _TinyModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def flops(self, input_shape: tuple[int, ...]) -> int:
        return 0


class _RecordingAlgorithm(BaseAlgorithm):
    """Reconstructs its input and keeps every batch object it was handed."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        self.seen: list[torch.Tensor] = []

    def training_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        self.seen.append(batch)
        return {"loss": (self.model(batch) - batch).pow(2).mean()}

    def validation_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        self.seen.append(batch)
        return {"loss": (self.model(batch) - batch).pow(2).mean()}


class _FixedDataset(data.Dataset):
    def __init__(self, n: int = 32) -> None:
        generator = torch.Generator().manual_seed(0)
        self._items = [torch.randn(8, generator=generator) for _ in range(n)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._items[index]


class _SyntheticDataset(BaseDataset):
    def build_dataset(self) -> data.Dataset:
        return _FixedDataset()


@pytest.mark.unit
def test_trainer_hands_the_algorithm_a_batch_moved_to_its_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both loops must route every batch through `move_to_device(batch, self.device)`.

    On CPU the move itself is a no-op, so this asserts the *call path* instead: that each
    batch is passed to `move_to_device` with the Trainer's own device, and that the object
    the algorithm receives is the return value rather than the untouched loader batch. The
    spy returns a distinct clone so dropping that return value fails the test — which is
    exactly how the original bug read (model moved, batch not).
    """
    devices: list[torch.device] = []
    returned: list[Any] = []

    def spy(batch: Any, device: torch.device) -> Any:
        devices.append(device)
        moved = move_to_device(batch, device).clone()
        returned.append(moved)
        return moved

    monkeypatch.setattr(engine.trainer, "move_to_device", spy)

    algorithm = _RecordingAlgorithm(_TinyModel())
    trainer = Trainer(
        algorithm=algorithm,
        train_dataset=_SyntheticDataset(),
        config=TrainerConfig(max_steps=2, batch_size=4, log_every=1000, seed=0),
        output_dir=tmp_path,
        val_dataset=_SyntheticDataset(),
        device=CPU,
    )

    assert trainer.train() == 2
    assert "loss" in trainer.validate()

    val_loader = trainer.val_loader
    assert val_loader is not None
    assert len(devices) == 2 + len(val_loader)  # every train step and every val batch
    assert all(device is trainer.device for device in devices)
    assert all(seen is moved for seen, moved in zip(algorithm.seen, returned, strict=True))
    assert all(seen.device == trainer.device for seen in algorithm.seen)
