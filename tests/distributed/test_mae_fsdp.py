from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.utils.data as data

from algorithms.mae import MAE
from data.base import BaseDataset
from distributed.parallel_dims import ParallelDims
from engine.config import TrainerConfig
from engine.trainer import Trainer
from models.vit import ViT3D


class _SingleScaleVolumes(data.Dataset):
    """Stands in for miao configured for one level per sample, axis order "lzyx"."""

    levels = 1

    def __init__(self, n: int = 16) -> None:
        generator = torch.Generator().manual_seed(0)
        self._items = [
            torch.randn(self.levels, 16, 16, 16, generator=generator) for _ in range(n)
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"img": self._items[index]}


class _ThreeScaleVolumes(_SingleScaleVolumes):
    levels = 3


class _SyntheticVolumeDataset(BaseDataset):
    @property
    def sample_axes(self) -> str | None:
        return "lzyx"

    def build_dataset(self) -> data.Dataset:
        return _SingleScaleVolumes()


class _ThreeScaleDataset(_SyntheticVolumeDataset):
    def build_dataset(self) -> data.Dataset:
        return _ThreeScaleVolumes()


def _build(output_dir: str, world_size: int, max_steps: int, dataset: BaseDataset) -> Trainer:
    encoder = ViT3D(
        img_size=(16, 16, 16), patch_size=(8, 8, 8), in_channels=1,
        embed_dim=32, depth=2, num_heads=4,
    )
    # input_axes deliberately omitted: MAE must take "lzyx" from the dataset.
    algorithm = MAE(
        encoder, dataset, mask_ratio=0.5, norm_pix_loss=False,
        decoder_embed_dim=16, decoder_depth=1, decoder_num_heads=2,
    )
    dims = ParallelDims(dp_shard=world_size)
    return Trainer(
        algorithm=algorithm,
        train_dataset=dataset,
        config=TrainerConfig(
            max_steps=max_steps, batch_size=2, lr=1e-3, log_every=1000, seed=0
        ),
        output_dir=Path(output_dir),
        dims=dims,
        mesh=dims.build_mesh("cpu"),
    )


def _algorithm_of(trainer: Trainer) -> MAE:
    """The trainer's algorithm, narrowed back to MAE.

    `Trainer.algorithm` is a `BaseAlgorithm`, and attribute lookups on an `nn.Module` fall
    through to `__getattr__`, whose return type is `Tensor | Module` — so reaching MAE's own
    attributes needs the type restated here rather than at every call site.
    """
    algorithm = trainer.algorithm
    assert isinstance(algorithm, MAE)
    return algorithm


def _mae_under_fsdp_reduces_loss(
    rank: int, world_size: int, output_dir: str
) -> tuple[float, float]:
    trainer = _build(output_dir, world_size, 25, _SyntheticVolumeDataset())
    batch = next(iter(trainer.train_loader))
    before = trainer.algorithm(batch)["loss"].item()
    trainer.train()
    after = trainer.algorithm(batch)["loss"].item()
    return before, after


def _axes_come_from_the_dataset(rank: int, world_size: int, output_dir: str) -> str:
    trainer = _build(output_dir, world_size, 1, _SyntheticVolumeDataset())
    return _algorithm_of(trainer).input_axes


def _encoder_params_are_sharded(rank: int, world_size: int, output_dir: str) -> bool:
    from torch.distributed.tensor import DTensor

    trainer = _build(output_dir, world_size, 1, _SyntheticVolumeDataset())
    # `encoder` is the same module the trainer sharded (MAE holds one object under both names),
    # so this also shows MAE's own view of the encoder sees the sharded parameters.
    return isinstance(_algorithm_of(trainer).encoder.patch_embed.weight, DTensor)


def _unsharded_parameter_names(rank: int, world_size: int, output_dir: str) -> list[str]:
    from torch.distributed.tensor import DTensor

    trainer = _build(output_dir, world_size, 1, _SyntheticVolumeDataset())
    return [
        name
        for name, param in trainer.algorithm.named_parameters()
        if not isinstance(param, DTensor)
    ]


def _multi_scale_batch_is_rejected(rank: int, world_size: int, output_dir: str) -> bool:
    trainer = _build(output_dir, world_size, 1, _ThreeScaleDataset())
    try:
        trainer.train()
    except ValueError as error:
        return "single-scale" in str(error)
    return False


@pytest.mark.cpu_dist
def test_mae_under_fsdp_reduces_loss(run_distributed):
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        results = run_distributed(_mae_under_fsdp_reduces_loss, world_size=2, args=(tmp,))
    for before, after in results:
        assert after < before


@pytest.mark.cpu_dist
def test_axes_come_from_the_dataset(run_distributed):
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        results = run_distributed(_axes_come_from_the_dataset, world_size=2, args=(tmp,))
    assert results == ["lzyx", "lzyx"]


@pytest.mark.cpu_dist
def test_encoder_params_are_sharded(run_distributed):
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        assert all(run_distributed(_encoder_params_are_sharded, world_size=2, args=(tmp,)))


@pytest.mark.cpu_dist
def test_the_decoder_is_sharded_along_with_the_encoder(run_distributed):
    # MAE holds parameters outside the model, and clip_grad_norm_ and AdamW both refuse to mix
    # sharded DTensors with plain tensors, so a sharded run has to cover the decoder too.
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        results = run_distributed(_unsharded_parameter_names, world_size=2, args=(tmp,))
    assert results == [[], []]


@pytest.mark.cpu_dist
def test_multi_scale_batch_is_rejected_through_the_trainer(run_distributed):
    # Proves the encoder's single-scale contract holds through the real training loop, not just
    # when prepare_input is called directly.
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        assert all(run_distributed(_multi_scale_batch_is_rejected, world_size=2, args=(tmp,)))
