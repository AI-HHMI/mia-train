"""LoRA under FSDP2: a module whose parameters have mixed `requires_grad`.

This is the one part of adaptation that no single-process test can reach, and it is worth pinning
rather than trusting. `fully_shard` builds its `FSDPParamGroup` from the parameter set it finds and
reduce-scatters gradients per group, so a group holding 98% frozen tensors is an unusual shape
for it: torch 2.13 handles it (`_fsdp_param_group.py` filters trainable params when resolving
mixed-precision dtypes, and its post-backward reduce skips parameters with no gradient), but that
is an implementation detail of a version, not a contract, and a regression would show up here as a
hang or a wrong gradient rather than as an error.

Also covers grad-norm clipping, which is where the mix actually bites: `clip_grad_norm_` is called
over `algorithm.parameters()`, so it sees frozen sharded DTensors with `grad = None` alongside
trainable ones with DTensor gradients.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.utils.data as data

from algorithms.base import BaseAlgorithm
from data.base import BaseDataset
from distributed.parallel_dims import ParallelDims
from engine.config import LoRAConfig, TrainerConfig
from engine.lora import apply_lora
from engine.trainer import Trainer
from models.dinov3_vit3d import DinoVisionTransformer3D

TINY = dict(
    img_size=16,
    patch_size=8,
    in_chans=1,
    embed_dim=32,
    depth=2,
    num_heads=4,
    n_storage_tokens=4,
    layerscale_init=1.0e-05,
    mask_k_bias=True,
    pos_embed_rope_type="superposition",
    pos_embed_rope_dtype="fp32",
)


class _Volumes(data.Dataset):
    def __init__(self, n: int = 16) -> None:
        generator = torch.Generator().manual_seed(0)
        self._items = [torch.randn(1, 16, 16, 16, generator=generator) for _ in range(n)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"img": self._items[index]}


class _Dataset(BaseDataset):
    @property
    def sample_axes(self) -> str | None:
        return "lzyx"

    def build_dataset(self) -> data.Dataset:
        return _Volumes()


class _MeanOfPatches(BaseAlgorithm):
    """A stand-in for SimMIM: drives the encoder and owns a head that must train at full rank."""

    def __init__(self, model: DinoVisionTransformer3D, dataset: BaseDataset) -> None:
        super().__init__(model, dataset)
        self.head = torch.nn.Linear(TINY["embed_dim"], 1)

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        volumes = batch["img"].unsqueeze(1)  # (B, L=1, C, ...) for prepare_input
        tokens, _ = self.model.patch_features(self.model.prepare_input(volumes, "lczyx"))
        return {"loss": self.head(tokens).pow(2).mean()}

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.training_step(batch)


def _build(output_dir: str, world_size: int, max_steps: int) -> Trainer:
    torch.manual_seed(0)
    model = DinoVisionTransformer3D(**TINY)
    apply_lora(model, LoRAConfig(rank=4, alpha=8.0))
    dims = ParallelDims(dp_shard=world_size)
    return Trainer(
        algorithm=_MeanOfPatches(model, _Dataset()),
        train_dataset=_Dataset(),
        config=TrainerConfig(
            max_steps=max_steps,
            batch_size=2,
            lr=1e-2,
            log_every=1000,
            seed=0,
            measure_mfu=False,
            grad_clip_norm=1.0,
        ),
        output_dir=Path(output_dir),
        dims=dims,
        mesh=dims.build_mesh("cpu"),
    )


def _adapters_train_and_base_weights_do_not(
    rank: int, world_size: int, output_dir: str
) -> tuple[bool, bool, bool]:
    trainer = _build(output_dir, world_size, 3)
    named = dict(trainer.algorithm.named_parameters())
    base = named["model.blocks.0.attn.qkv.weight"]
    adapter = named["model.blocks.0.attn.qkv.lora_b"]

    before_base = base.detach().to_local().clone()
    before_adapter = adapter.detach().to_local().clone()
    trainer.train()

    base_unchanged = torch.equal(before_base, base.detach().to_local())
    adapter_moved = not torch.equal(before_adapter, adapter.detach().to_local())
    # A frozen parameter must not have accumulated a gradient either: that is memory spent on a
    # tensor no param group holds, which is exactly the failure the freeze/LoRA conflict produces.
    no_frozen_grads = all(
        parameter.grad is None
        for parameter in trainer.algorithm.parameters()
        if not parameter.requires_grad
    )
    return base_unchanged, adapter_moved, no_frozen_grads


def _loss_decreases(rank: int, world_size: int, output_dir: str) -> tuple[float, float]:
    trainer = _build(output_dir, world_size, 20)
    batch = next(iter(trainer.train_loader))
    before = trainer.algorithm(batch)["loss"].item()
    trainer.algorithm.zero_grad(set_to_none=True)
    trainer.train()
    after = trainer.algorithm(batch)["loss"].item()
    return before, after


def _grad_norm_is_finite(rank: int, world_size: int, output_dir: str) -> float:
    """Clipping mixes frozen DTensors carrying no gradient with trainable ones that do."""
    trainer = _build(output_dir, world_size, 1)
    batch = next(iter(trainer.train_loader))
    trainer.algorithm(batch)["loss"].backward()
    total = torch.nn.utils.clip_grad_norm_(trainer.algorithm.parameters(), 1.0)
    return float(total)


def _algorithm_head_is_not_frozen(rank: int, world_size: int, output_dir: str) -> bool:
    """LoRA is applied to the model, so a strategy's own head must be untouched by it."""
    trainer = _build(output_dir, world_size, 1)
    named = dict(trainer.algorithm.named_parameters())
    return named["head.weight"].requires_grad and named["head.bias"].requires_grad


@pytest.mark.cpu_dist
def test_adapters_train_and_base_weights_do_not(run_distributed):
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        results = run_distributed(
            _adapters_train_and_base_weights_do_not, world_size=2, args=(tmp,)
        )
    for base_unchanged, adapter_moved, no_frozen_grads in results:
        assert base_unchanged, "a frozen base weight was updated under FSDP"
        assert adapter_moved, "the adapter did not train under FSDP"
        assert no_frozen_grads, "a frozen parameter accumulated a gradient"


@pytest.mark.cpu_dist
def test_a_mostly_frozen_model_still_learns(run_distributed):
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        results = run_distributed(_loss_decreases, world_size=2, args=(tmp,))
    for before, after in results:
        assert after < before


@pytest.mark.cpu_dist
def test_grad_clipping_tolerates_the_frozen_mix(run_distributed):
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        results = run_distributed(_grad_norm_is_finite, world_size=2, args=(tmp,))
    for total in results:
        assert total > 0.0 and torch.isfinite(torch.tensor(total))


@pytest.mark.cpu_dist
def test_the_algorithms_own_head_trains_at_full_rank(run_distributed):
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        results = run_distributed(_algorithm_head_is_not_frozen, world_size=2, args=(tmp,))
    assert all(results)
