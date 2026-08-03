from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import DataLoader, DistributedSampler

from algorithms.base import BaseAlgorithm
from data.base import BaseDataset
from distributed.parallel_dims import ParallelDims
from distributed.parallelize import parallelize_model
from utils.metrics import MetricLogger, reduce_metrics

from .checkpoint import CheckpointManager
from .config import TrainerConfig
from .optimizer import build_lr_scheduler, build_optimizer

_AUTOCAST_DTYPES = {"bf16": torch.bfloat16}


class Trainer:
    """Runs the training loop, calling `algorithm.training_step(batch)` without inspecting it.

    Owns parallelism application, optimizer and LR schedule, mixed precision, metric logging,
    and DCP checkpointing — the pieces DESIGN.md requires the engine to handle exactly once.
    """

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        train_dataset: BaseDataset,
        config: TrainerConfig,
        output_dir: Path,
        dims: ParallelDims | None = None,
        mesh: DeviceMesh | None = None,
        val_dataset: BaseDataset | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.dims = dims or ParallelDims()
        self.mesh = mesh
        self.output_dir = output_dir
        self.device = device or torch.device("cpu")

        torch.manual_seed(config.seed)

        if mesh is not None:
            parallelize_model(algorithm.model, mesh, self.dims)
        self.algorithm = algorithm.to(self.device)

        self.optimizer = build_optimizer(self.algorithm, config)
        self.scheduler = build_lr_scheduler(self.optimizer, config)
        self.checkpoints = CheckpointManager(
            self.algorithm, self.optimizer, output_dir / "checkpoints"
        )

        dp_rank = self.dims.dp_rank(mesh) if mesh is not None else 0
        self.train_loader = train_dataset.build_dataloader(
            batch_size=config.batch_size,
            rank=dp_rank,
            world_size=self.dims.dp_world_size,
            num_workers=config.num_workers,
        )
        self.val_loader = (
            val_dataset.build_dataloader(
                batch_size=config.batch_size,
                rank=dp_rank,
                world_size=self.dims.dp_world_size,
                shuffle=False,
                num_workers=config.num_workers,
            )
            if val_dataset is not None
            else None
        )

        self.logger = MetricLogger(
            log_dir=output_dir / "tensorboard",
            is_primary=not dist.is_initialized() or dist.get_rank() == 0,
        )

    def _autocast(self) -> Any:
        dtype = _AUTOCAST_DTYPES.get(self.config.precision)
        if dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=dtype)

    def _endless_batches(self, loader: DataLoader, start_epoch: int = 0) -> Iterator[Any]:
        """Yield batches forever, re-seeding the sampler each pass.

        `DistributedSampler` permutes identically every epoch unless `set_epoch` is called, so
        without this a step-based run would revisit the same batch order indefinitely.
        """
        epoch = start_epoch
        while True:
            if isinstance(loader.sampler, DistributedSampler):
                loader.sampler.set_epoch(epoch)
            yield from loader
            epoch += 1

    def train(self) -> int:
        """Train until `max_steps`, resuming from the newest checkpoint if one exists."""
        step = self.checkpoints.load_latest()
        for _ in range(step):
            self.scheduler.step()

        batches = self._endless_batches(self.train_loader, start_epoch=step)
        self.algorithm.train()

        while step < self.config.max_steps:
            batch = next(batches)
            with self._autocast():
                metrics = self.algorithm.training_step(batch)
            metrics["loss"].backward()

            if self.config.grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.algorithm.parameters(), self.config.grad_clip_norm
                )
                metrics["grad_norm"] = grad_norm

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % self.config.log_every == 0:
                logged = reduce_metrics(metrics)
                # `get_last_lr()` is typed `list[float | Tensor]` because tensor LRs are legal.
                logged["lr"] = float(self.scheduler.get_last_lr()[0])
                self.logger.log(step, logged)

            if self.config.checkpoint_every and step % self.config.checkpoint_every == 0:
                self.checkpoints.save(step)

            if self.config.val_every and step % self.config.val_every == 0:
                self.logger.log(step, self.validate(), prefix="val")
                self.algorithm.train()

        self.logger.close()
        return step

    def validate(self) -> dict[str, float]:
        """Average `algorithm.validation_step` metrics over the whole validation loader."""
        if self.val_loader is None:
            raise ValueError("Trainer was constructed without a val_dataset")

        self.algorithm.eval()
        totals: dict[str, float] = {}
        batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                with self._autocast():
                    metrics = self.algorithm.validation_step(batch)
                for name, value in reduce_metrics(metrics).items():
                    totals[name] = totals.get(name, 0.0) + value
                batches += 1
        return {name: total / max(1, batches) for name, total in totals.items()}
