from __future__ import annotations

import contextlib
import itertools
import time
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
from distributed.parallelize import parallelize_algorithm
from utils.device import move_to_device
from utils.hardware_flops import peak_flops
from utils.metrics import MetricLogger, reduce_metrics

from .activation_checkpoint import apply_activation_checkpointing
from .checkpoint import CheckpointManager
from .config import TrainerConfig
from .mfu import ThroughputMeter, measure_step_flops
from .optimizer import build_lr_scheduler, build_optimizer, is_stem
from .profiler import StepProfiler, annotate, current_rank, should_profile

_AUTOCAST_DTYPES = {"bf16": torch.bfloat16}


class Trainer:
    """Runs the training loop, calling the algorithm on each batch without inspecting it.

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

        if config.activation_checkpointing:
            # Before sharding, and independent of it: FSDP2 hooks a module's forward to gather
            # its parameters, so wrapping afterwards would put the recomputation outside the
            # gather. Memory pressure is per-GPU and exists on one device as much as on eight.
            apply_activation_checkpointing(algorithm)

        if mesh is not None:
            # The algorithm, not just its model: an algorithm's own parameters (MAE's decoder)
            # have to be sharded alongside the model's, or grad clipping mixes DTensors with
            # plain tensors.
            parallelize_algorithm(algorithm, mesh, self.dims)
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

        self.is_primary = not dist.is_initialized() or dist.get_rank() == 0
        self.logger = MetricLogger(
            log_dir=output_dir / "tensorboard",
            is_primary=self.is_primary,
        )

    def _autocast(self) -> Any:
        dtype = _AUTOCAST_DTYPES.get(self.config.precision)
        if dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=dtype)

    def _backbone_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        """The model's parameters outside its input stem -- what a warm-up holds fixed.

        Scoped to `algorithm.model`, so an algorithm's own head is untouched however it is named.
        The stem test is `optimizer.is_stem`, the same one the layerwise learning rate uses, so the
        two cannot disagree about where the backbone begins.
        """
        return [
            (name, parameter)
            for name, parameter in self.algorithm.model.named_parameters()
            if not is_stem(name)
        ]

    def _set_backbone_frozen(self, frozen: bool) -> None:
        """Toggle the backbone's `requires_grad`, reporting the parameter count once.

        Applied *after* `build_optimizer`, never before: `build_param_groups` skips parameters that
        do not require grad, so freezing first would leave the backbone out of the optimizer
        permanently and the groups would have to be rebuilt at the boundary -- which changes the
        shape a checkpoint's optimizer state reloads into. Toggling afterwards keeps the group
        structure fixed for the whole run; AdamW simply allocates no state for a parameter whose
        grad stays None, and `zero_grad(set_to_none=True)` guarantees it does.
        """
        parameters = self._backbone_parameters()
        for _, parameter in parameters:
            parameter.requires_grad_(not frozen)
        if self.is_primary:
            count = sum(p.numel() for _, p in parameters)
            verb = "froze" if frozen else "unfroze"
            print(f"[freeze] {verb} {len(parameters)} backbone tensors ({count/1e6:.1f}M params)",
                  flush=True)

    def _peak_flops(self) -> tuple[float | None, str]:
        """One GPU's peak throughput at this run's precision, and why it is unknown if it is."""
        if self.config.peak_tflops is not None:
            return self.config.peak_tflops * 1e12, ""
        if self.device.type != "cuda":
            return None, f"no peak-FLOPS figure for device type {self.device.type!r}"
        return peak_flops(torch.cuda.get_device_name(self.device), self.config.precision)

    def _build_throughput_meter(self, batch: Any) -> ThroughputMeter:
        """Measure one step's FLOPs on `batch`, leaving no trace on the run.

        Every rank measures, because the forward and backward it runs contain the same collectives
        a real step does — a rank that skipped the probe would leave the others waiting in an
        all-gather. Each rank counts its own local work, which is exactly the numerator a per-GPU
        utilization figure wants.

        The RNG state is restored afterwards so that enabling this cannot change what the run
        trains on. The probe draws from the same global generator as mask sampling and
        augmentation, so without the restore, two runs with one seed and different `measure_mfu`
        would silently diverge from the first step.
        """
        peak, reason = self._peak_flops()
        if peak is None and self.is_primary:
            print(f"[mfu] utilization not reported: {reason}", flush=True)

        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            step_flops = measure_step_flops(self.algorithm, batch, self._autocast)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        if self.is_primary:
            peak_note = f"{peak / 1e12:.1f} TFLOP/s peak" if peak else "peak unknown"
            print(
                f"[mfu] measured {step_flops / 1e12:.3f} TFLOP per step per rank, {peak_note}",
                flush=True,
            )
        return ThroughputMeter(
            step_flops=step_flops,
            peak_flops=peak,
            samples_per_step=self.config.batch_size * self.dims.dp_world_size,
        )

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
        resumed_from = step
        for _ in range(step):
            self.scheduler.step()

        batches = self._endless_batches(self.train_loader, start_epoch=step)
        self.algorithm.train()

        # Decided from the *restored* step, not from zero: a job resumed past the boundary must
        # come back unfrozen, and one resumed inside the warm-up must come back frozen. Getting
        # this from `config` alone would silently retrain a resumed run with a frozen backbone.
        backbone_frozen = step < self.config.freeze_backbone_steps
        if backbone_frozen:
            self._set_backbone_frozen(True)

        meter: ThroughputMeter | None = None
        if self.config.measure_mfu and step < self.config.max_steps:
            # Probe the batch the first step is about to train on, then put it back, so measuring
            # does not consume one. `_build_throughput_meter` restores the RNG, and together those
            # make the probe invisible: the run sees the same batches in the same order with the
            # same random draws whether or not it is enabled.
            probe_batch = move_to_device(next(batches), self.device)
            batches = itertools.chain([probe_batch], batches)
            meter = self._build_throughput_meter(probe_batch)
            meter.start()

        logged_at = step
        # Host seconds spent blocked on the input pipeline since the last log, and the wall clock
        # they are a fraction of. Reported as `data_wait_frac` on every run, not only profiled
        # ones: it is the single number that separates "the GPU is busy and slow" from "the GPU is
        # idle waiting for a batch", it costs two clock reads per step, and needing a trace to
        # answer a yes/no question that cheap would be the wrong trade.
        data_seconds = 0.0
        window_start = time.perf_counter()

        with StepProfiler(
            self.output_dir,
            start_step=self.config.profile_start_step,
            active=self.config.profile_steps,
            profile_memory=self.config.profile_memory,
            enabled=self.config.profile
            and should_profile(current_rank(), self.config.profile_all_ranks),
        ) as profiler:
            while step < self.config.max_steps:
                # Timed on the host, with no device synchronization: what is being measured is how
                # long this process sat in `next()` waiting for a worker to hand over a sample,
                # which is a host-side wait by construction. Forcing a sync to "improve" it would
                # only fold the previous step's device queue into the number.
                wait_start = time.perf_counter()
                with annotate("data_wait"):
                    raw_batch = next(batches)
                data_seconds += time.perf_counter() - wait_start

                with annotate("h2d"):
                    batch = move_to_device(raw_batch, self.device)

                with annotate("forward"), self._autocast():
                    # Through __call__, not training_step: `BaseAlgorithm.forward` aliases it, and
                    # plain replication only all-reduces gradients from forward hooks.
                    metrics = self.algorithm(batch)
                with annotate("backward"):
                    metrics["loss"].backward()

                if self.config.grad_clip_norm is not None:
                    with annotate("grad_clip"):
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.algorithm.parameters(), self.config.grad_clip_norm
                        )
                    metrics["grad_norm"] = grad_norm

                with annotate("optimizer"):
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                step += 1

                if backbone_frozen and step >= self.config.freeze_backbone_steps:
                    # After the step that completes the warm-up, so the boundary step is the last
                    # one trained frozen -- matching `freeze_backbone_steps` as a count of frozen
                    # steps rather than as the index of the first joint one.
                    self._set_backbone_frozen(False)
                    backbone_frozen = False

                if step % self.config.log_every == 0:
                    logged = reduce_metrics(metrics)
                    # `get_last_lr()` is typed `list[float | Tensor]` because tensor LRs are legal.
                    logged["lr"] = float(self.scheduler.get_last_lr()[0])
                    now = time.perf_counter()
                    logged["data_wait_frac"] = data_seconds / max(now - window_start, 1e-9)
                    data_seconds, window_start = 0.0, now
                    if meter is not None:
                        # Sampled here, after `reduce_metrics` has already synchronized on
                        # `.item()`, so the window is bounded by real device work rather than by a
                        # queue depth. The step count is measured rather than assumed to be
                        # `log_every`: resuming mid-cadence makes the first window shorter.
                        window = meter.window(step - logged_at)
                        if window is not None:
                            logged.update(window.as_metrics())
                    logged_at = step
                    self.logger.log(step, logged)

                if self.config.checkpoint_every and step % self.config.checkpoint_every == 0:
                    self.checkpoints.save(step)

                if self.config.val_every and step % self.config.val_every == 0:
                    self.logger.log(step, self.validate(), prefix="val")
                    self.algorithm.train()
                    # Validation runs a whole loader inside one training step's timing window, and
                    # it is not training time. Rebasing here keeps it out of `data_wait_frac`
                    # rather than letting one window report a fraction that describes the
                    # validation pass.
                    data_seconds, window_start = 0.0, time.perf_counter()

                profiler.step()

        # Save the final state unless the last step happened to land on the cadence. Otherwise
        # finishing a run discards up to `checkpoint_every - 1` steps of training, since
        # `max_steps` need not be a multiple of it. Skipped when nothing was trained, so
        # re-running a finished run does not rewrite an identical checkpoint.
        if (
            self.config.checkpoint_every
            and step > resumed_from
            and step % self.config.checkpoint_every != 0
        ):
            self.checkpoints.save(step)

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
                    metrics = self.algorithm.validation_step(move_to_device(batch, self.device))
                for name, value in reduce_metrics(metrics).items():
                    totals[name] = totals.get(name, 0.0) + value
                batches += 1
        return {name: total / max(1, batches) for name, total in totals.items()}
