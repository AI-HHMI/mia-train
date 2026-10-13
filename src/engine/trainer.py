from __future__ import annotations

import contextlib
import itertools
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import DataLoader, DistributedSampler

from algorithms.base import BaseAlgorithm
from data.augment import BatchPhotometric
from data.base import BaseDataset
from distributed.grad_norm import clip_grad_norm_
from distributed.parallel_dims import ParallelDims
from distributed.parallelize import apply_tensor_parallel, shard_algorithm
from utils.device import move_to_device
from utils.hardware_flops import peak_flops
from utils.metrics import MetricLogger, reduce_metrics

from .activation_checkpoint import apply_activation_checkpointing
from .checkpoint import CheckpointManager, checkpoint_step
from .config import TrainerConfig
from .mfu import ThroughputMeter, measure_step_flops
from .optimizer import build_lr_scheduler, build_optimizer, is_stem, optimizer_step
from .profiler import StepProfiler, annotate, current_rank, should_profile

_AUTOCAST_DTYPES = {"bf16": torch.bfloat16}


def _stream_seed(seed: int, dp_rank: int, step: int) -> int:
    """The seed of one data-parallel rank's random streams for an attempt starting at `step`.

    Rank 0 at step 0 keeps `seed` itself, so a fresh single-GPU run draws what it always did. The
    strides are odd 32-bit constants (the golden ratio, and murmur3's), not powers of two: torch's
    CPU generator keeps only a seed's low 32 bits -- seeds 5 and 5 + 2**32 draw identically -- and
    odd strides move those bits for every rank and every restart.
    """
    return (seed + dp_rank * 0x9E3779B9 + step * 0x85EBCA6B) % 2**32


def _fingerprint(batch: Any) -> tuple[float, ...]:
    """Each tensor's float64 sum and sum of squares, in a fixed order: equal only for equal data."""
    tensors: list[torch.Tensor] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, torch.Tensor):
            tensors.append(obj)
        elif isinstance(obj, dict):
            for key in sorted(obj):
                walk(obj[key])
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)

    walk(batch)
    sums: list[float] = []
    for tensor in tensors:
        values = tensor.detach().to(torch.float64)
        sums += [values.sum().item(), (values * values).sum().item()]
    return tuple(sums)


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
        photometric: BatchPhotometric | None = None,
    ) -> None:
        self.config = config
        self.dims = dims or ParallelDims()
        self.mesh = mesh
        self.output_dir = output_dir
        self.device = device or torch.device("cpu")
        # The half of `[augment]` that runs here rather than in a dataloader worker -- see
        # `data.augment.split_augmentation`. Held by the trainer rather than wrapped around the
        # dataset because the point is to apply it *after* the batch reaches the device, and it is
        # applied only in `train()`: `validate()` must measure the model on the data as it is, and
        # keeping the call out of that loop makes it so structurally rather than by convention.
        self.photometric = (
            photometric if photometric is not None and photometric.enabled() else None
        )

        # Tensor parallelism, then activation checkpointing, then sharding — in that order, and
        # each boundary is load-bearing.
        #
        # TP comes first because `checkpoint_wrapper` re-parents a block under
        # `_checkpoint_wrapped_module`, and a parallel plan's paths are resolved through
        # `named_children()`; applied afterwards it would match nothing, silently, and the run
        # would train fully replicated (see `distributed.parallelize.apply_tensor_parallel`).
        #
        # Sharding comes last because FSDP2 hooks a module's forward to gather its parameters, so
        # checkpointing afterwards would put the recomputation outside the gather. Memory pressure
        # is per-GPU and exists on one device as much as on eight, which is why the middle step
        # does not depend on there being a mesh at all.
        if mesh is not None:
            apply_tensor_parallel(algorithm.model, mesh, self.dims)

        if config.activation_checkpointing:
            apply_activation_checkpointing(algorithm)

        if mesh is not None:
            # The algorithm, not just its model: an algorithm's own parameters (MAE's decoder)
            # have to be sharded alongside the model's, or grad clipping mixes DTensors with
            # plain tensors.
            shard_algorithm(algorithm, mesh, self.dims)
        self.algorithm = algorithm.to(self.device)

        self.optimizer = build_optimizer(self.algorithm, config)
        self.scheduler = build_lr_scheduler(self.optimizer, config)
        self.checkpoints = CheckpointManager(
            self.algorithm, self.optimizer, output_dir / "checkpoints"
        )
        if config.checkpoint_every:
            # Now rather than at the first save, `checkpoint_every` steps into the run.
            self.checkpoints.check_supported()

        # Compiled into a handle of its own rather than over `self.algorithm`, and last, once the
        # optimizer already holds its parameter groups.
        #
        # `torch.compile` returns an `OptimizedModule` that renames every parameter to
        # `_orig_mod.<name>`, and `engine.optimizer` decides what a parameter *is* from its name:
        # `_depth` treats anything not under the `model.` prefix as sitting above the backbone.
        # Rebinding `self.algorithm` to the wrapper would therefore make every encoder tensor look
        # like a head, silently disabling `layerwise_lr_decay` -- and it would disagree with
        # `is_stem`, which matches by substring and would still fire, which is exactly the split
        # `is_stem` exists to prevent. Keeping the wrapper separate leaves the optimizer, the
        # freeze warm-up, the FLOP probe and DCP all reading the real module. The two share their
        # Parameters, so there is still only one set of weights and nothing to keep in sync.
        self._forward = torch.compile(self.algorithm) if config.compile else self.algorithm

        dp_rank = self.dims.dp_rank(mesh) if mesh is not None else 0
        self._dp_rank = dp_rank
        # Seeded per data-parallel rank, once everything that must be identical on every rank
        # exists (`build_trainer` initialised the model under `config.seed` alone); `train()`
        # reseeds with the step it resumes from. See `_seed_streams`.
        self._seed_streams(step=0)
        self._batches_checked = False
        self.train_loader = train_dataset.build_dataloader(
            batch_size=config.batch_size,
            rank=dp_rank,
            world_size=self.dims.dp_world_size,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor,
            # The sampler's shuffled order follows the run's seed too, identical on every rank.
            seed=config.seed,
        )
        self.val_loader = (
            val_dataset.build_dataloader(
                batch_size=config.batch_size,
                rank=dp_rank,
                world_size=self.dims.dp_world_size,
                shuffle=False,
                # Keep the partial batch. Dropped, a rank's share smaller than one batch yielded no
                # validation at all, on every rank alike -- 32 samples over 8 ranks at batch 8 ran
                # 80 empty validations in one 200k-step run. The sampler pads every share to the
                # same length, so the ranks still run the same number of batches.
                drop_last=False,
                num_workers=config.num_workers,
                persistent_workers=config.persistent_workers,
                prefetch_factor=config.prefetch_factor,
            )
            if val_dataset is not None
            else None
        )

        # Per dataset, not one shared hook: `[data]` and `[val_data]` are separate sections and may
        # disagree about deferring, and finishing a validation batch with the training set's rules
        # would be silent. Identity unless a dataset deferred work to the device.
        self._finish_train = train_dataset.finish_batch
        self._finish_val = val_dataset.finish_batch if val_dataset is not None else None

        self.is_primary = not dist.is_initialized() or dist.get_rank() == 0
        self.logger = MetricLogger(
            log_dir=output_dir / "tensorboard",
            is_primary=self.is_primary,
        )

    def _seed_streams(self, step: int) -> None:
        """Seed torch (CPU and CUDA), numpy and Python's `random` for this rank and attempt.

        Per data-parallel rank, because everything drawn from here on must differ between ranks:
        the DataLoader takes each epoch's worker seeds from torch's generator, and a dataset that
        samples at random rather than by index (miao's `sampling: random`) would otherwise draw
        the same crops on every rank -- one sample, copied dp_world_size times -- as would the
        augmentation and masks drawn on the device. The ranks of one tensor-parallel group share a
        data-parallel rank, so they still draw one batch. numpy and `random` too, because with
        `num_workers = 0` miao draws from numpy's global state in this process, which nothing else
        seeds.

        Per attempt, because checkpoints hold no generator state: without the restored step a
        resumed run would replay its data, augmentation and masks from step 0, and a run resumed
        at step S of T would train min(S, T - S) of its steps on batches it had already seen.
        """
        seed = _stream_seed(self.config.seed, self._dp_rank, step)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _check_rank_batches(self, batch: Any) -> None:
        """Stop the run if its first batch breaks data parallelism, which no metric shows.

        Data-parallel ranks must hold different data, and the ranks of one tensor-parallel group
        the same data. Every rank training on one crop looks like an ordinary run at batch 1, so
        the first batch's fingerprint is compared across ranks once. Identical batches on some
        data-parallel ranks can be legitimate -- a sampler padding a set smaller than the world --
        so only all of them holding one batch is refused.
        """
        self._batches_checked = True
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return
        gathered: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (self._dp_rank, _fingerprint(batch)))
        by_dp_rank: dict[int, set[tuple[float, ...]]] = {}
        for dp_rank, fingerprint in gathered:
            by_dp_rank.setdefault(dp_rank, set()).add(fingerprint)
        split = sorted(rank for rank, prints in by_dp_rank.items() if len(prints) > 1)
        if split:
            raise RuntimeError(
                f"the tensor-parallel ranks of data-parallel rank(s) {split} hold different first "
                "batches; a tensor-parallel group must train on one batch"
            )
        if len(by_dp_rank) > 1 and len({next(iter(p)) for p in by_dp_rank.values()}) == 1:
            raise RuntimeError(
                f"all {len(by_dp_rank)} data-parallel ranks hold the same first batch, so the run "
                "would train on one sample copied to every rank; their data must differ (is the "
                "dataset drawing from an unseeded or shared generator?)"
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
        latest = self.checkpoints.latest_checkpoint()
        # A checkpoint saved inside the frozen-backbone warm-up, up to and including its last step,
        # holds no optimizer state for the backbone: frozen parameters get no gradient, so AdamW
        # never created any. A fresh optimizer's template does have it, and DCP's strict load
        # refuses the difference ("Missing key ... optim.state.model..."), so the checkpoint is
        # loaded into the layout it was saved with. Only then: a frozen template on a later
        # checkpoint would silently drop the backbone's moments.
        saved_frozen = (
            latest is not None
            and 0 < checkpoint_step(latest) <= self.config.freeze_backbone_steps
        )
        if saved_frozen:
            self._set_backbone_frozen(True)
        step = self.checkpoints.load(latest) if latest is not None else 0
        resumed_from = step
        for _ in range(step):
            self.scheduler.step()
        # Before the first `next(batches)`, which is when the DataLoader draws its worker seeds:
        # an attempt resumed at a later step must not replay the data of the first one.
        self._seed_streams(step)

        batches = self._endless_batches(self.train_loader, start_epoch=step)
        self.algorithm.train()

        # Decided from the *restored* step, not from zero: a job resumed past the boundary must
        # come back unfrozen, and one resumed inside the warm-up must come back frozen. Getting
        # this from `config` alone would silently retrain a resumed run with a frozen backbone.
        backbone_frozen = step < self.config.freeze_backbone_steps
        if backbone_frozen and not saved_frozen:
            self._set_backbone_frozen(True)
        elif saved_frozen and not backbone_frozen:
            # Saved at the warm-up's last step, frozen; the backbone trains from here.
            self._set_backbone_frozen(False)

        meter: ThroughputMeter | None = None
        if self.config.measure_mfu and step < self.config.max_steps:
            # Probe the batch the first step is about to train on, then put it back, so measuring
            # does not consume one. `_build_throughput_meter` restores the RNG, and together those
            # make the probe invisible: the run sees the same batches in the same order with the
            # same random draws whether or not it is enabled.
            probe_batch = self._finish_train(
                move_to_device(next(batches), self.device), self.device
            )
            self._check_rank_batches(probe_batch)
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
                    # Moved first, then finished: what crosses the bus is the deferred form, which
                    # for `miao`'s deferred images is the crop in its stored dtype rather than a
                    # normalized float four times its size.
                    batch = self._finish_train(
                        move_to_device(raw_batch, self.device), self.device
                    )
                if not self._batches_checked:
                    self._check_rank_batches(batch)

                if self.photometric is not None:
                    # After the transfer, so the intensity and noise passes run on the device
                    # instead of costing a worker 165 ms of CPU per sample -- which is what made
                    # the input pipeline's tail, and with it 27% of a step.
                    with annotate("augment"):
                        batch = self.photometric(batch)

                with annotate("forward"), self._autocast():
                    # Through __call__, not training_step: `BaseAlgorithm.forward` aliases it, and
                    # plain replication only all-reduces gradients from forward hooks. Through
                    # `_forward` rather than `self.algorithm`, so that `[trainer].compile` reaches
                    # the step; the two are the same object unless it is set.
                    metrics = self._forward(batch)
                with annotate("backward"):
                    metrics["loss"].backward()

                if self.config.grad_clip_norm is not None:
                    with annotate("grad_clip"):
                        # `distributed.grad_norm`, not torch's: under tensor parallelism the
                        # gradients span two device meshes and torch's takes one norm over all of
                        # them at once. Identical to torch's when they do not.
                        grad_norm = clip_grad_norm_(
                            self.algorithm.parameters(), self.config.grad_clip_norm
                        )
                    metrics["grad_norm"] = grad_norm

                with annotate("optimizer"):
                    optimizer_step(self.optimizer)
                    # On `self.algorithm`, not the compiled `_forward`: it reads the parameters the
                    # step just wrote, and none of it belongs in the traced graph.
                    self.algorithm.after_optimizer_step()
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
                    stalled = data_seconds / max(now - window_start, 1e-9)
                    logged["data_wait_frac"] = stalled
                    if dist.is_initialized():
                        # The worst rank, not the mean, and not rank 0's own figure. Every rank
                        # synchronizes with every other one each step, so a single rank waiting on
                        # its input pipeline holds all of them there -- and the ranks that are
                        # merely waiting record no stall of their own, because from their side the
                        # time is spent inside a collective. Reported from rank 0 alone, this
                        # metric therefore reads ~0 during precisely the steps it exists to
                        # explain. That is not hypothetical: it is what a per-sample cost that
                        # depends on the crop, like the affinity task's connected components,
                        # produces.
                        worst = torch.tensor(stalled, device=self.device)
                        dist.all_reduce(worst, op=dist.ReduceOp.MAX)
                        logged["data_wait_frac_max"] = float(worst.item())
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
                    # Deliberately the module, not `_forward`: `torch.compile` only wraps a
                    # module's `__call__`, and `OptimizedModule` proxies every other attribute
                    # straight through, so `_forward.validation_step` *is* this same method. Naming
                    # it directly says so, and leaves the all-gather that
                    # `register_fsdp_forward_method` installed as the only thing standing between
                    # a sharded parameter and this call.
                    assert self._finish_val is not None  # implied by val_loader
                    metrics = self.algorithm.validation_step(
                        self._finish_val(move_to_device(batch, self.device), self.device)
                    )
                for name, value in reduce_metrics(metrics).items():
                    totals[name] = totals.get(name, 0.0) + value
                batches += 1
        return {name: total / max(1, batches) for name, total in totals.items()}
