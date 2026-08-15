"""Model FLOPs utilization: the fraction of a GPU's arithmetic capacity a training step uses.

MFU is the one number that says whether a run is limited by arithmetic or by everything else —
data loading, collectives, launch overhead, memory bandwidth. Loss curves cannot distinguish a
run that is twice as slow as it should be from one that is not.

The numerator is *measured*, not derived from `BaseModel.flops()`, and that is the whole design
decision here. `flops()` returns one full-resolution forward pass of the backbone, which is not
what a step costs in five of this repo's six algorithms:

  - `mae` encodes only the unmasked ~25% of tokens but runs its decoder over the *full* grid, so
    the customary `3 x flops()` overstates a step by ~2.8x.
  - `dinov3` runs a no-grad teacher over 2 global crops plus a student over 2 global and 8 local
    crops at a second resolution, and `3 x flops()` understates it by ~4.2x.
  - `semantic_seg` and `affinity_seg` upsample to full voxel resolution *before* convolving, so
    the head can cost several times the backbone — up to 5.5x at small crops.
  - `muvit_mae` masks a joint multi-level sequence and runs one decoder per level.

Worse, none of that is a fixed correction factor: attention is quadratic in token count, so at
ViT-L over a 512-cube attention is 84% of the backbone and dropping to 25% of tokens costs 0.094x
rather than 0.25x. A measured count follows the code instead of tracking six formulas that would
drift the first time an algorithm changed.

What the measurement does *not* see is FlashAttention-4, which is why counting runs under
`counting_kernels` — see that context manager for the measurement showing attention silently
counted as zero.

One caveat worth stating plainly: with `[trainer].activation_checkpointing` enabled the counted
step includes the recomputed forward, so the reported figure is hardware FLOPs utilization rather
than the model FLOPs utilization of the PaLM definition. Checkpointing is off by default, and
when it is off the two coincide.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from layers.common.attention import counting_kernels as counting_kernels_common
from layers.dinov3.attention import counting_kernels as counting_kernels_dinov3


@contextlib.contextmanager
def _counted_kernels(module: nn.Module) -> Iterator[None]:
    """Put every attention layer in the repo, of either implementation, on a countable kernel.

    Both are entered because the repo has two unrelated attention modules -- the from-scratch one
    `ViT3D`/`MuViT3D` use and the DINOv3 port -- and each has its own FlashAttention-4 switch with
    no common base class. Entering only one is silent: it rewires nothing on the other's models and
    the count comes back short by the whole attention term, which is most of a step at long
    context. `tests/unit/test_engine_mfu.py` pins that every registered model is actually covered,
    so a third implementation cannot be added without the omission failing a test.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(counting_kernels_common(module))
        stack.enter_context(counting_kernels_dinov3(module))
        yield


def measure_step_flops(
    algorithm: torch.nn.Module,
    batch: Any,
    autocast: Any,
) -> int:
    """FLOPs a single training step executes on this rank, forward and backward.

    Backward is counted rather than assumed: the textbook "backward is twice forward" rule is an
    approximation the first layer breaks, since it needs no input gradient. Measured on an H100
    for a 4-block ViT3D the true ratio is 2.83, and for `dinov3` it is nowhere near 3 because the
    teacher forward has no backward at all.

    `autocast` is the trainer's own autocast factory, so the count reflects the dtype the run
    actually trains in.

    The step is run for its arithmetic alone, so every trace of it is undone. Three kinds of state
    would otherwise leak into training, and the third is the one that is easy to miss:

    - **Gradients**, which would be added to the first real step's. Zeroed.
    - **The RNG**, drawn from by mask sampling and augmentation. Two runs with the same seed and
      different `measure_mfu` would otherwise diverge from the first step.
    - **Buffers.** `BaseAlgorithm.training_step` is free to mutate its own state, and `dinov3`
      does: it increments a registered `_step` counter that drives the teacher temperature and
      momentum schedules. Probing would advance those schedules by one step for the whole run and
      persist the shift into checkpoints. Buffers are snapshotted and copied back rather than
      special-cased per algorithm, since the engine deliberately knows nothing about what any
      algorithm keeps.

    Parameters are left alone because nothing steps the optimizer here.
    """
    buffers = {name: buffer.detach().clone() for name, buffer in algorithm.named_buffers()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        with _counted_kernels(algorithm), autocast():
            with FlopCounterMode(display=False) as counter:
                metrics = algorithm(batch)
                metrics["loss"].backward()
    finally:
        algorithm.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        with torch.no_grad():
            for name, buffer in algorithm.named_buffers():
                buffer.copy_(buffers[name])

    return int(counter.get_total_flops())


@dataclass(frozen=True)
class Throughput:
    """One measurement window's rates. `mfu` is None when the device has no tabulated peak."""

    mfu: float | None
    tflops_per_s: float
    samples_per_s: float

    def as_metrics(self) -> dict[str, float]:
        metrics = {
            "tflops_per_s": self.tflops_per_s,
            "samples_per_s": self.samples_per_s,
        }
        if self.mfu is not None:
            metrics["mfu"] = self.mfu
        return metrics


class ThroughputMeter:
    """Turns per-window wall-clock into MFU, tracked between logging points.

    Timed across a whole logging window rather than per step, because the training loop issues no
    explicit synchronization: a `t1 - t0` around one iteration would measure how long it took to
    *enqueue* the kernels, not to run them. The window boundary is placed immediately after
    `reduce_metrics`, whose `.item()` forces a device synchronization anyway, so the measurement
    is honest without adding a single sync of its own.

    The first window is discarded. It carries cuDNN autotuning, allocator growth and the first
    batches through the input pipeline, none of which recur, and including it would understate a
    run's steady state.
    """

    def __init__(
        self,
        step_flops: int,
        peak_flops: float | None,
        samples_per_step: int,
    ) -> None:
        self._step_flops = step_flops
        self._peak_flops = peak_flops
        self._samples_per_step = samples_per_step
        self._last: float | None = None
        self._warmed = False

    def start(self) -> None:
        self._last = time.perf_counter()
        self._warmed = False

    def window(self, steps: int) -> Throughput | None:
        """Rates since the previous call, or None for the first (discarded) window."""
        now = time.perf_counter()
        previous, self._last = self._last, now
        if previous is None or not self._warmed:
            self._warmed = True
            return None

        elapsed = now - previous
        flops_per_s = self._step_flops * steps / elapsed
        return Throughput(
            mfu=(flops_per_s / self._peak_flops) if self._peak_flops else None,
            tflops_per_s=flops_per_s / 1e12,
            samples_per_s=self._samples_per_step * steps / elapsed,
        )
