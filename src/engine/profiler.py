"""A torch.profiler trace of a few representative training steps.

`mfu.py` answers *how much* of the GPU a step uses; this answers *what the rest of the time went
to*. The two are complementary and the second is only worth reaching for once the first says
something is wrong -- which for this repo's volumetric runs it does: an affinity fine-tune at
256^3 reports ~6% MFU, and the reason is not visible in any scalar the run already logs.

The measurement that motivates the design: two runs of the same model on the same data, differing
only in the decoder head, measured 24.5 and 15.1 TFLOP per step -- and the *cheaper* one was
slower. That is only possible if arithmetic is a minority of the step, so the interesting
quantities are all things a FLOP counter scores at zero: host stalls waiting on the input
pipeline, device-to-host synchronizations, collectives, and memory-bandwidth-bound elementwise
work. A trace shows all four on one timeline; nothing else does.

Two things are deliberately *not* here:

  - **A summary.** Reducing a trace to a table means deciding in advance which question is being
    asked, and the reason to open a profiler at all is that the question is not yet known. The
    trace is written in the format Perfetto and TensorBoard already read.
  - **`with_stack`.** Python stack attribution on CUDA costs a large multiple of the step it is
    measuring and routinely produces traces that will not load. The `record_function` regions the
    trainer emits give the same attribution at the granularity that matters here, for free.

Enabled runs pay for the profiled window only. `torch.profiler.schedule` leaves the profiler in
its `NONE` state outside the window, where `step()` is a counter increment, so the remaining
99.99% of a 100k-step run is unaffected.

Reading a trace back is the other half of the feature and lives here too:

    python -m engine.profiler <run directory>

prints where the step went, without a GUI. A trace is also a Chrome-trace JSON, so the same file
opens at https://ui.perfetto.dev by dragging it in -- the command prints its path for that. It is
deliberately *not* readable in this repo's TensorBoard: the timeline is a separate artifact from
the scalars, and rendering it in TensorBoard needs the third-party `torch-tb-profiler` plugin,
which is not a dependency here.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


class StepProfiler:
    """Traces `active` steps once, after `start_step`, and writes them to `output_dir/profile`.

    The window is placed by *step*, not at the start of training, because the first steps of a run
    are the least representative ones it has: cuDNN picks convolution algorithms on first sight of
    a shape, the caching allocator is still growing, FSDP has not yet settled its bucketing, and
    the input pipeline's workers are still filling their prefetch queues. A trace of step 3 would
    mostly describe those.

    `start_step` counts steps taken *by this process*, not the absolute training step, so a job
    resumed at step 60000 profiles its own 50th step rather than never profiling at all.
    """

    def __init__(
        self,
        output_dir: Path,
        start_step: int,
        active: int,
        profile_memory: bool = False,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._profiler: Any = None
        if not enabled:
            return

        trace_dir = output_dir / "profile"
        trace_dir.mkdir(parents=True, exist_ok=True)
        self._trace_dir = trace_dir
        self._profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # One warmup step inside the window as well: entering the profiled state itself
            # allocates its buffers and installs the CUPTI callbacks, and the step that pays for
            # that is not one worth keeping.
            schedule=torch.profiler.schedule(
                skip_first=start_step, wait=0, warmup=1, active=active, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
            # Shapes are what make a trace readable after the fact: "aten::conv3d took 40 ms" is
            # not actionable until you can see it ran on (1, 64, 256, 256, 256). The overhead is
            # a few percent inside the window and nothing outside it.
            record_shapes=True,
            profile_memory=profile_memory,
            with_stack=False,
        )

    def __enter__(self) -> StepProfiler:
        if self._profiler is not None:
            self._profiler.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._profiler is not None:
            self._profiler.__exit__(*exc)
            self._profiler = None

    def step(self) -> None:
        """Advance the schedule. A no-op outside the profiled window."""
        if self._profiler is not None:
            self._profiler.step()

    @property
    def trace_dir(self) -> Path | None:
        return self._trace_dir if self._enabled else None


def should_profile(rank: int, all_ranks: bool) -> bool:
    """Whether this rank writes a trace.

    Rank 0 alone by default. A trace is tens to hundreds of megabytes, so eight of them is a real
    cost on shared storage, and for the questions that motivate profiling here -- is the step
    waiting on data, on a sync, on a collective? -- one rank answers them.

    `all_ranks` exists for the question one rank cannot answer: whether the ranks are *balanced*.
    A collective is only as fast as its slowest participant, so a rank that spends 50 ms in
    all-gather may be waiting rather than communicating, and telling those apart needs every
    rank's timeline. Volumetric crops make this a live concern -- the affinity target
    construction's cost depends on how many objects a crop happens to contain, which differs per
    rank on every step.
    """
    return all_ranks or rank == 0


@contextlib.contextmanager
def annotate(name: str) -> Any:
    """Name a region so it appears as one row in the trace.

    A thin wrapper over `torch.profiler.record_function` so call sites read as intent rather than
    as profiler plumbing, and so there is one place to change if the annotation mechanism ever
    does. Costs roughly a microsecond when no profiler is active, against step times measured in
    hundreds of milliseconds.
    """
    with torch.profiler.record_function(name):
        yield


def current_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


# ----------------------------------------------------------------------------------------------
# Reading a trace back.
#
# A profiler that needs a GUI to answer "is this run data-bound?" is a profiler nobody runs twice.
# What follows turns a trace into the half-dozen numbers that answer it, using nothing but the
# standard library, so it works over ssh on a login node with no port forwarding.
# ----------------------------------------------------------------------------------------------


def find_trace(target: Path) -> Path:
    """Locate the trace file, given a run directory, a `profile/` directory, or the file itself.

    All three are accepted because all three are things a user reasonably has to hand: the run
    directory is what the training job printed, `profile/` is what they get from tab-completing
    into it, and the file itself is what they would name after copying it somewhere.
    """
    if target.is_file():
        return target
    for candidate in (target / "profile", target):
        if candidate.is_dir():
            traces = sorted(candidate.glob("*.pt.trace.json*"))
            if traces:
                # Newest wins: re-profiling a resumed run leaves the earlier trace in place.
                return max(traces, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(
        f"no *.pt.trace.json under {target}. Pass a run directory containing a profile/ "
        "subdirectory, or the trace file itself. A run only writes one if it set "
        "[trainer].profile = true and reached profile_start_step."
    )


def _load(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return list(json.load(handle)["traceEvents"])


def _covered(spans: list[tuple[float, float]]) -> float:
    """Total time during which at least one span is open, in microseconds.

    Summing durations instead would double-count: kernels on different CUDA streams overlap, and
    the question here is how much of the window the device was doing *anything*, not how much
    work was queued.
    """
    if not spans:
        return 0.0
    spans.sort()
    total, start, end = 0.0, *spans[0]
    for span_start, span_end in spans[1:]:
        if span_start > end:
            total += end - start
            start, end = span_start, span_end
        else:
            end = max(end, span_end)
    return total + end - start


def summarize(trace_path: Path) -> str:
    """Where one training step went, as a report.

    Three views, because no single one is enough. The device timeline says how much of the step
    the GPU was busy at all -- idle time there is the signature of a host that cannot keep up,
    whether because it is waiting on data or blocked in a synchronization. The annotated regions
    say which part of the step owns that time. The kernel table says which single operation to go
    and look at, which is routinely not one anybody would have guessed.
    """
    events = [e for e in _load(trace_path) if e.get("ph") == "X" and e.get("dur") is not None]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_category[str(event.get("cat", "?"))].append(event)

    device = (
        by_category["kernel"] + by_category["gpu_memcpy"] + by_category["gpu_memset"]
    )
    if not device:
        return f"{trace_path}\n\nNo GPU activity in this trace -- was the run on CPU?"

    steps = [e for e in events if str(e.get("name", "")).startswith("ProfilerStep")]
    n = len(steps) or 1
    step_ms = (
        sum(e["dur"] for e in steps) / 1000.0 / n
        if steps
        else (max(e["ts"] + e["dur"] for e in device) - min(e["ts"] for e in device)) / 1000.0 / n
    )
    window_us = max(e["ts"] + e["dur"] for e in device) - min(e["ts"] for e in device)
    busy_ms = _covered([(e["ts"], e["ts"] + e["dur"]) for e in device]) / 1000.0
    window_ms = window_us / 1000.0

    out = [
        f"{trace_path}",
        "",
        f"  profiled steps        {n}",
        f"  mean step             {step_ms:8.1f} ms",
        f"  GPU busy              {busy_ms / n:8.1f} ms/step   {busy_ms / window_ms * 100:5.1f}%",
        f"  GPU idle              {(window_ms - busy_ms) / n:8.1f} ms/step   "
        f"{(window_ms - busy_ms) / window_ms * 100:5.1f}%   <- host could not keep the device fed",
        "",
    ]

    for category, title, note in (
        ("user_annotation", "host wall clock", "regions nest, so these overlap"),
        ("gpu_user_annotation", "device time", "backward is unattributed: autograd runs on its "
                                               "own thread, which record_function does not follow"),
    ):
        totals: dict[str, float] = defaultdict(float)
        for event in by_category.get(category, []):
            if not str(event["name"]).startswith("ProfilerStep"):
                totals[str(event["name"])] += event["dur"] / 1000.0
        if not totals:
            continue
        out += [f"  --- annotated regions, {title} ({note}) ---",
                f"  {'region':<34}{'ms/step':>9}{'% step':>9}"]
        for name, total in sorted(totals.items(), key=lambda kv: -kv[1])[:12]:
            out.append(f"  {name[:33]:<34}{total / n:>9.1f}{total / n / step_ms * 100:>8.1f}%")
        out.append("")

    syncs = [
        e for e in by_category.get("cuda_runtime", [])
        if "Synchronize" in str(e["name"])
    ]
    if syncs:
        blocked = sum(e["dur"] for e in syncs) / 1000.0
        out += [
            f"  host blocked in cudaStreamSynchronize & co: {blocked / n:.1f} ms/step over "
            f"{len(syncs) / n:.0f} calls/step",
            "",
        ]

    kernels: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for event in device:
        entry = kernels[str(event["name"])]
        entry[0] += event["dur"] / 1000.0
        entry[1] += 1
    out += ["  --- slowest kernels ---", f"  {'kernel':<62}{'ms/step':>9}{'% step':>9}"]
    for name, (total, _count) in sorted(kernels.items(), key=lambda kv: -kv[1][0])[:12]:
        out.append(f"  {name[:61]:<62}{total / n:>9.2f}{total / n / step_ms * 100:>8.1f}%")

    out += [
        "",
        "  For the full timeline, open this file at https://ui.perfetto.dev (drag it in):",
        f"    {trace_path}",
    ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m engine.profiler",
        description="Summarize a torch.profiler trace written by a mia-train run.",
    )
    parser.add_argument(
        "run",
        type=Path,
        help="a run directory, its profile/ subdirectory, or a .pt.trace.json file",
    )
    print(summarize(find_trace(parser.parse_args().run)))


if __name__ == "__main__":
    main()
