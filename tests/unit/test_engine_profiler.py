"""Pins the profiler's off-by-default contract and its rank policy.

The properties that matter are all about what happens when profiling is *not* wanted: a disabled
profiler must create no directory, must not touch torch.profiler, and `step()` on it must be a
no-op, because every ordinary run of this repo executes that path a hundred thousand times. The
schedule arithmetic is pinned too, since `profile_start_step` counting from the wrong origin
would silently trace the unrepresentative first steps it exists to skip.

Nothing here starts a real CUDA profiler: CUPTI is unavailable on a CPU test runner, and the
behaviour worth testing is the wrapper's, not torch's.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import torch

from engine.config import TrainerConfig
from engine.profiler import StepProfiler, annotate, find_trace, should_profile, summarize


def _write_trace(path: Path, gzipped: bool = False) -> Path:
    """A minimal but structurally real chrome trace: one step, one kernel, one annotation."""
    events = {
        "traceEvents": [
            {"ph": "X", "cat": "user_annotation", "name": "ProfilerStep#7",
             "ts": 0, "dur": 1000, "pid": 1, "tid": 1},
            {"ph": "X", "cat": "user_annotation", "name": "forward",
             "ts": 10, "dur": 400, "pid": 1, "tid": 1},
            {"ph": "X", "cat": "gpu_user_annotation", "name": "forward",
             "ts": 20, "dur": 300, "pid": 1, "tid": 7},
            {"ph": "X", "cat": "kernel", "name": "sgemm", "ts": 20, "dur": 250,
             "pid": 1, "tid": 7},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaStreamSynchronize",
             "ts": 300, "dur": 90, "pid": 1, "tid": 1},
            # No "dur": must be ignored rather than crash the reader.
            {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "python"}},
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzipped:
        with gzip.open(path, "wt") as handle:
            json.dump(events, handle)
    else:
        path.write_text(json.dumps(events), encoding="utf-8")
    return path


@pytest.mark.unit
def test_disabled_profiler_writes_nothing(tmp_path: Path) -> None:
    with StepProfiler(tmp_path, start_step=0, active=1, enabled=False) as profiler:
        for _ in range(5):
            profiler.step()
    assert not (tmp_path / "profile").exists()
    assert profiler.trace_dir is None


@pytest.mark.unit
def test_profiling_is_off_by_default() -> None:
    config = TrainerConfig(max_steps=10, batch_size=1)
    assert config.profile is False
    assert config.profile_all_ranks is False
    assert config.profile_memory is False


@pytest.mark.unit
def test_only_rank_zero_profiles_by_default() -> None:
    assert should_profile(0, all_ranks=False)
    assert not should_profile(1, all_ranks=False)
    assert not should_profile(7, all_ranks=False)


@pytest.mark.unit
def test_all_ranks_profile_when_asked() -> None:
    assert all(should_profile(rank, all_ranks=True) for rank in range(8))


@pytest.mark.unit
def test_annotate_is_transparent_when_no_profiler_is_running() -> None:
    """The annotation wrapper must not change the value or the shape of what it encloses."""
    tensor = torch.ones(4)
    with annotate("region"):
        doubled = tensor * 2
    assert torch.equal(doubled, torch.full((4,), 2.0))


@pytest.mark.unit
def test_enabled_profiler_creates_its_output_directory(tmp_path: Path) -> None:
    """Built, but never entered: constructing must not require CUPTI or a GPU."""
    profiler = StepProfiler(tmp_path, start_step=50, active=6, enabled=True)
    assert profiler.trace_dir == tmp_path / "profile"
    assert profiler.trace_dir.is_dir()


@pytest.mark.unit
def test_schedule_skips_then_traces_the_requested_steps(tmp_path: Path) -> None:
    """`profile_start_step` steps pass, one warms up, then `profile_steps` are recorded.

    Pinned against torch's own schedule rather than reimplemented, so a change to the meaning of
    `skip_first` upstream fails here instead of silently moving the window.
    """
    schedule = torch.profiler.schedule(skip_first=3, wait=0, warmup=1, active=2, repeat=1)
    actions = [schedule(step) for step in range(9)]
    none, warmup = torch.profiler.ProfilerAction.NONE, torch.profiler.ProfilerAction.WARMUP

    assert actions[:3] == [none, none, none]
    assert actions[3] == warmup
    assert actions[4] == torch.profiler.ProfilerAction.RECORD
    assert actions[5] == torch.profiler.ProfilerAction.RECORD_AND_SAVE
    assert actions[6:] == [none, none, none]


@pytest.mark.unit
def test_negative_profile_start_step_is_rejected() -> None:
    with pytest.raises(ValueError, match="profile_start_step"):
        TrainerConfig(max_steps=10, batch_size=1, profile_start_step=-1)


@pytest.mark.unit
def test_zero_profile_steps_is_rejected() -> None:
    """Zero is not "disabled" -- `profile = false` is, and the message has to say so."""
    with pytest.raises(ValueError, match="profile_steps"):
        TrainerConfig(max_steps=10, batch_size=1, profile_steps=0)


@pytest.mark.unit
@pytest.mark.parametrize("handed", ["run_dir", "profile_dir", "file"])
def test_trace_is_found_however_it_is_named(tmp_path: Path, handed: str) -> None:
    """A run directory, its profile/ subdirectory and the file itself are all valid targets."""
    trace = _write_trace(tmp_path / "profile" / "host_1.pt.trace.json")
    target = {"run_dir": tmp_path, "profile_dir": tmp_path / "profile", "file": trace}[handed]
    assert find_trace(target) == trace


@pytest.mark.unit
def test_newest_trace_wins(tmp_path: Path) -> None:
    """Re-profiling a resumed run leaves the older trace in place; the newer one is the answer."""
    old = _write_trace(tmp_path / "profile" / "host_1.pt.trace.json")
    new = _write_trace(tmp_path / "profile" / "host_2.pt.trace.json")
    import os

    os.utime(old, (1_000_000, 1_000_000))
    assert find_trace(tmp_path) == new


@pytest.mark.unit
def test_missing_trace_says_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="profile"):
        find_trace(tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize("gzipped", [False, True])
def test_summary_reports_the_numbers_that_matter(tmp_path: Path, gzipped: bool) -> None:
    suffix = ".pt.trace.json.gz" if gzipped else ".pt.trace.json"
    trace = _write_trace(tmp_path / "profile" / f"host_1{suffix}", gzipped=gzipped)
    report = summarize(trace)

    assert "profiled steps        1" in report
    # The kernel runs 250 of the 1000 us the trace spans, so the device is busy a quarter of it.
    assert "GPU busy" in report and "GPU idle" in report
    assert "forward" in report and "sgemm" in report
    # One blocking sync, 90 us of it -- the count is what tells a user where to look.
    assert "cudaStreamSynchronize" in report
    assert "ui.perfetto.dev" in report


@pytest.mark.unit
def test_summary_survives_a_trace_with_no_gpu(tmp_path: Path) -> None:
    """A CPU-only run still produces a trace; reading it must explain itself, not divide by zero."""
    path = tmp_path / "cpu.pt.trace.json"
    path.write_text(
        json.dumps({"traceEvents": [
            {"ph": "X", "cat": "cpu_op", "name": "aten::add", "ts": 0, "dur": 5, "pid": 1, "tid": 1}
        ]}),
        encoding="utf-8",
    )
    assert "No GPU activity" in summarize(path)
