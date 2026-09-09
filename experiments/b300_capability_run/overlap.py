"""Measure how much of a step's collective time was hidden behind compute.

The profiler's op table reports `Self CUDA`, a *sum of kernel durations*. Two runs with
identical tables can have every collective hidden or every collective exposed, so that table
cannot answer the question -- and its rows double-count, since `record_param_comms`, `nccl:*`
and `ncclDevKernel_*` are three views of the same kernel. Only the timeline can: NCCL kernels
occupy their own streams, so overlap is two streams busy at the same instant.

Usage: python overlap.py trace.json[.gz]
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict


def union(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge spans, so a busy interval covered by two kernels is counted once."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def total(spans: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in spans)


def intersect(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            out.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return out


def main(path: str) -> None:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]

    comm: list[tuple[float, float]] = []
    compute: list[tuple[float, float]] = []
    by_stream: dict[int, float] = defaultdict(float)
    for event in events:
        if event.get("cat") != "kernel" or "dur" not in event:
            continue
        span = (event["ts"], event["ts"] + event["dur"])
        name = event["name"]
        by_stream[event.get("args", {}).get("stream", -1)] += event["dur"]
        (comm if "nccl" in name.lower() else compute).append(span)

    comm, compute = union(comm), union(compute)
    if not comm or not compute:
        print("no kernels of one class; nothing to compare")
        return
    busy = union(comm + compute)
    hidden = total(intersect(comm, compute))
    comm_total, compute_total = total(comm), total(compute)
    window = busy[-1][1] - busy[0][0]

    def row(label: str, seconds: float, share: str = "") -> None:
        print(f"{label:<24}{seconds / 1e6:8.3f} s  {share}")

    row("profiled window", window)
    row("gpu busy (either)", total(busy), f"{total(busy) / window:6.1%} of window")
    row("compute kernels", compute_total)
    row("collective kernels", comm_total, f"{comm_total / window:6.1%} of window")
    row("  overlapped w/ compute", hidden, f"{hidden / comm_total:6.1%} of collectives")
    row("  EXPOSED", comm_total - hidden, f"{(comm_total - hidden) / window:6.1%} of window")
    row("idle (neither)", window - total(busy))
    print("\nexposed time by collective (which ones failed to hide):")
    by_kind: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for event in events:
        if event.get("cat") != "kernel" or "dur" not in event:
            continue
        name = event["name"]
        if "nccl" not in name.lower():
            continue
        kind = name.split("_")[1] if name.startswith("ncclDevKernel_") else name.split("(")[0]
        by_kind[kind].append((event["ts"], event["ts"] + event["dur"]))
    for kind, spans in sorted(by_kind.items(), key=lambda kv: -total(union(kv[1]))):
        merged = union(spans)
        kind_total = total(merged)
        kind_exposed = kind_total - total(intersect(merged, compute))
        print(
            f"  {kind:<16} {len(spans):5d} calls  {kind_total / 1e6:7.3f} s total  "
            f"{kind_exposed / 1e6:7.3f} s exposed  ({kind_exposed / kind_total:5.1%})"
        )

    print("\nkernel time by stream (sum, not wall clock):")
    for stream, duration in sorted(by_stream.items(), key=lambda kv: -kv[1]):
        print(f"  stream {stream:3d}  {duration / 1e6:8.3f} s")


if __name__ == "__main__":
    main(sys.argv[1])
