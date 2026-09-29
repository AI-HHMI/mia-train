#!/usr/bin/env python
"""Turn the sweep's JSON lines into the markdown tables in README.md.

    python experiments/b300_capability_run/table.py /nrs/.../b300_capability/*.jsonl

Regenerating the table rather than hand-copying it is the point: the sweep is re-run whenever the
model, the head or the parallelism changes, and a README whose numbers were typed once drifts from
the file they came from without anything saying so.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COLUMNS = (
    ("crop", lambda r: f"{r['size']}³"),
    ("tokens", lambda r: f"{r['tokens']:,}"),
    ("global batch", lambda r: str(r["global_batch"])),
    ("s/step", lambda r: f"{r['step_seconds']:.2f}"),
    ("peak GiB", lambda r: f"{r['peak_reserved_gib']:.1f}"),
    ("Mvoxel/s", lambda r: f"{r['voxels_per_s'] / 1e6:.1f}"),
    ("samples/s", lambda r: f"{r['samples_per_s']:.2f}"),
    ("encoder MFU", lambda r: f"{r['encoder_mfu']:.3f}" if r.get("encoder_mfu") else "-"),
)


def rows(path: Path) -> list[dict]:
    return sorted(
        (json.loads(line) for line in path.read_text().splitlines() if line.strip()),
        key=lambda r: r["size"],
    )


def main(paths: list[str]) -> int:
    for name in paths:
        path = Path(name)
        records = rows(path)
        if not records:
            print(f"### {path.stem}\n\n(no records)\n")
            continue
        first = records[0]
        print(
            f"### {path.stem} — dp_replicate {first['dp_replicate']} x dp_shard "
            f"{first['dp_shard']} x tp {first['tp']} = {first['world_size']} GPUs, "
            f"{first['decoder']} head\n"
        )
        print("| " + " | ".join(head for head, _ in COLUMNS) + " |")
        print("|" + "|".join(["---"] * len(COLUMNS)) + "|")
        for record in records:
            print("| " + " | ".join(render(record) for _, render in COLUMNS) + " |")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
