#!/usr/bin/env python3
"""Round-over-round table of pseudo-label quality, from `mia_pseudolabel.py oracle` output.

    python experiments/pseudo_labeling/oracle_table.py /nrs/.../oracle_*.json

Reads the withheld-ground-truth scores for each round and prints them side by side, because the
question this experiment exists to answer -- did iterating actually improve the labels, or did the
model merely become more confident about the same mistakes -- is only visible as a trend.

Reading the table:

  precision up, instance up      the loop is working: better targets, and more of them
  precision flat, instance up    confirmation bias. The teacher asserts more and knows no more,
                                 which is what iterating invites and what the val curve alone
                                 cannot distinguish from real progress
  precision down, merges up      the threshold is too permissive to bootstrap on: a merge
                                 corrupts every pair spanning two fused objects
  enrichment near 1              abstention is untargeted -- it is discarding as much easy
                                 interior as hard boundary, so it costs signal and buys little

**These numbers select nothing.** They come from labels the experiment pretended not to have, so
using them to pick a threshold, a stopping round or a winning arm would leak ground truth and the
result would not transfer to data that is genuinely unlabelled. Selection stays on seed100.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLUMNS = [
    ("label_name", "round", "{}"),
    ("blocks", "blocks", "{}"),
    ("pair_precision", "precision", "{:.4f}"),
    ("pair_recall", "recall", "{:.4f}"),
    ("frac_instance", "instance", "{:.3f}"),
    ("frac_ignore", "abstain", "{:.3f}"),
    ("merged_pseudo", "merges", "{}"),
    ("split_gt", "splits", "{}"),
    ("boundary_enrichment", "enrich", "{:.2f}"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("oracle_json", type=Path, nargs="+")
    args = parser.parse_args()

    rounds = [json.loads(p.read_text()) for p in args.oracle_json]

    widths = [max(len(head), max(len(fmt.format(r[key])) for r in rounds))
              for key, head, fmt in COLUMNS]
    print("  ".join(h.rjust(w) for (_, h, _), w in zip(COLUMNS, widths, strict=True)))
    print("  ".join("-" * w for w in widths))
    for r in rounds:
        print("  ".join(fmt.format(r[key]).rjust(w)
                        for (key, _, fmt), w in zip(COLUMNS, widths, strict=True)))

    for r in rounds:
        print(f"\n{r['label_name']}: teacher {r['teacher_run']} step {r['teacher_step']}")

    if len(rounds) >= 2:
        first, last = rounds[0], rounds[-1]
        d_prec = last["pair_precision"] - first["pair_precision"]
        d_inst = last["frac_instance"] - first["frac_instance"]
        print(f"\n{first['label_name']} -> {last['label_name']}: "
              f"precision {d_prec:+.4f}, instance coverage {d_inst:+.3f}, "
              f"merges {last['merged_pseudo'] - first['merged_pseudo']:+d}")
        if d_inst > 0.01 and d_prec < 0.002:
            print("  ^ coverage grew without precision: the signature of confirmation bias, not "
                  "improvement. Check the val curve before reading later rounds as progress.")


if __name__ == "__main__":
    main()
