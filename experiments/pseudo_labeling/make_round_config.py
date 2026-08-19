#!/usr/bin/env python3
"""Emit one round's miao data config: real labels from `base`, pseudo-labels from `train_100`.

    python experiments/pseudo_labeling/make_round_config.py \
        --sidecar-root /nrs/.../pseudo_r1 --out /nrs/.../round1.yaml --gt-weight 0.3

The mix is the whole point of the file. `base` carries five cubes of *real* ground truth and
`train_100` carries however many blocks the round pseudo-labelled, and both go into one training
set -- as in NoisyStudent, where the student sees labelled and pseudo-labelled data together
rather than pseudo-labels alone. `--gt-weight` is the share of samples drawn from real labels;
the rest is split evenly over the pseudo blocks.

Two structural points that are easy to get wrong:

  * **One volume entry per block, not per cube.** A pseudo-label array is cube-shaped but only a
    few blocks of it were ever written; everything else reads as `ignore`. A single entry per
    cube with a cube-sized bounding box would sample mostly unlabelled space, and the loss would
    silently be computed over almost nothing. Each block therefore gets its own entry with its
    own `bounding_box`, which also guarantees no crop spans two blocks -- the thing that makes
    independently-numbered blocks safe.

  * **`label_key` is per volume.** That is what lets one config mix real and pseudo labels at
    all: the `base` entries point at `labels/public_gt-cell-nisb` and the pseudo entries at
    `labels/pseudo_rN`, in the same file, with no new dataset code anywhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO / "configs" / "data" / "nisb_base.yaml"
GT_LABEL_KEY = "labels/public_gt-cell-nisb"


def base_volumes(weight_total: float) -> list[dict]:
    """The five real-ground-truth training cubes, sharing `weight_total` between them."""
    config = yaml.safe_load(BASE_CONFIG.read_text())
    volumes = config["volumes"]
    share = weight_total / len(volumes)
    out = []
    for volume in volumes:
        entry = dict(volume)
        entry["weight"] = share
        entry["label_key"] = GT_LABEL_KEY
        out.append(entry)
    return out


def pseudo_volumes(sidecar_root: Path, weight_total: float) -> tuple[list[dict], dict]:
    """One entry per pseudo-labelled block, sharing `weight_total` between them."""
    manifests = sorted(sidecar_root.glob("*.zarr/pseudolabel.json"))
    if not manifests:
        raise SystemExit(f"no */pseudolabel.json under {sidecar_root}; run `build` first")

    blocks = []
    for manifest in manifests:
        meta = json.loads(manifest.read_text())
        container = manifest.parent
        for box, block in zip(meta["bounding_boxes"], meta["blocks"], strict=True):
            blocks.append((container, meta["label_name"], box, block))

    share = weight_total / len(blocks)
    volumes = []
    for container, label_name, box, block in blocks:
        volumes.append({
            "name": f"{container.stem} {block['block'].replace('.zarr', '')}",
            "path": str(container),
            "image_key": "raw",
            "label_key": f"labels/{label_name}",
            "zarr_version": "zarr3",
            "weight": share,
            "normalize": True,
            "bounding_box": [[int(lo), int(hi)] for lo, hi in box],
        })

    first = json.loads(manifests[0].read_text())
    provenance = {
        "cubes": len(manifests), "blocks": len(blocks),
        "teacher_run": first.get("run"), "teacher_step": first.get("step"),
        "cc_logit": first.get("cc_logit"), "tau_bg": first.get("tau_bg"),
        "tau_fg": first.get("tau_fg"), "min_size": first.get("min_size"),
        "mean_frac_instance": round(sum(
            b["frac_instance"] for _, _, _, b in blocks) / len(blocks), 4),
        "mean_frac_ignore": round(sum(
            b["frac_ignore"] for _, _, _, b in blocks) / len(blocks), 4),
    }
    return volumes, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sidecar-root", type=Path, required=True,
                        help="directory of <cube>.zarr sidecars written by mia_pseudolabel build")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gt-weight", type=float, default=0.3,
                        help="share of training samples drawn from real ground truth")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="must match the teacher's training size: RoPE normalises coordinates "
                             "by the runtime grid extent, so a different size rescales every "
                             "positional relationship the model learned")
    parser.add_argument("--samples-per-epoch", type=int, default=1000)
    args = parser.parse_args()

    if not 0.0 <= args.gt_weight <= 1.0:
        raise SystemExit(f"--gt-weight must be in [0, 1], got {args.gt_weight}")

    volumes = base_volumes(args.gt_weight)
    pseudo, provenance = pseudo_volumes(args.sidecar_root, 1.0 - args.gt_weight)
    volumes += pseudo

    config = {
        "resolutions": [[9, 9, 20]],
        "output_axes": "lcxyz",
        "patch_size": [args.patch_size] * 3,
        "samples_per_epoch": args.samples_per_epoch,
        "volumes": volumes,
    }

    header = (
        "# GENERATED by experiments/pseudo_labeling/make_round_config.py -- do not edit by hand.\n"
        f"# real-label cubes : 5 (weight {args.gt_weight})\n"
        f"# pseudo blocks    : {provenance['blocks']} over {provenance['cubes']} cubes "
        f"(weight {1.0 - args.gt_weight})\n"
        f"# teacher          : {provenance['teacher_run']} step {provenance['teacher_step']}\n"
        f"# filters          : cc_logit={provenance['cc_logit']} tau_bg={provenance['tau_bg']} "
        f"tau_fg={provenance['tau_fg']} min_size={provenance['min_size']}\n"
        f"# pseudo-label mix : {provenance['mean_frac_instance']:.3f} instance, "
        f"{provenance['mean_frac_ignore']:.3f} abstained (mean over blocks)\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header + yaml.safe_dump(config, sort_keys=False))
    print(header + f"wrote {args.out} ({len(volumes)} volume entries)")


if __name__ == "__main__":
    main()
