#!/usr/bin/env python3
"""Emit one data-engine round's miao config: the ground-truth volumes plus every pseudo block.

    python experiments/sam_lmd_v1/make_round_config.py --sidecar-root <root> --label-name sam_r1 \\
        --out <round.yaml> --gt-weight 0.5 --expect 87 [--verify]

The mixture is the point of the file. The four ground-truth finetune volumes of lmd_ssl_v1 keep
`--gt-weight` of the samples between them; every pseudo-labelled block shares the rest equally. The
paper found oversampling its manually annotated masks best when mixing them with automatic ones,
and the NoisyStudent-style affinity experiment in this repo trained at 0.2; 0.5 is the cautious
starting point for a first round whose labels' quality is unmeasured on unlabeled data.

Two structural points inherited from `experiments/pseudo_labeling/make_round_config.py`, because
they are what make sparse pseudo-labels safe to sample:

  * **One volume entry per block, not per volume.** A sidecar's label array is volume-shaped, but
    only its labelled blocks were ever written; everything else reads as ignore. An entry per block
    with the block's own `bounding_box` keeps every crop inside labelled space and guarantees no
    crop spans two blocks, which is what lets each block number its instances independently.

  * **`label_key` is per volume.** The ground-truth entries point at the published label groups
    and the pseudo entries at `labels/sam_rN` inside the sidecars, in one file.

Blocks in which the teacher kept no mask at all are left out: a crop from one can only ever produce
a step with nothing to prompt for, which is a full encoder pass for zero gradient.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GT_CONFIG = REPO / "experiments" / "lmd_ssl_v1" / "lmd_finetune_singlescale.yaml"

RESOLUTION = [8.0, 8.0, 8.0]
PATCH = 256


def miao_box(covered: list[list[int]], shape: list[int], margin: int = 1) -> list[list[int]]:
    """The `bounding_box` a block gets: its covered region plus `margin` voxels per side.

    miao places a patch centre at least `patch // 2 + 1` from a box's near face and `patch // 2`
    from its far face, so a box exactly one patch wide has no valid centre and the dataset refuses
    to build (`Valid center range collapsed`). A block whose lattice covers exactly one tile on
    some axis -- every block of a thin serial-section volume does on z -- is exactly that wide.
    One voxel of slack per side is the least that admits a centre; what it costs is at most one
    voxel of never-labelled space (reading as -1, i.e. "not an object") along a crop's face.
    Clamped to the volume, which is why the volume's shape is needed here.
    """
    return [
        [max(0, int(lo) - margin), min(int(extent), int(hi) + margin)]
        for (lo, hi), extent in zip(covered, shape, strict=True)
    ]


def gt_volumes(weight_total: float) -> list[dict]:
    """lmd_ssl_v1's four finetune volumes, exactly as arms 1/2 trained on them, sharing weight."""
    config = yaml.safe_load(GT_CONFIG.read_text())
    volumes = [dict(v) for v in config["volumes"]]
    for volume in volumes:
        volume["weight"] = round(weight_total / len(volumes), 6)
    return volumes


def shape_xyz(manifest: dict) -> list[int]:
    """The volume's level-0 spatial shape in the config's (xyz) order, from storage order."""
    storage, out = manifest["storage_axes"], manifest["output_spatial_axes"]
    return [int(manifest["spatial_shape_level0"][storage.index(axis)]) for axis in out]


def pseudo_volumes(root: Path, label_name: str, weight_total: float, expect: int | None
                   ) -> tuple[list[dict], dict]:
    manifests = sorted(root.glob(f"*.zarr/pseudolabel_{label_name}.json"))
    if expect is not None and len(manifests) != expect:
        raise SystemExit(
            f"{len(manifests)} manifest(s) for {label_name!r} under {root}, expected {expect}: a "
            "partial round would train on less data than intended and confound the arms"
        )
    if not manifests:
        raise SystemExit(f"no */pseudolabel_{label_name}.json under {root}; run `label` first")

    blocks, empty = [], 0
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        for block in manifest["blocks"]:
            if block["instances"] == 0:
                empty += 1
                continue
            blocks.append((manifest, block))
    if not blocks:
        raise SystemExit(f"every labelled block under {root} is empty; nothing to train on")

    share = round(weight_total / len(blocks), 8)
    volumes = []
    for manifest, block in blocks:
        entry = {
            "name": f"{manifest['volume']}/block-{block['index']:04d}",
            "path": manifest["sidecar"],
            "image_key": manifest["image_key"],
            "label_key": manifest["label_key"],
            "zarr_version": manifest["zarr_version"],
            "weight": share,
            "normalize": manifest.get("normalize", True),
            "bounding_box": miao_box(block["covered_box_xyz"], shape_xyz(manifest)),
        }
        # The corpus's per-volume intensity windows travel with the block; without them a uint16
        # ExM volume would be divided by 65535 and the model would see a black crop.
        for key in ("normalize_min", "normalize_max"):
            if manifest.get(key) is not None:
                entry[key] = manifest[key]
        volumes.append(entry)

    first = json.loads(manifests[0].read_text())
    claimed = [b["claimed_fraction"] for _, b in blocks]
    provenance = {
        "volumes": len(manifests), "blocks": len(blocks), "empty_blocks": empty,
        "teacher_run": first["run"], "teacher_step": first["step"],
        "amg": first["amg"], "lattice_block": first["lattice_block"],
        "instances": sum(b["instances"] for _, b in blocks),
        "mean_claimed_fraction": round(sum(claimed) / len(claimed), 4),
    }
    return volumes, provenance


def build(root: Path, label_name: str, gt_weight: float, expect: int | None,
          samples_per_epoch: int) -> tuple[dict, dict]:
    if not 0.0 <= gt_weight <= 1.0:
        raise SystemExit(f"--gt-weight must be in [0, 1], got {gt_weight}")
    volumes = gt_volumes(gt_weight)
    pseudo, provenance = pseudo_volumes(root, label_name, 1.0 - gt_weight, expect)
    config = {
        "resolutions": [RESOLUTION],
        "patch_size": [PATCH] * 3,
        "output_axes": "lcxyz",
        "samples_per_epoch": samples_per_epoch,
        "volumes": volumes + pseudo,
    }
    return config, provenance


def verify(path: Path) -> None:
    """Build the dataset miao would build, so a bad entry fails here and not after a queue wait."""
    sys.path.insert(0, str(REPO / "src"))
    import components  # noqa: F401,PLC0415
    from data.registry import DataRegistry  # noqa: PLC0415

    dataset = DataRegistry.build("miao_volumes", config_path=str(path))
    _ = dataset.dataset
    print(f"verified: {len(dataset.config.volumes)} entries build at a {PATCH}-cube")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--label-name", required=True, help="e.g. sam_r1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gt-weight", type=float, default=0.5,
                        help="share of samples drawn from the four ground-truth volumes")
    parser.add_argument("--expect", type=int, default=None,
                        help="refuse unless exactly this many volumes have a manifest")
    parser.add_argument("--samples-per-epoch", type=int, default=10_000)
    parser.add_argument("--verify", action="store_true",
                        help="also build the dataset through mia-train's registry")
    args = parser.parse_args()

    config, provenance = build(args.sidecar_root, args.label_name, args.gt_weight, args.expect,
                               args.samples_per_epoch)
    header = (
        "# GENERATED by experiments/sam_lmd_v1/make_round_config.py -- do not edit by hand.\n"
        f"# ground truth   : 4 volumes of lmd_ssl_v1's finetune split (weight {args.gt_weight})\n"
        f"# pseudo-labels  : {provenance['blocks']} blocks over {provenance['volumes']} volumes "
        f"(weight {round(1.0 - args.gt_weight, 6)}), {provenance['empty_blocks']} empty dropped\n"
        f"# teacher        : {provenance['teacher_run']} step {provenance['teacher_step']}\n"
        f"# mask generator : {json.dumps(provenance['amg'])}\n"
        f"# pseudo content : {provenance['instances']} instances, mean claimed fraction "
        f"{provenance['mean_claimed_fraction']}\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header + yaml.safe_dump(config, sort_keys=False))
    print(header + f"wrote {args.out} ({len(config['volumes'])} volume entries)")
    if args.verify:
        verify(args.out)


if __name__ == "__main__":
    main()
