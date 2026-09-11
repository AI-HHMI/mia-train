#!/usr/bin/env python3
"""Segment-everything a promptable model over blocks of unlabeled volumes -> instance pseudo-labels.

    # label K blocks of one unlabeled volume into a sidecar container (GPU)
    python experiments/sam_lmd_v1/pseudolabel.py label <run_dir> --volume <name> \\
        --out <sidecar root> --label-name sam_r1 --blocks 1 [--step N] [--block 512]

    # the same pass over blocks of a GROUND-TRUTH volume, scored against its labels (GPU)
    python experiments/sam_lmd_v1/pseudolabel.py diagnose <run_dir> --volume kasthuri15_ac3 \\
        --out <dir>/kasthuri15_ac3.json --blocks 1

    # one table over a directory of diagnose outputs (CPU)
    python experiments/sam_lmd_v1/pseudolabel.py summarize <dir> [--out summary.json]

This is the fully automatic stage of the Segment Anything data engine, with the model standing in
for the annotators: a grid of clicks per tile, the IoU head and the stability score deciding which
masks are kept, NMS removing duplicates, tiles reconciled into one labelling. All of that is
`algorithms/promptable/amg.PromptGridPredictor`, reached exactly as `predict.py` reaches it --
through `load_algorithm` and `volume_predictor()` -- so a pseudo-label is produced by the same code
that produces a scored prediction, with the thresholds turned towards precision (see LABEL_AMG).

What is added here is only what an unlabeled volume needs and `predict.py` does not provide: a
full-extent bounding box (the tile lattice requires one), a block rather than a whole volume (a
volume can be 40 000 voxels across; a 512-cube of the 8 nm lattice labels in minutes), and a place
to write the result that is not the read-only published store (`blocks.py`).

Everything that decides what a block's labels are is recorded in its sidecar's manifest: the
teacher run and step, every mask-generator setting, and per block the region covered, the
instance count and the claimed fraction. `make_round_config.py` reads those manifests; nothing
else needs to be told what was labelled.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

from blocks import (  # noqa: E402
    block_edges,
    create_label_array,
    create_sidecar,
    nested_choice,
    partition,
    permute_box,
    spatial_axes_metadata,
    volume_seed,
)

CORPUS = HERE / "unlabeled_corpus.yaml"
GT_CONFIG = REPO / "experiments" / "lmd_ssl_v1" / "lmd_finetune_singlescale.yaml"

#: Mask-generator settings for LABELLING, applied over the run's own with `--override` semantics.
#: Stricter than the eval settings on the two confidence gates, because a training target is
#: judged by its precision where a scored prediction is judged by pq: a missed object costs a
#: pseudo-label nothing (the voxels simply stay unclaimed and are never prompted for) while a
#: merged or truncated one is trained on. The grid density and tile reconciliation are the values
#: promptable_seg_v1/RESULTS.md found best, and `predict_eval.sh` uses the same ones.
LABEL_AMG: dict[str, Any] = {
    "points_per_side": 14,
    "pred_iou_thresh": 0.7,
    "stability_thresh": 0.8,
    "nms_iou": 0.7,
    "tile_merge": "propagate",
    "propagate_min_coverage": 0.8,
    "points_per_batch": 64,
}


# ------------------------------------------------------------------------------ resolving a volume


def _spatial(config: Any) -> str:
    return "".join(axis for axis in config.output_axes if axis in "xyz")


def resolve_volume(config: Any, name: str) -> dict[str, Any]:
    """The facts about one volume the lattice and the sidecar need, read through miao.

    A volume without a `bounding_box` is given its full level-0 extent as one, in the config's
    output axis order, because `VolumeGrid` insists on a box and for an unlabeled volume the
    whole thing is the region of interest. Everything geometric comes back in STORAGE axis order,
    the order the label array will be written in.
    """
    from miao.dataset import VolumeDataset

    try:
        volume = next(v for v in config.volumes if v.name == name)
    except StopIteration:
        raise SystemExit(
            f"no volume named {name!r}; the config holds {[v.name for v in config.volumes]}"
        ) from None
    single = config.model_copy(update={"volumes": [volume]})
    info = VolumeDataset(single)._volumes[0]
    storage_axes: str = info.img_spatial_axes
    level0 = min(info.image_meta.scales)
    meta0 = info.image_meta.scales[level0]
    spatial_shape = [int(meta0.shape[i]) for i in info.img_spatial_idx]

    if volume.bounding_box is None:
        full = [[0, extent] for extent in spatial_shape]
        volume = volume.model_copy(
            update={"bounding_box": permute_box(full, storage_axes, _spatial(config))}
        )
        single = config.model_copy(update={"volumes": [volume]})
        info = VolumeDataset(single)._volumes[0]

    lattice_nm = [float(v) for v in config.resolutions[0]] if config.resolutions else None
    if lattice_nm is None:
        raise SystemExit("the data config sets no `resolutions`; the lattice needs one")
    return {
        "volume": volume,
        "config": single,
        "info": info,
        "storage_axes": storage_axes,
        "output_spatial_axes": _spatial(config),
        "spatial_shape": spatial_shape,
        "voxel_nm": [float(meta0.scale_factors[i]) for i in info.img_spatial_idx],
        "translation_nm": [float(meta0.translation_or_zeros()[i]) for i in info.img_spatial_idx],
        "axes_meta": spatial_axes_metadata(info.image_meta.axes, info.img_spatial_idx),
        # The lattice voxel, permuted from the config's spatial order into storage order.
        "lattice_nm": [lattice_nm[_spatial(config).index(axis)] for axis in storage_axes],
        "box": [[int(lo), int(hi)] for lo, hi in np.asarray(info.bounding_box).tolist()],
    }


def plan_blocks(resolved: dict[str, Any], lattice_block: int, count: int
                ) -> list[tuple[int, list[list[int]]]]:
    """The first `count` cells of this volume's fixed partition, storage order."""
    box = resolved["box"]
    edges = block_edges(
        lattice_block, resolved["lattice_nm"], resolved["voxel_nm"],
        [hi - lo for lo, hi in box],
    )
    cells = partition(box, edges)
    return nested_choice(cells, volume_seed(resolved["volume"].name), count)


# --------------------------------------------------------------------------------- the model pass


def load_model(run_dir: Path, step: int | None, amg: dict[str, Any]) -> tuple[Any, int, dict]:
    from predict import load_algorithm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overrides = [f"algorithm.{key}={_toml(value)}" for key, value in amg.items()]
    algorithm, loaded, resolved = load_algorithm(run_dir, device, step, overrides=overrides)
    return algorithm, loaded, resolved


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def run_block(algorithm: Any, resolved_cfg: dict, config_resolved: dict[str, Any],
              box: list[list[int]]) -> tuple[Any, Any, float]:
    """One block through the mask generator -> (grid, VolumePrediction, seconds)."""
    from predict import resolve_patch
    from prediction.dense import select_predictor
    from prediction.grid import VolumeGrid

    name = config_resolved["volume"].name
    patch = resolve_patch(config_resolved["config"], resolved_cfg, name, None)
    grid = VolumeGrid(config_resolved["config"], name, patch, box=box)
    device = next(algorithm.parameters()).device
    started = time.perf_counter()
    prediction = select_predictor(algorithm).run(grid, device)
    return grid, prediction, time.perf_counter() - started


def amg_settings(algorithm: Any, amg: dict[str, Any]) -> dict[str, Any]:
    """Every generator knob the run ended up with, for the manifest."""
    settings = dict(algorithm.amg_settings)
    settings.update(amg)
    return settings


# --------------------------------------------------------------------------------------- label


def sidecar_path(root: Path, name: str) -> Path:
    return root / (name.replace("/", "__") + ".zarr")


def cmd_label(args: argparse.Namespace) -> None:
    import zarr
    from miao.config import load_config

    from prediction.grid import resample_labels

    config = load_config(args.corpus)
    resolved = resolve_volume(config, args.volume)
    volume = resolved["volume"]
    out = sidecar_path(args.out, volume.name)
    manifest_path = out / f"pseudolabel_{args.label_name}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    done = {block["index"] for block in manifest["blocks"]} if manifest else set()

    planned = plan_blocks(resolved, args.block, args.blocks)
    todo = [(index, box) for index, box in planned if index not in done]
    print(f"{volume.name}: shape {resolved['spatial_shape']} ({resolved['storage_axes']}), "
          f"{resolved['voxel_nm']} nm, {len(planned)} block(s) planned, {len(todo)} to label",
          flush=True)
    if not todo:
        print("  already complete -- nothing to do", flush=True)
        return

    amg = dict(LABEL_AMG)
    amg.update(args.amg)
    algorithm, step, resolved_cfg = load_model(args.run_dir, args.step, amg)

    create_sidecar(Path(volume.path), out, volume.image_key, volume.zarr_version)
    if manifest is None:
        array = create_label_array(
            out, args.label_name, resolved["axes_meta"], resolved["spatial_shape"],
            resolved["voxel_nm"], resolved["translation_nm"], volume.zarr_version,
        )
        manifest = {
            "volume": volume.name, "source": volume.path, "sidecar": str(out),
            "image_key": volume.image_key, "zarr_version": volume.zarr_version,
            "label_name": args.label_name, "label_key": f"labels/{args.label_name}",
            "storage_axes": resolved["storage_axes"],
            "output_spatial_axes": resolved["output_spatial_axes"],
            "spatial_shape_level0": resolved["spatial_shape"],
            "voxel_level0_nm": resolved["voxel_nm"],
            "normalize": volume.normalize, "normalize_min": volume.normalize_min,
            "normalize_max": volume.normalize_max,
            "run": args.run_dir.name, "run_dir": str(args.run_dir), "step": step,
            "lattice_block": args.block, "amg": amg_settings(algorithm, amg),
            "blocks": [],
        }
    else:
        # Resuming a partly labelled volume: the array already holds the finished blocks.
        array = zarr.open_array(str(out / "labels" / args.label_name / "s0"), mode="r+")
        if manifest["run"] != args.run_dir.name or manifest["step"] != step:
            raise SystemExit(
                f"{manifest_path} was written by {manifest['run']} step {manifest['step']}; "
                f"refusing to add blocks from {args.run_dir.name} step {step} to it"
            )

    for index, box in todo:
        grid, prediction, seconds = run_block(algorithm, resolved_cfg, resolved, box)
        # Nearest-neighbour from the lattice back to the source's own voxels, ids intact (an id
        # survives an index map, not an interpolation). int32 first: ids are dense and small, and
        # the native block can be eight times the lattice block's voxels.
        native = resample_labels(
            prediction.array.astype(np.int32), tuple(int(v) for v in grid.native_extent)
        )
        covered = grid.native_box()
        array[tuple(slice(lo, hi) for lo, hi in covered)] = native
        claimed = float((prediction.array > 0).mean())
        entry = {
            "index": index,
            "box_storage": box,
            "native_box_storage": covered,
            # The region actually labelled, exactly. make_round_config.py derives the miao
            # `bounding_box` from it, with the slack miao's centre rule needs.
            "covered_box_xyz": permute_box(
                covered, resolved["storage_axes"], resolved["output_spatial_axes"]
            ),
            "output_shape": list(prediction.array.shape),
            "tiles": len(grid.tiles),
            "instances": int(prediction.attrs["instances"]),
            "claimed_fraction": round(claimed, 6),
            "seconds": round(seconds, 1),
        }
        manifest["blocks"].append(entry)
        manifest["blocks"].sort(key=lambda block: block["index"])
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"  block {index}: {entry['tiles']} tiles, {entry['instances']} instances, "
              f"{100 * claimed:.1f}% claimed, {seconds:.0f} s -> {covered}", flush=True)
        del native
    print(f"wrote {manifest_path} ({len(manifest['blocks'])} blocks)", flush=True)


# ------------------------------------------------------------------------------------ diagnose


def compare_labellings(pred: np.ndarray, truth: np.ndarray, *, iou_thresh: float = 0.5,
                       part: float = 0.1, min_truth_voxels: int = 512) -> dict[str, Any]:
    """How a pseudo-labelling relates to the ground truth on the same lattice.

    Precision is the number that matters for a training target: the fraction of pseudo-masks whose
    best-matching true object reaches `iou_thresh`. A *merge* is a pseudo-mask of which two or more
    true objects each make up at least `part`; recall is over true objects of at least
    `min_truth_voxels` (the same floor the strategy prompts for). Truth ids are factorised first --
    this corpus stores 64-bit segment ids, and a pair key built by multiplication would overflow.
    """
    p = pred.ravel().astype(np.int64)
    t_values, t = np.unique(truth.ravel(), return_inverse=True)
    t = t.astype(np.int64)
    n_truth = int(t_values.size)
    background = {int(np.searchsorted(t_values, v)) for v in t_values if v <= 0}

    claimed = p > 0
    n_claimed = int(claimed.sum())
    pseudo_sizes = np.bincount(p[claimed]) if n_claimed else np.zeros(1, dtype=np.int64)
    truth_sizes = np.bincount(t, minlength=n_truth)
    pseudo_ids = np.flatnonzero(pseudo_sizes)
    pseudo_ids = pseudo_ids[pseudo_ids > 0]
    truth_ids = np.array(
        [i for i in np.flatnonzero(truth_sizes >= min_truth_voxels) if i not in background],
        dtype=np.int64,
    )
    result: dict[str, Any] = {
        "pseudo_instances": int(pseudo_ids.size),
        "truth_instances": int(truth_ids.size),
        "claimed_fraction": float(claimed.mean()),
        "truth_foreground_fraction": float(np.isin(t, list(background), invert=True).mean()),
    }
    if pseudo_ids.size == 0:
        result.update(precision=0.0, recall=0.0, merges=0, merge_rate=0.0,
                      claimed_on_background=0.0, mean_best_iou=0.0)
        return result

    pairs, counts = np.unique(p[claimed] * n_truth + t[claimed], return_counts=True)
    pid, tid = pairs // n_truth, pairs % n_truth
    on_object = np.isin(tid, list(background), invert=True)
    inter = counts[on_object].astype(np.float64)
    union = pseudo_sizes[pid[on_object]] + truth_sizes[tid[on_object]] - inter
    iou = inter / union

    best_pseudo = np.zeros(pseudo_sizes.size)
    np.maximum.at(best_pseudo, pid[on_object], iou)
    best_truth = np.zeros(n_truth)
    np.maximum.at(best_truth, tid[on_object], iou)

    significant = (inter / pseudo_sizes[pid[on_object]]) >= part
    partners = np.bincount(pid[on_object][significant], minlength=pseudo_sizes.size)
    merges = int((partners >= 2).sum())

    result.update(
        precision=float((best_pseudo[pseudo_ids] >= iou_thresh).mean()),
        recall=float((best_truth[truth_ids] >= iou_thresh).mean()) if truth_ids.size else 0.0,
        merges=merges,
        merge_rate=merges / pseudo_ids.size,
        claimed_on_background=float(counts[~on_object].sum() / n_claimed),
        mean_best_iou=float(best_pseudo[pseudo_ids].mean()),
    )
    return result


def cmd_diagnose(args: argparse.Namespace) -> None:
    from miao.config import load_config

    config = load_config(args.gt_config)
    resolved = resolve_volume(config, args.volume)
    if resolved["volume"].label_key is None:
        raise SystemExit(f"{args.volume} has no label_key in {args.gt_config}; nothing to diagnose")
    amg = dict(LABEL_AMG)
    amg.update(args.amg)
    algorithm, step, resolved_cfg = load_model(args.run_dir, args.step, amg)

    report: dict[str, Any] = {
        "volume": args.volume, "gt_config": str(args.gt_config), "run": args.run_dir.name,
        "step": step, "lattice_block": args.block, "amg": amg_settings(algorithm, amg),
        "blocks": [],
    }
    for index, box in plan_blocks(resolved, args.block, args.blocks):
        grid, prediction, seconds = run_block(algorithm, resolved_cfg, resolved, box)
        truth = grid.read_ground_truth()
        scores = compare_labellings(prediction.array, truth, min_truth_voxels=args.min_truth_voxels)
        scores.update(index=index, native_box_storage=grid.native_box(), tiles=len(grid.tiles),
                      seconds=round(seconds, 1))
        report["blocks"].append(scores)
        print(f"  block {index}: {scores['pseudo_instances']} pseudo vs "
              f"{scores['truth_instances']} true  precision@0.5 {scores['precision']:.3f}  "
              f"recall {scores['recall']:.3f}  "
              f"merges {scores['merges']}  claimed {100 * scores['claimed_fraction']:.1f}% "
              f"(truth fg {100 * scores['truth_foreground_fraction']:.1f}%)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}", flush=True)


def cmd_summarize(args: argparse.Namespace) -> None:
    """Pool every block of every diagnose report in a directory into one line per volume + total."""
    reports = sorted(args.directory.glob("*.json"))
    reports = [p for p in reports if p.name != "summary.json"]
    if not reports:
        raise SystemExit(f"no diagnose reports under {args.directory}")
    rows, totals = [], {"pseudo": 0, "truth": 0, "hits": 0.0, "found": 0.0, "merges": 0}
    for path in reports:
        report = json.loads(path.read_text())
        blocks = report["blocks"]
        pseudo = sum(b["pseudo_instances"] for b in blocks)
        truth = sum(b["truth_instances"] for b in blocks)
        hits = sum(b["precision"] * b["pseudo_instances"] for b in blocks)
        found = sum(b["recall"] * b["truth_instances"] for b in blocks)
        merges = sum(b["merges"] for b in blocks)
        rows.append({
            "volume": report["volume"], "blocks": len(blocks), "pseudo_instances": pseudo,
            "truth_instances": truth, "precision": hits / max(pseudo, 1),
            "recall": found / max(truth, 1), "merges": merges,
            "merge_rate": merges / max(pseudo, 1),
            "claimed_fraction": float(np.mean([b["claimed_fraction"] for b in blocks])),
        })
        for key, value in (("pseudo", pseudo), ("truth", truth), ("hits", hits),
                           ("found", found), ("merges", merges)):
            totals[key] += value
    summary = {
        "run": json.loads(reports[0].read_text())["run"],
        "step": json.loads(reports[0].read_text())["step"],
        "amg": json.loads(reports[0].read_text())["amg"],
        "volumes": rows,
        "pooled": {
            "pseudo_instances": totals["pseudo"], "truth_instances": totals["truth"],
            "precision": totals["hits"] / max(totals["pseudo"], 1),
            "recall": totals["found"] / max(totals["truth"], 1),
            "merges": totals["merges"], "merge_rate": totals["merges"] / max(totals["pseudo"], 1),
        },
    }
    print(f"{'volume':30s} {'blocks':>6s} {'pseudo':>7s} {'truth':>6s} {'prec@.5':>8s} "
          f"{'recall':>7s} {'merges':>7s} {'claimed':>8s}")
    for row in rows:
        print(f"{row['volume'][:30]:30s} {row['blocks']:6d} {row['pseudo_instances']:7d} "
              f"{row['truth_instances']:6d} {row['precision']:8.3f} {row['recall']:7.3f} "
              f"{row['merges']:7d} {100 * row['claimed_fraction']:7.1f}%")
    pooled = summary["pooled"]
    print(f"{'POOLED':30s} {'':6s} {pooled['pseudo_instances']:7d} {pooled['truth_instances']:6d} "
          f"{pooled['precision']:8.3f} {pooled['recall']:7.3f} {pooled['merges']:7d}")
    out = args.out or (args.directory / "summary.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


# ------------------------------------------------------------------------------------------ CLI


def _amg_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("mask generator", "override LABEL_AMG for this run")
    group.add_argument("--pred-iou", type=float, dest="pred_iou_thresh")
    group.add_argument("--stability", type=float, dest="stability_thresh")
    group.add_argument("--points-per-side", type=int)
    group.add_argument("--points-per-batch", type=int)
    group.add_argument("--nms-iou", type=float)
    group.add_argument("--tile-merge", choices=("canvas", "none", "propagate"))
    group.add_argument("--propagate-min-coverage", type=float)


def _collect_amg(args: argparse.Namespace) -> dict[str, Any]:
    keys = ("pred_iou_thresh", "stability_thresh", "points_per_side", "points_per_batch",
            "nms_iou", "tile_merge", "propagate_min_coverage")
    return {key: getattr(args, key) for key in keys if getattr(args, key, None) is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("run_dir", type=Path, help="a mia-train run directory of a promptable model")
        p.add_argument("--volume", required=True)
        p.add_argument("--step", type=int, default=None, help="checkpoint step; default newest")
        p.add_argument("--blocks", type=int, default=1, help="how many blocks of this volume")
        p.add_argument("--block", type=int, default=512,
                       help="block edge in voxels of the training lattice (8 nm)")
        _amg_arguments(p)

    label = sub.add_parser("label", help="pseudo-label blocks of an unlabeled volume")
    common(label)
    label.add_argument("--corpus", type=Path, default=CORPUS)
    label.add_argument("--out", type=Path, required=True, help="sidecar root directory")
    label.add_argument("--label-name", required=True, help="e.g. sam_r1")

    diagnose = sub.add_parser("diagnose", help="label blocks of a GT volume and score them")
    common(diagnose)
    diagnose.add_argument("--gt-config", type=Path, default=GT_CONFIG)
    diagnose.add_argument("--out", type=Path, required=True, help="JSON report to write")
    diagnose.add_argument("--min-truth-voxels", type=int, default=512)

    summarize = sub.add_parser("summarize", help="pool a directory of diagnose reports")
    summarize.add_argument("directory", type=Path)
    summarize.add_argument("--out", type=Path, default=None)

    args = parser.parse_args()
    if args.command in ("label", "diagnose"):
        args.amg = _collect_amg(args)
    {"label": cmd_label, "diagnose": cmd_diagnose, "summarize": cmd_summarize}[args.command](args)


if __name__ == "__main__":
    main()
