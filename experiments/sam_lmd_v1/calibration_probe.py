"""Is the IoU head's score any guide to a grid mask's true IoU? Per-candidate evidence, on truth.

    # one GT volume, the SAME block the round's L1 diagnostic scored (GPU)
    python experiments/sam_lmd_v1/calibration_probe.py probe <run dir> --step N \
        --volume kasthuri15_ac3 --out <dir>/kasthuri15_ac3.npz [--block 512] [--max-tiles 27]

    # pool every volume's records in a directory into one report (CPU, seconds)
    python experiments/sam_lmd_v1/calibration_probe.py summarize <dir> \
        [--pred-iou 0.7 --stability 0.8]

The round diagnostics (`pseudolabel.py diagnose`) judge the labeller's OUTPUT: what survives the
gates, containment, NMS and tile reconciliation, scored at voxel resolution against truth. That
number was 0.15-0.19 precision@0.5 for every teacher in both versions of the recipe, and it cannot
say which stage lost the precision. This probe judges the INPUT to all of that: every candidate
the decoder produces for every grid click on every tile of the same block, before any filter,
with the truth beside it. Per candidate it records

  pred_iou        what the IoU head said (the gate's evidence)
  stability       the logits' stability score (the gate's other evidence)
  area            the mask's size in mask-grid cells; size_ok = within the generator's bounds
  iou_best        true IoU against the best-matching true object, on the mask grid, thresholded
                  exactly as `losses.mask_iou` (the quantity the head is trained to predict)
  iou_voxel       the same match scored at voxel resolution (`losses.voxel_mask_iou`, exact)
  iou_clicked     true IoU against the object UNDER the click (0 for an off-object click)
  partners        how many true objects each make up >= 10% of the mask (>= 2 is a merge, the
                  diagnostic's own definition)
  passes          whether the labelling gates as configured would keep it

plus, per click, the id and size of the object under it (0 = background). True objects are the
connected components of the tile's labels, as `PromptTargets` defines them for training, so
`iou_best` is the number the head was trained to predict. From that, `summarize` answers the
questions the diagnostic could not: how well pred_iou tracks iou_best (calibration bins,
correlation); the precision of what the gates pass BEFORE any tile reconciliation; whether any
threshold on the head would give precise masks; what fraction of clicks the decoder can answer
with a >= 0.5 mask at all (the oracle ceiling); and what off-object clicks score.

The truth volumes trained the teacher, so every level here is an in-sample upper bound; the
comparison between stages is what this measures.
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

from pseudolabel import (  # noqa: E402
    GT_CONFIG,
    LABEL_AMG,
    amg_settings,
    load_model,
    plan_blocks,
    resolve_volume,
)

#: A true object "takes part" in a mask when it holds at least this share of it (the diagnostic's
#: `compare_labellings(part=0.1)`); two or more partners is a merge.
PART = 0.1

FIELDS = (
    "tile", "click", "candidate", "click_id", "click_size", "pred_iou", "stability", "area",
    "size_ok", "iou_best", "iou_voxel", "iou_clicked", "best_id", "best_size", "partners",
    "passes",
)


# ------------------------------------------------------------------------------------ scoring


def candidate_scores(
    predicted: torch.Tensor, hard: torch.Tensor, soft: torch.Tensor, clicked_row: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """`(N, cells)` bool candidates against `(M, cells)` truth masks -> per-candidate truth facts.

    `hard` is the pooled truth thresholded at 0.5 (what `mask_iou` scores against), `soft` the
    area-pooled fraction (what `voxel_mask_iou` scores against, exactly). `clicked_row` is `(N,)`
    the row of `hard` for the object under each candidate's click, or -1 for none. Every pairwise
    quantity is one matmul, as `amg.pairwise_intersection` does it. `best_row` is -1 for a
    candidate that meets no object.
    """
    n = predicted.shape[0]
    device = predicted.device
    pred = predicted.to(torch.float32)
    area = pred.sum(-1)
    if hard.shape[0] == 0:
        zeros = torch.zeros(n, device=device)
        return {
            "area": area, "iou_best": zeros, "iou_voxel": zeros, "iou_clicked": zeros,
            "best_row": torch.full((n,), -1, device=device, dtype=torch.long),
            "partners": torch.zeros(n, device=device, dtype=torch.long),
        }
    hard_f = hard.to(torch.float32)
    hard_size = hard_f.sum(-1)
    soft_size = soft.sum(-1)

    inter = pred @ hard_f.T                                    # (N, M)
    union = area[:, None] + hard_size[None, :] - inter
    iou = inter / union.clamp_min(1.0)
    iou_best, best_row = iou.max(dim=1)

    inter_v = pred @ soft.T
    union_v = area[:, None] + soft_size[None, :] - inter_v
    iou_v = inter_v / union_v.clamp_min(1e-6)
    # The voxel-resolution score of the SAME match, so the two columns describe one object.
    iou_voxel = iou_v.gather(1, best_row.unsqueeze(1)).squeeze(1)

    has_click = clicked_row >= 0
    iou_clicked = torch.zeros(n, device=device)
    if has_click.any():
        iou_clicked[has_click] = iou[has_click].gather(
            1, clicked_row[has_click].unsqueeze(1)
        ).squeeze(1)

    partners = ((inter / area.clamp_min(1.0)[:, None]) >= PART).sum(dim=1)
    best_row = torch.where(iou_best > 0, best_row, torch.full_like(best_row, -1))
    return {
        "area": area, "iou_best": iou_best, "iou_voxel": iou_voxel, "iou_clicked": iou_clicked,
        "best_row": best_row, "partners": partners,
    }


def _lookup(table: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
    """`table[row]` with -1 rows reading as 0."""
    out = torch.zeros_like(row)
    valid = row >= 0
    if valid.any():
        out[valid] = table[row[valid]]
    return out


# -------------------------------------------------------------------------------------- probe


def cmd_probe(args: argparse.Namespace) -> None:
    from miao.config import load_config

    from algorithms.affinity.targets import relabel_connected_cc3d
    from algorithms.promptable.amg import point_grid, stability_score
    from algorithms.promptable.targets import pooled_masks
    from layers.common.prompt import FOREGROUND
    from layers.common.rope import voxel_coords
    from predict import resolve_patch
    from prediction.grid import VolumeGrid

    config = load_config(args.gt_config)
    resolved = resolve_volume(config, args.volume)
    if resolved["volume"].label_key is None:
        raise SystemExit(f"{args.volume} has no label_key in {args.gt_config}; nothing to score")
    amg = dict(LABEL_AMG)
    amg.update(args.amg)
    algorithm, step, resolved_cfg = load_model(args.run_dir, args.step, amg)
    predictor = algorithm.volume_predictor()
    device = next(algorithm.parameters()).device
    stride = tuple(int(s) for s in algorithm.mask_stride)

    name = resolved["volume"].name
    patch = resolve_patch(resolved["config"], resolved_cfg, name, None)
    (index, box), = plan_blocks(resolved, args.block, 1)
    grid = VolumeGrid(resolved["config"], name, patch, box=box)

    truth_np = grid.read_ground_truth()
    # Dense ids: this corpus stores 64-bit segment ids, and everything below indexes by id.
    values, inverse = np.unique(truth_np, return_inverse=True)
    dense = np.arange(values.size, dtype=np.int64)
    dense[values <= 0] = 0
    truth = torch.from_numpy(dense[inverse].reshape(truth_np.shape)).to(device)
    del truth_np, inverse

    tiles = grid.tiles
    if args.max_tiles and len(tiles) > args.max_tiles:
        picks = np.linspace(0, len(tiles) - 1, args.max_tiles).round().astype(int)
        tiles = [tiles[i] for i in sorted(set(picks.tolist()))]
    handle = grid.image_handle()
    points_per_side = int(predictor.points_per_side)
    batch = int(predictor.points_per_batch)
    print(f"{name}: block {index} {box} -> {len(grid.tiles)} tiles, probing {len(tiles)}; "
          f"{points_per_side}^3 clicks per tile, mask stride {stride}", flush=True)

    columns: dict[str, list[np.ndarray]] = {field: [] for field in FIELDS}
    started = time.perf_counter()
    bounds: tuple[int, float] | None = None
    for tile_index, (native, out) in enumerate(tiles):
        volume = torch.from_numpy(grid.read_image(handle, native)[None, None]).to(device)
        extent = tuple(int(v) for v in volume.shape[-3:])
        min_cells, max_cells = predictor._bounds(extent)
        bounds = (int(min_cells), float(max_cells))
        window = tuple(slice(o, o + p) for o, p in zip(out, patch, strict=True))
        # Connected components of the tile, as training defines an object (`PromptTargets`).
        truth_tile = relabel_connected_cc3d(truth[window])
        ids = torch.unique(truth_tile)
        ids = ids[ids > 0]
        n_cells = int(np.prod([e // s for e, s in zip(extent, stride, strict=True)]))
        if ids.numel():
            soft = pooled_masks(truth_tile[None], ids[None], stride)[0].reshape(ids.numel(), -1)
        else:
            soft = torch.zeros((0, n_cells), device=device)
        hard = soft > 0.5
        hard_size = hard.sum(-1)

        points = point_grid(extent, points_per_side).to(device)
        voxel = points.round().long().clamp_min(0)
        voxel = torch.minimum(voxel, torch.tensor(extent, device=device) - 1)
        click_id = truth_tile[tuple(voxel.T)]
        clicked_row = torch.full_like(click_id, -1)
        if ids.numel():
            position = torch.searchsorted(ids, click_id).clamp_max(ids.numel() - 1)
            on_object = (click_id > 0) & (ids[position] == click_id)
            clicked_row = torch.where(on_object, position, clicked_row)
        click_size = _lookup(hard_size, clicked_row)
        coords = voxel_coords(points, extent).unsqueeze(1)
        labels = torch.full((points.shape[0], 1), FOREGROUND, device=device)

        kept = 0
        with torch.autocast(device.type, dtype=torch.bfloat16):
            image, image_coords, token_grid = algorithm.encode(volume)
            for start in range(0, points.shape[0], batch):
                stop = min(start + batch, points.shape[0])
                logits, ious = algorithm.decode_points(
                    image.expand(stop - start, -1, -1), image_coords, token_grid,
                    coords[start:stop], labels[start:stop], multimask=True,
                )
                logits, ious = logits.float(), ious.float()
                b, k = ious.shape
                flat = logits.flatten(0, 1)                      # (b*k, *cells)
                passes = predictor._passes(flat, ious.flatten(), extent)
                stab = stability_score(flat, predictor.stability_delta)
                rows = torch.arange(start, stop, device=device).repeat_interleave(k)
                scored = candidate_scores(flat.flatten(1) > 0, hard, soft, clicked_row[rows])
                best_row = scored["best_row"]
                record = {
                    "tile": torch.full_like(rows, tile_index),
                    "click": rows,
                    "candidate": torch.arange(k, device=device).repeat(b),
                    "click_id": click_id[rows],
                    "click_size": click_size[rows],
                    "pred_iou": ious.flatten(),
                    "stability": stab,
                    "area": scored["area"],
                    "size_ok": (scored["area"] >= min_cells) & (scored["area"] <= max_cells),
                    "iou_best": scored["iou_best"],
                    "iou_voxel": scored["iou_voxel"],
                    "iou_clicked": scored["iou_clicked"],
                    "best_id": _lookup(ids, best_row) if ids.numel() else torch.zeros_like(rows),
                    "best_size": _lookup(hard_size, best_row),
                    "partners": scored["partners"],
                    "passes": passes,
                }
                for field in FIELDS:
                    columns[field].append(record[field].detach().cpu().numpy())
                kept += int(passes.sum())
        del soft, hard, image
        on = int((clicked_row >= 0).sum())
        print(f"  tile {tile_index + 1}/{len(tiles)}: {ids.numel()} true objects, "
              f"{on}/{points.shape[0]} clicks on an object, {kept} candidates pass the gates  "
              f"({time.perf_counter() - started:.0f} s)", flush=True)

    arrays = {field: np.concatenate(columns[field]) for field in FIELDS}
    meta = {
        "volume": name, "gt_config": str(args.gt_config), "run": args.run_dir.name,
        "step": step, "lattice_block": args.block, "block_index": index, "box_storage": box,
        "tiles_in_block": len(grid.tiles), "tiles_probed": len(tiles), "patch": patch,
        "mask_stride": list(stride), "points_per_side": points_per_side,
        "size_bounds_cells": bounds, "amg": amg_settings(algorithm, amg), "part": PART,
        "regenerate": " ".join(["python", "experiments/sam_lmd_v1/calibration_probe.py", "probe",
                                str(args.run_dir), "--step", str(step), "--volume", name,
                                "--out", str(args.out), "--block", str(args.block),
                                "--max-tiles", str(args.max_tiles)]),
        "seconds": round(time.perf_counter() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, meta=json.dumps(meta), **arrays)
    print(f"wrote {args.out}: {arrays['pred_iou'].size} candidates", flush=True)


# ---------------------------------------------------------------------------------- summarize


def load_records(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path) as data:
        meta = json.loads(str(data["meta"]))
        arrays = {field: data[field] for field in FIELDS}
    return meta, arrays


def gate(arrays: dict[str, np.ndarray], pred_iou: float, stability: float) -> np.ndarray:
    """The labelling gates re-applied at chosen thresholds, size bounds as recorded."""
    return (
        (arrays["pred_iou"] >= pred_iou) & (arrays["stability"] >= stability)
        & arrays["size_ok"].astype(bool)
    )


def _rate(mask: np.ndarray, condition: np.ndarray) -> float:
    return float(condition[mask].mean()) if mask.any() else float("nan")


def summarize_records(
    arrays: dict[str, np.ndarray], pred_iou: float, stability: float, iou_thresh: float = 0.5,
) -> dict[str, Any]:
    on_object = arrays["click_id"] > 0
    keep = gate(arrays, pred_iou, stability)
    good = arrays["iou_best"] >= iou_thresh
    merged = arrays["partners"] >= 2

    # One row per click: the head's own pick and the oracle pick among the K candidates.
    order = np.lexsort((arrays["candidate"], arrays["click"], arrays["tile"]))
    k = int(arrays["candidate"].max()) + 1
    per_click = {field: arrays[field][order].reshape(-1, k)
                 for field in ("pred_iou", "iou_best", "click_id")}
    n_clicks = per_click["pred_iou"].shape[0]
    top = per_click["pred_iou"].argmax(axis=1)
    top_iou = per_click["iou_best"][np.arange(n_clicks), top]
    oracle_iou = per_click["iou_best"].max(axis=1)
    click_on = per_click["click_id"][:, 0] > 0
    max_pred = per_click["pred_iou"].max(axis=1)

    bins = np.linspace(0.0, 1.0, 11)
    which = np.clip(np.digitize(arrays["pred_iou"], bins) - 1, 0, 9)
    calibration = []
    for b in range(10):
        inside = which == b
        calibration.append({
            "pred_iou_bin": [round(float(bins[b]), 1), round(float(bins[b + 1]), 1)],
            "candidates": int(inside.sum()),
            "mean_true_iou": _rate(inside, arrays["iou_best"]),
            "precision": _rate(inside, good),
            "merge_rate": _rate(inside, merged),
        })
    thresholds = []
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        k_t = gate(arrays, t, stability)
        thresholds.append({
            "pred_iou": t, "kept": int(k_t.sum()), "precision": _rate(k_t, good),
            "mean_true_iou": _rate(k_t, arrays["iou_best"]), "merge_rate": _rate(k_t, merged),
            "from_offobject_clicks": _rate(k_t, ~on_object),
        })

    def corr(select: np.ndarray) -> float:
        if select.sum() < 3:
            return float("nan")
        return float(np.corrcoef(arrays["pred_iou"][select], arrays["iou_best"][select])[0, 1])

    drift = on_object & (arrays["iou_best"] > 0) & (arrays["best_id"] != arrays["click_id"])
    return {
        "clicks": int(n_clicks),
        "clicks_on_object": float(click_on.mean()),
        "candidates": int(arrays["pred_iou"].size),
        "gate": {"pred_iou": pred_iou, "stability": stability},
        "passing": {
            "candidates": int(keep.sum()),
            "per_click": float(keep.sum() / max(n_clicks, 1)),
            "precision": _rate(keep, good),
            "precision_voxel": _rate(keep, arrays["iou_voxel"] >= iou_thresh),
            "mean_true_iou": _rate(keep, arrays["iou_best"]),
            "mean_pred_iou": _rate(keep, arrays["pred_iou"]),
            "merge_rate": _rate(keep, merged),
            "from_offobject_clicks": _rate(keep, ~on_object),
            "best_is_not_clicked_object": _rate(keep & on_object, drift),
        },
        "all_candidates": {
            "precision": float(good.mean()),
            "mean_true_iou": float(arrays["iou_best"].mean()),
            "merge_rate": float(merged.mean()),
            "head_mae": float(np.abs(arrays["pred_iou"] - arrays["iou_best"]).mean()),
            "corr_pred_true": corr(np.ones_like(on_object)),
            "corr_pred_true_on_object": corr(on_object),
        },
        "per_click": {
            "head_top_mean_iou_on_object": _rate(click_on, top_iou),
            "oracle_mean_iou_on_object": _rate(click_on, oracle_iou),
            "head_top_at_least_0.5_on_object": _rate(click_on, top_iou >= iou_thresh),
            "oracle_at_least_0.5_on_object": _rate(click_on, oracle_iou >= iou_thresh),
            "offobject_max_pred_iou_mean": _rate(~click_on, max_pred),
            "offobject_max_pred_iou_at_least_gate": _rate(~click_on, max_pred >= pred_iou),
            "on_object_max_pred_iou_at_least_gate": _rate(click_on, max_pred >= pred_iou),
        },
        "calibration": calibration,
        "thresholds": thresholds,
    }


def _fmt(value: Any, width: int = 6, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:{width}.{digits}f}" if np.isfinite(value) else " " * (width - 3) + "nan"
    return f"{value:{width}d}"


def cmd_summarize(args: argparse.Namespace) -> None:
    paths = sorted(p for p in args.directory.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no probe records under {args.directory}")
    reports: dict[str, Any] = {}
    pooled: dict[str, list[np.ndarray]] = {field: [] for field in FIELDS}
    offset = 0
    for path in paths:
        meta, arrays = load_records(path)
        reports[meta["volume"]] = {
            "meta": meta, **summarize_records(arrays, args.pred_iou, args.stability)}
        # Tiles renumbered so clicks stay distinct across volumes in the pooled view.
        shifted = dict(arrays)
        shifted["tile"] = arrays["tile"] + offset
        offset += int(arrays["tile"].max()) + 1
        for field in FIELDS:
            pooled[field].append(shifted[field])
    merged = {field: np.concatenate(pooled[field]) for field in FIELDS}
    reports["POOLED"] = summarize_records(merged, args.pred_iou, args.stability)

    first = next(iter(reports.values()))["meta"]
    print(f"teacher {first['run']} step {first['step']}; gates pred_iou >= {args.pred_iou}, "
          f"stability >= {args.stability}; precision = true IoU >= 0.5 on the mask grid\n")
    print(f"{'volume':26s} {'clicks':>7s} {'onObj':>6s} {'pass/clk':>8s} {'prec':>6s} "
          f"{'precVox':>7s} {'trueIoU':>7s} {'predIoU':>7s} {'merge':>6s} {'offObj':>6s} "
          f"{'drift':>6s} | {'corr':>6s} {'MAE':>6s} | {'top>=.5':>7s} {'orc>=.5':>7s} "
          f"{'offPass':>7s}")
    for volume, report in reports.items():
        p, a, c = report["passing"], report["all_candidates"], report["per_click"]
        print(f"{volume[:26]:26s} {report['clicks']:7d} {_fmt(report['clicks_on_object'])} "
              f"{_fmt(p['per_click'], 8, 2)} {_fmt(p['precision'])} "
              f"{_fmt(p['precision_voxel'], 7)} {_fmt(p['mean_true_iou'], 7)} "
              f"{_fmt(p['mean_pred_iou'], 7)} {_fmt(p['merge_rate'])} "
              f"{_fmt(p['from_offobject_clicks'])} {_fmt(p['best_is_not_clicked_object'])} | "
              f"{_fmt(a['corr_pred_true'])} {_fmt(a['head_mae'])} | "
              f"{_fmt(c['head_top_at_least_0.5_on_object'], 7)} "
              f"{_fmt(c['oracle_at_least_0.5_on_object'], 7)} "
              f"{_fmt(c['offobject_max_pred_iou_at_least_gate'], 7)}")
    print("\ncalibration (POOLED): pred_iou bin -> candidates, mean true IoU, precision, "
          "merge rate")
    for row in reports["POOLED"]["calibration"]:
        lo, hi = row["pred_iou_bin"]
        print(f"  [{lo:.1f}, {hi:.1f})  {row['candidates']:8d}  {_fmt(row['mean_true_iou'])}  "
              f"{_fmt(row['precision'])}  {_fmt(row['merge_rate'])}")
    print("\nhead threshold sweep (POOLED, stability gate fixed): kept, precision, mean true IoU, "
          "merge rate, share from off-object clicks")
    for row in reports["POOLED"]["thresholds"]:
        print(f"  pred_iou >= {row['pred_iou']:.2f}  {row['kept']:8d}  {_fmt(row['precision'])}  "
              f"{_fmt(row['mean_true_iou'])}  {_fmt(row['merge_rate'])}  "
              f"{_fmt(row['from_offobject_clicks'])}")
    out = args.out or (args.directory / "summary.json")
    out.write_text(json.dumps(reports, indent=2))
    print(f"\nwrote {out}")


# ------------------------------------------------------------------------------------------ CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="score every grid candidate on one GT block")
    probe.add_argument("run_dir", type=Path)
    probe.add_argument("--step", type=int)
    probe.add_argument("--volume", required=True)
    probe.add_argument("--out", type=Path, required=True)
    probe.add_argument("--gt-config", type=Path, default=GT_CONFIG)
    probe.add_argument("--block", type=int, default=512, help="lattice block, as the diagnostic")
    probe.add_argument("--max-tiles", type=int, default=0, help="0 = every tile of the block")
    probe.add_argument("--points-per-side", type=int)
    probe.add_argument("--points-per-batch", type=int)
    probe.set_defaults(func=cmd_probe)

    summarize = sub.add_parser("summarize", help="pool a directory of probe records")
    summarize.add_argument("directory", type=Path)
    summarize.add_argument("--pred-iou", type=float, default=LABEL_AMG["pred_iou_thresh"])
    summarize.add_argument("--stability", type=float, default=LABEL_AMG["stability_thresh"])
    summarize.add_argument("--out", type=Path)
    summarize.set_defaults(func=cmd_summarize)

    args = parser.parse_args()
    if args.command == "probe":
        args.amg = {key: getattr(args, key) for key in ("points_per_side", "points_per_batch")
                    if getattr(args, key) is not None}
    args.func(args)


if __name__ == "__main__":
    main()
