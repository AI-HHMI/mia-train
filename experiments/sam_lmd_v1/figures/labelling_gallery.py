"""A teacher's pseudo-labels beside the ground truth, on the blocks the tables scored.

    python experiments/sam_lmd_v1/figures/labelling_gallery.py label <run_dir> --step N \\
        --volume V --out <dir> [--gt-config ...] [--block 1024] [--min-mask-voxels 4096] \\
        [--min-truth-voxels 4096]
    python experiments/sam_lmd_v1/figures/labelling_gallery.py render <dir> [--volume V ...]

`label` runs the labeller exactly as `pseudolabel.py diagnose` does (same block partition, same
generator settings, `tile_merge = consensus`) on the first block of the volume, and saves what the
diagnostic throws away: the raw image, the ground truth, the assembled labelling of the whole block
and, separately, the labelling of the block's central window alone (the model's masks before any
window is reconciled with another). Everything is saved on the training lattice subsampled by two
(a 1024^3 block at 4 nm becomes 512^3 at 8 nm; the 4 nm lattice is a 2x upsampling of 8 nm data,
so nothing of the image is lost and the 16 nm mask cells stay 2 voxels wide). `<volume>.json`
holds the block's scores computed on the FULL lattice, so they can be checked against the table.
`<volume>_windows.npz` keeps every window's own labelling on the mask-cell grid (`origins` and
`shape` in cells, `cell_nm`, arrays `w0..wN`), so a different assembly rule can be tried on them
with `amg.consensus_labelling` and no model pass.

`render` draws, for each volume, three sections through the block: the image | the ground truth |
the assembled pseudo-labels, each piece coloured like the true object holding most of its voxels
so a neurite the model got right has one colour in both panels and a split neurite is that colour
cut by black lines | an error map: merge (>= 2 true objects each >= 10% of the piece) red, matched
(IoU >= 0.5) green, partial (one object, but under half of it) orange, spill (mostly background,
or impure) purple, true foreground the model left unlabelled dark grey. A second figure shows the
central window alone in the same layout. `gallery_overview.png` puts one section per volume on
one page.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE.parent))

SHORT = {"hemibrain_ellipsoid_body": "hemibrain", "liconn_mouse_dg": "liconn",
         "kasthuri15_ac3": "kasthuri", "zebrafish_fish2_quadcube1": "zebrafish"}
VOLUMES = ["hemibrain_ellipsoid_body", "kasthuri15_ac3", "liconn_mouse_dg",
           "zebrafish_fish2_quadcube1"]
SUB = 2               # saved lattice = training lattice / SUB
PART = 0.1            # `compare_labellings`' share for a merge partner
IOU = 0.5
PIECE_CLASSES = ("matched", "partial", "merge", "spill")

# Error-map colours (dataviz palette): green matched, red merge, orange partial, purple spill.
CLASS_COLOURS = {"matched": "#3f9a4c", "merge": "#d8342c", "partial": "#eb8a2f",
                 "spill": "#8e5bb5", "unlabelled": "#4a4a4a"}
CLASS_ORDER = ["matched", "partial", "merge", "spill", "unlabelled"]
LEGEND = {"matched": "matched a true object (IoU >= 0.5)", "partial": "part of one object (< half)",
          "merge": "merge (>= 2 objects)", "spill": "mostly background, or impure",
          "unlabelled": "true object, no label"}
TEXT = "#0b0b0b"
SURFACE = "#fcfcfb"


# ------------------------------------------------------------------------------------------ label


def paste_image(grid, handle) -> np.ndarray:
    """The block's image on the output lattice, read tile by tile through the grid's own reader."""
    image = np.zeros(grid.output_shape, dtype=np.float32)
    for native, out in grid.tiles:
        tile = grid.read_image(handle, native)
        window = tuple(slice(o, o + p) for o, p in zip(out, grid.patch, strict=True))
        image[window] = tile
    return image


def central_window(grid) -> tuple[list[list[int]], tuple[int, ...]]:
    """The native box of the tile nearest the block's centre, and its output origin."""
    centre = np.array(grid.output_shape) / 2
    half = np.array(grid.patch) / 2
    native, out = min(grid.tiles, key=lambda t: np.abs(np.array(t[1]) + half - centre).sum())
    box = [[int(o), int(o + r)] for o, r in zip(native, grid.read, strict=True)]
    return box, tuple(int(v) for v in out)


def relabel(labels: np.ndarray) -> np.ndarray:
    """Consecutive int32 ids (0 stays 0, negatives -> 0): the corpus stores 64-bit ids."""
    values, inverse = np.unique(labels, return_inverse=True)
    inverse = inverse.reshape(labels.shape).astype(np.int32)
    keep = values > 0
    lut = np.zeros(values.size, dtype=np.int32)
    lut[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return lut[inverse]


def subsample(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a[::SUB, ::SUB, ::SUB])


def parse_literal(text: str) -> Any:
    """`--amg key=value`: true/false, an int, a float, else the bare string."""
    if text in ("true", "false"):
        return text == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def cmd_label(args: argparse.Namespace) -> None:
    from miao.config import load_config
    from pseudolabel import (
        LABEL_AMG,
        compare_labellings,
        load_model,
        plan_blocks,
        resolve_volume,
        run_block,
    )

    from predict import resolve_patch
    from prediction.dense import select_predictor
    from prediction.grid import VolumeGrid

    args.out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.gt_config)
    resolved = resolve_volume(config, args.volume)
    amg = dict(LABEL_AMG)
    amg["min_mask_voxels"] = args.min_mask_voxels
    if args.pred_iou_thresh is not None:
        amg["pred_iou_thresh"] = args.pred_iou_thresh
    for item in args.amg or []:
        key, _, literal = item.partition("=")
        amg[key] = parse_literal(literal)
    print(f"generator settings: {amg}", flush=True)
    algorithm, step, resolved_cfg = load_model(args.run_dir, args.step, amg)
    (index, box), = plan_blocks(resolved, args.block, 1)
    print(f"{args.volume}: block {index} {box} of run {args.run_dir.name} step {step}", flush=True)

    patch = resolve_patch(resolved["config"], resolved_cfg, args.volume, None)
    grid = VolumeGrid(resolved["config"], args.volume, patch, box=box)
    handle = grid.image_handle()
    started = time.perf_counter()
    image = paste_image(grid, handle)
    truth = grid.read_ground_truth()
    print(f"  image + truth read in {time.perf_counter() - started:.0f} s; "
          f"output {grid.output_shape}, {len(grid.tiles)} windows", flush=True)

    # The central window alone: the model's masks before any reconciliation between windows.
    window_box, window_origin = central_window(grid)
    single = VolumeGrid(resolved["config"], args.volume, patch, box=window_box)
    if len(single.tiles) != 1:
        raise SystemExit(f"central window box {window_box} gave {len(single.tiles)} tiles, not 1")
    device = next(algorithm.parameters()).device
    started = time.perf_counter()
    window_pred = select_predictor(algorithm).run(single, device).array
    window_seconds = time.perf_counter() - started
    slices = tuple(slice(o, o + p) for o, p in zip(window_origin, patch, strict=True))
    window_scores = compare_labellings(
        window_pred, truth[slices], min_truth_voxels=args.min_truth_voxels
    )
    print(f"  central window ({window_seconds:.0f} s): precision {window_scores['precision']:.3f} "
          f"recall {window_scores['recall']:.3f}", flush=True)

    base = args.out / f"{args.volume}_base.npz"
    meta = {
        "volume": args.volume, "run": args.run_dir.name, "step": step, "block_index": index,
        "block_box_storage": box, "native_box_storage": grid.native_box(),
        "storage_axes": resolved["storage_axes"], "lattice_nm": resolved["lattice_nm"],
        "saved_nm": [v * SUB for v in resolved["lattice_nm"]],
        "output_shape": list(grid.output_shape), "windows": len(grid.tiles),
        "patch": list(patch), "amg": amg, "window_box_storage": window_box,
        "window_origin_output": list(window_origin), "window_seconds": round(window_seconds, 1),
        "window_scores": window_scores,
    }
    np.savez_compressed(
        base,
        image=(np.clip(subsample(image), 0, 1) * 255).astype(np.uint8),
        truth=subsample(relabel(truth)),
        window=subsample(relabel(window_pred)),
        window_origin=np.array(window_origin) // SUB,
        window_shape=np.array(patch) // SUB,
    )
    (args.out / f"{args.volume}.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {base}", flush=True)
    del window_pred
    render_volume(args.out, args.volume)          # the window figure, before the long pass

    # Keep every window's own labelling: the assembly rule can then be re-run on them offline
    # (`amg.consensus_labelling(tiles, shape, ...)`) in minutes instead of another model pass.
    # The generator calls `consensus_labelling` by module-global name, so a wrapper on the module
    # attribute sees the same arguments. The maps are on the MASK-CELL grid (origins in cells,
    # `shape` = the block in cells), which is what the rule works on; they are saved as they are.
    # (Dumps written before 2026-09-15 11:00 were subsampled by two, i.e. lossy; regenerate.)
    import algorithms.promptable.amg as amg_module

    captured: dict[str, Any] = {}
    real_consensus = amg_module.consensus_labelling

    def spy(tiles, shape, *rest, **kwargs):
        captured["origins"] = np.array([[int(v) for v in origin] for origin, _ in tiles])
        captured["maps"] = [np.asarray(local.cpu() if hasattr(local, "cpu") else local)
                            .astype(np.int32) for _, local in tiles]
        captured["shape"] = [int(v) for v in shape]
        return real_consensus(tiles, shape, *rest, **kwargs)

    amg_module.consensus_labelling = spy
    started = time.perf_counter()
    _, prediction, seconds = run_block(algorithm, resolved_cfg, resolved, box)
    amg_module.consensus_labelling = real_consensus
    if captured:
        np.savez_compressed(
            args.out / f"{args.volume}_windows.npz", origins=captured["origins"],
            shape=np.array(captured["shape"]),
            cell_nm=np.array([n * st for n, st in zip(resolved["lattice_nm"], algorithm.mask_stride,
                                                        strict=True)]),
            **{f"w{i}": m for i, m in enumerate(captured["maps"])},
        )
        print(f"  kept {len(captured['maps'])} window labellings", flush=True)
    scores = compare_labellings(prediction.array, truth, min_truth_voxels=args.min_truth_voxels)
    print(f"  assembled block ({seconds:.0f} s): {scores['pseudo_instances']} pieces vs "
          f"{scores['truth_instances']} objects  precision {scores['precision']:.3f}  "
          f"recall {scores['recall']:.3f}  merges {scores['merges']}  "
          f"fragments {scores['fragments']}", flush=True)
    assembled = args.out / f"{args.volume}_assembled.npz"
    np.savez_compressed(assembled, assembled=subsample(relabel(prediction.array)))
    meta.update(assembled_seconds=round(seconds, 1), assembled_scores=scores)
    (args.out / f"{args.volume}.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {assembled} (total {time.perf_counter() - started:.0f} s)", flush=True)
    render_volume(args.out, args.volume)


# ----------------------------------------------------------------------------------------- render


def classify(pred: np.ndarray, truth: np.ndarray, min_truth: int
             ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Per piece: its majority true object, and its error class (see module docstring).

    Returns (majority truth id per pseudo id, class index per pseudo id, counts per class). Class
    indices follow CLASS_ORDER; entry 0 (pseudo id 0) is meaningless and never drawn. A piece is
    a merge before it is anything else; `compare_labellings` counts a merge whose IoU with one
    partner reaches 0.5 as both, but for a picture one class per piece is what reads.
    """
    p = pred.ravel().astype(np.int64)
    t = truth.ravel().astype(np.int64)
    n_truth = int(t.max()) + 1
    claimed = p > 0
    pseudo_sizes = np.bincount(p[claimed], minlength=int(p.max()) + 1)
    truth_sizes = np.bincount(t, minlength=n_truth)
    pairs, counts = np.unique(p[claimed] * n_truth + t[claimed], return_counts=True)
    pid, tid = pairs // n_truth, pairs % n_truth

    majority = np.zeros(pseudo_sizes.size, dtype=np.int64)
    best_count = np.zeros(pseudo_sizes.size, dtype=np.int64)
    for i in np.argsort(counts):                    # last write wins -> the largest overlap
        majority[pid[i]] = tid[i]
        best_count[pid[i]] = counts[i]

    on_object = tid > 0
    inter = counts[on_object].astype(np.float64)
    union = pseudo_sizes[pid[on_object]] + truth_sizes[tid[on_object]] - inter
    best_iou = np.zeros(pseudo_sizes.size)
    np.maximum.at(best_iou, pid[on_object], inter / union)
    significant = inter / pseudo_sizes[pid[on_object]] >= PART
    partners = np.bincount(pid[on_object][significant], minlength=pseudo_sizes.size)
    # A piece is also a merge when it holds most of two or more counted true objects, however
    # small each is next to the piece: the 10%-of-the-piece rule alone lets a piece that swallowed
    # fifty objects pass as "impure" because no single object is a tenth of it.
    counted = truth_sizes[tid[on_object]] >= min_truth
    swallowed = (inter / truth_sizes[tid[on_object]] >= 0.5) & counted
    holds = np.bincount(pid[on_object][swallowed], minlength=pseudo_sizes.size)
    purity = best_count / np.maximum(pseudo_sizes, 1)

    klass = np.full(pseudo_sizes.size, CLASS_ORDER.index("spill"))
    klass[(purity >= 0.9) & (majority > 0)] = CLASS_ORDER.index("partial")
    klass[best_iou >= IOU] = CLASS_ORDER.index("matched")
    klass[(partners >= 2) | (holds >= 2)] = CLASS_ORDER.index("merge")
    ids = np.flatnonzero(pseudo_sizes)
    ids = ids[ids > 0]
    tally = {name: int((klass[ids] == CLASS_ORDER.index(name)).sum()) for name in PIECE_CLASSES}
    tally["pieces"] = int(ids.size)
    # Recall as the table counts it: true objects of >= min_truth voxels with a piece at IoU >= 0.5.
    best_truth = np.zeros(n_truth)
    np.maximum.at(best_truth, tid[on_object], inter / union)
    truth_ids = np.flatnonzero(truth_sizes >= min_truth)
    truth_ids = truth_ids[truth_ids > 0]
    tally["truth_objects"] = int(truth_ids.size)
    tally["truth_found"] = int((best_truth[truth_ids] >= IOU).sum())
    return majority, klass, tally


def palette(n: int, seed: int = 7) -> np.ndarray:
    """n distinct, mid-bright colours; id 0 (background) is white."""
    from matplotlib.colors import hsv_to_rgb

    rng = np.random.default_rng(seed)
    hsv = np.stack(
        [rng.random(n), 0.45 + 0.5 * rng.random(n), 0.55 + 0.4 * rng.random(n)], axis=1
    )
    rgb = hsv_to_rgb(hsv)
    rgb[0] = 1.0
    return rgb


def boundaries(section2d: np.ndarray) -> np.ndarray:
    """Pixels where the label changes to a neighbour (both sides drawn, 2 px wide)."""
    edge = np.zeros(section2d.shape, dtype=bool)
    rows = section2d[:-1, :] != section2d[1:, :]
    cols = section2d[:, :-1] != section2d[:, 1:]
    edge[:-1, :] |= rows
    edge[1:, :] |= rows
    edge[:, :-1] |= cols
    edge[:, 1:] |= cols
    return edge


def section(volume: np.ndarray, axes: str, z: int) -> np.ndarray:
    """Section z perpendicular to the sectioning axis, as (rows=y, cols=x)."""
    plane = np.take(volume, z, axis=axes.index("z"))
    return plane if axes.replace("z", "") == "yx" else plane.T


def hex_rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def draw_row(axes_row, image2d, truth2d, pred2d, truth_rgb, majority, klass, prefix: str) -> None:
    """One section: image | truth | pieces coloured by majority truth | error map."""
    grey = np.repeat(image2d[..., None].astype(np.float32) / 255, 3, axis=-1)
    axes_row[0].imshow(grey, interpolation="nearest")
    axes_row[0].set_title(f"{prefix}image")

    truth_img = truth_rgb[truth2d]
    truth_img[boundaries(truth2d) & (truth2d > 0)] *= 0.35
    axes_row[1].imshow(truth_img, interpolation="nearest")
    axes_row[1].set_title("ground truth")

    pred_img = truth_rgb[majority[pred2d]]
    pred_img[(pred2d > 0) & (majority[pred2d] == 0)] = 0.82     # a piece on true background
    pred_img[pred2d == 0] = 1.0
    pred_img[boundaries(pred2d) & (pred2d > 0)] = 0.0
    axes_row[2].imshow(pred_img, interpolation="nearest")
    axes_row[2].set_title("model, coloured by the true object under each piece")

    error = np.ones(pred2d.shape + (3,), dtype=np.float32)
    error[(pred2d == 0) & (truth2d > 0)] = hex_rgb(CLASS_COLOURS["unlabelled"])
    for name in PIECE_CLASSES:
        chosen = (pred2d > 0) & (klass[pred2d] == CLASS_ORDER.index(name))
        error[chosen] = hex_rgb(CLASS_COLOURS[name])
    error = 0.55 * error + 0.45 * grey
    error[boundaries(pred2d) & (pred2d > 0)] *= 0.5
    axes_row[3].imshow(error, interpolation="nearest")
    axes_row[3].set_title("what each piece is")
    for ax in axes_row:
        ax.set_xticks([])
        ax.set_yticks([])


def legend_handles():
    from matplotlib.patches import Patch

    return [Patch(facecolor=CLASS_COLOURS[k], label=LEGEND[k]) for k in CLASS_ORDER]


def tally_text(tally: dict[str, int]) -> str:
    return (f"{tally['matched']} matched, {tally['partial']} partial, {tally['merge']} merges, "
            f"{tally['spill']} spills of {tally['pieces']} pieces; "
            f"{tally['truth_found']} of {tally['truth_objects']} true objects found")


def figure(image, truth, pred, axes: str, nm: float, truth_rgb, min_truth: int, title: str,
           out: Path) -> None:
    import matplotlib.pyplot as plt

    majority, klass, tally = classify(pred, truth, min_truth)
    depth = pred.shape[axes.index("z")]
    fig, axs = plt.subplots(3, 4, figsize=(14, 10.8))
    for row, frac in zip(axs, (0.3, 0.5, 0.7), strict=True):
        z = int(depth * frac)
        draw_row(row, section(image, axes, z), section(truth, axes, z), section(pred, axes, z),
                 truth_rgb, majority, klass, f"section {z * nm / 1000:.2f} um:  ")
    fig.suptitle(f"{title}  --  {tally_text(tally)}", fontsize=10)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "text.color": TEXT, "axes.labelcolor": TEXT,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "axes.titlesize": 9})


def cmd_oracle(args: argparse.Namespace) -> None:
    """The same pictures for a PERFECT model: every window's masks are the ground truth's own
    connected components, pooled to the mask grid, then glued by the labeller's consensus rule.
    What survives is the cost of the mask grid and of the gluing alone; whatever the real arms
    lose beyond it is within-window prediction error."""
    import torch
    from miao.config import load_config
    from pseudolabel import (
        LABEL_AMG,
        assemble_oracle,
        compare_labellings,
        oracle_tile_masks,
        plan_blocks,
        resolve_volume,
    )

    from algorithms.promptable.amg import tile_labelling, upsample_cells
    from prediction.grid import VolumeGrid

    args.out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.gt_config)
    resolved = resolve_volume(config, args.volume)
    if resolved["volume"].label_key is None:
        raise SystemExit(f"{args.volume} has no label_key in {args.gt_config}; nothing to draw")
    patch = [args.patch] * 3
    if sorted(int(v) for v in config.patch_size) != patch:
        raise SystemExit(
            f"{args.gt_config} has patch_size {list(config.patch_size)} but the window is {args.patch}: "
            "VolumeGrid reads each tile at the config's patch, so the lattice would silently change "
            "scale. Use the generated crop copy (data/lmd_finetune_singlescale_crop<N>.yaml)."
        )
    stride = (args.mask_stride,) * 3
    amg = dict(LABEL_AMG)
    amg.update(oracle=True, pred_iou_thresh=1.0, stability_thresh=1.0,
               min_mask_voxels=args.min_mask_voxels, min_support=args.min_support,
               agree_thresh=args.agree_thresh, mask_stride=args.mask_stride)
    (index, box), = plan_blocks(resolved, args.block, 1)
    grid = VolumeGrid(resolved["config"], args.volume, patch, box=box)
    print(f"{args.volume}: block {index} {box}, oracle at window {args.patch}, "
          f"{len(grid.tiles)} windows, output {grid.output_shape}", flush=True)
    handle = grid.image_handle()
    started = time.perf_counter()
    image = paste_image(grid, handle)
    truth_np = grid.read_ground_truth()
    values, inverse = np.unique(truth_np, return_inverse=True)
    dense = np.arange(values.size, dtype=np.int64)
    dense[values <= 0] = 0
    truth = torch.from_numpy(dense[inverse].reshape(truth_np.shape))
    del truth_np, inverse
    print(f"  image + truth read in {time.perf_counter() - started:.0f} s", flush=True)

    # The central window alone: perfect masks on the mask grid, painted as one window is.
    window_box, window_origin = central_window(grid)
    slices = tuple(slice(o, o + p) for o, p in zip(window_origin, patch, strict=True))
    started = time.perf_counter()
    masks, scores = oracle_tile_masks(truth[slices], stride, args.min_mask_voxels)
    window_pred = upsample_cells(tile_labelling(masks, scores).numpy(), stride)
    window_seconds = time.perf_counter() - started
    window_scores = compare_labellings(
        window_pred, truth[slices].numpy(), min_truth_voxels=args.min_truth_voxels
    )
    print(f"  central window: {masks.shape[0]} perfect masks; precision "
          f"{window_scores['precision']:.3f} recall {window_scores['recall']:.3f}", flush=True)

    meta = {
        "volume": args.volume, "run": f"oracle__w{args.patch}_r0", "step": 0,
        "block_index": index, "block_box_storage": box, "native_box_storage": grid.native_box(),
        "storage_axes": resolved["storage_axes"], "lattice_nm": resolved["lattice_nm"],
        "saved_nm": [v * SUB for v in resolved["lattice_nm"]],
        "output_shape": list(grid.output_shape), "windows": len(grid.tiles),
        "patch": list(patch), "amg": amg, "window_box_storage": window_box,
        "window_origin_output": list(window_origin), "window_seconds": round(window_seconds, 1),
        "window_scores": window_scores,
    }
    truth_saved = subsample(relabel(truth.numpy()))
    np.savez_compressed(
        args.out / f"{args.volume}_base.npz",
        image=(np.clip(subsample(image), 0, 1) * 255).astype(np.uint8),
        truth=truth_saved,
        window=subsample(relabel(window_pred)),
        window_origin=np.array(window_origin) // SUB,
        window_shape=np.array(patch) // SUB,
    )
    (args.out / f"{args.volume}.json").write_text(json.dumps(meta, indent=2))
    del window_pred, image

    started = time.perf_counter()
    stats: dict[str, Any] = {}
    labels, instances = assemble_oracle(
        grid.tiles, grid.patch, grid.output_shape, truth, stride=stride,
        tile_merge="consensus", agree_thresh=args.agree_thresh, min_support=args.min_support,
        min_mask_voxels=args.min_mask_voxels, report=stats,
    )
    seconds = time.perf_counter() - started
    scores = compare_labellings(labels.numpy(), truth.numpy(), min_truth_voxels=args.min_truth_voxels)
    print(f"  assembled block ({seconds:.0f} s): {stats.get('tile_masks_drawn', 0)} perfect masks "
          f"-> {scores['pseudo_instances']} pieces vs {scores['truth_instances']} objects  "
          f"precision {scores['precision']:.3f}  recall {scores['recall']:.3f}  "
          f"merges {scores['merges']}  fragments {scores['fragments']}", flush=True)
    np.savez_compressed(args.out / f"{args.volume}_assembled.npz",
                        assembled=subsample(relabel(labels.numpy())))
    meta.update(assembled_seconds=round(seconds, 1), assembled_scores=scores)
    (args.out / f"{args.volume}.json").write_text(json.dumps(meta, indent=2))
    render_volume(args.out, args.volume)


def render_volume(directory: Path, volume: str) -> None:
    style()
    meta = json.loads((directory / f"{volume}.json").read_text())
    base = np.load(directory / f"{volume}_base.npz")
    image, truth, window = base["image"], base["truth"], base["window"]
    origin, shape = base["window_origin"], base["window_shape"]
    axes = meta["storage_axes"]
    nm = meta["saved_nm"][0]
    min_truth = int(meta["amg"]["min_mask_voxels"] // SUB**3)
    truth_rgb = palette(int(truth.max()) + 1)
    short = SHORT.get(volume, volume)
    arm = meta["run"].split("__")[1].split("_r0")[0]
    gate = f"gate {meta['amg'].get('pred_iou_thresh', '?')}"

    win = tuple(slice(int(o), int(o + s)) for o, s in zip(origin, shape, strict=True))
    ws = meta["window_scores"]
    figure(image[win], truth[win], window, axes, nm, truth_rgb, min_truth,
           f"{short}: ONE {shape[0] * nm / 1000:.0f} um window alone, {arm} step {meta['step']}, "
           f"{gate} (table: precision {ws['precision']:.2f}, recall {ws['recall']:.2f})",
           directory / f"gallery_{short}_window.png")

    assembled_path = directory / f"{volume}_assembled.npz"
    if assembled_path.exists():
        assembled = np.load(assembled_path)["assembled"]
        sc = meta["assembled_scores"]
        figure(image, truth, assembled, axes, nm, truth_rgb, min_truth,
               f"{short}: the ASSEMBLED {assembled.shape[0] * nm / 1000:.0f} um block "
               f"({meta['windows']} windows), {arm} step {meta['step']}, {gate} "
               f"(table: precision {sc['precision']:.2f}, recall {sc['recall']:.2f})",
               directory / f"gallery_{short}_block.png")


def render_overview(directory: Path, volumes: list[str]) -> Path | None:
    style()
    import matplotlib.pyplot as plt

    ready = [v for v in volumes if (directory / f"{v}_assembled.npz").exists()]
    if not ready:
        return None
    fig, axs = plt.subplots(len(ready), 4, figsize=(14, 3.6 * len(ready)), squeeze=False)
    for row, volume in zip(axs, ready, strict=True):
        meta = json.loads((directory / f"{volume}.json").read_text())
        base = np.load(directory / f"{volume}_base.npz")
        image, truth = base["image"], base["truth"]
        assembled = np.load(directory / f"{volume}_assembled.npz")["assembled"]
        min_truth = int(meta["amg"]["min_mask_voxels"] // SUB**3)
        majority, klass, tally = classify(assembled, truth, min_truth)
        axes = meta["storage_axes"]
        z = assembled.shape[axes.index("z")] // 2
        draw_row(row, section(image, axes, z), section(truth, axes, z),
                 section(assembled, axes, z), palette(int(truth.max()) + 1), majority, klass,
                 f"{SHORT.get(volume, volume)}, middle section:  ")
        sc = meta["assembled_scores"]
        row[2].set_title("model, coloured by the true object under it")
        row[3].set_title(
            f"{tally['matched']} matched, {tally['partial']} partial, {tally['merge']} merges, "
            f"{tally['spill']} spills of {tally['pieces']} pieces\n{tally['truth_found']} of "
            f"{tally['truth_objects']} objects found; table precision {sc['precision']:.2f}, "
            f"recall {sc['recall']:.2f}", fontsize=7.5)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = directory / "gallery_overview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def cmd_render(args: argparse.Namespace) -> None:
    volumes = args.volume or [v for v in VOLUMES if (args.directory / f"{v}_base.npz").exists()]
    for volume in volumes:
        render_volume(args.directory, volume)
    render_overview(args.directory, volumes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    label = sub.add_parser("label", help="label one scored block and save image/truth/labellings")
    label.add_argument("run_dir", type=Path)
    label.add_argument("--step", type=int, default=None)
    label.add_argument("--volume", required=True)
    label.add_argument("--out", type=Path, required=True)
    label.add_argument(
        "--gt-config", type=Path,
        default=REPO / "experiments/sam_lmd_v1/data/lmd_finetune_singlescale_4nm.yaml",
    )
    label.add_argument("--block", type=int, default=1024, help="block edge in lattice voxels")
    label.add_argument("--min-mask-voxels", type=int, default=4096)
    label.add_argument("--min-truth-voxels", type=int, default=4096)
    label.add_argument("--pred-iou-thresh", type=float, default=None,
                       help="the head's gate; default the labeller's (LABEL_AMG, 0.7)")
    label.add_argument("--amg", action="append", default=None, metavar="KEY=VALUE",
                       help="any other generator setting, e.g. prefer=part, "
                            "split_tiled_wholes=true, consistency_clicks=4")
    oracle = sub.add_parser("oracle", help="the same pictures for PERFECT per-window masks glued "
                                            "by the consensus rule (no model)")
    oracle.add_argument("--volume", required=True)
    oracle.add_argument("--out", type=Path, required=True)
    oracle.add_argument("--gt-config", type=Path,
                        default=REPO / "experiments/lmd_ssl_v1/lmd_finetune_singlescale.yaml")
    oracle.add_argument("--block", type=int, default=512, help="block edge in lattice voxels")
    oracle.add_argument("--patch", type=int, default=256, help="window edge in lattice voxels")
    oracle.add_argument("--mask-stride", type=int, default=4)
    oracle.add_argument("--min-mask-voxels", type=int, default=512)
    oracle.add_argument("--min-truth-voxels", type=int, default=512)
    oracle.add_argument("--agree-thresh", type=float, default=0.5)
    oracle.add_argument("--min-support", type=int, default=1)
    render = sub.add_parser("render", help="draw the figures from saved arrays")
    render.add_argument("directory", type=Path)
    render.add_argument("--volume", action="append", default=None)
    args = parser.parse_args()
    {"label": cmd_label, "oracle": cmd_oracle, "render": cmd_render}[args.command](args)


if __name__ == "__main__":
    main()
