"""The leaderboard's segmentations side by side: SAM arm, affinity rows, ground truth, per volume.

    python experiments/sam_lmd_v1/figures/leaderboard_gallery.py --out <dir> \\
        --sam arm4=/nrs/.../eval/arm4_8nm_gb16_r0/test:5000 \\
        --mws 1c=/nrs/.../lmd1_neuron_artifacts/arm1/test:50000 \\
        --mws 2c=/nrs/.../mws_artifacts/test:50000 \\
        --volumes kasthuri15_ac4 liconn_mouse_hippocampus liconn_expid82

Every artifact is what `predict.py` (SAM) or the mutex-watershed pipeline (affinities) wrote for the
`lmd_ssl_v1_neuron_instance` task: the same lattice, the same region, ids per voxel. Each is shown
AFTER its own size filter, the one the leaderboard scored it with (`NAME=DIR:MIN_SIZE`; ids with
fewer voxels than MIN_SIZE become background, exactly mia-evals' `size_filter`), so the panels
show what the panoptic-quality numbers were computed on. The ground truth is the `.gt.zarr` beside
the SAM artifact, and the image is read through the same `VolumeGrid` that produced the lattice.

Per volume: `leaderboard_<volume>.png`, two rows on the middle section -- image | ground truth |
one panel per model coloured by the true object under each piece (a correct object keeps its
colour across all panels; black lines are piece boundaries), then the error map per model
(matched / partial / merge / spill / unlabelled truth, as in `labelling_gallery.py`). Panel titles
carry the piece tally and, for the affinity rows, the recorded per-volume panoptic quality.
`leaderboard_overview.png` stacks the first row of every volume.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import labelling_gallery as gallery  # noqa: E402

RECORDS = Path("/groups/scicompsoft/home/orhane/projects/mia-evals/leaderboard/"
               "lmd_ssl_v1_neuron_instance/records")
RECORD_LABELS = {"1c": "1c_step50000_mws", "2c": "2c_step50000_mws"}
SHORT = {"kasthuri15_ac4": "kasthuri ac4", "liconn_mouse_hippocampus": "liconn hippocampus",
         "liconn_expid82": "liconn expid82", "zebrafish_fish2_doublecube1": "zebrafish doublecube1"}


def parse_source(text: str) -> tuple[str, Path, int]:
    name, _, rest = text.partition("=")
    directory, _, min_size = rest.rpartition(":")
    return name, Path(directory), int(min_size)


def read_labels(path: Path, stride: int = 1) -> np.ndarray:
    """The whole array, or every `stride`-th voxel per axis (for volumes too big to hold)."""
    import zarr

    node = zarr.open(str(path), mode="r")
    array = node if hasattr(node, "shape") else node[list(node.array_keys())[0]]
    if stride == 1:
        return np.asarray(array[:])
    return np.asarray(array[tuple(slice(None, None, stride) for _ in array.shape)])


def size_filter(labels: np.ndarray, min_size: int) -> np.ndarray:
    """mia-evals' rule: an id with fewer than `min_size` voxels becomes background."""
    if min_size <= 0:
        return labels
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = True
    return np.where(keep[labels], labels, 0)


def recorded_pq(model: str, volume: str) -> float | None:
    label = RECORD_LABELS.get(model)
    if label is None or not (RECORDS / f"{label}.json").exists():
        return None
    record = json.loads((RECORDS / f"{label}.json").read_text())
    return record.get("per_volume", {}).get(volume, {}).get("voxel_instance", {}).get("pq")


def read_image(gt_config: Path, volume: str, expected: tuple[int, ...],
               stride: int = 1) -> tuple[np.ndarray, str]:
    """The block's image on the prediction lattice, and the storage axes, as predict.py saw them."""
    from miao.config import load_config
    from pseudolabel import resolve_volume

    from prediction.grid import VolumeGrid

    config = load_config(gt_config)
    with contextlib.redirect_stdout(io.StringIO()):
        resolved = resolve_volume(config, volume)
        grid = VolumeGrid(config, volume, [256, 256, 256])
    read = tuple(-(-o // stride) for o in grid.output_shape)
    if read != tuple(expected):
        raise SystemExit(f"{volume}: grid output {grid.output_shape} (stride {stride} -> {read}) "
                         f"!= artifact {expected}")
    image = gallery.paste_image(grid, grid.image_handle())
    if stride > 1:
        image = image[::stride, ::stride, ::stride]
    return (np.clip(image, 0, 1) * 255).astype(np.uint8), resolved["storage_axes"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sam", required=True, metavar="NAME=DIR:MIN_SIZE")
    parser.add_argument("--mws", action="append", default=[], metavar="NAME=DIR:MIN_SIZE")
    parser.add_argument("--gt-config", type=Path,
                        default=REPO / "experiments/lmd_ssl_v1/lmd_val_singlescale.yaml")
    parser.add_argument("--volumes", nargs="+", required=True)
    parser.add_argument("--min-truth", type=int, default=512,
                        help="true objects counted for recall, in lattice voxels")
    parser.add_argument("--sections", type=int, default=1,
                        help="sections per volume (1 = the middle; 3 = at 30/50/70%% of the depth)")
    parser.add_argument("--suffix", default="", help="appended to every output file name")
    parser.add_argument("--no-overview", action="store_true")
    parser.add_argument("--read-stride", type=int, default=1,
                        help="read every n-th voxel per axis (doublecube1 at 1920^3 needs 2)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gallery.style()
    sources = [parse_source(args.sam)] + [parse_source(m) for m in args.mws]
    sam_name, sam_dir, _ = sources[0]
    overview_rows = []

    for volume in args.volumes:
        truth = gallery.relabel(read_labels(sam_dir / f"{volume}.gt.zarr", args.read_stride))
        image, axes = read_image(args.gt_config, volume, truth.shape, args.read_stride)
        preds = {}
        for name, directory, min_size in sources:
            labels = read_labels(directory / f"{volume}.zarr", args.read_stride).astype(np.int64)
            preds[name] = gallery.relabel(size_filter(labels, min_size))
            print(f"{volume} {name}: {int(preds[name].max())} pieces after size filter "
                  f"{min_size}", flush=True)

        # Statistics on the lattice halved (as the labelling gallery does); the section at full.
        sub = gallery.subsample
        truth_s = sub(truth)
        min_truth = max(1, args.min_truth // (gallery.SUB * args.read_stride) ** 3)
        stats = {name: gallery.classify(sub(p), truth_s, min_truth) for name, p in preds.items()}
        truth_rgb = gallery.palette(int(truth.max()) + 1)
        depth = truth.shape[axes.index("z")]
        fractions = [0.5] if args.sections == 1 else list(np.linspace(0.3, 0.7, args.sections))
        short = SHORT.get(volume, volume)
        nm = 8.0 * args.read_stride

        # Two rows per section: pieces coloured by the true object under them, then the error map.
        fig, axs = plt.subplots(2 * len(fractions), 2 + len(preds),
                                figsize=(3.5 * (2 + len(preds)), 3.7 * 2 * len(fractions)))
        for k, frac in enumerate(fractions):
            z = int(depth * frac)
            top, bottom = axs[2 * k], axs[2 * k + 1]
            image2d, truth2d = gallery.section(image, axes, z), gallery.section(truth, axes, z)
            grey = np.repeat(image2d[..., None].astype(np.float32) / 255, 3, axis=-1)
            top[0].imshow(grey, interpolation="nearest")
            top[0].set_ylabel(f"section {z * nm / 1000:.2f} um", fontsize=9)
            truth_img = truth_rgb[truth2d]
            truth_img[gallery.boundaries(truth2d) & (truth2d > 0)] *= 0.35
            top[1].imshow(truth_img, interpolation="nearest")
            if k == 0:
                top[0].set_title(f"{short}: image", fontsize=9)
                top[1].set_title(f"ground truth, {stats[sam_name][2]['truth_objects']} objects",
                                 fontsize=9)
            for ax in (bottom[0], bottom[1]):
                ax.axis("off")
            for column, (name, pred) in enumerate(preds.items(), start=2):
                majority, klass, tally = stats[name]
                pred2d = gallery.section(pred, axes, z)
                img = truth_rgb[majority[pred2d]]
                img[(pred2d > 0) & (majority[pred2d] == 0)] = 0.82
                img[pred2d == 0] = 1.0
                img[gallery.boundaries(pred2d) & (pred2d > 0)] = 0.0
                top[column].imshow(img, interpolation="nearest")
                if k == 0:
                    pq = recorded_pq(name, volume)
                    head = name + (f", leaderboard pq {pq:.3f}" if pq is not None else "")
                    top[column].set_title(
                        f"{head}\n{tally['matched']} matched, {tally['partial']} partial, "
                        f"{tally['merge']} merges, {tally['spill']} spills; "
                        f"{tally['truth_found']}/{tally['truth_objects']} found", fontsize=8)
                error = np.ones(pred2d.shape + (3,), dtype=np.float32)
                error[(pred2d == 0) & (truth2d > 0)] = gallery.hex_rgb(
                    gallery.CLASS_COLOURS["unlabelled"])
                for cls in gallery.PIECE_CLASSES:
                    chosen = (pred2d > 0) & (klass[pred2d] == gallery.CLASS_ORDER.index(cls))
                    error[chosen] = gallery.hex_rgb(gallery.CLASS_COLOURS[cls])
                error = 0.55 * error + 0.45 * grey
                error[gallery.boundaries(pred2d) & (pred2d > 0)] *= 0.5
                bottom[column].imshow(error, interpolation="nearest")
                if k == 0:
                    overview_rows.append((short, name, img, tally, recorded_pq(name, volume)))
            if k == 0:
                overview_rows.append((short, "image", grey, None, None))
                overview_rows.append((short, "truth", truth_img, stats[sam_name][2], None))
        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.legend(handles=gallery.legend_handles(), loc="lower center", ncol=5, frameon=False)
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        out = args.out / f"leaderboard_{volume}{args.suffix}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}", flush=True)

    if args.no_overview:
        return
    # Overview: one row per volume -- image | truth | each model, coloured by the true object.
    order = ["image", "truth"] + [s[0] for s in sources]
    volumes = [SHORT.get(v, v) for v in args.volumes]
    fig, axs = plt.subplots(
        len(volumes), len(order), figsize=(3.5 * len(order), 3.7 * len(volumes)), squeeze=False
    )
    for r, short in enumerate(volumes):
        for c, name in enumerate(order):
            entry = next(e for e in overview_rows if e[0] == short and e[1] == name)
            axs[r, c].imshow(entry[2], interpolation="nearest")
            if name == "image":
                title = f"{short}: image"
            elif name == "truth":
                title = f"ground truth ({entry[3]['truth_objects']} objects)"
            else:
                pq = entry[4]
                title = name + (f", leaderboard pq {pq:.3f}" if pq is not None else "") + \
                    f"\n{entry[3]['matched']} matched, {entry[3]['merge']} merges, " \
                    f"{entry[3]['truth_found']}/{entry[3]['truth_objects']} found"
            axs[r, c].set_title(title, fontsize=8)
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
    fig.tight_layout()
    out = args.out / f"leaderboard_overview{args.suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
