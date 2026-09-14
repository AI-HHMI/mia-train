"""How big is a neurite compared with the encoder's token? Pictures and numbers, per volume.

    python experiments/sam_lmd_v1/figures/token_scale.py --out <dir>

Reads the single-tile diagnostic block (256^3 at 8 nm) of each finetune volume through the same
`VolumeGrid` the labeller uses, and produces:

  token_scale_slices.png   for hemibrain and liconn: the central section with the 16-voxel token
                           grid drawn over it; the same section with every ground-truth object
                           boundary drawn; and a map of how many distinct objects each token of
                           that layer contains.
  token_scale_stats.png    for all four volumes: the distribution of local object thickness
                           (twice the distance from an object voxel to the nearest voxel of a
                           different label, over all object voxels), with the mask cell (32 nm),
                           the 4 nm-read token (64 nm) and the 8 nm token (128 nm) marked; and the
                           distribution of distinct objects per token.
  token_scale_stats.json   the numbers behind both.

The point it tests: a click has to pick one object out of the token(s) it lands in and trace it
through tokens it shares with its neighbours. Where objects are thinner than a token and several
share one, that is hard; where an object spans many tokens, it is easy. Nothing here involves a
model -- it is a property of the data at the encoder's scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE.parent))

VOLUMES = ["hemibrain_ellipsoid_body", "liconn_mouse_dg", "kasthuri15_ac3",
           "zebrafish_fish2_quadcube1"]
PICTURED = ["hemibrain_ellipsoid_body", "liconn_mouse_dg"]
TOKEN = 16          # voxels per encoder patch
CELL = 4            # voxels per mask cell
NM = 8.0            # the lattice voxel

# Reference palette (dataviz skill, light surface): slots 1-4 = blue, orange, aqua, yellow (the
# adjacent-pair order that passes its validator); text tokens for annotations.
SERIES = {"hemibrain_ellipsoid_body": "#2a78d6", "liconn_mouse_dg": "#eb6834",
          "kasthuri15_ac3": "#1baf7a", "zebrafish_fish2_quadcube1": "#eda100"}
GRID_COLOUR = "#2a78d6"
BOUNDARY_COLOUR = "#eb6834"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
SURFACE = "#fcfcfb"
SHORT = {"hemibrain_ellipsoid_body": "hemibrain", "liconn_mouse_dg": "liconn",
         "kasthuri15_ac3": "kasthuri", "zebrafish_fish2_quadcube1": "zebrafish"}


def load_block(name: str) -> tuple[np.ndarray, np.ndarray, str]:
    """(image 256^3 in [0,1], labels 256^3 int64, storage axes) of the single-tile block."""
    from miao.config import load_config
    from pseudolabel import GT_CONFIG, plan_blocks, resolve_volume

    from prediction.grid import VolumeGrid

    cfg = load_config(GT_CONFIG)
    resolved = resolve_volume(cfg, name)
    (_, box), = plan_blocks(resolved, 288, 1)
    grid = VolumeGrid(resolved["config"], name, [256, 256, 256], box=box)
    if len(grid.tiles) != 1:
        raise SystemExit(f"{name}: expected one tile in a 288 block, got {len(grid.tiles)}")
    handle = grid.image_handle()
    native, _ = grid.tiles[0]
    image = grid.read_image(handle, native)
    labels = grid.read_ground_truth().astype(np.int64)
    if image.shape != labels.shape:
        raise SystemExit(f"{name}: image {image.shape} vs labels {labels.shape}")
    labels[labels < 0] = 0
    return image, labels, resolved["storage_axes"]


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    """Object voxels with a 6-neighbour of a different label (background counts as different)."""
    fg = labels > 0
    edge = np.zeros_like(fg)
    for axis in range(labels.ndim):
        a = np.moveaxis(labels, axis, 0)
        e = np.moveaxis(edge, axis, 0)
        diff = a[1:] != a[:-1]
        e[1:] |= diff
        e[:-1] |= diff
    return edge & fg


def thickness_nm(labels: np.ndarray) -> np.ndarray:
    """Twice the distance (nm) from every object voxel to the nearest voxel of another label."""
    from scipy.ndimage import distance_transform_edt

    edge = boundary_mask(labels)
    fg = labels > 0
    # Distance to the nearest edge voxel, measured from the edge voxels themselves (0 there);
    # +0.5 voxel so an edge voxel reads as half a voxel from the membrane, not zero.
    dist = distance_transform_edt(~edge)
    return (2.0 * (dist[fg] + 0.5) * NM).astype(np.float32)


def objects_per_token(labels: np.ndarray) -> np.ndarray:
    """(16, 16, 16) count of distinct object ids inside each 16^3 token."""
    n = labels.shape[0] // TOKEN
    out = np.zeros((n, n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                block = labels[i * TOKEN:(i + 1) * TOKEN, j * TOKEN:(j + 1) * TOKEN,
                               k * TOKEN:(k + 1) * TOKEN]
                ids = np.unique(block)
                out[i, j, k] = int((ids > 0).sum())
    return out


def section(volume: np.ndarray, axes: str) -> np.ndarray:
    """The middle section perpendicular to the sectioning axis z, as a 2-D array (rows, cols)."""
    z = axes.index("z")
    mid = volume.shape[z] // 2
    plane = np.take(volume, mid, axis=z)
    return plane if axes.replace("z", "") == "yx" else plane.T


def token_layer(counts: np.ndarray, axes: str) -> np.ndarray:
    z = axes.index("z")
    layer = np.take(counts, counts.shape[z] // 2, axis=z)
    return layer if axes.replace("z", "") == "yx" else layer.T


def draw_grid(ax, shape: tuple[int, int], step: int, colour: str, alpha: float, lw: float) -> None:
    for x in range(0, shape[1] + 1, step):
        ax.axvline(x - 0.5, color=colour, alpha=alpha, lw=lw)
    for y in range(0, shape[0] + 1, step):
        ax.axhline(y - 0.5, color=colour, alpha=alpha, lw=lw)


def scale_bar(ax, nm: float, shape: tuple[int, int]) -> None:
    px = nm / NM
    x0, y0 = shape[1] - px - 8, shape[0] - 10
    ax.plot([x0, x0 + px], [y0, y0], color="white", lw=3, solid_capstyle="butt")
    ax.text(x0 + px / 2, y0 - 4, f"{int(nm)} nm", color="white", ha="center", va="bottom",
            fontsize=8)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 9, "text.color": TEXT, "axes.labelcolor": TEXT,
                         "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.edgecolor": "#c3c2b7",
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

    stats: dict[str, dict] = {}
    blocks: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    for name in VOLUMES:
        image, labels, axes = load_block(name)
        blocks[name] = (image, labels, axes)
        thick = thickness_nm(labels)
        counts = objects_per_token(labels)
        occupied = counts[counts > 0]
        stats[name] = {
            "object_voxel_fraction": float((labels > 0).mean()),
            "objects_in_block": int(np.unique(labels[labels > 0]).size),
            "thickness_nm_median": float(np.median(thick)),
            "thickness_nm_p25": float(np.percentile(thick, 25)),
            "thickness_nm_p75": float(np.percentile(thick, 75)),
            "fraction_of_object_voxels_within_32nm_of_edge": float((thick <= 64).mean()),
            "fraction_of_object_voxels_within_64nm_of_edge": float((thick <= 128).mean()),
            "objects_per_token_mean": float(occupied.mean()),
            "tokens_with_2_or_more_objects": float((occupied >= 2).mean()),
            "tokens_with_4_or_more_objects": float((occupied >= 4).mean()),
            "thickness_hist_nm": np.histogram(thick, bins=np.arange(0, 1040, 16))[0].tolist(),
            "objects_per_token_hist": np.bincount(occupied, minlength=13)[:13].tolist(),
        }
        print(f"{SHORT[name]:10s} objects {stats[name]['objects_in_block']:5d}  thickness median "
              f"{stats[name]['thickness_nm_median']:5.0f} nm  <=64 nm from edge "
              f"{100 * stats[name]['fraction_of_object_voxels_within_64nm_of_edge']:4.0f}%  "
              f"objects/token {stats[name]['objects_per_token_mean']:.2f}  tokens with >=2: "
              f"{100 * stats[name]['tokens_with_2_or_more_objects']:3.0f}%  >=4: "
              f"{100 * stats[name]['tokens_with_4_or_more_objects']:3.0f}%", flush=True)
    (args.out / "token_scale_stats.json").write_text(json.dumps(stats, indent=2))

    # ---------------------------------------------------------------- figure 1: the sections
    fig, axs = plt.subplots(len(PICTURED), 3, figsize=(15, 5.6 * len(PICTURED)))
    ramp = LinearSegmentedColormap.from_list("blue_seq", ["#eef4fc", "#2a78d6", "#0e2f5c"])
    for row, name in enumerate(PICTURED):
        image, labels, axes = blocks[name]
        img2 = section(image, axes)
        lab2 = section(labels, axes)
        # 2-D boundaries of the section itself (label change between neighbouring pixels).
        e = np.zeros(lab2.shape, dtype=bool)
        e[1:, :] |= lab2[1:, :] != lab2[:-1, :]
        e[:, 1:] |= lab2[:, 1:] != lab2[:, :-1]
        ax = axs[row, 0]
        ax.imshow(img2, cmap="gray", vmin=np.percentile(img2, 1), vmax=np.percentile(img2, 99))
        draw_grid(ax, img2.shape, TOKEN, GRID_COLOUR, 0.55, 0.6)
        scale_bar(ax, 500, img2.shape)
        ax.set_title(f"{SHORT[name]}: section at 8 nm, encoder tokens (16 voxels = 128 nm)",
                     loc="left", fontsize=10)
        ax = axs[row, 1]
        ax.imshow(img2, cmap="gray", vmin=np.percentile(img2, 1), vmax=np.percentile(img2, 99))
        overlay = np.zeros((*e.shape, 4))
        overlay[e] = matplotlib.colors.to_rgba(BOUNDARY_COLOUR, 0.9)
        ax.imshow(overlay, interpolation="nearest")
        draw_grid(ax, img2.shape, TOKEN, GRID_COLOUR, 0.55, 0.6)
        scale_bar(ax, 500, img2.shape)
        ax.set_title("ground-truth object boundaries (orange) over the same tokens", loc="left",
                     fontsize=10)
        ax = axs[row, 2]
        layer = token_layer(objects_per_token(labels), axes)
        im = ax.imshow(layer, cmap=ramp, vmin=0, vmax=10, interpolation="nearest")
        for (i, j), v in np.ndenumerate(layer):
            ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=7,
                    color="white" if v >= 5 else TEXT)
        ax.set_title("distinct objects inside each token of this layer", loc="left", fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("objects per token", color=TEXT2)
        for a in axs[row]:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle("Objects against the encoder's token: hemibrain neuropil vs liconn expansion "
                 "microscopy", x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.5)
    fig.savefig(args.out / "token_scale_slices.png", dpi=110)
    plt.close(fig)

    # ---------------------------------------------------------------- figure 2: the numbers
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    ax = axs[0]
    edges = np.arange(0, 1040, 16)
    centres = edges[:-1] + 8
    for name in VOLUMES:
        h = np.array(stats[name]["thickness_hist_nm"], dtype=float)
        cdf = np.cumsum(h) / h.sum()
        ax.plot(centres, cdf, color=SERIES[name], lw=2, label=SHORT[name])
    marks = ((32, "mask cell 32 nm"), (64, "token at 4 nm: 64 nm"), (128, "token at 8 nm: 128 nm"))
    for x, lab in marks:
        ax.axvline(x, color="#c3c2b7", lw=1, ls="--")
        ax.text(x + 4, 0.03, lab, rotation=90, color=TEXT2, fontsize=8, va="bottom")
    ax.set_xlim(0, 1040)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("local thickness: 2 x distance from an object voxel to the nearest other label "
                  "(nm)")
    ax.set_ylabel("fraction of object voxels at or below")
    ax.set_title("How thin the objects are, per volume", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    ax = axs[1]
    width = 0.2
    xs = np.arange(1, 13)
    for k, name in enumerate(VOLUMES):
        h = np.array(stats[name]["objects_per_token_hist"], dtype=float)[1:13]
        ax.bar(xs + (k - 1.5) * width, h / h.sum(), width=width * 0.9, color=SERIES[name],
               label=SHORT[name], linewidth=0)
    ax.set_xticks(xs)
    ax.set_xlabel("distinct objects inside one token (16^3 voxels, 128 nm)")
    ax.set_ylabel("fraction of occupied tokens")
    ax.set_title("How many objects share a token", loc="left", fontsize=11)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        axs[0].spines[spine].set_visible(False)
        axs[1].spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "token_scale_stats.png", dpi=110)
    plt.close(fig)
    print(f"wrote {args.out}/token_scale_slices.png, token_scale_stats.png, token_scale_stats.json")


if __name__ == "__main__":
    main()
