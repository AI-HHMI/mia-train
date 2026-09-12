"""Segment everything: a promptable model run over a whole volume without anyone prompting it.

The trained model answers "segment *this*". Whole-volume segmentation asks it that question at
every point of a regular grid, keeps the answers it was confident about, and reconciles them into
one labelling. That is Segment Anything's automatic mask generator, and it is how SA-1B was made.
Restated here for volumes, with the departures a third dimension forces.

**It is a search, not a field.** Every other prediction in this repo is a fixed number of channels
per voxel that overlapping tiles can be averaged across. This produces a *set of masks*, and two
tiles' sets are reconciled by identity -- "is this the same object" -- which is why it lives behind
`BaseAlgorithm.volume_predictor` rather than inside `prediction.dense`'s blending loop.

**Everything below happens on the mask grid**, at `patch_size / mask_upscale` (stride 4 by
default), and only the final labelling is brought to voxel resolution. That is 64x cheaper than
working at voxel resolution throughout and, measured on this corpus, costs 0.013 IoU at the
model's current quality (`sam3d/eval_checkpoint.py`).

Three places this deviates from the reference, each for a measured reason:

  * **Grid density.** The reference prompts a 32x32 grid and gets ~100 masks per image, about ten
    prompts per object. A 256-cube here holds a median of 138 objects, so 12^3 = 1728 prompts is
    the equivalent density; 32^3 would be 32768 and almost entirely redundant.
  * **Duplicates are decided by mask IoU, not box IoU.** The reference suppresses duplicates by box
    overlap for speed. In 3D that is wrong: a neurite's bounding box is often most of the tile while
    the object is 1% of it, so two unrelated processes crossing a tile have near-identical boxes and
    one would be wrongly suppressed. Boxes are used only to skip pairs that cannot overlap.
  * **Nesting is resolved explicitly.** The model is deliberately ambiguity-aware -- one click
    yields whole, part and subpart -- and an integer labelling cannot hold all three. Which level
    to keep is a decision (`prefer`), made by containment, rather than an accident of which
    candidate happened to score higher.

The thresholds are not the reference's. Its 0.88 predicted-IoU gate assumes a far better model
than a first training run produces, and would reject nearly everything here. They are tuned against
a score on a held-out volume via `predict.py --override`, which is why they are constructor
arguments on the strategy rather than constants.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from layers.common.prompt import FOREGROUND
from layers.common.rope import voxel_coords
from prediction.grid import VolumeGrid
from prediction.types import VolumePrediction

PREFER = ("whole", "part")
CONSISTENCY_PICK = ("top", "best")


# ---------------------------------------------------------------------------------------------
# Model-free pieces. Each is a pure function of tensors, which is what makes them testable against
# a brute-force reference without a model or a store.
# ---------------------------------------------------------------------------------------------


def point_grid(extent: Sequence[int], points_per_side: int) -> torch.Tensor:
    """`points_per_side^rank` voxel positions spread evenly over a crop -> `(N, rank)`.

    Cell-centred, as the reference's `build_point_grid`: `points_per_side` points per axis, each in
    the middle of its cell, so no point sits on a face and the first and last are half a cell in.
    Voxel indices, not normalised coordinates, so a test can read them.
    """
    if points_per_side < 1:
        raise ValueError(f"points_per_side must be at least 1, got {points_per_side}")
    axes = [
        (torch.arange(points_per_side, dtype=torch.float32) + 0.5) * (size / points_per_side) - 0.5
        for size in extent
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, len(extent))


def stability_score(logits: torch.Tensor, delta: float) -> torch.Tensor:
    """IoU between the mask thresholded at `+delta` and at `-delta` -> one score per mask.

    A mask whose boundary moves a lot under a small change of threshold is a mask the model was not
    sure about, whatever IoU it *predicted*. The reference's second filter, and independent of the
    IoU head: this is measured on the logits, that is a learned estimate.
    """
    loose = (logits > -delta).flatten(-3).sum(-1)
    tight = (logits > delta).flatten(-3).sum(-1)
    return tight / loose.clamp_min(1)


def mask_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Tight bounding boxes of `(N, *spatial)` boolean masks -> `(N, 2, rank)`, lo/hi exclusive.

    An empty mask gets a zero-extent box at the origin, which overlaps nothing, so it drops out of
    every pairwise step without a special case.
    """
    rank = masks.dim() - 1
    boxes = masks.new_zeros((masks.shape[0], 2, rank), dtype=torch.long)
    for axis in range(rank):
        others = tuple(d for d in range(1, rank + 1) if d != axis + 1)
        present = masks.amax(dim=others) if others else masks
        any_present = present.any(dim=1)
        index = torch.arange(present.shape[1], device=masks.device)
        first = torch.where(present, index, present.shape[1]).amin(dim=1)
        last = torch.where(present, index, -1).amax(dim=1) + 1
        boxes[:, 0, axis] = torch.where(any_present, first, 0)
        boxes[:, 1, axis] = torch.where(any_present, last, 0)
    return boxes


def boxes_intersect(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`(Na, 2, rank)` x `(Nb, 2, rank)` boxes -> `(Na, Nb)` bool, True where they overlap."""
    lo = torch.maximum(a[:, None, 0], b[None, :, 0])
    hi = torch.minimum(a[:, None, 1], b[None, :, 1])
    return (hi > lo).all(dim=-1)


def pairwise_intersection(a: torch.Tensor, b: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """Voxel counts of `A_i & B_j` for `(Na, *s)` and `(Nb, *s)` boolean masks -> `(Na, Nb)`.

    A matmul over flattened masks -- `A @ B.T` counts the shared ones -- chunked over rows so the
    float copies stay a few hundred megabytes on a dense tile rather than a few gigabytes.
    """
    flat_b = b.flatten(1).to(torch.float32)
    out = []
    for start in range(0, a.shape[0], chunk):
        flat_a = a[start : start + chunk].flatten(1).to(torch.float32)
        out.append(flat_a @ flat_b.T)
    return torch.cat(out, dim=0) if out else a.new_zeros((0, b.shape[0]), dtype=torch.float32)


def mask_nms(masks: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Greedy non-maximum suppression on MASK IoU -> indices kept, in descending score order.

    Boxes decide only which pairs to look at; the decision is the masks' own IoU. See the module
    docstring for why box IoU is the wrong criterion in 3D.
    """
    if masks.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=masks.device)
    # Stable, so equal scores keep their input order and a run is reproducible. Which of two
    # exactly-tied duplicates survives is then arbitrary but fixed, rather than arbitrary and
    # different each time.
    order = scores.argsort(descending=True, stable=True)
    masks, boxes = masks[order], mask_boxes(masks[order])
    areas = masks.flatten(1).sum(-1).to(torch.float32)
    alive = torch.ones(masks.shape[0], dtype=torch.bool, device=masks.device)
    keep = []
    for index in range(masks.shape[0]):
        if not alive[index]:
            continue
        keep.append(index)
        later = torch.arange(index + 1, masks.shape[0], device=masks.device)
        later = later[alive[later] & boxes_intersect(boxes[index : index + 1], boxes[later])[0]]
        if later.numel() == 0:
            continue
        inter = pairwise_intersection(masks[index : index + 1], masks[later])[0]
        iou = inter / (areas[index] + areas[later] - inter).clamp_min(1)
        alive[later[iou > iou_threshold]] = False
    return order[torch.tensor(keep, dtype=torch.long, device=masks.device)]


def resolve_containment(
    masks: torch.Tensor,
    scores: torch.Tensor,
    prefer: str,
    threshold: float,
    duplicate_iou: float,
) -> torch.Tensor:
    """Drop one of each nested pair -> indices kept.

    `A` is nested in `B` when `|A & B| / |A| >= threshold` AND their IoU is below
    `duplicate_iou`. `prefer="whole"` keeps `B`, the container -- right when the ground truth is
    cell-level and a nucleus inside a cell must not become a second instance. `prefer="part"`
    keeps `A`.

    The IoU clause is what separates this from NMS, by definition rather than by ordering. Two
    masks with high IoU are *duplicates*: two answers to the same question, and the one that scored
    higher should win, which is NMS's job. A part inside a whole has high containment and LOW
    IoU: it is a different answer, and which one to keep is a decision about level, not quality.
    Without the clause, this step would also adjudicate duplicates -- and it would do so by size,
    keeping the largest of several near-identical wholes rather than the best-scoring.

    Run this BEFORE NMS. Greedy NMS keeps whichever of two overlapping masks scored higher, and a
    part that is most of its whole can score higher than the whole; if NMS sees them first it keeps
    the part and discards the whole, and nothing downstream can bring the whole back.

    The boundary this draws is deliberate and worth stating: a part covering at least
    `duplicate_iou` of its whole *is* a duplicate here, and the IoU head decides between them.
    That is the head's job -- it is trained to predict each mask's IoU against the truth, and a
    75%-part is a mask whose true IoU is ~0.75 -- so the design leans on the head being
    calibrated, which the training runs measure (`*_iou_error`) and which was 0.004 at 50k steps.
    """
    if prefer not in PREFER:
        raise ValueError(f"prefer must be one of {PREFER}, got {prefer!r}")
    count = masks.shape[0]
    if count == 0:
        return torch.zeros(0, dtype=torch.long, device=masks.device)
    areas = masks.flatten(1).sum(-1).to(torch.float32)
    boxes = mask_boxes(masks)
    alive = torch.ones(count, dtype=torch.bool, device=masks.device)
    # Small first, so each mask is tested against everything that could contain it.
    for index in areas.argsort().tolist():
        if not alive[index]:
            continue
        others = torch.arange(count, device=masks.device)
        others = others[
            alive[others] & (others != index)
            & boxes_intersect(boxes[index : index + 1], boxes[others])[0]
        ]
        if others.numel() == 0:
            continue
        inter = pairwise_intersection(masks[index : index + 1], masks[others])[0]
        inside = inter / areas[index].clamp_min(1)
        iou = inter / (areas[index] + areas[others] - inter).clamp_min(1)
        # A container is strictly larger and not a duplicate. Equal-sized or near-identical masks
        # are NMS's to settle, by score.
        nested = (inside >= threshold) & (iou < duplicate_iou) & (areas[others] > areas[index])
        containers = others[nested]
        if containers.numel() == 0:
            continue
        if prefer == "whole":
            alive[index] = False
        else:
            alive[containers] = False
    return torch.nonzero(alive, as_tuple=True)[0]


def interior_points(mask: torch.Tensor, logits: torch.Tensor, count: int) -> torch.Tensor:
    """`count` well-separated cells inside one `(*grid)` mask -> `(k, rank)` indices, k <= count.

    The first is the mask's most confident cell; each further one is the mask cell farthest from
    every point chosen so far (farthest-point sampling). So for a mask made of two lobes, the second
    point lands in the other lobe -- which is the whole reason to ask the model again from there.
    """
    cells = torch.nonzero(mask, as_tuple=False)
    if cells.shape[0] == 0 or count < 1:
        return cells[:0]
    chosen = [int(logits[mask].argmax())]
    if count > 1 and cells.shape[0] > 1:
        position = cells.to(torch.float32)
        nearest = ((position - position[chosen[0]]) ** 2).sum(-1)
        for _ in range(min(count, cells.shape[0]) - 1):
            farthest = int(nearest.argmax())
            chosen.append(farthest)
            nearest = torch.minimum(nearest, ((position - position[farthest]) ** 2).sum(-1))
    return cells[torch.tensor(chosen, device=cells.device)]


def tiled_wholes(
    masks: torch.Tensor,
    scores: torch.Tensor,
    containment_thresh: float,
    cover_thresh: float,
    duplicate_iou: float,
    min_parts: int = 2,
    part_overlap: float = 0.2,
) -> torch.Tensor:
    """Which `(N, *grid)` masks are tiled by two or more disjoint smaller masks -> `(N,)` bool.

    A merge -- one mask spanning two neighbouring objects -- usually arrives with its own evidence:
    the grid also clicked inside each object it spans, and those clicks produced confident masks of
    the objects on their own. `resolve_containment(prefer="whole")` would keep the merge and drop
    them, which is the right call for a nucleus inside a cell and the wrong one here. This tells
    the two cases apart by what the contained masks add up to: a nucleus is one part covering a
    fraction of its cell, a merge is several disjoint parts that together cover almost all of it.

    A mask is flagged when at least `min_parts` of its contained masks (`|A & M| / |A| >=
    containment_thresh`, not a duplicate of `M`, smaller than `M`), chosen greedily by score and
    pairwise overlapping by less than `part_overlap` of themselves, together cover at least
    `cover_thresh` of it. The parts themselves are left alone -- dropping the flagged whole is what
    lets them through containment and NMS as ordinary masks.
    """
    count = masks.shape[0]
    flagged = torch.zeros(count, dtype=torch.bool, device=masks.device)
    if count < min_parts + 1:
        return flagged
    areas = masks.flatten(1).sum(-1).to(torch.float32)
    boxes = mask_boxes(masks)
    for index in range(count):
        others = torch.arange(count, device=masks.device)
        others = others[
            (others != index) & (areas[others] < areas[index])
            & boxes_intersect(boxes[index : index + 1], boxes[others])[0]
        ]
        if others.numel() < min_parts:
            continue
        inter = pairwise_intersection(masks[index : index + 1], masks[others])[0]
        inside = inter / areas[others].clamp_min(1)
        iou = inter / (areas[index] + areas[others] - inter).clamp_min(1)
        parts = others[(inside >= containment_thresh) & (iou < duplicate_iou)]
        if parts.numel() < min_parts:
            continue
        # Greedy by score, so the confident answers define the tiling; a part overlapping what is
        # already tiled by more than `part_overlap` of itself is a duplicate answer, not a new part.
        union = torch.zeros_like(masks[index])
        accepted = 0
        for part in parts[scores[parts].argsort(descending=True)].tolist():
            overlap = (masks[part] & union).sum().to(torch.float32) / areas[part].clamp_min(1)
            if overlap > part_overlap:
                continue
            union |= masks[part]
            accepted += 1
        covered = (union & masks[index]).sum().to(torch.float32) / areas[index].clamp_min(1)
        flagged[index] = accepted >= min_parts and bool(covered >= cover_thresh)
    return flagged


TILE_MERGE = ("canvas", "none", "propagate", "consensus")

#: A mask counts as reaching into a shared region when at least this many of its cells lie there;
#: below it, two windows' boundary jitter would be read as a disagreement.
MIN_SHARED_CELLS = 8


def tile_labelling(masks: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """`(N, *tile)` masks -> `(*tile)` int32 map of local ids 1..N; a better score wins overlaps."""
    out = torch.zeros(masks.shape[1:], dtype=torch.int32, device=masks.device)
    for index in scores.argsort(descending=True, stable=True).tolist():
        out[masks[index] & (out == 0)] = index + 1
    return out


def consensus_labelling(
    tiles: Sequence[tuple[Sequence[int], torch.Tensor]],
    shape: Sequence[int],
    agree_thresh: float,
    min_support: int = 1,
    report: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, int]:
    """Every window's masks, reconciled by agreement between windows -> `(*shape)` labels, count.

    `tiles` holds, per window, its origin on the mask grid and its `tile_labelling` map. Two masks
    from two windows are the same object when they agree where BOTH windows looked: their IoU
    inside the two windows' shared region reaches `agree_thresh`, and each is the other's best
    match there (mutual best, so a mask that spans two objects cannot bridge them -- it is joined
    to the one it matches best and disputed on the other). Agreements are chained with union-find,
    which is how an object longer than a window gets one id.

    Then every window votes: a cell takes the class that the windows covering it assigned, and a
    cell that two windows assign to DIFFERENT classes is left unlabelled (0) rather than given to
    whichever window came first. A window that saw a cell and drew no mask there is not a vote
    against -- the grid misses about half of what it clicks on -- so unlabelled means disputed or
    unseen-by-any-mask, never "background by majority". `min_support` asks a cell to be claimed
    by at least that many windows, capped at the number of windows that actually covered it, so
    the block's borders are not erased for lack of a second look.

    The alternative this replaces painted first-come and joined on one-sided coverage; measured on
    this repo's data, that rule got WORSE with more windows (finer steps multiplied its mistakes and
    added no corrections), which is the signature of a rule that cannot use evidence.

    `report`, if given, receives counts of what happened to every mask that reaches into a shared
    region with at least `MIN_SHARED_CELLS` cells: `joined` (agreed with a mutual best match),
    `disagreed` (overlapped the other window's masks but did not agree), `unmet` (the other window
    drew nothing there at all). The last is the one no joining rule can fix: the object was found
    in one window and missed in the next.
    """
    if not 0.0 < agree_thresh <= 1.0:
        raise ValueError(f"agree_thresh must be in (0, 1], got {agree_thresh}")
    if min_support < 1:
        raise ValueError(f"min_support must be at least 1, got {min_support}")
    shape = tuple(int(v) for v in shape)
    stats = {"joined": 0, "disagreed": 0, "unmet": 0, "disagreed_best_iou_sum": 0.0}
    if report is not None:
        report.update(stats)
    if not tiles:
        return torch.zeros(shape, dtype=torch.int64), 0
    device = tiles[0][1].device
    rank = len(shape)

    offsets: list[int] = []
    total = 0
    boxes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for origin, local in tiles:
        if local.dim() != rank or len(origin) != rank:
            raise ValueError(
                f"tile at {tuple(origin)} with shape {tuple(local.shape)} is not rank {rank}"
            )
        offsets.append(total)
        total += int(local.max()) if local.numel() else 0
        lo = tuple(int(o) for o in origin)
        boxes.append((lo, tuple(o + e for o, e in zip(lo, local.shape, strict=True))))

    parent = list(range(total + 1))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    def in_region(index: int, lo: Sequence[int], hi: Sequence[int]) -> torch.Tensor:
        origin, local = tiles[index]
        window = tuple(
            slice(low - o, high - o) for low, high, o in zip(lo, hi, origin, strict=True)
        )
        ids = local[window].long()
        return torch.where(ids > 0, ids + offsets[index], torch.zeros_like(ids))

    for t in range(len(tiles)):
        for u in range(t + 1, len(tiles)):
            lo = [max(a, b) for a, b in zip(boxes[t][0], boxes[u][0], strict=True)]
            hi = [min(a, b) for a, b in zip(boxes[t][1], boxes[u][1], strict=True)]
            if any(high <= low for low, high in zip(lo, hi, strict=True)):
                continue
            a, b = in_region(t, lo, hi), in_region(u, lo, hi)
            # Sizes INSIDE the shared region: a mask is compared only where the other window
            # could have seen the same object, never on what lies beyond that window's edge.
            sizes_a = torch.bincount(a[a > 0], minlength=total + 1)
            sizes_b = torch.bincount(b[b > 0], minlength=total + 1)
            present_a = torch.nonzero(sizes_a >= MIN_SHARED_CELLS).flatten()
            present_b = torch.nonzero(sizes_b >= MIN_SHARED_CELLS).flatten()
            both = (a > 0) & (b > 0)
            if not bool(both.any()):
                stats["unmet"] += int(present_a.numel() + present_b.numel())
                continue
            ua, ia = torch.unique(a[both], return_inverse=True)
            ub, ib = torch.unique(b[both], return_inverse=True)
            inter = torch.zeros(ua.numel(), ub.numel(), device=device)
            inter.index_put_((ia, ib), torch.ones_like(ia, dtype=torch.float32), accumulate=True)
            size_a = sizes_a[ua].float()
            size_b = sizes_b[ub].float()
            iou = inter / (size_a[:, None] + size_b[None, :] - inter).clamp_min(1.0)
            best_b = iou.argmax(dim=1)
            best_a = iou.argmax(dim=0)
            rows = torch.arange(ua.numel(), device=device)
            mutual = best_a[best_b] == rows
            ok = mutual & (iou[rows, best_b] >= agree_thresh)
            for i, j in zip(ua[ok].tolist(), ub[best_b[ok]].tolist(), strict=True):
                union(i, j)
            if report is not None:
                cols = torch.arange(ub.numel(), device=device)
                ok_b = (best_b[best_a] == cols) & (iou[best_a, cols] >= agree_thresh)
                big_a, big_b = size_a >= MIN_SHARED_CELLS, size_b >= MIN_SHARED_CELLS
                stats["joined"] += int((ok & big_a).sum() + (ok_b & big_b).sum())
                stats["disagreed"] += int((~ok & big_a).sum() + (~ok_b & big_b).sum())
                stats["disagreed_best_iou_sum"] += float(
                    iou[rows, best_b][~ok & big_a].sum() + iou[best_a, cols][~ok_b & big_b].sum()
                )
                met = torch.isin(present_a, ua).sum() + torch.isin(present_b, ub).sum()
                stats["unmet"] += int(present_a.numel() + present_b.numel() - met)

    roots = torch.tensor([find(x) for x in range(total + 1)], dtype=torch.int64, device=device)
    roots[0] = 0

    first = torch.zeros(shape, dtype=torch.int64, device=device)
    support = torch.zeros(shape, dtype=torch.int32, device=device)
    seen = torch.zeros(shape, dtype=torch.int32, device=device)
    conflict = torch.zeros(shape, dtype=torch.bool, device=device)
    for (origin, local), offset in zip(tiles, offsets, strict=True):
        window = tuple(slice(o, o + e) for o, e in zip(origin, local.shape, strict=True))
        ids = local.long()
        classes = roots[torch.where(ids > 0, ids + offset, torch.zeros_like(ids))]
        has = classes > 0
        seen[window] += 1
        f, sup, con = first[window], support[window], conflict[window]
        new = has & (f == 0)
        f[new] = classes[new]
        sup[new] = 1
        same = has & ~new & (f == classes)
        sup[same] += 1
        con |= has & ~new & (f != classes)

    need = torch.minimum(torch.full_like(seen, min_support), seen)
    labels = torch.where(~conflict & (support >= need), first, torch.zeros_like(first))
    kept = torch.unique(labels)
    kept = kept[kept > 0]
    lut = torch.zeros(total + 1, dtype=torch.int64, device=device)
    lut[kept] = torch.arange(1, kept.numel() + 1, device=device)
    if report is not None:
        report.update(stats)
        report["disputed_cells"] = int(conflict.sum())
        report["tile_masks"] = total
    return lut[labels], int(kept.numel())


class Canvas:
    """The whole-volume labelling being assembled, on the mask grid, one tile at a time.

    Tiles overlap by half a patch, so the same object is seen by up to eight tiles and arrives as
    up to eight slightly different masks. Three ways of reconciling them are implemented, selected
    by `tile_merge`, because which is best is an empirical question this repo measures rather than
    assumes:

      * `"canvas"` -- each incoming mask looks at what is already painted beneath it: if one
        existing id covers at least `merge_fraction` of it, it *is* that object and takes its id;
        otherwise it is new. A geometric heuristic, and the baseline.
      * `"none"` -- every mask is new. The control: it shows what the heuristic is worth, since an
        object spanning two tiles is then guaranteed two ids.
      * `"propagate"` -- before a tile is searched, every object already painted in its region is
        handed back to the model as a dense mask prompt and *continued* by inference under its own
        id (`PromptGridPredictor._propagate`). Identity is carried by the prompt rather than
        inferred afterwards; the `"canvas"` rule remains the fallback for what the grid then
        discovers.
        Needs the accepted masks' logits, so the canvas keeps a second, float volume.

    Under every mode painting fills only unclaimed voxels, so the first tile to see a voxel decides
    it and a later tile's spill into a neighbour cannot re-label what the neighbour already owns.
    That is also what lets an object larger than a tile survive: cut by every tile edge, each
    fragment overlaps the last one's paint and inherits (or is prompted with) its id. The reference
    instead *discards* masks touching a crop edge -- right for photographs, where the full image is
    a base layer every object fits in, and wrong for a volume tiled because it fits nowhere. That
    rule is available too, as `PromptGridPredictor(edge_discard=True)`, so its cost can be measured.
    """

    def __init__(
        self,
        shape: Sequence[int],
        merge_fraction: float,
        device: torch.device,
        tile_merge: str = "canvas",
    ) -> None:
        if not 0.0 < merge_fraction <= 1.0:
            raise ValueError(f"merge_fraction must be in (0, 1], got {merge_fraction}")
        if tile_merge not in TILE_MERGE:
            raise ValueError(f"tile_merge must be one of {TILE_MERGE}, got {tile_merge!r}")
        self.labels = torch.zeros(tuple(shape), dtype=torch.int64, device=device)
        # Logits of the accepted mask at each painted voxel, kept only when something will read
        # them: a dense prompt built from ids alone would hand the model a binary mask, and it was
        # trained on the previous round's raw logits.
        self.logits = (
            torch.zeros(tuple(shape), dtype=torch.float32, device=device)
            if tile_merge == "propagate" else None
        )
        self.merge_fraction = merge_fraction
        self.tile_merge = tile_merge
        self.next_id = 1

    def _window(self, origin: Sequence[int], shape: Sequence[int]) -> tuple[slice, ...]:
        window = tuple(slice(o, o + e) for o, e in zip(origin, shape, strict=True))
        if tuple(self.labels[window].shape) != tuple(shape):
            raise ValueError(
                f"tile of {tuple(shape)} at {tuple(origin)} does not fit the canvas "
                f"{tuple(self.labels.shape)}"
            )
        return window

    def paint(
        self,
        mask: torch.Tensor,
        identifier: int,
        origin: Sequence[int],
        logits: torch.Tensor | None = None,
    ) -> int:
        """Paint one `(*tile)` mask under a known id into unclaimed voxels -> voxels painted."""
        window = self._window(origin, mask.shape)
        region = self.labels[window]
        fill = mask & (region == 0)
        region[fill] = identifier
        if self.logits is not None:
            if logits is None:
                raise ValueError("this canvas keeps logits, so every painted mask must bring them")
            self.logits[window][fill] = logits[fill]
        return int(fill.sum())

    def add(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        origin: Sequence[int],
        logits: torch.Tensor | None = None,
    ) -> None:
        """Paint one tile's `(N, *tile)` masks at `origin`, best score first, resolving identity.

        `"none"` allocates a fresh id for every mask; the other modes inherit an existing id when
        one covers at least `merge_fraction` of the mask.
        """
        window = self._window(origin, masks.shape[1:])
        region = self.labels[window]
        for index in scores.argsort(descending=True, stable=True).tolist():
            mask = masks[index]
            size = int(mask.sum())
            if size == 0:
                continue
            assigned = 0
            if self.tile_merge != "none":
                beneath = region[mask]
                beneath = beneath[beneath > 0]
                if beneath.numel():
                    ids, counts = beneath.unique(return_counts=True)
                    best = int(counts.argmax())
                    if counts[best].item() / size >= self.merge_fraction:
                        assigned = int(ids[best])
            if assigned == 0:
                assigned = self.next_id
                self.next_id += 1
            self.paint(mask, assigned, origin, None if logits is None else logits[index])

    def fragments_in(
        self, origin: Sequence[int], shape: Sequence[int]
    ) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        """Every object already painted in a tile's region -> `(id, mask, logits)` per object.

        What `"propagate"` hands back to the model. Masks and logits are in the tile's own frame,
        zero outside the fragment, so the prompt asserts only what an earlier tile established.
        """
        if self.logits is None:
            raise ValueError(
                "fragments need the logits canvas; construct with tile_merge='propagate'"
            )
        window = self._window(origin, shape)
        region, logits = self.labels[window], self.logits[window]
        out = []
        for identifier in region.unique().tolist():
            if identifier == 0:
                continue
            fragment = region == identifier
            out.append((identifier, fragment, torch.where(fragment, logits, 0.0)))
        return out

    def claimed(self, origin: Sequence[int], shape: Sequence[int]) -> torch.Tensor:
        """`(*tile)` bool: which of a tile's voxels are already painted."""
        return self.labels[self._window(origin, shape)] > 0

    @property
    def instances(self) -> int:
        return self.next_id - 1


def touches_tile_face(
    masks: torch.Tensor, interior_faces: Sequence[tuple[bool, bool]]
) -> torch.Tensor:
    """`(N,)` bool: does each `(N, *tile)` mask reach a tile face that is NOT the volume's boundary?

    `interior_faces[axis] = (low, high)` says whether that face is interior to the volume. The
    reference's crop rule: a mask cut by a tile edge is a fragment whose whole the neighbouring tile
    can see -- unless the edge is the volume's own, where there is no neighbour and the object
    genuinely ends.
    """
    hit = torch.zeros(masks.shape[0], dtype=torch.bool, device=masks.device)
    for axis, (low, high) in enumerate(interior_faces):
        if low:
            hit |= masks.select(axis + 1, 0).flatten(1).any(dim=1)
        if high:
            hit |= masks.select(axis + 1, -1).flatten(1).any(dim=1)
    return hit


# ---------------------------------------------------------------------------------------------
# The predictor: the pieces above, driven by a trained strategy over `predict.py`'s tiles.
# ---------------------------------------------------------------------------------------------


class PromptGridPredictor:
    """`VolumePredictor` for a promptable strategy: a grid of clicks per tile, reconciled.

    `algorithm` must expose `encode`, `decode_points` and `mask_stride`, which is
    `PromptableSegmentation`'s inference surface. Every threshold is an argument because every
    threshold has to be tuned after training against a score, not copied from the reference.

    How tiles are reconciled is `tile_merge` (see `Canvas`) plus two orthogonal switches:
    `edge_discard` drops masks touching an interior tile face before painting (the reference's
    rule), and `skip_claimed_clicks` leaves out grid points that land on voxels an earlier tile or a
    propagated object already painted -- fewer decodes and fewer duplicate candidates, at the cost
    of the grid's chance to disagree with what is already there.
    """

    def __init__(
        self,
        algorithm: Any,
        *,
        points_per_side: int = 12,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.5,
        stability_thresh: float = 0.8,
        stability_delta: float = 1.0,
        nms_iou: float = 0.7,
        min_mask_voxels: int = 512,
        max_mask_fraction: float = 0.95,
        prefer: str = "whole",
        containment_thresh: float = 0.8,
        merge_fraction: float = 0.5,
        tile_merge: str = "canvas",
        edge_discard: bool = False,
        skip_claimed_clicks: bool = False,
        propagate_min_coverage: float = 0.8,
        consistency_clicks: int = 0,
        consistency_thresh: float = 0.5,
        consistency_pick: str = "top",
        split_tiled_wholes: bool = False,
        tiled_cover_thresh: float = 0.8,
        agree_thresh: float = 0.5,
        min_support: int = 1,
    ) -> None:
        if prefer not in PREFER:
            raise ValueError(f"prefer must be one of {PREFER}, got {prefer!r}")
        if tile_merge not in TILE_MERGE:
            raise ValueError(f"tile_merge must be one of {TILE_MERGE}, got {tile_merge!r}")
        if not 0.0 < agree_thresh <= 1.0:
            raise ValueError(f"agree_thresh must be in (0, 1], got {agree_thresh}")
        if min_support < 1:
            raise ValueError(f"min_support must be at least 1, got {min_support}")
        if points_per_batch < 1:
            raise ValueError(f"points_per_batch must be at least 1, got {points_per_batch}")
        if not 0.0 < propagate_min_coverage <= 1.0:
            raise ValueError(
                f"propagate_min_coverage must be in (0, 1], got {propagate_min_coverage}"
            )
        if consistency_clicks < 0:
            raise ValueError(f"consistency_clicks must not be negative, got {consistency_clicks}")
        if consistency_pick not in CONSISTENCY_PICK:
            raise ValueError(
                f"consistency_pick must be one of {CONSISTENCY_PICK}, got {consistency_pick!r}"
            )
        if not 0.0 < consistency_thresh <= 1.0 or not 0.0 < tiled_cover_thresh <= 1.0:
            raise ValueError("consistency_thresh and tiled_cover_thresh must be in (0, 1]")
        self.consistency_clicks = consistency_clicks
        self.consistency_thresh = consistency_thresh
        self.consistency_pick = consistency_pick
        self.split_tiled_wholes = split_tiled_wholes
        self.tiled_cover_thresh = tiled_cover_thresh
        self.agree_thresh = agree_thresh
        self.min_support = min_support
        self.algorithm = algorithm
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_thresh = stability_thresh
        self.stability_delta = stability_delta
        self.nms_iou = nms_iou
        self.min_mask_voxels = min_mask_voxels
        self.max_mask_fraction = max_mask_fraction
        self.prefer = prefer
        self.containment_thresh = containment_thresh
        self.merge_fraction = merge_fraction
        self.tile_merge = tile_merge
        self.edge_discard = edge_discard
        self.skip_claimed_clicks = skip_claimed_clicks
        self.propagate_min_coverage = propagate_min_coverage

    def settings(self) -> dict[str, Any]:
        """Every knob, for the artifact's attrs -- a labelling is not interpretable without them."""
        return {
            "points_per_side": self.points_per_side,
            "pred_iou_thresh": self.pred_iou_thresh,
            "stability_thresh": self.stability_thresh,
            "stability_delta": self.stability_delta,
            "nms_iou": self.nms_iou,
            "min_mask_voxels": self.min_mask_voxels,
            "max_mask_fraction": self.max_mask_fraction,
            "prefer": self.prefer,
            "containment_thresh": self.containment_thresh,
            "merge_fraction": self.merge_fraction,
            "tile_merge": self.tile_merge,
            "edge_discard": self.edge_discard,
            "skip_claimed_clicks": self.skip_claimed_clicks,
            "propagate_min_coverage": self.propagate_min_coverage,
            "consistency_clicks": self.consistency_clicks,
            "consistency_thresh": self.consistency_thresh,
            "consistency_pick": self.consistency_pick,
            "split_tiled_wholes": self.split_tiled_wholes,
            "tiled_cover_thresh": self.tiled_cover_thresh,
            "agree_thresh": self.agree_thresh,
            "min_support": self.min_support,
        }

    def _bounds(self, extent: Sequence[int]) -> tuple[int, float]:
        cells = math.prod(self.algorithm.mask_stride)
        smallest = max(1, self.min_mask_voxels // cells)
        return smallest, self.max_mask_fraction * math.prod(extent) / cells

    def _passes(
        self, logits: torch.Tensor, ious: torch.Tensor, extent: Sequence[int]
    ) -> torch.Tensor:
        """The three filters, on `(N, *mask grid)` logits and `(N,)` predicted IoUs."""
        min_cells, max_cells = self._bounds(extent)
        area = (logits > 0).flatten(1).sum(-1)
        return (
            (ious >= self.pred_iou_thresh)
            & (stability_score(logits, self.stability_delta) >= self.stability_thresh)
            & (area >= min_cells)
            & (area <= max_cells)
        )

    @torch.no_grad()
    def decode_grid(
        self,
        image: torch.Tensor,
        image_coords: torch.Tensor,
        grid: tuple[int, ...],
        extent: Sequence[int],
        skip: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The click grid against one embedding -> surviving `(N, *)` masks, scores, logits.

        Decode in batches; keep what passes the three filters; then remove nesting and duplicates.
        Filtering happens per batch so that only survivors are ever held as full masks. `skip` is an
        optional `(*mask grid)` bool of voxels whose clicks are to be left out.
        """
        stride = tuple(self.algorithm.mask_stride)
        device = image.device
        points = point_grid(extent, self.points_per_side).to(device)
        if skip is not None:
            cells = (points.round().long() // torch.tensor(stride, device=device)).clamp_min(0)
            cells = torch.minimum(cells, torch.tensor(skip.shape, device=device) - 1)
            points = points[~skip[tuple(cells.T)]]
        coords = voxel_coords(points, extent).unsqueeze(1)
        labels = torch.full((points.shape[0], 1), FOREGROUND, device=device)

        kept: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for start in range(0, points.shape[0], self.points_per_batch):
            batch_coords = coords[start : start + self.points_per_batch]
            batch_labels = labels[start : start + self.points_per_batch]
            logits, ious = self.algorithm.decode_points(
                image.expand(batch_coords.shape[0], -1, -1), image_coords, grid,
                batch_coords, batch_labels, multimask=True,
            )
            logits, ious = logits.flatten(0, 1).float(), ious.flatten()
            survives = self._passes(logits, ious, extent)
            if survives.any():
                kept.append((logits[survives] > 0, ious[survives], logits[survives]))

        shape = tuple(e // s for e, s in zip(extent, stride, strict=True))
        if not kept:
            return (
                torch.zeros((0, *shape), dtype=torch.bool, device=device),
                torch.zeros(0, device=device),
                torch.zeros((0, *shape), device=device),
            )
        masks = torch.cat([k[0] for k in kept])
        scores = torch.cat([k[1] for k in kept])
        logits = torch.cat([k[2] for k in kept])
        # The merge-aware filters, BEFORE containment: a rejected merge must not have already
        # suppressed the parts it was made of. Tiling first, because it costs no decodes and
        # leaves fewer masks for the clicks to check.
        if self.split_tiled_wholes and masks.shape[0]:
            merged = tiled_wholes(
                masks, scores, self.containment_thresh, self.tiled_cover_thresh, self.nms_iou
            )
            masks, scores, logits = masks[~merged], scores[~merged], logits[~merged]
        if self.consistency_clicks > 0 and masks.shape[0]:
            agreement = self._consistency(image, image_coords, grid, extent, masks, logits)
            keep = agreement >= self.consistency_thresh
            masks, scores, logits = masks[keep], scores[keep], logits[keep]
        # Nesting first, duplicates second -- see `resolve_containment` for why the order matters.
        keep = resolve_containment(
            masks, scores, self.prefer, self.containment_thresh, duplicate_iou=self.nms_iou
        )
        masks, scores, logits = masks[keep], scores[keep], logits[keep]
        keep = mask_nms(masks, scores, self.nms_iou)
        return masks[keep], scores[keep], logits[keep]

    @torch.no_grad()
    def _consistency(
        self,
        image: torch.Tensor,
        image_coords: torch.Tensor,
        grid: tuple[int, ...],
        extent: Sequence[int],
        masks: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Ask the model again from inside each mask -> `(N,)` agreement, the min over the clicks.

        The gates judge a mask by itself -- the IoU head's confidence, the boundary's stability --
        and a mask that merges two neighbouring objects passes both: it is a clean, stable,
        plausible object, and nothing about it alone says it is two. This uses the model's own
        promptability as an independent witness. `consistency_clicks` well-separated interior
        points of the mask are clicked one at a time (the grid's prompt type, asked from a
        different place), and each answer is compared with the mask. One object gives back the
        same mask from anywhere inside it; a merge gives back a lobe from a click in that lobe.

        `consistency_pick` decides which of the three candidates an answer is: `"top"`, the one
        the IoU head ranks first, or `"best"`, the one most like the mask. `"best"` only rejects
        masks the model cannot reproduce from inside at all; `"top"` also rejects a merge whose
        lobe the model is more confident about than the whole -- which is the case the diagnostic
        shows, since in this corpus a confident sub-part candidate is itself the anomaly (the
        targets are whole cells, so the head learns to score sub-parts low). Reduced by `min`: an
        object has to be claimed from every part of itself.
        """
        device = masks.device
        stride = torch.tensor(self.algorithm.mask_stride, device=device)
        owners, cells = [], []
        for index in range(masks.shape[0]):
            points = interior_points(masks[index], logits[index], self.consistency_clicks)
            owners.append(torch.full((points.shape[0],), index, device=device))
            cells.append(points)
        owner = torch.cat(owners)
        clicks = torch.cat(cells).to(torch.float32) * stride + (stride - 1).float() / 2
        coords = voxel_coords(clicks, extent).unsqueeze(1)
        labels = torch.full((clicks.shape[0], 1), FOREGROUND, device=device)

        agreement = torch.ones(masks.shape[0], device=device)
        for start in range(0, clicks.shape[0], self.points_per_batch):
            batch_coords = coords[start : start + self.points_per_batch]
            batch_owner = owner[start : start + self.points_per_batch]
            out, ious = self.algorithm.decode_points(
                image.expand(batch_coords.shape[0], -1, -1), image_coords, grid,
                batch_coords, labels[start : start + self.points_per_batch], multimask=True,
            )
            candidates = out.float() > 0                       # (b, K, *grid)
            own = masks[batch_owner].unsqueeze(1)              # (b, 1, *grid)
            inter = (candidates & own).flatten(2).sum(-1).to(torch.float32)
            union = (candidates | own).flatten(2).sum(-1).to(torch.float32)
            iou = inter / union.clamp_min(1)                   # (b, K)
            if self.consistency_pick == "top":
                picked = iou.gather(1, ious.argmax(dim=1, keepdim=True)).squeeze(1)
            else:
                picked = iou.max(dim=1).values
            agreement.scatter_reduce_(0, batch_owner, picked, reduce="amin", include_self=True)
        return agreement

    @torch.no_grad()
    def segment_tile(self, volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One `(1, C, *extent)` tile searched from scratch -> `(N, *stride grid)` masks, scores."""
        image, image_coords, grid = self.algorithm.encode(volume)
        masks, scores, _ = self.decode_grid(image, image_coords, grid, tuple(volume.shape[-3:]))
        return masks, scores

    @torch.no_grad()
    def _propagate(
        self,
        canvas: Canvas,
        origin: Sequence[int],
        image: torch.Tensor,
        image_coords: torch.Tensor,
        grid: tuple[int, ...],
        extent: Sequence[int],
    ) -> int:
        """Continue every object already painted in this tile's region, by inference -> count.

        Each fragment goes back to the model as the dense prompt it was trained to refine from --
        its own accepted logits, zero where nothing is known -- together with one click at the
        fragment's most confident voxel, and is decoded through the single-mask token, the regime
        every mask-prompted training round used. What comes back is painted under the fragment's
        own id, so identity crosses the seam by assertion rather than by an overlap threshold.

        **A continuation is not gated like a discovery.** The grid's masks earn their place through
        the IoU head and the stability score. A propagated mask cannot: two things about the prompt
        sit outside the training distribution -- it is truncated at the previous tile's edge, and
        its click is a redundant interior one where training always supplied one from the error
        region -- and the IoU head, trained to score error-correction rounds, reports 0.2-0.4 for
        them regardless of quality. Measured (`sam3d/propagate_probe.py`): with the grid's gates
        applied, 0 of 34 continuations passed while covering 90-100% of their fragments, and
        propagation was a silent no-op that scored identically to `"canvas"`.

        So a continuation is accepted on the evidence that actually validates it: the fragment's
        identity was already established by a mask that DID pass the gates, and the model has now
        confirmed that fragment by covering at least `propagate_min_coverage` of it. Only the size
        bounds still apply. The mask-only refinement round the training recipe lacks is what would
        make the head trustworthy here and let this gate be retired.
        """
        stride = torch.tensor(self.algorithm.mask_stride, device=image.device)
        shape = tuple(e // int(s) for e, s in zip(extent, stride.tolist(), strict=True))
        min_cells, max_cells = self._bounds(extent)
        painted = 0
        for identifier, fragment, logits in canvas.fragments_in(origin, shape):
            cell = torch.nonzero(logits == logits.max(), as_tuple=False)[0]
            click = (cell * stride + (stride - 1).float() / 2).unsqueeze(0)
            out, _ = self.algorithm.decode_points(
                image, image_coords, grid,
                voxel_coords(click, extent).unsqueeze(1),
                torch.full((1, 1), FOREGROUND, device=image.device),
                multimask=False,
                mask_input=logits[None, None],
            )
            out = out[0, 0].float()
            mask = out > 0
            area = int(mask.sum())
            covered = int((mask & fragment).sum()) / max(int(fragment.sum()), 1)
            if covered >= self.propagate_min_coverage and min_cells <= area <= max_cells:
                painted += canvas.paint(mask, identifier, origin, out)
        return painted

    def run(self, grid: VolumeGrid, device: torch.device) -> VolumePrediction:
        stride = tuple(self.algorithm.mask_stride)
        for name, values in (("output", grid.output_shape), ("patch", grid.patch)):
            if any(v % s for v, s in zip(values, stride, strict=True)):
                raise ValueError(
                    f"{name} shape {tuple(values)} is not divisible by the mask stride {stride}; "
                    "the masks are assembled on the mask grid and every tile must land on it"
                )
        canvas_shape = [v // s for v, s in zip(grid.output_shape, stride, strict=True)]
        # "consensus" keeps every window's labelling and reconciles them once, at the end; the
        # other modes paint into one canvas as they go.
        canvas = (
            None if self.tile_merge == "consensus"
            else Canvas(canvas_shape, self.merge_fraction, device, tile_merge=self.tile_merge)
        )
        collected: list[tuple[tuple[int, ...], torch.Tensor]] = []
        handle = grid.image_handle()
        tiles = grid.tiles
        print(f"{len(tiles)} tiles of {grid.patch} -> output {grid.output_shape}; "
              f"{self.points_per_side}^3 prompts per tile, masks at stride {stride}, "
              f"tile_merge={self.tile_merge}"
              f"{', edge_discard' if self.edge_discard else ''}"
              f"{', skip_claimed_clicks' if self.skip_claimed_clicks else ''}", flush=True)
        for index, (native, out) in enumerate(tiles):
            volume = torch.from_numpy(grid.read_image(handle, native)[None, None]).to(device)
            extent = tuple(volume.shape[-3:])
            origin = [o // s for o, s in zip(out, stride, strict=True)]
            shape = tuple(e // s for e, s in zip(extent, stride, strict=True))
            with torch.autocast(device.type, dtype=torch.bfloat16):
                image, image_coords, token_grid = self.algorithm.encode(volume)
                if canvas is not None and self.tile_merge == "propagate":
                    self._propagate(canvas, origin, image, image_coords, token_grid, extent)
                skip = (
                    canvas.claimed(origin, shape)
                    if canvas is not None and self.skip_claimed_clicks else None
                )
                masks, scores, logits = self.decode_grid(
                    image, image_coords, token_grid, extent, skip=skip
                )
            if self.edge_discard and masks.shape[0]:
                interior = [
                    (o > 0, o + p < full)
                    for o, p, full in zip(out, grid.patch, grid.output_shape, strict=True)
                ]
                keep = ~touches_tile_face(masks, interior)
                masks, scores, logits = masks[keep], scores[keep], logits[keep]
            if canvas is not None:
                canvas.add(masks, scores, origin, logits if canvas.logits is not None else None)
                so_far = canvas.instances
            else:
                collected.append((tuple(origin), tile_labelling(masks, scores)))
                so_far = sum(int(t.max()) for _, t in collected)
            if (index + 1) % 10 == 0 or index + 1 == len(tiles):
                what = "instances" if canvas is not None else "tile masks"
                print(f"  {index + 1}/{len(tiles)}  {so_far} {what} so far", flush=True)
        if canvas is not None:
            labels, instances = canvas.labels, canvas.instances
        else:
            report: dict[str, Any] = {}
            labels, instances = consensus_labelling(
                collected, canvas_shape, self.agree_thresh, self.min_support, report=report
            )
            checks = report["joined"] + report["disagreed"] + report["unmet"]
            mean_iou = report["disagreed_best_iou_sum"] / max(report["disagreed"], 1)
            print(f"  consensus: {report['tile_masks']} tile masks -> {instances} objects; of "
                  f"{checks} mask-in-shared-region checks {report['joined']} joined, "
                  f"{report['disagreed']} disagreed (best IoU {mean_iou:.2f}), "
                  f"{report['unmet']} met no mask in the other window; "
                  f"{report['disputed_cells']} cells disputed", flush=True)
        # Back to voxel resolution with nearest-neighbour: the labelling has no information below
        # the mask stride, and interpolating ids would invent new ones.
        labels = F.interpolate(
            labels[None, None].float(), size=tuple(grid.output_shape), mode="nearest"
        )[0, 0].to(torch.int64)
        array: np.ndarray = labels.cpu().numpy()
        return VolumePrediction(
            array=array,
            kind="instances",
            attrs={
                "background_id": 0,
                "instances": instances,
                "mask_stride": list(stride),
                "generator": "prompt_grid",
                **self.settings(),
            },
        )
