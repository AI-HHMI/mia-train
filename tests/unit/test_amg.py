"""The automatic mask generator's pieces, each against a brute-force reference.

Every function here is pure and model-free, which is what makes it checkable exactly: the
reference implementations below are the slow, obvious ones, and the module must agree with them
bit for bit. The end-to-end tests drive the predictor with an *oracle* strategy that answers every
click with the true object under it, so the generator's job reduces to something with a known
answer -- recover every object the grid touched, once each.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from algorithms.promptable.amg import (
    Canvas,
    PromptGridPredictor,
    boxes_intersect,
    mask_boxes,
    mask_nms,
    pairwise_intersection,
    point_grid,
    resolve_containment,
    stability_score,
)


def _random_masks(n: int, shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """Blobby random masks: a few random boxes each, so they overlap in interesting ways."""
    generator = torch.Generator().manual_seed(seed)
    masks = torch.zeros((n, *shape), dtype=torch.bool)
    for index in range(n):
        for _ in range(int(torch.randint(1, 3, (1,), generator=generator))):
            lo = [int(torch.randint(0, s - 1, (1,), generator=generator)) for s in shape]
            hi = [
                int(torch.randint(low + 1, size + 1, (1,), generator=generator))
                for low, size in zip(lo, shape, strict=True)
            ]
            masks[index][tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))] = True
    return masks


@pytest.mark.unit
def test_point_grid_is_cell_centred_and_inside_the_crop():
    points = point_grid((64, 32, 16), 4)
    assert points.shape == (64, 3)
    # Half a cell in from each face, and symmetric about the centre.
    assert points[:, 0].min() == pytest.approx(64 / 8 - 0.5)
    assert points[:, 0].max() == pytest.approx(64 - 64 / 8 - 0.5)
    for axis, extent in enumerate((64, 32, 16)):
        assert (points[:, axis].min() + points[:, axis].max()) == pytest.approx(extent - 1)
    assert (points >= 0).all() and (points[:, 0] < 64).all()


@pytest.mark.unit
def test_stability_is_the_iou_between_two_thresholds():
    logits = torch.zeros(1, 1, 4, 4, 4)
    logits[..., :2, :, :] = 3.0    # 32 cells solidly in
    logits[..., 2, :, :] = 0.5     # 16 cells that flip with the threshold
    logits[..., 3, :, :] = -3.0
    # tight (> 1): 32; loose (> -1): 48.
    assert stability_score(logits, delta=1.0).item() == pytest.approx(32 / 48)
    assert stability_score(torch.full((1, 1, 2, 2, 2), 5.0), delta=1.0).item() == 1.0


@pytest.mark.unit
def test_mask_boxes_match_nonzero_extents_and_empty_masks_overlap_nothing():
    masks = _random_masks(12, (9, 7, 11), seed=1)
    masks[3] = False  # an empty one
    boxes = mask_boxes(masks)
    for index in range(12):
        where = np.nonzero(masks[index].numpy())
        if masks[index].any():
            expected = [[int(w.min()) for w in where], [int(w.max()) + 1 for w in where]]
            assert boxes[index].tolist() == expected
        else:
            assert boxes[index].tolist() == [[0, 0, 0], [0, 0, 0]]
    overlap = boxes_intersect(boxes, boxes)
    assert not overlap[3].any() and not overlap[:, 3].any(), "an empty box must overlap nothing"


@pytest.mark.unit
def test_box_intersection_and_pairwise_intersection_match_brute_force():
    a, b = _random_masks(7, (6, 6, 6), seed=2), _random_masks(5, (6, 6, 6), seed=3)
    boxes_a, boxes_b = mask_boxes(a), mask_boxes(b)
    expected_boxes = torch.zeros(7, 5, dtype=torch.bool)
    expected_inter = torch.zeros(7, 5)
    for i in range(7):
        for j in range(5):
            expected_inter[i, j] = (a[i] & b[j]).sum()
            expected_boxes[i, j] = all(
                min(boxes_a[i, 1, d], boxes_b[j, 1, d]) > max(boxes_a[i, 0, d], boxes_b[j, 0, d])
                for d in range(3)
            )
    assert torch.equal(boxes_intersect(boxes_a, boxes_b), expected_boxes)
    torch.testing.assert_close(pairwise_intersection(a, b, chunk=3), expected_inter)


def _greedy_nms_reference(masks: torch.Tensor, scores: torch.Tensor, threshold: float) -> list[int]:
    order = scores.argsort(descending=True).tolist()
    kept: list[int] = []
    for index in order:
        area = masks[index].sum().item()
        duplicate = False
        for other in kept:
            inter = (masks[index] & masks[other]).sum().item()
            union = area + masks[other].sum().item() - inter
            if union and inter / union > threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(index)
    return kept


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mask_nms_matches_a_brute_force_greedy_reference(seed):
    masks = _random_masks(25, (8, 8, 8), seed=seed)
    # Add near-duplicates so suppression actually fires.
    masks = torch.cat([masks, masks[:6]])
    masks[25:] ^= _random_masks(6, (8, 8, 8), seed=seed + 10) & (torch.rand(6, 8, 8, 8) < 0.05)
    scores = torch.rand(masks.shape[0], generator=torch.Generator().manual_seed(seed))
    assert mask_nms(masks, scores, 0.7).tolist() == _greedy_nms_reference(masks, scores, 0.7)


@pytest.mark.unit
def test_two_crossing_rods_survive_mask_nms_where_box_nms_would_kill_one():
    # The 3D case the reference's box NMS gets wrong: two thin processes crossing a tile have
    # near-identical bounding boxes and almost no voxels in common.
    masks = torch.zeros(2, 16, 16, 16, dtype=torch.bool)
    masks[0, :, 7:9, 7:9] = True          # a rod along x
    masks[1, 7:9, :, 7:9] = True          # a rod along y, through the same centre
    boxes = mask_boxes(masks)
    # Their boxes barely differ in extent along the third axis, and coincide there.
    assert boxes_intersect(boxes, boxes).all()
    kept = mask_nms(masks, torch.tensor([0.9, 0.8]), iou_threshold=0.7)
    assert sorted(kept.tolist()) == [0, 1]


@pytest.mark.unit
def test_containment_keeps_the_level_asked_for():
    cell = torch.zeros(12, 12, 12, dtype=torch.bool)
    cell[1:11, 1:11, 1:11] = True
    nucleus = torch.zeros_like(cell)
    nucleus[4:8, 4:8, 4:8] = True
    other = torch.zeros_like(cell)
    other[0:1, 0:12, 0:12] = True                # a separate slab, contained in nothing
    masks = torch.stack([cell, nucleus, other])
    # The part scores HIGHER: the choice of level must not follow score.
    scores = torch.tensor([0.7, 0.9, 0.8])

    assert sorted(resolve_containment(masks, scores, "whole", 0.8, 0.7).tolist()) == [0, 2]
    assert sorted(resolve_containment(masks, scores, "part", 0.8, 0.7).tolist()) == [1, 2]
    with pytest.raises(ValueError, match="prefer"):
        resolve_containment(masks, scores, "nested", 0.8, 0.7)


@pytest.mark.unit
def test_near_duplicates_are_left_for_nms_not_settled_by_size():
    # Two masks that are almost the same object -- one a voxel-shell larger -- are duplicates, not a
    # nesting. Containment must leave both so NMS can keep the one that SCORED higher, rather than
    # keeping the larger one by construction, which would bias every object toward its fattest
    # candidate.
    small = torch.zeros(12, 12, 12, dtype=torch.bool)
    small[2:10, 2:10, 2:10] = True
    big = torch.zeros_like(small)
    big[1:11, 1:11, 1:11] = True                       # IoU 512/1000 -> above a 0.5 duplicate line
    masks = torch.stack([big, small])
    scores = torch.tensor([0.6, 0.9])                  # the SMALL one is the better mask
    kept = resolve_containment(masks, scores, "whole", 0.8, duplicate_iou=0.5)
    assert sorted(kept.tolist()) == [0, 1], "a duplicate pair is not for this step to decide"
    assert mask_nms(masks[kept], scores[kept], 0.5).tolist() == [1], "NMS keeps the better score"


@pytest.mark.unit
def test_canvas_gives_one_object_one_id_across_overlapping_tiles():
    # A 24-long object seen by two 16-wide tiles overlapping by 8: each tile holds a fragment,
    # and the fragments must end up with ONE id. Then a second object touching the first must NOT
    # be absorbed into it.
    canvas = Canvas((32, 8, 8), merge_fraction=0.5, device=torch.device("cpu"))
    tile_a = torch.zeros(1, 16, 8, 8, dtype=torch.bool)
    tile_a[0, 4:16, 2:6, 2:6] = True                      # object voxels 4..15 in tile A's frame
    canvas.add(tile_a, torch.tensor([0.9]), origin=(0, 0, 0))
    tile_b = torch.zeros(2, 16, 8, 8, dtype=torch.bool)
    tile_b[0, 0:20 - 8, 2:6, 2:6] = True                  # the same object, voxels 8..19 globally
    tile_b[1, 20 - 8:24 - 8, 2:6, 2:6] = True             # a neighbour abutting it, voxels 20..23
    canvas.add(tile_b, torch.tensor([0.9, 0.8]), origin=(8, 0, 0))

    labels = canvas.labels
    assert canvas.instances == 2
    ids_first = labels[4:20, 2:6, 2:6].unique().tolist()
    ids_second = labels[20:24, 2:6, 2:6].unique().tolist()
    assert ids_first == [1], f"one object got {ids_first}"
    assert ids_second == [2], f"the neighbour got {ids_second}"
    assert (labels[24:] == 0).all() and (labels[:4] == 0).all()


@pytest.mark.unit
def test_canvas_lets_the_first_tile_keep_a_voxel_it_already_owns():
    canvas = Canvas((8, 8, 8), merge_fraction=0.5, device=torch.device("cpu"))
    first = torch.zeros(1, 8, 8, 8, dtype=torch.bool)
    first[0, :4] = True
    canvas.add(first, torch.tensor([0.9]), origin=(0, 0, 0))
    # A later, lower-quality mask that spills two planes into the first object: too little overlap
    # to be the same object (2 of 6 planes), so it is new, and it may not repaint what it spills on.
    spill = torch.zeros(1, 8, 8, 8, dtype=torch.bool)
    spill[0, 2:8] = True
    canvas.add(spill, torch.tensor([0.5]), origin=(0, 0, 0))
    assert canvas.labels[:4].unique().tolist() == [1]
    assert canvas.labels[4:].unique().tolist() == [2]


class _Oracle:
    """A promptable strategy that answers every click with the true object beneath it.

    Three candidates per click, as the real decoder gives: the object, the piece of it below the
    click plane (a "part"), and an empty mask. Predicted IoUs are what a calibrated IoU head would
    say -- 0.9 for the whole, the part's TRUE IoU against the whole for the part, 0.0 for empty --
    because the generator relies on that head to settle duplicates, and an oracle that scored a
    75%-part equal to its whole would be modelling a broken head, not a model. The generator must
    keep exactly one mask per object touched, choose the whole over the part, and drop the empty.
    """

    mask_stride = (2, 2, 2)

    def __init__(self, labels: torch.Tensor) -> None:
        self.labels = labels  # (X, Y, Z) at voxel resolution

    def encode(self, volume):
        extent = tuple(volume.shape[-3:])
        grid = tuple(e // 8 for e in extent)
        return torch.zeros(1, 1, 4), torch.zeros(1, 1, 3), grid

    def decode_points(self, image, image_coords, grid, coords, labels, multimask, mask_input=None):
        extent = torch.tensor(self.labels.shape, dtype=torch.float32)
        voxels = ((coords[:, 0] + 1) * extent / 2 - 0.5).round().long()
        pooled_shape = tuple(e // 2 for e in self.labels.shape)
        out = torch.full((coords.shape[0], 3, *pooled_shape), -10.0)
        ious = torch.zeros(coords.shape[0], 3)
        for index, voxel in enumerate(voxels.tolist()):
            identifier = int(self.labels[tuple(voxel)])
            if identifier == 0:
                continue
            whole = (self.labels == identifier)
            part = whole.clone()
            part[voxel[0]:] = False
            for k, mask in enumerate((whole, part)):
                pooled = torch.nn.functional.avg_pool3d(mask.float()[None, None], 2)[0, 0] > 0.5
                out[index, k][pooled] = 10.0
                # A calibrated head: the whole is a good mask; the part is exactly as good as the
                # fraction of the object it covers.
                ious[index, k] = 0.9 if k == 0 else 0.9 * mask.sum().item() / whole.sum().item()
        return out, ious


def _labels_with_blobs() -> torch.Tensor:
    labels = torch.zeros(32, 32, 32, dtype=torch.long)
    labels[2:14, 2:14, 2:14] = 11
    labels[18:30, 18:30, 4:16] = 22
    labels[4:12, 20:30, 20:30] = 33
    return labels


@pytest.mark.unit
def test_the_generator_recovers_every_object_the_grid_touches_exactly_once():
    labels = _labels_with_blobs()
    predictor = PromptGridPredictor(
        _Oracle(labels), points_per_side=8, points_per_batch=16, pred_iou_thresh=0.5,
        stability_thresh=0.5, min_mask_voxels=8, prefer="whole",
    )
    masks, scores = predictor.segment_tile(torch.zeros(1, 1, 32, 32, 32))

    touched = {int(labels[tuple(p.round().long().tolist())]) for p in point_grid((32,) * 3, 8)}
    touched.discard(0)
    assert masks.shape[0] == len(touched) == 3, "one mask per object hit, no parts, no empties"
    # Each surviving mask is a whole object, pooled -- not a half.
    pooled_truth = {
        i: torch.nn.functional.avg_pool3d((labels == i).float()[None, None], 2)[0, 0] > 0.5
        for i in touched
    }
    for mask in masks:
        assert any(torch.equal(mask, truth) for truth in pooled_truth.values())


class _FakeGrid:
    """Two tiles of 32 overlapping by 16 along the first axis, over a 48x32x32 output."""

    patch = [32, 32, 32]
    output_shape = (48, 32, 32)

    def __init__(self, labels: torch.Tensor) -> None:
        self.labels = labels

    @property
    def tiles(self):
        return [((0, 0, 0), (0, 0, 0)), ((16, 0, 0), (16, 0, 0))]

    def image_handle(self):
        return None

    def read_image(self, handle, origin):
        return np.zeros((32, 32, 32), dtype=np.float32)


class _TiledOracle(_Oracle):
    """The oracle, but the labels it consults are the tile's slice of a larger volume."""

    def __init__(self, full: torch.Tensor) -> None:
        super().__init__(full)
        self.full = full
        self.origin = (0, 0, 0)

    def decode_points(self, image, image_coords, grid, coords, labels, multimask, mask_input=None):
        window = tuple(slice(o, o + 32) for o in self.origin)
        self.labels = self.full[window]
        return super().decode_points(
            image, image_coords, grid, coords, labels, multimask, mask_input
        )


@pytest.mark.unit
def test_whole_volume_run_assembles_one_id_per_object_across_tiles():
    full = torch.zeros(48, 32, 32, dtype=torch.long)
    full[4:28, 4:12, 4:12] = 5          # spans both tiles: cut by the first tile's far face
    full[36:46, 16:28, 16:28] = 6       # only in the second tile
    full[2:10, 20:30, 4:14] = 7         # only in the first
    oracle = _TiledOracle(full)
    predictor = PromptGridPredictor(
        oracle, points_per_side=8, points_per_batch=16, pred_iou_thresh=0.5, stability_thresh=0.5,
        min_mask_voxels=8, merge_fraction=0.3,
    )
    grid = _FakeGrid(full)

    # The predictor reads tiles in order; the oracle needs to know which one it is answering for.
    # Hooked on `encode`, which `run()` calls once per tile before any decoding.
    original = oracle.encode
    served: list[int] = []

    def per_tile(volume):
        oracle.origin = grid.tiles[len(served)][0]
        served.append(1)
        return original(volume)

    oracle.encode = per_tile
    result = predictor.run(grid, torch.device("cpu"))

    assert result.kind == "instances"
    assert result.array.shape == (48, 32, 32) and result.array.dtype == np.int64
    assert result.attrs["instances"] == 3
    for identifier in (5, 6, 7):
        predicted = np.unique(result.array[(full == identifier).numpy()])
        assert len(predicted) == 1 and predicted[0] != 0, f"object {identifier} -> {predicted}"
    # Distinct objects got distinct ids, and background stayed background.
    ids = {int(np.unique(result.array[(full == i).numpy()])[0]) for i in (5, 6, 7)}
    assert len(ids) == 3
    assert (result.array[(full == 0).numpy()] == 0).mean() > 0.95


# ---------------------------------------------------------------------------------------------
# Tile reconciliation modes. The controls here are what make the whole-volume comparison
# interpretable: `none` must split a spanning object, `canvas` must join it, `propagate` must join
# it by inference, and the reference's edge rule must lose what it is documented to lose.
# ---------------------------------------------------------------------------------------------

from algorithms.promptable.amg import touches_tile_face  # noqa: E402


def _spanning_tiles():
    """A 24-long object across two 16-wide tiles overlapping by 8, plus a neighbour in tile B."""
    tile_a = torch.zeros(1, 16, 8, 8, dtype=torch.bool)
    tile_a[0, 4:16, 2:6, 2:6] = True
    tile_b = torch.zeros(2, 16, 8, 8, dtype=torch.bool)
    tile_b[0, 0:12, 2:6, 2:6] = True
    tile_b[1, 12:16, 2:6, 2:6] = True
    return tile_a, tile_b


@pytest.mark.unit
def test_tile_merge_none_is_the_control_that_splits_a_spanning_object():
    canvas = Canvas((32, 8, 8), merge_fraction=0.5, device=torch.device("cpu"), tile_merge="none")
    tile_a, tile_b = _spanning_tiles()
    canvas.add(tile_a, torch.tensor([0.9]), origin=(0, 0, 0))
    canvas.add(tile_b, torch.tensor([0.9, 0.8]), origin=(8, 0, 0))
    assert canvas.instances == 3, "every mask a new id: the spanning object is cut at the seam"
    # Tile A painted global 4..16 as id 1; tile B's mask of the same object (global 8..20) could
    # only fill the unclaimed 16..20, under a fresh id -- the seam cuts the object at 16.
    assert canvas.labels[4:16, 2:6, 2:6].unique().tolist() == [1]
    assert canvas.labels[16:20, 2:6, 2:6].unique().tolist() == [2]
    assert canvas.labels[20:24, 2:6, 2:6].unique().tolist() == [3]


@pytest.mark.unit
def test_propagate_canvas_keeps_logits_and_hands_back_fragments_in_the_tile_frame():
    canvas = Canvas(
        (32, 8, 8), merge_fraction=0.5, device=torch.device("cpu"), tile_merge="propagate"
    )
    tile_a, _ = _spanning_tiles()
    logits = torch.full((1, 16, 8, 8), -3.0)
    logits[0, 4:16, 2:6, 2:6] = 5.0
    canvas.add(tile_a, torch.tensor([0.9]), origin=(0, 0, 0), logits=logits)
    assert canvas.logits is not None and canvas.logits[4:16, 2:6, 2:6].unique().tolist() == [5.0]
    assert canvas.logits[0:4].abs().max() == 0, "unpainted voxels carry no logit"

    fragments = canvas.fragments_in(origin=(8, 0, 0), shape=(16, 8, 8))
    assert [f[0] for f in fragments] == [1]
    _, fragment, fragment_logits = fragments[0]
    assert fragment.shape == (16, 8, 8)
    # Only global voxels 8..15 lie in tile B, i.e. B-frame 0..7.
    assert fragment[0:8, 2:6, 2:6].all() and not fragment[8:].any()
    assert fragment_logits[fragment].unique().tolist() == [5.0]
    assert fragment_logits[~fragment].abs().max() == 0, "the prompt asserts only what was painted"

    with pytest.raises(ValueError, match="every painted mask must bring them"):
        canvas.paint(tile_a[0], 7, (0, 0, 0))


@pytest.mark.unit
def test_canvas_without_logits_refuses_fragments():
    canvas = Canvas((8, 8, 8), merge_fraction=0.5, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="tile_merge='propagate'"):
        canvas.fragments_in((0, 0, 0), (8, 8, 8))
    with pytest.raises(ValueError, match="tile_merge"):
        Canvas((8, 8, 8), 0.5, torch.device("cpu"), tile_merge="union_find")


@pytest.mark.unit
def test_touches_tile_face_ignores_faces_on_the_volume_boundary():
    masks = torch.zeros(3, 8, 8, 8, dtype=torch.bool)
    masks[0, 0:3, 3:5, 3:5] = True      # reaches the LOW face on axis 0
    masks[1, 5:8, 3:5, 3:5] = True      # reaches the HIGH face on axis 0
    masks[2, 3:5, 3:5, 3:5] = True      # interior
    both_interior = [(True, True)] * 3
    assert touches_tile_face(masks, both_interior).tolist() == [True, True, False]
    # The low face is the volume's own boundary: nothing lies beyond it, so a mask ending there is
    # not a fragment.
    low_is_boundary = [(False, True), (True, True), (True, True)]
    assert touches_tile_face(masks, low_is_boundary).tolist() == [False, True, False]


@pytest.mark.unit
@pytest.mark.parametrize("tile_merge", ["canvas", "propagate"])
def test_whole_volume_run_joins_a_spanning_object_under_each_merge_mode(tile_merge):
    full = torch.zeros(48, 32, 32, dtype=torch.long)
    full[4:28, 4:12, 4:12] = 5          # spans both tiles
    full[36:46, 16:28, 16:28] = 6
    full[2:10, 20:30, 4:14] = 7
    oracle = _TiledOracle(full)
    predictor = PromptGridPredictor(
        oracle, points_per_side=8, points_per_batch=16, pred_iou_thresh=0.5, stability_thresh=0.5,
        min_mask_voxels=8, merge_fraction=0.3, tile_merge=tile_merge,
    )
    grid = _FakeGrid(full)
    served: list[int] = []
    original_encode = oracle.encode

    def encode_for_tile(volume):
        oracle.origin = grid.tiles[len(served)][0]
        served.append(1)
        return original_encode(volume)

    oracle.encode = encode_for_tile
    result = predictor.run(grid, torch.device("cpu"))
    assert result.attrs["instances"] == 3 and result.attrs["tile_merge"] == tile_merge
    for identifier in (5, 6, 7):
        predicted = np.unique(result.array[(full == identifier).numpy()])
        assert len(predicted) == 1 and predicted[0] != 0, f"object {identifier} -> {predicted}"


@pytest.mark.unit
def test_tile_merge_none_end_to_end_splits_exactly_the_spanning_object():
    full = torch.zeros(48, 32, 32, dtype=torch.long)
    full[4:40, 4:12, 4:12] = 5          # crosses tile A's far face at 32 -> two ids under `none`
    full[36:46, 16:28, 16:28] = 6       # inside tile B only -> one id regardless
    oracle = _TiledOracle(full)
    predictor = PromptGridPredictor(
        oracle, points_per_side=8, points_per_batch=16, pred_iou_thresh=0.5, stability_thresh=0.5,
        min_mask_voxels=8, tile_merge="none",
    )
    grid = _FakeGrid(full)
    served: list[int] = []
    original_encode = oracle.encode

    def encode_for_tile(volume):
        oracle.origin = grid.tiles[len(served)][0]
        served.append(1)
        return original_encode(volume)

    oracle.encode = encode_for_tile
    result = predictor.run(grid, torch.device("cpu"))
    assert len(np.unique(result.array[(full == 5).numpy()])) == 2, "the seam must cut it"
    assert len(np.unique(result.array[(full == 6).numpy()])) == 1


@pytest.mark.unit
def test_skip_claimed_clicks_decodes_fewer_prompts_and_changes_nothing_it_should_not():
    full = torch.zeros(48, 32, 32, dtype=torch.long)
    full[4:28, 4:12, 4:12] = 5
    full[36:46, 16:28, 16:28] = 6
    decodes: dict[bool, int] = {}
    for skip in (False, True):
        oracle = _TiledOracle(full)
        original_decode = oracle.decode_points
        count = [0]

        def counting_decode(image, image_coords, grid, coords, labels, multimask, mask_input=None,
                            _orig=original_decode, _count=count):
            _count[0] += coords.shape[0]
            return _orig(image, image_coords, grid, coords, labels, multimask, mask_input)

        oracle.decode_points = counting_decode
        predictor = PromptGridPredictor(
            oracle, points_per_side=8, points_per_batch=16, pred_iou_thresh=0.5,
            stability_thresh=0.5, min_mask_voxels=8, tile_merge="propagate",
            skip_claimed_clicks=skip,
        )
        grid = _FakeGrid(full)
        served: list[int] = []
        original_encode = oracle.encode

        def encode_for_tile(volume, _o=oracle, _g=grid, _s=served, _e=original_encode):
            _o.origin = _g.tiles[len(_s)][0]
            _s.append(1)
            return _e(volume)

        oracle.encode = encode_for_tile
        result = predictor.run(grid, torch.device("cpu"))
        assert result.attrs["instances"] == 2
        decodes[skip] = count[0]
    assert decodes[True] < decodes[False], "clicks on already-painted voxels were not decoded"


@pytest.mark.unit
def test_a_continuation_is_accepted_on_coverage_not_on_the_iou_head():
    """The oracle here reports predicted IoU 0.0 for every mask, failing the grid's gate outright.

    A grid discovery must be rejected; a propagated continuation of an already-accepted fragment
    must be painted, because what validates it is that it covers the fragment -- the IoU head is out
    of distribution for mask-prompted decodes and would silently veto every continuation.
    """
    full = torch.zeros(48, 32, 32, dtype=torch.long)
    full[4:40, 4:12, 4:12] = 5          # crosses the seam at 32

    class _Unsure(_TiledOracle):
        def decode_points(self, *args, **kwargs):
            out, ious = super().decode_points(*args, **kwargs)
            return out, torch.zeros_like(ious)     # "no confidence" from the head

    oracle = _Unsure(full)
    grid = _FakeGrid(full)
    served: list[int] = []
    original_encode = oracle.encode

    def encode_for_tile(volume):
        oracle.origin = grid.tiles[len(served)][0]
        served.append(1)
        return original_encode(volume)

    oracle.encode = encode_for_tile

    # With pred_iou_thresh=0.0 the GRID accepts everything (so tile A paints the fragment), and the
    # question is purely whether propagation in tile B continues it under the same id.
    predictor = PromptGridPredictor(
        oracle, points_per_side=8, points_per_batch=16, pred_iou_thresh=0.0, stability_thresh=0.0,
        min_mask_voxels=8, tile_merge="propagate", propagate_min_coverage=0.8,
    )
    canvas = Canvas((24, 16, 16), 0.5, torch.device("cpu"), tile_merge="propagate")
    # Tile A's grid paints the object's first half under id 1.
    volume_a = torch.zeros(1, 1, 32, 32, 32)
    image, coords, token_grid = predictor.algorithm.encode(volume_a)
    masks, scores, logits = predictor.decode_grid(image, coords, token_grid, (32, 32, 32))
    canvas.add(masks, scores, (0, 0, 0), logits)
    assert canvas.instances == 1
    # Tile B: propagation alone must extend id 1 across the seam, with the head saying 0.0.
    image, coords, token_grid = predictor.algorithm.encode(volume_a)   # advances the oracle's tile
    painted = predictor._propagate(canvas, (8, 0, 0), image, coords, token_grid, (32, 32, 32))
    assert painted > 0, "a continuation covering its fragment must be painted despite pred IoU 0"
    assert canvas.labels[16:20, 2:6, 2:6].unique().tolist() == [1], "beyond the seam, still id 1"
    assert canvas.instances == 1

    with pytest.raises(ValueError, match="propagate_min_coverage"):
        PromptGridPredictor(oracle, propagate_min_coverage=0.0)
