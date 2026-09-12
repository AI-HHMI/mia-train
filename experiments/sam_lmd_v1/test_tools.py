"""The model-free parts of the SAM data engine, each against an obvious reference.

    python -m pytest experiments/sam_lmd_v1/test_tools.py -q

Run from the repository root so `pyproject.toml` puts `src/` on the path. Everything here is CPU,
seconds, and touches no store under /groups: the block partition is checked for disjointness and
coverage, the nested draw for its prefix property, and a sidecar written to a temporary directory is
read back THROUGH miao's own store opener -- the reader that will consume it in training -- in both
zarr formats the corpus uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from blocks import (  # noqa: E402
    IGNORE,
    block_edges,
    create_label_array,
    create_sidecar,
    nested_choice,
    partition,
    permute_box,
    volume_seed,
)

pytestmark = pytest.mark.unit


def test_block_edges_follow_the_voxel_size_and_clamp_to_the_volume():
    # 512 lattice voxels of 8 nm on a 4 nm store is 1024 native voxels; on a 25 nm axis, 164.
    assert block_edges(512, (8, 8, 8), (4.0, 4.0, 25.0), (5000, 5000, 5000)) == [1024, 1024, 164]
    # A thin axis is spanned whole rather than refused.
    assert block_edges(512, (8, 8, 8), (8.0, 8.0, 8.0), (3000, 3000, 300)) == [512, 512, 300]


def test_partition_is_disjoint_centred_and_covers_whole_cells():
    box = [[10, 1300], [0, 700], [5, 105]]
    cells = partition(box, [512, 512, 300])
    # 2 x 1 x 1 cells: the z axis (100 voxels) is shorter than its edge and holds one cell.
    assert len(cells) == 2
    for cell in cells:
        assert cell[2] == [5, 105]
        assert cell[1][1] - cell[1][0] == 512 and cell[0][1] - cell[0][0] == 512
    # Centred: the 1290 - 1024 = 266 leftover voxels split 133 to each face.
    assert cells[0][0] == [10 + 133, 10 + 133 + 512]
    # Disjoint on the split axis, and every cell inside the box.
    spans = sorted(cell[0] for cell in cells)
    assert spans[0][1] <= spans[1][0]
    for cell in cells:
        for (lo, hi), (blo, bhi) in zip(cell, box, strict=True):
            assert blo <= lo < hi <= bhi


def test_nested_choice_is_a_prefix_of_one_permutation():
    cells = partition([[0, 4096], [0, 4096], [0, 4096]], [512, 512, 512])
    assert len(cells) == 512
    seed = volume_seed("some/volume")
    one = nested_choice(cells, seed, 1)
    four = nested_choice(cells, seed, 4)
    sixteen = nested_choice(cells, seed, 16)
    assert one == four[:1] and four == sixteen[:4]
    assert len({index for index, _ in sixteen}) == 16, "distinct cells"
    # More than exist: everything, once.
    assert len(nested_choice(cells[:3], seed, 100)) == 3
    assert volume_seed("a") != volume_seed("b") and volume_seed("a") == volume_seed("a")


def test_permute_box_reorders_storage_to_xyz():
    zyx = [[1, 2], [3, 4], [5, 6]]
    assert permute_box(zyx, "zyx", "xyz") == [[5, 6], [3, 4], [1, 2]]
    assert permute_box(zyx, "xyz", "xyz") == zyx
    with pytest.raises(ValueError):
        permute_box(zyx, "zyx", "xy")


def _fake_source(root: Path, zarr_version: str) -> Path:
    """A store with the metadata shape a real lmd volume has, in the requested format."""
    source = root / "source.zarr"
    (source / "raw").mkdir(parents=True)
    if zarr_version == "zarr3":
        (source / "zarr.json").write_text(json.dumps({
            "zarr_format": 3, "node_type": "group",
            "attributes": {"ome": {"version": "0.5", "multiscales": [{"axes": []}]}},
        }))
        (source / "raw" / "zarr.json").write_text(json.dumps({
            "zarr_format": 3, "node_type": "group", "attributes": {"ome": {"version": "0.5"}},
        }))
    else:
        (source / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        (source / ".zattrs").write_text(json.dumps({"multiscales": [{"version": "0.4"}]}))
        (source / "raw" / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (source / "raw" / "marker").write_text("the image tree")
    return source


@pytest.mark.parametrize("zarr_version", ["zarr3", "zarr2"])
def test_sidecar_round_trips_through_miao_in_both_formats(tmp_path: Path, zarr_version: str):
    miao_store = pytest.importorskip("miao.store")

    source = _fake_source(tmp_path, zarr_version)
    out = tmp_path / "sidecar.zarr"
    create_sidecar(source, out, "raw", zarr_version)
    assert (out / "raw").is_symlink() and (out / "raw" / "marker").read_text() == "the image tree"
    root_meta = "zarr.json" if zarr_version == "zarr3" else ".zattrs"
    assert (out / root_meta).read_text() == (source / root_meta).read_text()

    axes = [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"]
    array = create_label_array(
        out, "sam_r1", axes, shape=(40, 300, 260), scale=(30.0, 8.0, 8.0),
        translation=(0.0, 0.0, 0.0), zarr_version=zarr_version, chunks=(16, 128, 128),
    )
    block = np.arange(1, 1 + 20 * 130 * 130, dtype=np.int32).reshape(20, 130, 130)
    array[10:30, 100:230, 50:180] = block

    handle = miao_store.open_store(out / "labels" / "sam_r1" / "s0", zarr_version)
    assert tuple(handle.shape) == (40, 300, 260)
    back = np.asarray(handle[10:30, 100:230, 50:180])
    np.testing.assert_array_equal(back, block)
    # Unwritten space reads as the ignore value, from the fill value alone.
    assert int(np.asarray(handle[0:5, 0:5, 0:5]).max()) == IGNORE
    assert np.asarray(handle[0:5, 0:5, 0:5]).dtype == np.int32

    # A second label name accumulates in the listing rather than replacing the first.
    create_label_array(out, "sam_r2", axes, (40, 300, 260), (30.0, 8.0, 8.0), (0.0, 0.0, 0.0),
                       zarr_version)
    listing = (out / "labels" / ("zarr.json" if zarr_version == "zarr3" else ".zattrs")).read_text()
    assert "sam_r1" in listing and "sam_r2" in listing

    # And miao's OME reader accepts the group as a one-level pyramid with the scale we declared.
    from miao.zarr_meta import read_ome_metadata

    meta = read_ome_metadata(out, "labels/sam_r1", zarr_version)
    assert meta.axis_names == ["z", "y", "x"]
    assert list(meta.scales) == [0]
    assert meta.scales[0].scale_factors == [30.0, 8.0, 8.0]
    assert meta.scales[0].shape == [40, 300, 260]


def test_a_nonzero_translation_is_declared_and_a_zero_one_is_not(tmp_path: Path):
    axes = [{"name": a, "type": "space", "unit": "nanometer"} for a in "xyz"]
    source = _fake_source(tmp_path, "zarr3")
    out = tmp_path / "s.zarr"
    create_sidecar(source, out, "raw", "zarr3")
    create_label_array(out, "a", axes, (8, 8, 8), (8.0, 8.0, 8.0), (0.0, 0.0, 0.0), "zarr3")
    create_label_array(out, "b", axes, (8, 8, 8), (8.0, 8.0, 8.0), (4.0, 0.0, 0.0), "zarr3")
    a = json.loads((out / "labels" / "a" / "zarr.json").read_text())
    b = json.loads((out / "labels" / "b" / "zarr.json").read_text())
    kinds = lambda meta: [t["type"] for t in  # noqa: E731
                          meta["attributes"]["ome"]["multiscales"][0]["datasets"][0]
                          ["coordinateTransformations"]]
    assert kinds(a) == ["scale"]
    assert kinds(b) == ["scale", "translation"]


# ------------------------------------------------------------------ scoring pseudo-labels vs truth


def _compare(pred, truth, **kwargs):
    from pseudolabel import compare_labellings

    return compare_labellings(np.asarray(pred), np.asarray(truth), **kwargs)


def test_a_perfect_pseudo_labelling_scores_perfectly():
    truth = np.zeros((20, 20, 20), dtype=np.int64)
    truth[2:10, 2:10, 2:10] = 7
    truth[12:18, 12:18, 2:18] = 9
    pred = np.where(truth == 7, 1, np.where(truth == 9, 2, 0))
    scores = _compare(pred, truth, min_truth_voxels=100)
    assert scores["pseudo_instances"] == 2 and scores["truth_instances"] == 2
    assert scores["precision"] == 1.0 and scores["recall"] == 1.0
    assert scores["merges"] == 0 and scores["claimed_on_background"] == 0.0
    assert scores["mean_best_iou"] == pytest.approx(1.0)


def test_a_merge_is_counted_and_a_missed_object_lowers_recall():
    truth = np.zeros((20, 20, 20), dtype=np.int64)
    truth[2:10, 2:10, 2:10] = 7        # 512 voxels
    truth[10:14, 2:10, 2:10] = 9       # 256 voxels, touching 7
    truth[2:18, 14:18, 2:18] = 11      # 1024 voxels, never predicted
    pred = np.zeros_like(truth)
    pred[2:14, 2:10, 2:10] = 1         # one mask over both 7 and 9
    scores = _compare(pred, truth, min_truth_voxels=100)
    assert scores["merges"] == 1 and scores["merge_rate"] == 1.0
    # Best match is object 7: IoU 512 / 768.
    assert scores["mean_best_iou"] == pytest.approx(512 / 768)
    assert scores["precision"] == 1.0            # 0.667 >= 0.5 -- a merge can still be a "hit"
    assert scores["recall"] == pytest.approx(1 / 3)  # only 7 is recovered at IoU >= 0.5


def test_masks_on_background_are_not_hits_and_huge_truth_ids_do_not_overflow():
    truth = np.zeros((16, 16, 16), dtype=np.int64)
    truth[8:16, 8:16, 8:16] = 2**40 + 3          # a 64-bit segment id, as this corpus stores them
    pred = np.zeros_like(truth)
    pred[0:6, 0:6, 0:6] = 1                       # entirely on background
    pred[8:16, 8:16, 8:16] = 2                    # exactly the object
    scores = _compare(pred, truth, min_truth_voxels=100)
    assert scores["pseudo_instances"] == 2 and scores["precision"] == 0.5
    assert scores["claimed_on_background"] == pytest.approx(216 / (216 + 512))
    assert scores["recall"] == 1.0


def test_no_pseudo_masks_is_reported_not_divided_by_zero():
    truth = np.ones((4, 4, 4), dtype=np.int64)
    scores = _compare(np.zeros_like(truth), truth, min_truth_voxels=1)
    assert scores["pseudo_instances"] == 0 and scores["precision"] == 0.0
    assert scores["truth_instances"] == 1


def test_block_planning_on_a_resolved_volume_is_nested_and_inside_the_box():
    from pseudolabel import plan_blocks

    class Volume:
        name = "parent/crop"

    resolved = {
        "volume": Volume(), "box": [[0, 1000], [20, 3020], [0, 4000]],
        "lattice_nm": [8.0, 8.0, 8.0], "voxel_nm": [25.0, 10.0, 10.0],
    }
    # Edges: 512 * 8 / 25 = 164 in z, 410 in y and x.
    one = plan_blocks(resolved, 512, 1)
    four = plan_blocks(resolved, 512, 4)
    assert one == four[:1]
    for _, box in four:
        assert box[0][1] - box[0][0] == 164 and box[1][1] - box[1][0] == 410
        for (lo, hi), (blo, bhi) in zip(box, resolved["box"], strict=True):
            assert blo <= lo < hi <= bhi


def test_the_miao_box_adds_one_voxel_of_slack_and_clamps_to_the_volume():
    from make_round_config import miao_box, shape_xyz

    # Exactly one 256-tile wide on x: miao would refuse the covered box itself.
    assert miao_box([[72, 328], [1672, 1928], [374, 648]], [2000, 2000, 1364]) == \
        [[71, 329], [1671, 1929], [373, 649]]
    # At the volume's faces the slack is clamped rather than reaching outside the array.
    assert miao_box([[0, 256], [1744, 2000], [0, 100]], [2000, 2000, 100]) == \
        [[0, 257], [1743, 2000], [0, 100]]
    manifest = {"storage_axes": "zyx", "output_spatial_axes": "xyz",
                "spatial_shape_level0": [100, 1024, 2048]}
    assert shape_xyz(manifest) == [2048, 1024, 100]


# ---------------------------------------------------------------- calibration_probe.py, model-free


def _brute_force_scores(pred, hard, soft, clicked_row, part):
    import torch

    n, m = pred.shape[0], hard.shape[0]
    iou = torch.zeros(n, m)
    iou_v = torch.zeros(n, m)
    for i in range(n):
        for j in range(m):
            inter = (pred[i] & hard[j]).sum().float()
            union = (pred[i] | hard[j]).sum().float()
            iou[i, j] = inter / max(float(union), 1.0)
            p = pred[i].float()
            inter_v = (p * soft[j]).sum()
            iou_v[i, j] = inter_v / (p.sum() + soft[j].sum() - inter_v).clamp_min(1e-6)
    best, best_row = iou.max(dim=1)
    best_row = torch.where(best > 0, best_row, torch.full_like(best_row, -1))
    partners = torch.zeros(n, dtype=torch.long)
    clicked = torch.zeros(n)
    for i in range(n):
        area = max(float(pred[i].sum()), 1.0)
        partners[i] = sum(
            int(float((pred[i] & hard[j]).sum()) / area >= part) for j in range(m)
        )
        if clicked_row[i] >= 0:
            clicked[i] = iou[i, clicked_row[i]]
    return {
        "iou_best": best, "iou_voxel": iou_v[torch.arange(n), best_row.clamp_min(0)],
        "iou_clicked": clicked, "best_row": best_row, "partners": partners,
    }


def test_candidate_scores_match_a_brute_force_loop():
    import torch
    from calibration_probe import PART, candidate_scores

    torch.manual_seed(0)
    cells, n, m = 200, 7, 5
    soft = torch.zeros(m, cells)
    # Five disjoint true objects of different sizes, soft-pooled edges at their ends.
    starts = [0, 30, 70, 120, 160]
    for j, s in enumerate(starts):
        soft[j, s:s + 25] = 1.0
        soft[j, s + 25:s + 30] = 0.3            # below the 0.5 threshold -> not in `hard`
    hard = soft > 0.5
    pred = torch.zeros(n, cells, dtype=torch.bool)
    pred[0, 0:25] = True                        # exactly object 0
    pred[1, 0:55] = True                        # objects 0 and 1 merged
    pred[2, 10:20] = True                       # part of object 0
    pred[3, 190:200] = True                     # nothing
    pred[4, 30:50] = True                       # most of object 1
    pred[5] = True                              # everything
    pred[6, 0:2] = True                         # a sliver of object 0
    clicked_row = torch.tensor([0, 0, 0, -1, 1, 2, 0])
    got = candidate_scores(pred, hard, soft, clicked_row)
    want = _brute_force_scores(pred, hard, soft, clicked_row, PART)
    for key in ("iou_best", "iou_voxel", "iou_clicked"):
        assert torch.allclose(got[key].cpu(), want[key], atol=1e-5), key
    assert torch.equal(got["best_row"].cpu(), want["best_row"])
    assert torch.equal(got["partners"].cpu(), want["partners"])
    assert got["partners"][1] == 2 and got["partners"][5] == 5 and got["partners"][3] == 0
    assert torch.equal(got["area"].cpu(), pred.sum(-1).float())


def test_candidate_scores_with_no_true_objects_is_all_zero():
    import torch
    from calibration_probe import candidate_scores

    pred = torch.ones(3, 10, dtype=torch.bool)
    got = candidate_scores(pred, torch.zeros(0, 10, dtype=torch.bool), torch.zeros(0, 10),
                           torch.full((3,), -1))
    assert torch.equal(got["best_row"], torch.full((3,), -1))
    assert float(got["iou_best"].sum()) == 0.0 and int(got["partners"].sum()) == 0


def test_summarize_records_reads_the_gate_and_the_oracle_correctly():
    from calibration_probe import FIELDS, summarize_records

    # Two clicks, three candidates each. Click 0 is on object 7; click 1 is off-object.
    rows = [
        # FIELDS order: tile click cand click_id click_size pred stab area size_ok
        #               iou_best iou_voxel iou_clicked best_id best_size partners passes
        (0, 0, 0, 7, 100, 0.90, 0.9, 50, 1, 0.20, 0.18, 0.20, 7, 100, 1, 1),  # top pick, bad
        (0, 0, 1, 7, 100, 0.60, 0.9, 50, 1, 0.80, 0.75, 0.80, 7, 100, 1, 0),  # good, rated low
        (0, 0, 2, 7, 100, 0.75, 0.5, 50, 1, 0.55, 0.50, 0.55, 7, 100, 1, 0),   # fails stability
        (0, 1, 0, 0, 0, 0.85, 0.9, 50, 1, 0.10, 0.10, 0.00, 3, 40, 2, 1),      # off-object, passes
        (0, 1, 1, 0, 0, 0.10, 0.9, 50, 1, 0.00, 0.00, 0.00, 0, 0, 0, 0),
        (0, 1, 2, 0, 0, 0.20, 0.9, 5, 0, 0.00, 0.00, 0.00, 0, 0, 0, 0),         # too small
    ]
    arrays = {field: np.array([r[i] for r in rows]) for i, field in enumerate(FIELDS)}
    report = summarize_records(arrays, pred_iou=0.7, stability=0.8)
    assert report["clicks"] == 2 and report["clicks_on_object"] == 0.5
    assert report["passing"]["candidates"] == 2          # rows 0 and 3
    assert report["passing"]["precision"] == 0.0         # neither reaches 0.5
    assert report["passing"]["from_offobject_clicks"] == 0.5
    assert report["passing"]["merge_rate"] == 0.5        # row 3 has two partners
    assert report["per_click"]["head_top_at_least_0.5_on_object"] == 0.0   # top pick was row 0
    assert report["per_click"]["oracle_at_least_0.5_on_object"] == 1.0     # row 1 exists
    assert report["per_click"]["offobject_max_pred_iou_at_least_gate"] == 1.0
    sweep = {row["pred_iou"]: row for row in report["thresholds"]}
    assert sweep[0.6]["kept"] == 3 and abs(sweep[0.6]["precision"] - 1 / 3) < 1e-9
    assert sweep[0.9]["kept"] == 1
    top_bin = report["calibration"][9]
    assert top_bin["candidates"] == 1 and abs(top_bin["mean_true_iou"] - 0.2) < 1e-9


# ------------------------------------------------------------------ pseudolabel.compare_labellings


def test_compare_labellings_counts_merges_fragments_share_and_purity():
    from pseudolabel import compare_labellings

    # 400 voxels as a (4, 10, 10) volume; ids deliberately non-dense.
    truth = np.zeros(400, dtype=np.int64)
    truth[0:100], truth[100:200], truth[200:300] = 11, 22, 33   # three objects, then background
    pred = np.zeros(400, dtype=np.int64)
    pred[0:60], pred[60:100] = 1, 2                                     # object 11 split 60 / 40
    pred[100:300] = 3                                                   # objects 22 and 33 merged
    pred[300:350] = 4                                                   # background only
    r = compare_labellings(pred.reshape(4, 10, 10), truth.reshape(4, 10, 10), min_truth_voxels=1)

    assert r["pseudo_instances"] == 4 and r["truth_instances"] == 3
    assert r["precision"] == pytest.approx(0.5)        # ids 1 (IoU 0.6) and 3 (IoU 0.5) reach 0.5
    assert r["recall"] == pytest.approx(1.0)           # every object has a mask at IoU >= 0.5
    assert r["merges"] == 1 and r["merge_rate"] == pytest.approx(0.25)
    assert r["fragments"] == 1 and r["fragment_rate"] == pytest.approx(1 / 3)
    assert r["truth_best_share"] == pytest.approx((0.6 + 1.0 + 1.0) / 3)
    assert r["pseudo_purity"] == pytest.approx((1.0 + 1.0 + 0.5 + 1.0) / 4)
    assert r["claimed_on_background"] == pytest.approx(50 / 350)
    assert r["mean_best_iou"] == pytest.approx((0.6 + 0.4 + 0.5 + 0.0) / 4)


def test_compare_labellings_with_nothing_claimed_reports_zeros_for_every_field():
    from pseudolabel import compare_labellings

    truth = np.ones((2, 4, 4), dtype=np.int64)
    r = compare_labellings(np.zeros_like(truth), truth, min_truth_voxels=1)
    for key in ("precision", "recall", "merge_rate", "fragment_rate", "truth_best_share",
                "pseudo_purity", "mean_best_iou"):
        assert r[key] == 0.0
    assert r["merges"] == 0 and r["fragments"] == 0 and r["pseudo_instances"] == 0


# ------------------------------------------------------------------------------ oracle assembly


def _truth_line(z_lo: int, z_hi: int, value: int, shape=(8, 8, 48)) -> np.ndarray:
    truth = np.zeros(shape, dtype=np.int64)
    truth[:, :, z_lo:z_hi] = value
    return truth


def test_oracle_tile_masks_are_the_components_at_mask_resolution():
    import torch
    from pseudolabel import oracle_tile_masks

    truth = _truth_line(4, 12, 7, shape=(8, 8, 16))
    truth[:, :, 12:16] = 9
    masks, scores = oracle_tile_masks(torch.from_numpy(truth), (4, 4, 4), min_voxels=1)
    assert masks.shape == (2, 2, 2, 4) and scores.tolist() == [1.0, 1.0]
    rows = sorted(masks[:, 0, 0].int().tolist())
    assert rows == [[0, 0, 0, 1], [0, 1, 1, 0]]          # z 12..16 -> cell 3; z 4..12 -> cells 1, 2
    tiny = np.zeros((8, 8, 16), dtype=np.int64)
    tiny[0, 0, 0] = 3                                     # a component below min_voxels is no mask
    masks, _ = oracle_tile_masks(torch.from_numpy(tiny), (4, 4, 4), min_voxels=8)
    assert masks.shape[0] == 0


def _lattice(step: int = 8):
    """Windows of 16 along z at `step`, one window wide in x and y; output 8 x 8 x 32."""
    patch, output = [8, 8, 16], (8, 8, 32)
    origins = list(range(0, output[2] - patch[2] + 1, step))
    tiles = [((0, 0, z), (0, 0, z)) for z in origins]
    return tiles, patch, output


def test_oracle_consensus_reassembles_a_long_object_perfectly():
    import torch
    from pseudolabel import assemble_oracle, compare_labellings

    tiles, patch, output = _lattice()
    truth = torch.from_numpy(_truth_line(4, 28, 5, shape=output))       # spans all three windows
    labels, count = assemble_oracle(tiles, patch, output, truth, stride=(4, 4, 4),
                                    tile_merge="consensus", min_mask_voxels=1)
    assert count == 1
    scores = compare_labellings(labels.numpy(), truth.numpy(), min_truth_voxels=1)
    assert scores["precision"] == 1.0 and scores["recall"] == 1.0 and scores["fragments"] == 0


def test_oracle_recall_knob_is_deterministic_and_only_ever_loses_pieces():
    import torch
    from pseudolabel import assemble_oracle, compare_labellings

    tiles, patch, output = _lattice()
    truth = torch.from_numpy(_truth_line(4, 28, 5, shape=output))
    kwargs = dict(stride=(4, 4, 4), tile_merge="consensus", min_mask_voxels=1, recall=0.5)
    a, _ = assemble_oracle(tiles, patch, output, truth, seed=3, **kwargs)
    b, _ = assemble_oracle(tiles, patch, output, truth, seed=3, **kwargs)
    assert torch.equal(a, b)
    outcomes = set()
    for seed in range(12):
        labels, _ = assemble_oracle(tiles, patch, output, truth, seed=seed, **kwargs)
        scores = compare_labellings(labels.numpy(), truth.numpy(), min_truth_voxels=1)
        # Perfect masks can be missing, never wrong: no merge, every mask pure.
        assert scores["merges"] == 0 and scores["pseudo_purity"] in (0.0, 1.0)
        outcomes.add((round(scores["precision"], 3), scores["fragments"]))
    assert len(outcomes) > 1


def test_every_cli_subcommand_is_dispatched(monkeypatch, tmp_path):
    """The dispatch table must know every subparser: the oracle sweep once died on a KeyError."""
    import pseudolabel

    seen = []
    for name in pseudolabel.COMMANDS:
        monkeypatch.setattr(pseudolabel, f"cmd_{name}", lambda args, name=name: seen.append(name))
    for name in pseudolabel.COMMANDS:
        pseudolabel.dispatch(argparse.Namespace(command=name))
    assert seen == list(pseudolabel.COMMANDS)
    # And every subparser main() builds is one of them.
    captured = []
    real = argparse.ArgumentParser.add_subparsers

    def spy(self, **kwargs):
        action = real(self, **kwargs)
        original = action.add_parser

        def add_parser(name, **kw):
            captured.append(name)
            return original(name, **kw)

        action.add_parser = add_parser
        return action

    monkeypatch.setattr(argparse.ArgumentParser, "add_subparsers", spy)
    monkeypatch.setattr(pseudolabel, "dispatch", lambda args: None)
    monkeypatch.setattr("sys.argv", ["pseudolabel.py", "summarize", str(tmp_path)])
    pseudolabel.main()
    assert set(captured) == set(pseudolabel.COMMANDS)
