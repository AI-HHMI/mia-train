"""The model-free parts of the SAM data engine, each against an obvious reference.

    python -m pytest experiments/sam_lmd_v1/test_tools.py -q

Run from the repository root so `pyproject.toml` puts `src/` on the path. Everything here is CPU,
seconds, and touches no store under /groups: the block partition is checked for disjointness and
coverage, the nested draw for its prefix property, and a sidecar written to a temporary directory is
read back THROUGH miao's own store opener -- the reader that will consume it in training -- in both
zarr formats the corpus uses.
"""

from __future__ import annotations

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
