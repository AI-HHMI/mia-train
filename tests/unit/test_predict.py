"""`prediction.grid` must read exactly what miao reads, and tile on a lattice that stitches.

The parity test is the important one. `prediction.grid` reproduces miao's read, resample and
normalisation rather than calling into it, because miao offers no "read this exact box" entry point
-- its samplers choose the origin. A reimplementation that drifted would feed the encoder something
subtly unlike its training data and still produce a plausible score, so the two are compared
directly on real volumes: a box narrow enough to leave miao a handful of legal positions, then the
same box read both ways and required to agree to the bit.

Marked `slow` because it opens stores under /groups. Everything else here is pure arithmetic,
plus the `VolumePredictor` dispatch in `prediction.dense` and the `--override` parsing in
`predict.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from prediction.dense import blend_weight
from prediction.grid import aligned_tiling, normalize, resample_labels

pytestmark = pytest.mark.unit

DATA_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "experiments/lmd_ssl_v1/lmd_val_singlescale.yaml"
)


# ----------------------------------------------------------------- the lattice


def test_tiling_lands_on_the_output_lattice():
    """Native stride R/2 must map to output stride exactly patch/2, or tiles cannot be stitched."""
    native, output, extent = aligned_tiling(480, 206, 256)
    assert native == [0, 103, 206]
    assert output == [0, 128, 256]
    assert extent == 512
    # Successive output origins differ by exactly patch/2, with no drift accumulating.
    assert {b - a for a, b in zip(output, output[1:], strict=False)} == {128}


def test_tiling_shrinks_rather_than_pads():
    """The tail that does not complete a stride is dropped: no voxel comes from invented data."""
    native, _, _ = aligned_tiling(480, 206, 256)
    covered = 206 + (len(native) - 1) * 103
    assert covered == 412 <= 480
    # One more tile would need native data past the box.
    assert native[-1] + 206 == covered


def test_an_odd_read_shape_is_refused():
    """An odd read cannot halve into a stride: the lattice would drift half a voxel per tile."""
    with pytest.raises(ValueError, match="must divide by 2"):
        aligned_tiling(480, 205, 256)


def test_a_region_smaller_than_one_patch_is_refused():
    with pytest.raises(ValueError, match="smaller than a patch"):
        aligned_tiling(100, 206, 256)


# ----------------------------------------------------------------- resampling and normalisation


def test_label_resampling_preserves_large_ids():
    """The failure this avoids: float32 cannot hold an id above 2**24.

    miao's nearest path casts to float32, which is how a store with tens of thousands of instances
    can deliver one per patch while the tensor is still int64.
    """
    big = np.array([[[2**40 + 1, 2**40 + 2]]], dtype=np.int64)
    out = resample_labels(big, (1, 1, 4))
    assert set(out.ravel().tolist()) == {2**40 + 1, 2**40 + 2}
    # For contrast: the float32 route collapses them into one id.
    assert len(set(np.float32(big).astype(np.int64).ravel().tolist())) == 1


def test_normalisation_follows_the_volume_not_the_dtype():
    """A windowed uint16 volume must not be divided by 255 or by 65535."""
    block = np.array([100, 219, 569, 919, 2000], dtype=np.uint16)
    windowed = normalize(block, np.dtype(np.uint16), 219.0, 919.0)
    assert windowed[0] == pytest.approx(0.0)          # clamped up to the window floor
    assert windowed[2] == pytest.approx(0.5)
    assert windowed[4] == pytest.approx(1.0)          # clamped down to the ceiling

    plain = normalize(block, np.dtype(np.uint16), None, None)
    assert plain[2] == pytest.approx(569 / 65535)


# ----------------------------------------------------------------- parity with miao


@pytest.mark.slow
@pytest.mark.parametrize(
    "volume", ["liconn_mouse_hippocampus", "kasthuri15_ac4", "liconn_expid82"]
)
def test_reader_matches_miao_exactly(volume: str) -> None:
    """Same patch, read both ways, required to agree to the bit.

    Covers the three shapes of the problem present in this eval set: a uint16 volume with an
    intensity window, an anisotropic uint8 volume whose axes are up-sampled in z and down-sampled in
    x and y at once, and a volume whose storage axis order differs from the config's.
    """
    pytest.importorskip("miao")
    if not DATA_CONFIG.is_file():
        pytest.skip(f"{DATA_CONFIG} not present")
    from miao.config import load_config
    from miao.dataset import VolumeDataset

    from predict import resolve_patch
    from prediction.grid import VolumeGrid

    base = load_config(DATA_CONFIG)
    out_axes = "".join(axis for axis in base.output_axes if axis in "xyz")
    grid = VolumeGrid(base, volume, resolve_patch(base, {}, volume, None))
    read = [int(r) for r in grid.info.scales.read_shapes[0]]

    # Slack of 8 rather than 1: miao's centre bounds are derived from the coarsest window and leave
    # a little more room than the read shape alone, by an amount that differs per volume. A handful
    # of legal positions is fine -- each one is compared at its own origin.
    entry = next(v for v in base.volumes if v.name == volume)
    box_storage = [[grid.box_low[a], grid.box_low[a] + read[a] + 8] for a in range(3)]
    box_output = [box_storage[grid.axes.index(axis)] for axis in out_axes]
    dataset = VolumeDataset(base.model_copy(update={
        "volumes": [entry.model_copy(update={"bounding_box": box_output})],
        "sampling": "sequential", "overlap": 0, "cache_bytes": 1 << 26,
    }))
    assert len(dataset) >= 1

    sample = dataset[0]
    theirs = sample["img"].numpy()[0, 0]
    low_nm = sample["bbox"].numpy()[0, 0]
    read_origin = tuple(
        int(round(low_nm[out_axes.index(axis)] / grid.image_voxel[i]))
        for i, axis in enumerate(grid.axes)
    )

    grid.read = read                        # compare at miao's own read shape, not the even one
    mine = np.transpose(
        grid.read_image(grid.image_handle(), read_origin),
        [grid.axes.index(axis) for axis in out_axes],
    )
    assert mine.shape == theirs.shape
    assert np.abs(mine - theirs).max() == 0.0


def test_blend_weight_generalises_the_cubic_closed_form():
    """A non-cubic patch must still get a centre-weighted map, and a cubic one the old values.

    The old `patch_weight(size)` took one integer and assumed a cube. Models trained at a
    non-cubic patch are legitimate -- an anisotropic volume is the obvious case -- so the weight is
    now per-axis, and this pins that it did not change the cubic answer.
    """
    cubic = blend_weight((6, 6, 6))
    assert cubic.shape == (6, 6, 6)
    # Chessboard distance to the outside: 1 at every face, rising to 3 at the centre of a 6-cube.
    assert cubic[0, 0, 0] == 1.0
    assert cubic[3, 3, 3] == 3.0
    assert cubic[0, 3, 3] == 1.0

    oblong = blend_weight((2, 8, 8))
    assert oblong.shape == (2, 8, 8)
    # The short axis caps the weight everywhere: no voxel is more than 1 from a face along it.
    assert oblong.max() == 1.0


def test_aligned_tiling_with_a_quarter_window_step():
    """`steps_per_patch=4`: the window advances by a quarter, native and output strides in step."""
    native, output, extent = aligned_tiling(480, 208, 256, steps_per_patch=4)
    assert native == [0, 52, 104, 156, 208, 260]          # 208 / 4 = 52 native voxels per step
    assert output == [0, 64, 128, 192, 256, 320]          # 256 / 4 = 64 output voxels per step
    assert extent == 256 + 5 * 64
    # Every native origin maps to its output origin by the one resampling factor 256 / 208.
    for n, o in zip(native, output, strict=True):
        assert n * 256 == o * 208
    # The default reproduces the half-window lattice exactly.
    assert aligned_tiling(480, 206, 256) == aligned_tiling(480, 206, 256, steps_per_patch=2)
    with pytest.raises(ValueError, match="divide by 4"):
        aligned_tiling(480, 206, 256, steps_per_patch=4)


def test_volume_grid_quarter_step_rounds_reads_and_keeps_tiles_on_the_truth_region():
    """The covering invariant of the test below, at a quarter-window step.

    Reads are rounded UP to a multiple of the step count (71 -> 72, 342 -> 344), so every native
    stride is whole and the tiles and `native_box()` still describe one region.
    """
    class FakeInfo:
        img_spatial_axes = "zyx"
        lbl_spatial_axes = "zyx"
        lbl_axes = "zyx"
        bounding_box = np.array([[0, 100], [0, 1024], [0, 1024]])
        img_level_voxels = {0: [29.0, 6.0, 6.0]}
        lbl_level_voxels = {0: [29.0, 6.0, 6.0]}

        class scales:
            chosen_levels = [0]
            label_chosen_levels = [0]
            read_shapes = [np.array([71, 342, 342])]

    from prediction.grid import VolumeGrid

    grid = VolumeGrid.__new__(VolumeGrid)
    grid.volume = None
    grid.patch = [256, 256, 256]
    grid.info = FakeInfo()
    grid.axes = "zyx"
    grid.rank = 3
    grid.image_level = 0
    grid.label_level = 0
    grid.image_voxel = FakeInfo.img_level_voxels[0]
    grid.steps_per_patch = 4
    VolumeGrid._resolve_geometry(grid, None, "fake")

    assert grid.read == [72, 344, 344]
    assert grid.output_origins[1][:3] == [0, 64, 128]
    assert grid.native_origins[1][:3] == [0, 86, 172]
    assert grid.native_origins[0] == [0, 18]                 # 100 slices, 72-slice tiles, stride 18
    counts = [len(o) for o in grid.native_origins]
    assert len(grid.tiles) == counts[0] * counts[1] * counts[2]
    tiles = grid.tiles
    for axis in range(3):
        starts = [native[axis] for native, _ in tiles]
        low, high = min(starts), max(starts) + grid.read[axis]
        assert [low, high] == grid.native_box()[axis]


def test_tiles_and_ground_truth_cover_the_same_region():
    """The invariant whose violation invalidated a whole scoring run.

    `VolumeGrid` decides a region once and two things then read from it: the image tiles, and the
    ground truth. They must be the same region. When centring was added to improve coverage, the
    tiles moved to the middle of the annotated box and `read_ground_truth` was left reading from its
    low corner -- a ~40-voxel offset between prediction and truth. Nothing raised. The scores were
    plausible and meaningless, and every test in the suite still passed: the reader parity tests
    compared image reads against miao, the oracle test compared the ground truth against itself, and
    the synthetic fixtures had no bounding box and so no offset to get wrong.

    Checked without touching a store, so it costs nothing and cannot be skipped for want of data.
    """
    class FakeInfo:
        img_spatial_axes = "zyx"
        lbl_spatial_axes = "zyx"
        lbl_axes = "zyx"
        # A box whose extent does not divide the stride, so the lattice must shrink and therefore
        # has somewhere to be centred. Kasthuri's shape: 100 native z slices, one 72-slice tile.
        bounding_box = np.array([[0, 100], [0, 1024], [0, 1024]])
        img_level_voxels = {0: [29.0, 6.0, 6.0]}
        lbl_level_voxels = {0: [29.0, 6.0, 6.0]}

        class scales:
            chosen_levels = [0]
            label_chosen_levels = [0]
            read_shapes = [np.array([71, 342, 342])]

    from prediction.grid import VolumeGrid

    grid = VolumeGrid.__new__(VolumeGrid)          # geometry only; no store is opened
    grid.volume = None
    grid.patch = [256, 256, 256]
    grid.info = FakeInfo()
    grid.axes = FakeInfo.img_spatial_axes
    grid.rank = 3
    grid.image_level = 0
    grid.label_level = 0
    grid.image_voxel = FakeInfo.img_level_voxels[0]
    VolumeGrid._resolve_geometry(grid, None, "fake")

    assert grid.native_offsets != [0, 0, 0], "the fixture must actually need centring"

    # The region the image tiles cover, per axis, from the tiles themselves.
    tiles = grid.tiles
    for axis in range(3):
        starts = [native[axis] for native, _ in tiles]
        low, high = min(starts), max(starts) + grid.read[axis]
        declared = grid.native_box()[axis]
        assert [low, high] == declared, (
            f"axis {axis}: tiles cover [{low}, {high}) but native_box says {declared}. Ground "
            "truth is read from native_box, so prediction and truth would describe different "
            "regions and every score would be meaningless."
        )


# ---------------------------------------------------------------------------------------------
# The VolumePredictor dispatch. These protect the dense path: every scored run in this repo was
# produced by `predict_volume`, and the dispatch must be a pure wrapper around it.
# ---------------------------------------------------------------------------------------------


class _FakeGrid:
    """Enough of `VolumeGrid` for `predict_volume`: a patch, two overlapping tiles, a fixed read."""

    patch = [8, 8, 8]
    output_shape = (12, 8, 8)
    effective_voxel = [1.0, 1.0, 1.0]
    axes = "zyx"
    box_coverage = 1.0

    def __init__(self, seed: int = 0) -> None:
        generator = np.random.default_rng(seed)
        self._tiles = [((0, 0, 0), (0, 0, 0)), ((4, 0, 0), (4, 0, 0))]
        self._reads = {origin: generator.random((8, 8, 8), dtype=np.float32)
                       for origin, _ in self._tiles}

    @property
    def tiles(self):
        return list(self._tiles)

    def image_handle(self):
        return None

    def read_image(self, handle, origin):
        return self._reads[tuple(origin)]


class _DenseAlgorithm:
    """A dense-output strategy in miniature: three channels of a deterministic function."""

    prediction_kind = "affinity"
    prediction_channels = 3
    squash_convention = "sigmoid(0.2 * logit)"

    def volume_predictor(self):
        return None

    def logits(self, volumes):
        x = volumes[:, 0]
        return torch.stack([x, x * 2 - 1, -x], dim=1)

    @staticmethod
    def squash(logits):
        return torch.sigmoid(0.2 * logits)


@pytest.mark.unit
def test_the_dense_predictor_is_byte_identical_to_the_bare_dense_path():
    from prediction.dense import DensePredictor, predict_volume

    algorithm = _DenseAlgorithm()
    device = torch.device("cpu")
    direct = predict_volume(algorithm, _FakeGrid(), device)
    wrapped = DensePredictor(algorithm).run(_FakeGrid(), device)

    assert wrapped.array.dtype == direct.dtype == np.float16
    assert np.array_equal(wrapped.array, direct), "the wrapper must not touch the numbers"
    assert wrapped.kind == "affinity"
    assert wrapped.attrs == {
        "convention": "sigmoid(0.2 * logit), blended in that space",
        "channels": 3,
    }


@pytest.mark.unit
def test_dispatch_prefers_the_strategy_s_own_predictor_and_falls_back_to_dense():
    from prediction.dense import DensePredictor, select_predictor

    assert isinstance(select_predictor(_DenseAlgorithm()), DensePredictor)

    class _Own:
        def run(self, grid, device):
            raise AssertionError("not called here")

    class _WithOwn(_DenseAlgorithm):
        def volume_predictor(self):
            return _Own()

    assert isinstance(select_predictor(_WithOwn()), _Own)


@pytest.mark.unit
def test_a_strategy_with_neither_predictor_nor_dense_protocol_is_refused_up_front():
    from prediction.dense import select_predictor

    class _Neither:
        def volume_predictor(self):
            return None

    with pytest.raises(SystemExit, match=r"lacks \['logits'.*volume_predictor"):
        select_predictor(_Neither())


@pytest.mark.unit
def test_overrides_patch_the_resolved_record_and_refuse_what_the_run_never_had():
    from predict import apply_overrides

    # Real registered names: the key check is against what the class accepts, so that a knob
    # added after a run was trained can still be set on that run's checkpoint.
    resolved = {
        "algorithm": {"name": "promptable_seg", "kwargs": {"pred_iou_thresh": 0.88,
                                                          "prefer": "part"}},
        "model": {"name": "dinov3_vit3d", "kwargs": {"use_fa4": True}},
        "data": {"name": "d", "kwargs": {}},
    }
    patched = apply_overrides(
        resolved,
        ["algorithm.pred_iou_thresh=0.7", 'algorithm.prefer="whole"', "model.use_fa4=false"],
    )
    assert patched["algorithm"]["kwargs"] == {"pred_iou_thresh": 0.7, "prefer": "whole"}
    assert patched["model"]["kwargs"] == {"use_fa4": False}
    # The caller's record is untouched: it is the run's history, not scratch space.
    assert resolved["algorithm"]["kwargs"]["pred_iou_thresh"] == 0.88

    with pytest.raises(SystemExit, match="accepts no 'typo'"):
        apply_overrides(resolved, ["algorithm.typo=1"])
    # Not in the run's record, but the class takes it: allowed, which is the whole point.
    later = apply_overrides(resolved, ["algorithm.nms_iou=0.6"])
    assert later["algorithm"]["kwargs"]["nms_iou"] == 0.6
    with pytest.raises(SystemExit, match="only"):
        apply_overrides(resolved, ["trainer.lr=1"])
    with pytest.raises(SystemExit, match="expected section.key=value"):
        apply_overrides(resolved, ["pred_iou_thresh=0.7"])
    with pytest.raises(SystemExit, match="not a TOML value"):
        apply_overrides(resolved, ["algorithm.prefer=whole"])  # an unquoted string


def _geometry_only_grid(chosen_levels, label_chosen_levels):
    """A `VolumeGrid` built without a store, as the tiles-vs-ground-truth test above does."""
    class FakeInfo:
        img_spatial_axes = "zyx"
        lbl_spatial_axes = "zyx"
        lbl_axes = "zyx"
        bounding_box = np.array([[0, 100], [0, 1024], [0, 1024]])
        img_level_voxels = {0: [29.0, 6.0, 6.0], 1: [58.0, 12.0, 12.0]}
        lbl_level_voxels = {0: [29.0, 6.0, 6.0]}

        class scales:
            read_shapes = [np.array([71, 342, 342])]

    FakeInfo.scales.chosen_levels = chosen_levels
    FakeInfo.scales.label_chosen_levels = label_chosen_levels

    from prediction.grid import VolumeGrid

    grid = VolumeGrid.__new__(VolumeGrid)
    grid.volume = None
    grid.patch = [256, 256, 256]
    grid.info = FakeInfo()
    grid.axes = FakeInfo.img_spatial_axes
    grid.rank = 3
    grid.image_level = int(chosen_levels[0])
    grid.label_level = (
        int(label_chosen_levels[0]) if label_chosen_levels is not None else None
    )
    grid.image_voxel = FakeInfo.img_level_voxels[grid.image_level]
    return grid


def test_a_volume_without_labels_still_gets_a_lattice():
    """Pseudo-labelling tiles unlabeled volumes: no `label_key`, so no label level to resolve.

    Mirrors the `label_level` branch in `VolumeGrid.__init__`; `read_ground_truth` is where the
    absence of labels is reported, and only when someone asks for them.
    """
    from prediction.grid import VolumeGrid

    grid = _geometry_only_grid(chosen_levels=[0], label_chosen_levels=None)
    VolumeGrid._resolve_geometry(grid, None, "unlabeled")
    assert grid.label_level is None
    assert len(grid.tiles) > 0


def test_a_volume_read_above_level_zero_is_refused_not_mislocated():
    """The lattice is in level-0 voxels and reads the chosen level with them.

    For a chosen level of 1 every tile would be read from twice the intended coordinate and cover
    half the intended extent, and the ground truth -- read in level-0 units -- would describe a
    different region. Nothing else would notice; the scores would be plausible. So it is refused.
    """
    from prediction.grid import VolumeGrid

    grid = _geometry_only_grid(chosen_levels=[1], label_chosen_levels=[0])
    with pytest.raises(SystemExit, match="pyramid level 1"):
        VolumeGrid._resolve_geometry(grid, None, "fine-store")
