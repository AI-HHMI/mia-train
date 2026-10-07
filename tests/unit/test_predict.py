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
from typing import Any

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
NISB_CONFIG = Path(__file__).resolve().parents[2] / "configs/data/nisb_base.yaml"
# A light-sheet time series stored t, c, z, y, x (48 frames, one channel); each label array
# annotates a single frame, so it is predicted one frame at a time via `fixed_axes`.
TIMESERIES = Path(
    "/groups/miaai/miaai/lmd-v0.0.1/data/lm-zebrafish-Betzig-mosaic-example_annotations_Thayer"
    "/crop-001_dsr_timeseries_48t_1c.zarr"
)
TIMESERIES_LABEL = "labels/manual_gt-cell-t10_postproofread"


# ----------------------------------------------------------------- the lattice


def test_tiling_lands_on_the_output_lattice():
    """Native stride R/2 must map to output stride exactly patch/2, or tiles cannot be stitched."""
    native, output, extent = aligned_tiling(480, 206, 256)
    assert native == [0, 103, 206]
    assert output == [0, 128, 256]
    assert extent == 512
    # Successive output origins differ by exactly patch/2, with no drift accumulating.
    assert {b - a for a, b in zip(output, output[1:], strict=False)} == {128}


def test_cover_reaches_the_box_at_the_stores_own_resolution():
    """read == patch with `cover`: a last tile flush with the far face covers the box, real data.

    The NISB case: 1350 z slices at a 256 patch stop at 1280 on half-window strides, and a 1024 x
    1024 x 512 window covers 2560 of 3000 -- so two patch sizes were scored on different regions.
    """
    native, output, extent = aligned_tiling(1000, 256, 256, cover=True)
    assert native == output == [0, 128, 256, 384, 512, 640, 744]
    assert extent == 1000
    assert aligned_tiling(1350, 256, 256, cover=True)[0][-1] + 256 == 1350
    assert aligned_tiling(3000, 1024, 1024, cover=True) == (
        [0, 512, 1024, 1536, 1976], [0, 512, 1024, 1536, 1976], 3000)
    # A box the strides already fill gains no tile.
    assert aligned_tiling(896, 256, 256, cover=True) == ([0, 128, 256, 384, 512, 640],) * 2 + (896,)
    # Off by default: the lattice every existing table was scored on.
    assert aligned_tiling(1000, 256, 256) == ([0, 128, 256, 384, 512, 640],) * 2 + (896,)
    # And on a resampled axis it changes nothing either way.
    assert aligned_tiling(480, 206, 256, cover=True) == aligned_tiling(480, 206, 256)


def test_tiling_shrinks_rather_than_pads():
    """Resampled: the tail that does not complete a stride is dropped, no voxel from invented data.

    A tile flush with the far face would start at native 274, which maps to output 340.3 -- off the
    lattice the other tiles share.
    """
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
    ("config", "volume", "box"),
    [
        (DATA_CONFIG, "liconn_mouse_hippocampus", None),
        (DATA_CONFIG, "kasthuri15_ac4", None),
        (DATA_CONFIG, "liconn_expid82", None),
        # The training config leaves NISB's cubes unboxed, and the grid needs a box.
        (NISB_CONFIG, "NISB base seed0", [[0, 3000], [0, 3000], [0, 1350]]),
    ],
)
def test_reader_matches_miao_exactly(config: Path, volume: str, box: list[list[int]] | None) -> None:
    """Same patch, read both ways: what the model is handed must be miao's sample, to the bit.

    Covers the four shapes of the problem present in these configs: a uint16 volume with an
    intensity window, an anisotropic uint8 volume whose axes are up-sampled in z and down-sampled in
    x and y at once, a volume whose storage axis order differs from the config's, and a store with
    a channel axis (NISB's `raw` is c, x, y, z), which the reader once indexed as if it had none.
    The third is why the comparison goes through `AxisOrder`, the transform inference applies: this
    test once transposed predict.py's read by hand before comparing, so the values agreed while the
    model, which never got that transpose, was handed every such tile with z and x exchanged.
    """
    pytest.importorskip("miao")
    if not config.is_file():
        pytest.skip(f"{config} not present")
    from miao.config import load_config

    base = load_config(config)
    if box is not None:
        base = base.model_copy(update={"volumes": [
            v.model_copy(update={"bounding_box": box}) if v.name == volume else v
            for v in base.volumes
        ]})
    _assert_reader_matches_miao(base, volume)


def _assert_reader_matches_miao(base: Any, volume: str) -> None:
    """One patch of `volume`, read by predict.py's reader and by miao, must agree to the bit."""
    from miao.dataset import VolumeDataset

    from predict import resolve_patch
    from prediction.grid import AxisOrder, VolumeGrid

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
    mine = AxisOrder(grid.axes, out_axes).to_model(
        grid.read_image(grid.image_handle(), read_origin)
    )
    assert mine.shape == theirs.shape
    assert np.abs(mine - theirs).max() == 0.0


def _timeseries_config(fixed_axes: dict[str, Any]) -> Any:
    """The time series as one entry pinned by `fixed_axes`, or a skip without it or its miao."""
    miao_config = pytest.importorskip("miao.config")
    if "fixed_axes" not in miao_config.VolumeConfig.model_fields:
        pytest.skip("the installed miao predates fixed_axes")
    if not TIMESERIES.is_dir():
        pytest.skip(f"{TIMESERIES} not present")
    return miao_config.MiaoConfig(
        volumes=[{
            "name": "timeseries", "path": str(TIMESERIES), "image_key": "raw",
            "label_key": TIMESERIES_LABEL, "zarr_version": "zarr3", "fixed_axes": fixed_axes,
            "bounding_box": [[0, 152], [0, 508], [0, 1466]],
        }],
        resolutions=[[200, 108, 108]], output_axes="lczyx", patch_size=[64, 128, 128],
    )


@pytest.mark.slow
def test_a_pinned_frame_is_read_as_miao_reads_it() -> None:
    """A time series pinned to one frame: its tiles, ground truth and provenance are that frame's.

    miao describes such a volume with the pinned axis removed, so a reader that opened the arrays
    without the pin would index t, c and z with the z, y, x window. The tiles must equal miao's
    sample to the bit, the ground truth must be frame 10 of the label array, and the attrs must
    name the frame: every frame shares the store path, so nothing else tells them apart.
    """
    base = _timeseries_config({"t": 10})
    _assert_reader_matches_miao(base, "timeseries")

    import zarr

    from predict import resolve_patch, shared_attrs
    from prediction.grid import VolumeGrid

    grid = VolumeGrid(base, "timeseries", resolve_patch(base, {}, "timeseries", None))
    window = tuple(slice(low, high) for low, high in grid.native_box())
    labels = zarr.open_array(str(TIMESERIES / TIMESERIES_LABEL / "s0"), mode="r")
    assert np.array_equal(grid.read_ground_truth(), np.asarray(labels[(10, 0, *window)]))
    assert shared_attrs(grid, Path("run"), 0, Path("data.yaml"))["source_fixed_axes"] == {"t": 10}


@pytest.mark.slow
def test_an_entry_pinning_several_frames_is_refused() -> None:
    """miao expands `fixed_axes: {t: [1, 10]}` into two volumes, and a prediction is one."""
    base = _timeseries_config({"t": [1, 10]})

    from predict import resolve_patch
    from prediction.grid import VolumeGrid

    with pytest.raises(SystemExit, match="give each frame its own entry"):
        VolumeGrid(base, "timeseries", resolve_patch(base, {}, "timeseries", None))


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


@pytest.mark.parametrize("cover_box", [False, True])
def test_volume_grid_covers_the_box_where_the_model_sees_native_voxels(cover_box):
    """8 nm store, 8 nm model, a box the strides do not fill.

    With `cover_box` the grid is the box, nothing centred; without, the centred sub-box -- the
    gary_comparison case, a 1000^3 test box scored on its central 896^3 (71.9%).
    """
    class FakeInfo:
        img_spatial_axes = "zyx"
        lbl_spatial_axes = "zyx"
        lbl_axes = "zyx"
        bounding_box = np.array([[0, 300], [0, 1000], [0, 1000]])
        img_level_voxels = {0: [8.0, 8.0, 8.0]}
        lbl_level_voxels = {0: [8.0, 8.0, 8.0]}

        class scales:
            chosen_levels = [0]
            label_chosen_levels = [0]
            read_shapes = [np.array([128, 256, 256])]

    from prediction.grid import VolumeGrid

    grid = VolumeGrid.__new__(VolumeGrid)
    grid.volume = None
    grid.patch = [128, 256, 256]
    grid.info = FakeInfo()
    grid.axes = "zyx"
    grid.rank = 3
    grid.image_level = 0
    grid.label_level = 0
    grid.image_voxel = FakeInfo.img_level_voxels[0]
    grid.steps_per_patch = 2
    grid.cover_box = cover_box
    VolumeGrid._resolve_geometry(grid, None, "fake")

    if cover_box:
        assert grid.covers_full_box and grid.box_coverage == 1.0
        assert grid.native_box() == [[0, 300], [0, 1000], [0, 1000]]
        assert tuple(grid.output_shape) == (300, 1000, 1000)
        assert grid.native_origins[0] == [0, 64, 128, 172]   # 172 = 300 - 128, flush with the face
    else:
        assert not grid.covers_full_box
        assert grid.native_box() == [[22, 278], [52, 948], [52, 948]]     # centred, 256 and 896
        assert tuple(grid.output_shape) == (256, 896, 896)
    for axis in range(3):
        starts = [native[axis] for native, _ in grid.tiles]
        assert [min(starts), max(starts) + grid.read[axis]] == grid.native_box()[axis]


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
    input_axes = "lczyx"                         # trained in the fake grid's own order
    offsets = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

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
        "model_axes": "zyx",
        "offsets": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }


@pytest.mark.unit
def test_axes_are_matched_by_name_and_an_affinity_channel_follows_its_axis():
    from prediction.grid import AxisOrder

    order = AxisOrder("zyx", "xyz")                 # a z, y, x store; a model trained x, y, z
    store = np.arange(2 * 3 * 4).reshape(2, 3, 4)          # z = 2, y = 3, x = 4
    assert order.to_model(store).shape == (4, 3, 2)
    assert np.array_equal(order.to_storage(order.to_model(store)), store)
    assert order.shape_to_storage((4, 3, 2)) == (2, 3, 4)
    six = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (10, 0, 0), (0, 10, 0), (0, 0, 10))
    # The model's +z channel (2) is the store's first axis, and so on.
    assert order.channel_order(six) == [2, 1, 0, 5, 4, 3]
    same = AxisOrder("xyz", "xyz")
    assert same.to_model(store) is store and same.channel_order(six) == list(range(6))
    with pytest.raises(SystemExit, match="not the same axes"):
        AxisOrder("zyx", "xyc")
    with pytest.raises(SystemExit, match="no counterpart"):
        order.channel_order(((1, 0, 0), (0, 1, 0), (0, 0, 3)))


class _StoreZYX:
    """One tile over a 6 x 8 x 10 store laid out z, y, x -- not the order the model trained in."""

    patch = [6, 8, 10]
    output_shape = (6, 8, 10)
    effective_voxel = [1.0, 1.0, 1.0]
    axes = "zyx"
    box_coverage = 1.0

    def __init__(self) -> None:
        self.read = np.random.default_rng(0).random((6, 8, 10), dtype=np.float32)

    @property
    def tiles(self):
        return [((0, 0, 0), (0, 0, 0))]

    def image_handle(self):
        return None

    def read_image(self, handle, origin):
        return self.read


class _Directional:
    """Trained on x, y, z: each channel the input minus its neighbour along one of ITS axes."""

    prediction_kind = "affinity"
    prediction_channels = 6
    squash_convention = "identity"
    input_axes = "lcxyz"
    offsets = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (0, 2, 0), (0, 0, 2))

    def __init__(self) -> None:
        self.seen: list[torch.Tensor] = []

    def volume_predictor(self):
        return None

    def logits(self, volumes):
        self.seen.append(volumes.clone())
        x = volumes[:, 0]
        return torch.stack([x - torch.roll(x, -distance, dims=1 + axis)
                            for distance in (1, 2) for axis in range(3)], dim=1)

    @staticmethod
    def squash(logits):
        return logits


@pytest.mark.unit
def test_a_tile_reaches_the_model_in_its_training_order_and_returns_in_the_stores():
    """A model trained on x, y, z tiles, a store laid out z, y, x: the model must be handed the tile
    transposed, as miao handed it every training sample, and its channels must come back meaning
    "along the store's axis i" -- the layout an affinity artifact promises mutex watershed."""
    from prediction.dense import predict_volume

    grid, algorithm = _StoreZYX(), _Directional()
    out = predict_volume(algorithm, grid, torch.device("cpu"))
    assert torch.equal(algorithm.seen[0][0, 0], torch.from_numpy(grid.read.transpose(2, 1, 0)))
    assert out.shape == (6, 6, 8, 10)
    for channel, (distance, axis) in enumerate((d, a) for d in (1, 2) for a in range(3)):
        expected = grid.read - np.roll(grid.read, -distance, axis=axis)
        assert np.allclose(out[channel], expected, atol=2e-3), f"channel {channel}"


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


def test_an_ome_artifact_carries_the_lattice_geometry_and_its_attrs(tmp_path):
    """One level, axes in storage order, scale = lattice voxel, translation = first voxel's
    centre; the attrs on the group and again on the array."""
    import zarr

    from prediction.artifact import LEVEL, multiscales, write_ome_artifact

    labels = np.arange(2 * 3 * 4, dtype=np.uint32).reshape(2, 3, 4)
    path = write_ome_artifact(
        tmp_path / "v.zarr", labels, axes="zyx", voxel_nm=[8.0, 8.0, 8.0],
        translation_nm=[112.0, 504.0, 504.0], attrs={"kind": "instances", "background_id": 0},
    )
    group = zarr.open_group(str(path), mode="r")
    ome = dict(group.attrs)["ome"]
    assert ome["version"] == "0.5" and len(ome["multiscales"]) == 1
    (dataset,) = ome["multiscales"][0]["datasets"]
    assert dataset["path"] == LEVEL
    assert dataset["coordinateTransformations"] == [
        {"type": "scale", "scale": [8.0, 8.0, 8.0]},
        {"type": "translation", "translation": [112.0, 504.0, 504.0]},
    ]
    assert [a["name"] for a in ome["multiscales"][0]["axes"]] == ["z", "y", "x"]
    assert dict(group.attrs)["kind"] == "instances"
    assert dict(group[LEVEL].attrs)["kind"] == "instances"
    assert np.array_equal(group[LEVEL][:], labels) and group[LEVEL].dtype == np.uint32

    # A channel-first array gets a leading channel axis with unit scale.
    ms = multiscales("zyx", [8.0, 8.0, 8.0], [0.0, 0.0, 0.0], channels=True, name="a")
    assert ms["multiscales"][0]["axes"][0] == {"name": "c", "type": "channel"}
    transforms = ms["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    assert transforms[0]["scale"] == [1.0, 8.0, 8.0, 8.0]
    with pytest.raises(ValueError, match="rank"):
        write_ome_artifact(tmp_path / "bad.zarr", np.zeros((2, 2)), axes="zyx", voxel_nm=[1, 1, 1],
                           translation_nm=[0, 0, 0], attrs={})


@pytest.mark.unit
def test_ome_geometry_places_the_first_lattice_voxel_by_the_stores_convention():
    """Lattice voxel 0 spans native voxels [low, low + v/e); its centre is low*v + (e - v)/2,
    plus the store's own level translation."""
    from types import SimpleNamespace

    from prediction.artifact import ome_geometry

    scales = {0: SimpleNamespace(translation_or_zeros=lambda: [0.0, 3.0, 0.0, 0.0])}  # c,z,y,x
    info = SimpleNamespace(image_meta=SimpleNamespace(scales=scales), img_spatial_idx=[1, 2, 3])
    grid = SimpleNamespace(
        info=info, image_level=0, axes="zyx",
        image_voxel=[29.0, 6.0, 6.0],            # kasthuri's native voxel
        effective_voxel=[8.0, 8.0, 8.0],         # the 8 nm lattice
        native_box=lambda: [[14, 86], [84, 939], [84, 939]],
    )
    g = ome_geometry(grid)
    assert g["axes"] == "zyx" and g["voxel_nm"] == [8.0, 8.0, 8.0]
    # z: 14 * 29 + (8 - 29) / 2 + 3;  y, x: 84 * 6 + (8 - 6) / 2
    assert g["translation_nm"] == pytest.approx([14 * 29 - 10.5 + 3.0, 84 * 6 + 1.0, 84 * 6 + 1.0])


@pytest.mark.unit
def test_labellings_are_written_as_the_narrowest_unsigned_type():
    """Neuroglancer has no int64; uint32 holds nearly every labelling and uint64 the rest."""
    from predict import unsigned_labels

    small = np.array([[0, 3], [7, 2**32 - 1]], dtype=np.int64)
    assert unsigned_labels(small).dtype == np.uint32
    assert np.array_equal(unsigned_labels(small), small)
    huge = np.array([0, 2**40], dtype=np.int64)          # a 64-bit hemibrain body id
    assert unsigned_labels(huge).dtype == np.uint64
    assert int(unsigned_labels(huge)[1]) == 2**40
    already = np.array([1, 2], dtype=np.uint32)
    assert unsigned_labels(already) is already
    with pytest.raises(ValueError, match="negative"):
        unsigned_labels(np.array([-1, 4], dtype=np.int64))
    with pytest.raises(TypeError, match="integer"):
        unsigned_labels(np.array([0.5]))


@pytest.mark.unit
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


def test_every_volume_in_the_config_is_predicted_unless_one_is_named():
    """`--volume` is optional: the data config already lists the split, so one command covers it."""
    from types import SimpleNamespace

    from predict import volumes_to_predict

    config = SimpleNamespace(volumes=[SimpleNamespace(name="a"), SimpleNamespace(name="b")])
    assert volumes_to_predict(config, None) == ["a", "b"]
    assert volumes_to_predict(config, "b") == ["b"]
    with pytest.raises(SystemExit, match="no volume named 'c'"):
        volumes_to_predict(config, "c")
