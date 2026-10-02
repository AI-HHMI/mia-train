"""`prediction.blockwise` must reproduce the whole-region dense path bit for bit, however it is cut.

A region too large for memory is predicted in blocks, by many workers, straight into one artifact.
That is only safe if the cut never shows in the numbers -- every voxel of a block must equal the
same voxel of `predict_volume` exactly -- and if a partial run can neither pass for a finished one
nor be quietly continued by a different prediction. Everything here runs on CPU over a miniature
lattice: 8^3 tiles at half-window steps over a 24 x 20 x 16 output, chunks of 4.
"""

from __future__ import annotations

import itertools
import json
import shutil

import numpy as np
import pytest
import torch
import zarr

from prediction.blockwise import block_boxes, predict_blocks
from prediction.dense import overlapping, predict_volume

pytestmark = pytest.mark.unit

CHUNKS = (3, 4, 4, 4)
BLOCK = (8, 8, 8)
GEOMETRY = {"axes": "zyx", "voxel_nm": [1.0, 1.0, 1.0], "translation_nm": [0.0, 0.0, 0.0]}
ATTRS = {"kind": "affinity", "convention": "sigmoid(0.2 * logit), blended in that space",
         "channels": 3, "step": 1}


class _Grid:
    """Enough of `VolumeGrid` for both paths: a lattice of half-overlapping tiles, fixed reads."""

    patch = [8, 8, 8]
    output_shape = (24, 20, 16)
    effective_voxel = [1.0, 1.0, 1.0]
    axes = "zyx"
    box_coverage = 1.0

    def __init__(self) -> None:
        generator = np.random.default_rng(0)
        starts = [range(0, s - p + 1, p // 2)
                  for s, p in zip(self.output_shape, self.patch, strict=True)]
        # C order, axis 0 outermost, as `VolumeGrid.tiles`; native origin = output origin here.
        self._tiles = [(origin, origin) for origin in itertools.product(*starts)]
        self._reads = {origin: generator.random((8, 8, 8), dtype=np.float32)
                       for origin, _ in self._tiles}

    @property
    def tiles(self):
        return list(self._tiles)

    def image_handle(self):
        return None

    def read_image(self, handle, origin):
        return self._reads[tuple(origin)]


class _Algorithm:
    """Three channels of a deterministic function of the tile; counts the tiles it runs."""

    prediction_kind = "affinity"
    prediction_channels = 3
    squash_convention = "sigmoid(0.2 * logit)"
    input_axes = "lczyx"                         # trained in the fake grid's own order
    offsets = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    def __init__(self) -> None:
        self.calls = 0

    def logits(self, volumes):
        self.calls += 1
        x = volumes[:, 0]
        return torch.stack([x, x * 2 - 1, -x], dim=1)

    @staticmethod
    def squash(logits):
        return torch.sigmoid(0.2 * logits)


class _FlushGrid(_Grid):
    """A box the half-window strides do not fill, at the store's own resolution: the last tile on
    each axis sits flush with the far face (`aligned_tiling` with read == patch and `cover`)."""

    output_shape = (26, 21, 17)

    def __init__(self) -> None:
        from prediction.grid import aligned_tiling

        generator = np.random.default_rng(0)
        starts = [aligned_tiling(s, p, p, cover=True)[1]
                  for s, p in zip(self.output_shape, self.patch, strict=True)]
        self._tiles = [(origin, origin) for origin in itertools.product(*starts)]
        self._reads = {origin: generator.random((8, 8, 8), dtype=np.float32)
                       for origin, _ in self._tiles}


def _predict(tmp_path, block=BLOCK, worker=0, workers=1, attrs=ATTRS, algorithm=None, grid=None):
    return predict_blocks(
        algorithm or _Algorithm(), grid or _Grid(), torch.device("cpu"), path=tmp_path / "v.zarr",
        attrs=attrs, geometry=GEOMETRY, block=block, worker=worker, workers=workers,
        chunks=CHUNKS,
    )


def _bits(array):
    return array.view(np.uint16)          # float16: equal bits, not merely equal values


def _whole():
    return predict_volume(_Algorithm(), _Grid(), torch.device("cpu"))


def _tiles_in(boxes):
    grid = _Grid()
    return sum(len(overlapping(grid.tiles, grid.patch, low,
                               tuple(h - lo for lo, h in zip(low, high, strict=True))))
               for low, high in boxes)


@pytest.mark.parametrize("block", [(4, 4, 4), (8, 12, 16), (12, 8, 4), (24, 20, 16), (99, 99, 99)])
def test_blocks_reproduce_the_whole_region_path_bit_for_bit(tmp_path, block):
    path = _predict(tmp_path, block)
    assert path == tmp_path / "v.zarr"
    blocked = zarr.open_array(str(path / "s0"), mode="r")[:]
    assert blocked.dtype == np.float16
    assert np.array_equal(_bits(blocked), _bits(_whole()))


@pytest.mark.parametrize("block", [(4, 4, 4), (8, 12, 16), (99, 99, 99)])
def test_a_flush_last_tile_leaves_no_seam_either(tmp_path, block):
    """It overlaps the tile before it by more than half a window; the cut must still not show."""
    grid = _FlushGrid()
    assert {origin[0] for origin, _ in grid.tiles} == {0, 4, 8, 12, 16, 18}
    path = _predict(tmp_path, block, grid=grid)
    whole = predict_volume(_Algorithm(), _FlushGrid(), torch.device("cpu"))
    assert whole.shape[1:] == _FlushGrid.output_shape
    assert np.array_equal(_bits(zarr.open_array(str(path / "s0"), mode="r")[:]), _bits(whole))


def test_workers_share_the_blocks_and_the_last_one_completes_the_artifact(tmp_path):
    assert _predict(tmp_path, worker=0, workers=3) is None
    assert (tmp_path / "v.zarr.partial").exists() and not (tmp_path / "v.zarr").exists()
    assert _predict(tmp_path, worker=2, workers=3) is None
    assert _predict(tmp_path, worker=1, workers=3) == tmp_path / "v.zarr"
    assert not (tmp_path / "v.zarr.partial").exists()
    assert not (tmp_path / "v.zarr.blocks").exists()

    group = zarr.open_group(str(tmp_path / "v.zarr"), mode="r")
    assert np.array_equal(_bits(group["s0"][:]), _bits(_whole()))
    attrs = dict(group.attrs)
    assert attrs["kind"] == "affinity" and attrs["step"] == 1
    assert attrs["blockwise"]["blocks"] == 3 * 3 * 2 and attrs["blockwise"]["device"] == "cpu"
    # Named for the final artifact, not the temporary directory it was written in.
    assert attrs["ome"]["multiscales"][0]["name"] == "v.zarr"
    assert dict(group["s0"].attrs)["blockwise"] == attrs["blockwise"]


def test_a_rerun_runs_only_the_blocks_not_marked_done(tmp_path):
    boxes = block_boxes(_Grid.output_shape, BLOCK, CHUNKS[1:])
    first = _Algorithm()
    _predict(tmp_path, worker=0, workers=2, algorithm=first)
    assert first.calls == _tiles_in(boxes[0::2])

    again = _Algorithm()
    assert _predict(tmp_path, worker=0, workers=2, algorithm=again) is None
    assert again.calls == 0, "every one of worker 0's blocks is marked done"

    (tmp_path / "v.zarr.blocks" / "2.json").unlink()       # as if killed while writing block 2
    redo = _Algorithm()
    _predict(tmp_path, worker=0, workers=2, algorithm=redo)
    assert redo.calls == _tiles_in([boxes[2]])

    # Any worker count finishes it, and redoes nothing already done.
    rest = _Algorithm()
    assert _predict(tmp_path, worker=0, workers=1, algorithm=rest) == tmp_path / "v.zarr"
    assert rest.calls == _tiles_in(boxes[1::2])
    assert np.array_equal(_bits(zarr.open_array(str(tmp_path / "v.zarr/s0"))[:]), _bits(_whole()))


def test_a_finished_artifact_is_left_alone(tmp_path):
    _predict(tmp_path)
    later = _Algorithm()
    assert _predict(tmp_path, algorithm=later) == tmp_path / "v.zarr"
    assert later.calls == 0


def test_a_partial_left_by_a_different_prediction_is_refused(tmp_path):
    _predict(tmp_path, worker=0, workers=2)
    with pytest.raises(SystemExit, match="different prediction"):
        _predict(tmp_path, worker=1, workers=2, attrs={**ATTRS, "step": 2})
    with pytest.raises(SystemExit, match="different prediction"):
        _predict(tmp_path, block=(8, 8, 16), worker=1, workers=2)


def test_markers_without_the_partial_they_describe_are_refused(tmp_path):
    """Completing would leave the marked blocks' chunks unwritten -- zero -- in a finished file."""
    _predict(tmp_path, worker=0, workers=2)
    shutil.rmtree(tmp_path / "v.zarr.partial")
    with pytest.raises(SystemExit, match="was missing"):
        _predict(tmp_path, worker=1, workers=2)
    assert not (tmp_path / "v.zarr.partial").exists() and not (tmp_path / "v.zarr").exists()


def test_blocks_from_two_gpu_types_are_not_joined(tmp_path):
    _predict(tmp_path, worker=0, workers=2)
    marker = tmp_path / "v.zarr.blocks" / "0.json"
    record = json.loads(marker.read_text())
    marker.write_text(json.dumps({**record, "device": "NVIDIA B300 SXM6 AC"}))
    with pytest.raises(SystemExit, match="2 GPU types"):
        _predict(tmp_path, worker=1, workers=2)
    assert not (tmp_path / "v.zarr").exists()


def test_a_block_never_shares_a_chunk_with_its_neighbour():
    with pytest.raises(SystemExit, match="not a multiple"):
        block_boxes((24, 20, 16), (6, 8, 8), (4, 4, 4))
    # The last block along an axis may end anywhere, and one spanning its axis needs no alignment.
    boxes = block_boxes((24, 20, 16), (8, 12, 100), (4, 4, 4))
    assert len(boxes) == 3 * 2 * 1
    assert boxes[0] == ((0, 0, 0), (8, 12, 16))
    assert boxes[-1] == ((16, 12, 0), (24, 20, 16))


def test_a_worker_index_outside_the_worker_count_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="--worker 2"):
        _predict(tmp_path, worker=2, workers=2)


def test_block_takes_one_number_or_one_per_axis():
    from predict import block_shape

    assert block_shape("1536", 3) == [1536, 1536, 1536]
    assert block_shape("512,1536,1536", 3) == [512, 1536, 1536]
    with pytest.raises(SystemExit, match="2 axes"):
        block_shape("512,512", 3)
    with pytest.raises(SystemExit, match="expected N"):
        block_shape("big", 3)
