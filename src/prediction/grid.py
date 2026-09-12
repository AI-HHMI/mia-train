"""One volume's aligned tile lattice, and the miao reads that fill it.

**Volumes are read through miao**, which owns this format: `miao.store.open_store` for the arrays,
and `VolumeDataset`'s resolved geometry for the read shapes, per-level voxel sizes and the permuted
bounding box. Reading them here instead would be a second implementation of the transform chain
that produced the training data, and it would drift. It is also the only thing that works on
every store in the corpus:
`em-drosophila-flyem-hemibrain` carries an illegal top-level `_source` key in its group metadata,
which zarr-python refuses outright and tensorstore ignores.

**Why this does its own tiling rather than using miao's `sampling = "sequential"`.** That grid
samples a volume; it does not tile it for reconstruction. Measured on `kasthuri15_ac4`: its patch
origins sit on the *native* 6 nm lattice (x steps alternate 1008 and 1026 nm) and each patch spans
2052 nm resampled to 256 voxels, i.e. 8.016 nm per voxel rather than 8.000. Successive patches are
independent resamplings of incommensurate physical boxes, so stitching them needs a second
resampling -- and an affinity cannot survive one: channel 0 asserts "the voxel one step along x is
mine", a statement about one specific voxel spacing.

So the lattice is built the other way round. Per axis, miao's own native read shape `R` is rounded
up to even; tiles start at native multiples of `R'/2` and therefore land at output multiples of
`patch/2`. Every tile is the same resampling operation, the output lattice is exactly uniform, and
one output voxel measures `R' * native / patch`. The scored region is the largest lattice-aligned
sub-box of `bounding_box`, centred in it: shrinking rather than padding, so nothing is predicted
from data invented to fill a tile, and the covered fraction is recorded as `box_coverage`.

**The ground truth is read here too**, on the same lattice. Putting a labelling on this grid needs
the volume's storage axis order, its image level, its *label* level -- which is not the same rung
(zebrafish labels are 16x16x30 nm against an 8x8x7.5 nm image) -- and the label's own channel axis
where it has one. That is the same transform chain the prediction went through, and duplicating it
in the scorer would be a second chance to get it wrong in a way no test compares.

Every geometric value taken from miao is in **storage** axis order (`img_spatial_axes`), not the
config's `output_axes`: miao permutes the config's xyz box into the store's zyx where they differ,
and mixing the two transposes a volume.

`tests/unit/test_predict.py` holds the parity test that reads a box both ways -- through miao's own
sampler and through `VolumeGrid` -- and requires the two to agree to the bit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def aligned_tiling(
    extent: int, read: int, patch: int, steps_per_patch: int = 2
) -> tuple[list[int], list[int], int]:
    """Native tile origins, output tile origins, and the output extent, all on one lattice.

    The window advances by `patch / steps_per_patch` output voxels: 2 (the default) is the
    half-window overlap every scored run used, 4 a quarter-window step. `read` and `patch` must
    both divide by `steps_per_patch` so a native stride of `read / steps` maps to an output stride
    of exactly `patch / steps` -- one lattice, no drift between tiles. The covered native extent
    is `read + (n - 1) * read / steps`, at most `extent`: the tail that does not complete a stride
    is dropped rather than padded, so no voxel is predicted from data invented to fill a tile.
    """
    if steps_per_patch < 1:
        raise ValueError(f"steps_per_patch must be at least 1, got {steps_per_patch}")
    if read % steps_per_patch:
        raise ValueError(
            f"read shape must divide by {steps_per_patch} to step by read / {steps_per_patch}, "
            f"got {read}"
        )
    if patch % steps_per_patch:
        raise ValueError(
            f"patch size must divide by {steps_per_patch} to step by patch / {steps_per_patch}, "
            f"got {patch}"
        )
    if extent < read:
        raise ValueError(
            f"the region spans {extent} native voxels but one patch needs {read}: the bounding box "
            "is smaller than a patch at this resolution. Predict at a finer target resolution, or "
            "with a model trained at a smaller patch."
        )
    stride = read // steps_per_patch
    count = (extent - read) // stride + 1
    return (
        [k * stride for k in range(count)],
        [k * (patch // steps_per_patch) for k in range(count)],
        patch + (count - 1) * (patch // steps_per_patch),
    )


def resample_image(block: np.ndarray, target: tuple[int, ...]) -> np.ndarray:
    """Trilinear to `target`, on raw values.

    miao normalises *after* resampling, so this must too, or the two disagree at the edges.
    """
    if tuple(block.shape) == target:
        return block.astype(np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(block)).float()[None, None]
    return (
        F.interpolate(tensor, size=target, mode="trilinear", align_corners=False)
        .squeeze(0).squeeze(0).numpy()
    )


def resample_labels(block: np.ndarray, target: tuple[int, ...]) -> np.ndarray:
    """Nearest to `target` by index map, which is the only resampling an id survives.

    Indexed rather than run through `F.interpolate`: miao's nearest path casts to float32 first, and
    float32 cannot represent an id above 2**24 -- the mechanism by which a store holding 32,039
    instances in a region delivers a median of one per patch while the tensor is still int64. An
    index map has no such ceiling, so this is deliberately *not* bit-identical to miao for ids past
    that point. It is identical wherever miao is correct.
    """
    if tuple(block.shape) == target:
        return block
    index = [
        np.minimum((np.arange(t) * s // t).astype(np.int64), s - 1)
        for t, s in zip(target, block.shape, strict=True)
    ]
    return block[np.ix_(*index)]


def normalize(
    block: np.ndarray, dtype: np.dtype, low: float | None, high: float | None
) -> np.ndarray:
    """miao's `_normalize_image_tensor`: a window if one is given, else divide by the dtype max."""
    out = block.astype(np.float32)
    if low is not None and high is not None:
        return (np.clip(out, low, high) - float(low)) / (float(high) - float(low))
    if np.issubdtype(dtype, np.integer):
        return out / float(np.iinfo(dtype).max)
    return out


def storage_axes_of(config: Any, volume_name: str) -> str:
    """The spatial axis order this volume is *stored* in, which is not the config's output order.

    Needed before a `VolumeGrid` exists, because the grid's patch size has to be expressed in this
    order and the grid takes the patch as an argument.
    """
    from miao.dataset import VolumeDataset

    single = config.model_copy(
        update={"volumes": [v for v in config.volumes if v.name == volume_name]}
    )
    if not single.volumes:
        raise SystemExit(
            f"no volume named {volume_name!r}; the config holds "
            f"{[v.name for v in config.volumes]}"
        )
    return str(VolumeDataset(single)._volumes[0].img_spatial_axes)


class VolumeGrid:
    """The aligned lattice for one volume, and the reads that fill it. All in storage axis order."""

    #: How many window steps span one patch: the window advances by `patch / steps_per_patch`.
    #: 2 is the half-window overlap every scored run used; a class default so a geometry built
    #: without `__init__` (the tests) has one.
    steps_per_patch: int = 2

    def __init__(
        self,
        config: Any,
        volume_name: str,
        patch: list[int],
        box: list[list[int]] | None = None,
        steps_per_patch: int = 2,
    ) -> None:
        """`box` restricts the region to a sub-box of the volume's own `bounding_box`.

        In level-0 image voxels, storage axis order, as `[[lo, hi], ...]`. Used to predict one
        block of a large volume at a time -- pseudo-labelling walks a volume in blocks because a
        whole one does not fit in memory -- and clipped to the annotated box, so a caller cannot
        widen the region past what the data config declares. `steps_per_patch` sets the window
        step, see `aligned_tiling`.
        """
        from miao.dataset import VolumeDataset
        from miao.store import create_context

        try:
            volume = next(v for v in config.volumes if v.name == volume_name)
        except StopIteration:
            raise SystemExit(
                f"no volume named {volume_name!r}; the config holds "
                f"{[v.name for v in config.volumes]}"
            ) from None
        self.volume = volume
        self.context = create_context(cache_bytes=config.cache_bytes)

        # One volume, so `_volumes[0]` is it. Taking the geometry from miao is deliberate: a
        # reimplementation would be a second transform chain to keep in step with the one that
        # produced the training data.
        single = config.model_copy(update={"volumes": [volume]})
        info = VolumeDataset(single)._volumes[0]
        self.info = info
        self.axes: str = info.img_spatial_axes
        self.patch = list(patch)
        self.rank = len(self.axes)
        self.image_level = int(info.scales.chosen_levels[0])
        # None for a volume with no `label_key`: pseudo-labelling runs this lattice over unlabeled
        # volumes, where there is no ground truth to read and `read_ground_truth` says so itself.
        self.label_level = (
            int(info.scales.label_chosen_levels[0])
            if info.scales.label_chosen_levels is not None else None
        )
        self.image_voxel = [float(v) for v in info.img_level_voxels[self.image_level]]
        self.steps_per_patch = int(steps_per_patch)

        self._resolve_geometry(box, volume_name)

    def _resolve_geometry(self, box: list[list[int]] | None, volume_name: str) -> None:
        """Compute the lattice from `self.info`, `self.patch` and `self.image_voxel`.

        Separated from `__init__` so the arithmetic can be exercised without opening a store: the
        one invariant that matters here -- that the image tiles and `native_box()` describe the
        *same* region -- was violated once and cost a whole scoring run, and the test for it should
        not depend on cluster storage being mounted. See
        `tests/unit/test_predict.py::test_tiles_and_ground_truth_cover_the_same_region`.
        """
        info = self.info
        if self.image_level != 0:
            # Every quantity below -- the bounding box, the tile origins, `native_box()` -- is in
            # LEVEL-0 voxels, and `read_image` indexes the chosen level's array with them. That is
            # one coordinate system only while the chosen level is 0. miao picks a coarser rung
            # when it matches the target resolution better (a 4 nm store asked for 8 nm reads its
            # level 1), and this lattice would then read from the wrong place and shrink every
            # tile by the level's factor, with nothing downstream able to tell. Refused rather than
            # generalised, because no volume this has been run on needs it: all eight eval volumes
            # and all 87 unlabeled pretraining volumes resolve to level 0 at 8 nm (measured).
            raise SystemExit(
                f"volume {volume_name!r} would be read at pyramid level {self.image_level}, but "
                "this lattice is expressed in level-0 voxels and only reads level 0. Predict at a "
                "target resolution the store's level 0 serves, or extend VolumeGrid to convert "
                "its box and origins into the chosen level's voxels (and its origin offset)."
            )
        if info.bounding_box is None:
            raise SystemExit(
                f"volume {volume_name!r} has no bounding_box. It is not optional for scoring: "
                "several volumes in this corpus annotate an offset sub-box, and the LICONN dentate "
                "gyrus and hippocampus blocks hold 0% ground truth in their own central block."
            )
        declared = np.asarray(info.bounding_box)
        low = [int(v) for v in declared[:, 0]]
        high = [int(v) for v in declared[:, 1]]
        if box is not None:
            if len(box) != len(low):
                raise SystemExit(
                    f"box has {len(box)} axes but volume {volume_name!r} has {len(low)}"
                )
            # Intersected, not replaced: a block outside the annotation has no ground truth, and
            # silently predicting there would produce a region that cannot be scored.
            low = [max(a, int(pair[0])) for a, pair in zip(low, box, strict=True)]
            high = [min(a, int(pair[1])) for a, pair in zip(high, box, strict=True)]
            if any(h <= lo for lo, h in zip(low, high, strict=True)):
                raise SystemExit(
                    f"the requested box {box} does not overlap volume {volume_name!r}'s annotated "
                    f"box {declared.tolist()}"
                )
        self.box_low = low
        self.box_extent = [h - lo for lo, h in zip(low, high, strict=True)]

        steps = self.steps_per_patch
        read = [int(r) for r in info.scales.read_shapes[0]]
        # Rounded up to a multiple of the step count so the native stride is a whole number of
        # voxels (for the default 2 this is the "make it even" rule every scored run used).
        self.read = [r + (-r) % steps for r in read]
        tiled = [
            aligned_tiling(extent, r, p, steps)
            for extent, r, p in zip(self.box_extent, self.read, self.patch, strict=True)
        ]
        self.native_origins = [t[0] for t in tiled]
        self.output_origins = [t[1] for t in tiled]
        self.output_shape = tuple(t[2] for t in tiled)
        self.native_extent = [
            r + (len(o) - 1) * (r // steps)
            for r, o in zip(self.read, self.native_origins, strict=True)
        ]
        # Centre the lattice in the bounding box rather than anchoring it at the low corner. The
        # covered extent is fixed by the lattice -- a tile that does not complete a stride is
        # dropped -- but *which* part is covered is free, and for a thin volume the difference is
        # large: Kasthuri AC4 annotates 100 native z slices and one 72-slice tile fits, so the
        # choice is between the first 72 and the middle 72.
        #
        # `native_box()` and `tiles` both apply this offset, and they must: the ground truth is
        # read from the former and the image from the latter, so a version of this that centred
        # only one of them compared a prediction against truth from a different region.
        self.native_offsets = [
            (extent - covered) // 2
            for extent, covered in zip(self.box_extent, self.native_extent, strict=True)
        ]
        self.covers_full_box = self.native_extent == self.box_extent
        self.box_coverage = float(
            np.prod([c / e for c, e in zip(self.native_extent, self.box_extent, strict=True)])
        )
        self.effective_voxel = [
            r * v / p
            for r, v, p in zip(self.read, self.image_voxel, self.patch, strict=True)
        ]

    @property
    def tiles(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """(native origin, output origin) per tile, storage axis order."""
        out: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        indices: list[tuple[int, ...]] = [()]
        for axis_origins in self.native_origins:
            indices = [prefix + (i,) for prefix in indices for i in range(len(axis_origins))]
        for index in indices:
            out.append((
                tuple(
                    self.box_low[a] + self.native_offsets[a] + self.native_origins[a][i]
                    for a, i in enumerate(index)
                ),
                tuple(self.output_origins[a][i] for a, i in enumerate(index)),
            ))
        return out

    def native_box(self) -> list[list[int]]:
        """The aligned sub-box actually covered, in image level-0 voxels, storage order."""
        return [
            [low + offset, low + offset + extent]
            for low, offset, extent in zip(
                self.box_low, self.native_offsets, self.native_extent, strict=True
            )
        ]

    def _open(self, key: str, level: int) -> Any:
        """A tensorstore handle for one array of this volume, through miao.

        miao owns this format, and using its opener is what makes the reads here the reads that
        produced the training data. It is also the only thing that works on every store in the
        corpus: `em-drosophila-flyem-hemibrain` carries an illegal top-level `_source` key in its
        group metadata, which zarr-python refuses and tensorstore ignores.
        """
        from miao.store import open_store

        return open_store(
            Path(self.volume.path) / key / f"s{level}", self.volume.zarr_version, self.context
        )

    def image_handle(self) -> Any:
        return self._open(self.volume.image_key, self.image_level)

    def read_image(self, handle: Any, origin: tuple[int, ...]) -> np.ndarray:
        """One tile: read, resample to the patch, normalise. Storage order throughout."""
        window = tuple(slice(o, o + r) for o, r in zip(origin, self.read, strict=True))
        block = np.asarray(handle[window])
        resampled = resample_image(block, tuple(self.patch))
        return normalize(
            resampled, np.dtype(block.dtype),
            self.volume.normalize_min if self.volume.normalize else None,
            self.volume.normalize_max if self.volume.normalize else None,
        )

    def read_ground_truth(self) -> np.ndarray:
        """The whole aligned region's labelling, on the output lattice, as int64.

        The label pyramid is resolved independently of the image's: zebrafish labels sit at
        16x16x30 nm against an 8x8x7.5 nm image, so the same physical box is a different number of
        voxels in each. The box is therefore converted through *physical* coordinates rather than by
        assuming the two arrays share a lattice.
        """
        if self.volume.label_key is None:
            raise SystemExit(
                f"volume {self.volume.name!r} has no label_key, so there is no ground truth to "
                "write beside the prediction."
            )
        info = self.info
        label_voxel = [float(v) for v in info.lbl_level_voxels[self.label_level]]
        label_axes: str = info.lbl_spatial_axes
        if label_axes != self.axes:
            raise SystemExit(
                f"volume {self.volume.name!r} stores its image as {self.axes!r} and its labels as "
                f"{label_axes!r}. Scoring across that permutation is not implemented, and guessing "
                "it would misalign every voxel while still returning a number."
            )

        handle = self._open(self.volume.label_key, self.label_level)
        shape = list(handle.shape)
        channel_axis = None
        if info.lbl_axes is not None and "c" in info.lbl_axes:
            channel_axis = info.lbl_axes.index("c")
            shape.pop(channel_axis)

        window: list[Any] = []
        native = self.native_box()
        for axis, (low, high) in enumerate(native):
            start = int(np.floor(low * self.image_voxel[axis] / label_voxel[axis] + 1e-6))
            stop = int(np.ceil(high * self.image_voxel[axis] / label_voxel[axis] - 1e-6))
            start = max(0, min(start, shape[axis] - 1))
            stop = max(start + 1, min(stop, shape[axis]))
            window.append(slice(start, stop))
        if channel_axis is not None:
            window.insert(channel_axis, 0)

        block = np.asarray(handle[tuple(window)])
        return resample_labels(block.astype(np.int64), self.output_shape)
