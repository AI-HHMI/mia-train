"""Run a trained checkpoint over a whole OME-NGFF volume and write the prediction as an artifact.

    python src/predict.py <run_dir> --data-config <miao.yaml> --volume <name> --out <dir> [--step N]

Writes two artifacts per volume, on one shared grid:

    <dir>/<volume>.zarr        the prediction, kind from the algorithm
    <dir>/<volume>.gt.zarr     the co-registered ground truth, kind="instances"

**Nothing about the run is assumed.** Patch size, channel count, what the channels *mean*, how they
are squashed for storage, and the spatial rank all come from the run's own resolved config and from
the algorithm object rebuilt from it. A model trained at patch 128, an algorithm emitting 64 class
scores instead of 6 affinities, a non-cubic patch -- none of those need a change here. The one thing
that is checked rather than adopted is the patch size: the data config and the model must agree,
because RoPE normalises coordinates by the *runtime* grid extent, so predicting at a patch size the
encoder was not trained at silently changes every position it sees.

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

**Why the ground truth is written here too.** Putting a labelling on this grid needs the volume's
storage axis order, its image level, its *label* level -- which is not the same rung (zebrafish
labels are 16x16x30 nm against an 8x8x7.5 nm image) -- and the label's own channel axis where it has
one. That is the same transform chain the prediction went through, and duplicating it in the scorer
would be a second chance to get it wrong in a way no test compares. Every input to the chain is
recorded in the artifact's attrs, so the result stays auditable.

Every geometric value taken from miao is in **storage** axis order (`img_spatial_axes`), not the
config's `output_axes`: miao permutes the config's xyz box into the store's zyx where they differ,
and mixing the two transposes a volume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import zarr


def aligned_tiling(extent: int, read: int, patch: int) -> tuple[list[int], list[int], int]:
    """Native tile origins, output tile origins, and the output extent, all on one lattice.

    `read` must be even so a native stride of `read / 2` maps to an output stride of exactly
    `patch / 2`. The covered native extent is `read + (n - 1) * read / 2`, at most `extent`: the
    tail that does not complete a stride is dropped rather than padded, so no voxel is predicted
    from data invented to fill a tile.
    """
    if read % 2:
        raise ValueError(f"read shape must be even to halve into a stride, got {read}")
    if patch % 2:
        raise ValueError(f"patch size must be even to halve into a stride, got {patch}")
    if extent < read:
        raise ValueError(
            f"the region spans {extent} native voxels but one patch needs {read}: the bounding box "
            "is smaller than a patch at this resolution. Predict at a finer target resolution, or "
            "with a model trained at a smaller patch."
        )
    stride = read // 2
    count = (extent - read) // stride + 1
    return (
        [k * stride for k in range(count)],
        [k * (patch // 2) for k in range(count)],
        patch + (count - 1) * (patch // 2),
    )


def blend_weight(shape: tuple[int, ...]) -> np.ndarray:
    """Confidence of a tile's own prediction: low at its faces, high at its centre.

    A tile sees no context beyond its border, so its edge voxels are its worst; weighting
    overlapping predictions this way hides the seams that would otherwise cut objects at tile
    boundaries -- which connected components then reports as split errors.

    The chessboard distance to the outside, which for a box of ones padded by one voxel is the
    per-axis distance minimised over axes. Numerically identical to BANIS'
    `distance_transform_cdt` of that input, verified for cubic sizes 8 through 512, and generalised
    here to a per-axis shape so a non-cubic patch works.
    """
    rank = len(shape)
    weight: np.ndarray | None = None
    for axis, extent in enumerate(shape):
        ramp = (
            np.minimum(np.arange(extent), extent - 1 - np.arange(extent)) + 1
        ).astype(np.float32)
        view = [1] * rank
        view[axis] = -1
        broadcast = ramp.reshape(view)
        weight = broadcast if weight is None else np.minimum(weight, broadcast)
    assert weight is not None                                # rank >= 1 by construction
    return np.broadcast_to(weight, shape).astype(np.float32)


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


def assert_checkpoint_is_fully_consumed(algorithm: Any, checkpoint_dir: Path) -> None:
    """Fail if the checkpoint holds model tensors this rebuild has nowhere to put.

    **DCP loads *into* a state dict, and skips whatever the template does not ask for, silently.**
    That makes an incomplete rebuild the worst kind of bug here: a model reconstructed without its
    LoRA adapter loads every base weight, ignores every `lora_a`/`lora_b`, and predicts with the
    *un-adapted* encoder -- which scores near the released-checkpoint baseline it started from. A
    plausible number, attributed to the wrong model, and nothing anywhere says so.

    Compared against `get_model_state_dict`, the same function `CheckpointManager` saves through,
    rather than against `named_parameters()`: an algorithm may hold one module under two names
    (`affinity_seg` exposes its encoder as both `model` and `encoder`), `named_parameters()`
    deduplicates by tensor identity and would report only one of them, and the other would then look
    orphaned. Only the `model.` half of the checkpoint is checked -- `optim.` and `train_state.` are
    not rebuilt here and are not meant to be.
    """
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    stored = set(FileSystemReader(checkpoint_dir).read_metadata().state_dict_metadata)
    have = {f"model.{key}" for key in get_model_state_dict(algorithm)}
    orphaned = sorted(key for key in stored if key.startswith("model.") and key not in have)
    if not orphaned:
        return

    hint = ""
    if any(".lora_" in key for key in orphaned):
        hint = (
            "\nThese are LoRA adapter tensors. The run trained an adapted encoder, so its "
            "resolved_config.json must carry a [lora] section for this rebuild to reproduce it. If "
            "the section is present and this still fires, the model and the checkpoint disagree "
            "about which projections were adapted."
        )
    raise SystemExit(
        f"{len(orphaned)} tensor(s) in {checkpoint_dir} have no slot in the rebuilt model, so DCP "
        f"would load the rest and ignore these without a word: {orphaned[:6]}"
        f"{' ...' if len(orphaned) > 6 else ''}{hint}"
    )


def load_algorithm(
    run_dir: Path, device: torch.device, step: int | None = None
) -> tuple[Any, int, dict[str, Any]]:
    """Rebuild the trained algorithm from a run directory. Returns (algorithm, step, resolved).

    `resolved_config.json` records each section as {"name": ..., "kwargs": {...}}, which is exactly
    what the registries take -- so the model is rebuilt from what the run really used rather than
    from a config file that may since have moved on. The resolved settings are returned too, because
    a caller needs them to know what the run's patch size was.
    """
    import components  # noqa: F401  (populates the registries)
    from algorithms.registry import AlgorithmRegistry
    from engine.checkpoint import CheckpointManager
    from engine.config import LoRAConfig
    from engine.lora import apply_lora
    from models.registry import ModelRegistry

    resolved = json.loads((run_dir / "resolved_config.json").read_text())
    model_cfg, algo_cfg, data_cfg = resolved["model"], resolved["algorithm"], resolved["data"]
    model = ModelRegistry.build(model_cfg["name"], **model_cfg["kwargs"])

    # Low-rank adaptation, in the same position `engine.run.build_trainer` applies it: on the bare
    # model, before the algorithm wraps it. `[lora]` is a top-level section rather than part of
    # `[model]`, so rebuilding from `model_cfg` alone produces a *plain* encoder -- see
    # `assert_checkpoint_is_fully_consumed` for what that costs. Absent from every run predating the
    # feature, hence the default: `LoRAConfig()` has rank 0 and is disabled.
    lora_cfg = LoRAConfig(**resolved.get("lora", {}))
    if lora_cfg.enabled():
        print(f"[lora] {apply_lora(model, lora_cfg).summary()}", flush=True)

    algorithm = AlgorithmRegistry.build(
        algo_cfg["name"], model, None,
        input_axes=data_cfg["kwargs"]["output_axes"],
        **algo_cfg["kwargs"],
    )
    algorithm.to(device).eval()

    optimizer = torch.optim.AdamW(algorithm.parameters(), lr=1e-4)
    manager = CheckpointManager(algorithm, optimizer, run_dir / "checkpoints")

    # Checked before loading, so a mismatch costs a second rather than a long job whose output is
    # quietly wrong. Only when the directory is actually there: a missing checkpoint is reported
    # below, and `load_step` names the steps that *do* exist, which is the more useful message.
    path = manager.latest_checkpoint() if step is None else run_dir / "checkpoints" / f"step_{step}"
    if path is not None and path.is_dir():
        assert_checkpoint_is_fully_consumed(algorithm, path)

    loaded = manager.load_latest() if step is None else manager.load_step(step)
    if loaded == 0:
        raise SystemExit(f"no checkpoint found under {run_dir / 'checkpoints'}")
    print(f"loaded step {loaded} from {run_dir.name}", flush=True)
    return algorithm, loaded, resolved


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


def resolve_patch(
    config: Any, resolved: dict[str, Any], volume_name: str, override: int | None
) -> list[int]:
    """The patch size to predict at, per axis in storage order.

    Taken from the data config, which states it per axis, and cross-checked against the model's own
    `img_size`. They must agree: RoPE normalises coordinates by the *runtime* grid extent, so an
    encoder fed a patch size other than the one it trained at sees every position shifted, silently
    and without any error. `--patch` overrides both, for deliberately probing that.
    """
    output_axes = "".join(axis for axis in config.output_axes if axis in "xyz")
    if len(config.patch_size) != len(output_axes):
        raise SystemExit(
            f"the data config's patch_size {list(config.patch_size)} has "
            f"{len(config.patch_size)} entries but its output_axes imply "
            f"{len(output_axes)} spatial axes ({output_axes})"
        )
    storage_axes = storage_axes_of(config, volume_name)
    by_axis = dict(zip(output_axes, (int(p) for p in config.patch_size), strict=True))
    patch = [by_axis[axis] for axis in storage_axes]

    if override is not None:
        print(f"[patch] overriding {patch} with {override} on every axis", flush=True)
        return [override] * len(storage_axes)

    img_size = resolved.get("model", {}).get("kwargs", {}).get("img_size")
    if img_size is not None:
        trained = (
            [int(img_size)] * len(storage_axes)
            if isinstance(img_size, (int, float))
            else [int(v) for v in img_size]
        )
        if sorted(trained) != sorted(patch):
            raise SystemExit(
                f"the data config asks for patch {patch} but the model was built with img_size "
                f"{img_size}. Predicting at a different patch size than the encoder trained at "
                "silently moves every position it sees, because RoPE normalises coordinates by the "
                "runtime grid extent. Fix the config, or pass --patch to override deliberately."
            )
    return patch


class VolumeGrid:
    """The aligned lattice for one volume, and the reads that fill it. All in storage axis order."""

    def __init__(
        self,
        config: Any,
        volume_name: str,
        patch: list[int],
        box: list[list[int]] | None = None,
    ) -> None:
        """`box` restricts the region to a sub-box of the volume's own `bounding_box`.

        In level-0 image voxels, storage axis order, as `[[lo, hi], ...]`. Used to predict one
        block of a large volume at a time -- pseudo-labelling walks a volume in blocks because a
        whole one does not fit in memory -- and clipped to the annotated box, so a caller cannot
        widen the region past what the data config declares.
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
        self.label_level = int(info.scales.label_chosen_levels[0])
        self.image_voxel = [float(v) for v in info.img_level_voxels[self.image_level]]

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

        read = [int(r) for r in info.scales.read_shapes[0]]
        self.read = [r + r % 2 for r in read]
        tiled = [
            aligned_tiling(extent, r, p)
            for extent, r, p in zip(self.box_extent, self.read, self.patch, strict=True)
        ]
        self.native_origins = [t[0] for t in tiled]
        self.output_origins = [t[1] for t in tiled]
        self.output_shape = tuple(t[2] for t in tiled)
        self.native_extent = [
            r + (len(o) - 1) * (r // 2)
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


@torch.no_grad()
def predict_volume(algorithm: Any, grid: VolumeGrid, device: torch.device) -> np.ndarray:
    """Blended predictions over the aligned region -> (channels, *output) float16.

    Channels, the squashing applied before blending, and the forward pass all come from the
    algorithm, so this serves any dense-output algorithm without knowing which one it has.

    Squashed *before* blending, not after: overlapping tiles are averaged in the stored
    representation. A weighted mean of logits is not the logit of a weighted mean of probabilities,
    so the order is part of the convention and is recorded with the data.
    """
    handle = grid.image_handle()
    channels = int(algorithm.prediction_channels)
    total = np.zeros((channels, *grid.output_shape), dtype=np.float32)
    weight = np.zeros((1, *grid.output_shape), dtype=np.float32)
    single = blend_weight(tuple(grid.patch))[None]

    tiles = grid.tiles
    print(f"{len(tiles)} tiles of {grid.patch} -> output {grid.output_shape} at "
          f"{[round(v, 3) for v in grid.effective_voxel]} nm/voxel ({grid.axes}); "
          f"{channels} channels of {algorithm.prediction_kind}; "
          f"lattice covers {100 * grid.box_coverage:.1f}% of the annotated box, centred",
          flush=True)
    for index, (native, out) in enumerate(tiles):
        volumes = torch.from_numpy(grid.read_image(handle, native)[None, None]).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = algorithm.logits(volumes)
        stored = algorithm.squash(logits.float())[0].cpu().numpy()

        window = tuple(slice(o, o + p) for o, p in zip(out, grid.patch, strict=True))
        total[(slice(None), *window)] += stored * single
        weight[(slice(None), *window)] += single
        if (index + 1) % 25 == 0 or index + 1 == len(tiles):
            print(f"  {index + 1}/{len(tiles)}", flush=True)

    # Divided a slab at a time. `(total / weight).astype(f16)` materialises a full float32 quotient
    # before downcasting, which at 7 gigavoxels over 6 channels is an extra 170 GB on top of the
    # accumulator. Slabbing costs nothing numerically and bounds the temporary at one slab.
    blended = np.empty(total.shape, dtype=np.float16)
    slab = max(1, grid.output_shape[0] // 16)
    for start in range(0, total.shape[1], slab):
        stop = start + slab
        blended[:, start:stop] = total[:, start:stop] / np.maximum(weight[:, start:stop], 1e-8)
    return blended


def shared_attrs(
    grid: VolumeGrid, run_dir: Path, step: int, data_config: Path
) -> dict[str, Any]:
    """Everything needed to reproduce this grid, on both artifacts so neither can drift."""
    return {
        "origin": [0] * grid.rank,
        "axes": grid.axes,
        "source_path": str(grid.volume.path),
        "source_image_key": grid.volume.image_key,
        "source_label_key": grid.volume.label_key,
        "image_level": grid.image_level,
        "label_level": grid.label_level,
        "native_box": grid.native_box(),
        "annotated_box": [
            [low, low + extent]
            for low, extent in zip(grid.box_low, grid.box_extent, strict=True)
        ],
        "read_shape": list(grid.read),
        "scale": [p / r for p, r in zip(grid.patch, grid.read, strict=True)],
        "effective_voxel_nm": [round(v, 6) for v in grid.effective_voxel],
        "covers_full_box": grid.covers_full_box,
        "box_coverage": round(grid.box_coverage, 6),
        "patch": list(grid.patch),
        "stride": [p // 2 for p in grid.patch],
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "step": step,
        "data_config": str(data_config),
        "volume": grid.volume.name,
    }


def write_array(path: Path, array: np.ndarray, **attrs: Any) -> Path:
    store = zarr.open(
        str(path), mode="w", shape=array.shape, dtype=array.dtype,
        chunks=tuple(min(256, s) for s in array.shape),
    )
    store[:] = array
    store.attrs.update(**attrs)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="a mia-train run directory")
    parser.add_argument("--data-config", type=Path, required=True,
                        help="the miao YAML describing the volume (e.g. an eval split's config)")
    parser.add_argument("--volume", type=str, required=True, help="which volume in it to predict")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory; artifacts are named after the volume")
    parser.add_argument("--step", type=int, default=None,
                        help="checkpoint step to load; default is the newest")
    parser.add_argument("--patch", type=int, default=None,
                        help="override the patch size on every axis. Defaults to the data config's "
                             "own patch_size, cross-checked against the model's img_size; an "
                             "override changes what the encoder's positions mean, so it is for "
                             "probing that deliberately, not for tuning")
    parser.add_argument("--truth-only", action="store_true",
                        help="write only the ground-truth artifact; needs no GPU and no checkpoint")
    args = parser.parse_args()

    from miao.config import load_config

    config = load_config(args.data_config)
    if config.resolutions is None:
        raise SystemExit(
            f"{args.data_config} sets no `resolutions`, so there is no single target resolution to "
            "predict at. Resolution sampling is a training-time device."
        )
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algorithm: Any = None
    step = -1
    resolved: dict[str, Any] = {}
    if not args.truth_only:
        algorithm, step, resolved = load_algorithm(args.run_dir, device, args.step)
        missing = [
            name for name in ("logits", "squash", "squash_convention",
                              "prediction_kind", "prediction_channels")
            if not hasattr(algorithm, name)
        ]
        if missing:
            raise SystemExit(
                f"{type(algorithm).__name__} lacks {missing}. Prediction over a whole volume needs "
                "an algorithm that declares what its output means and how to produce it; see "
                "`affinity_seg` for the members required."
            )

    grid = VolumeGrid(
        config, args.volume, resolve_patch(config, resolved, args.volume, args.patch)
    )

    if algorithm is not None:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        prediction = predict_volume(algorithm, grid, device)
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(0)
            print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB of "
                  f"{properties.total_memory / 2**30:.0f} GiB ({properties.name})", flush=True)

        path = write_array(
            args.out / f"{args.volume}.zarr", prediction,
            kind=str(algorithm.prediction_kind),
            convention=f"{algorithm.squash_convention}, blended in that space",
            channels=int(algorithm.prediction_channels),
            **shared_attrs(grid, args.run_dir, step, args.data_config),
        )
        print(f"wrote {path}  {prediction.shape} {prediction.dtype}", flush=True)

    truth = grid.read_ground_truth()
    instances = int((np.unique(truth) != 0).sum())
    path = write_array(
        args.out / f"{args.volume}.gt.zarr", truth,
        kind="instances",
        # 0 is background in every label store in this corpus, and -1 never occurs in one, so
        # nothing here is unannotated. Stated rather than assumed: for a thresholded-components
        # prediction 0 means "no edge survived", which is a different claim entirely.
        background_id=0,
        instances=instances,
        **shared_attrs(grid, args.run_dir, step, args.data_config),
    )
    print(f"wrote {path}  {truth.shape} int64, {instances} instances, "
          f"{100.0 * float((truth != 0).mean()):.1f}% annotated", flush=True)


if __name__ == "__main__":
    main()
