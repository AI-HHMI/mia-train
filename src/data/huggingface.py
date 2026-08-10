"""Semantic segmentation datasets published on the HuggingFace Hub.

One dataset class, parameterised. Hub segmentation datasets differ in only a few ways -- which
columns hold the image and the label, how those columns are encoded, the spatial rank, and what
identifies a group of correlated rows -- so those are configuration, and a specific dataset is a
`preset` rather than a subclass. `PRESETS` at the bottom is the whole of what "supporting CellMap"
amounts to; adding another Hub dataset is an entry there, or a `[data]` section that spells the
same fields out inline.

**Why these do not go through miao.** miao reads OME-NGFF and is built for chunked random access
into volumes far larger than memory. Hub datasets are the opposite shape of problem: many small,
complete arrays delivered row-wise. `datasets` caches them as memory-mapped Arrow shards, which is
both faster than a zarr round-trip and closer to what the data already is. Nothing here imports
miao.

`datasets` is an optional dependency, imported inside `build_dataset` rather than at module scope,
so `components.py` can register this in an install that lacks it and only a run that names it
pays. See pyproject's `cellmap` extra.

**Splitting is by group, not by row.** Rows from one specimen -- adjacent planes of a crop, or
crops sharing a fixation and imaging run -- are near-duplicates. Holding out whole groups is the
only split that measures generalisation to new tissue rather than memorisation of a texture.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.utils.data as data

from .base import BaseDataset
from .registry import DataRegistry

# Axis order a sample is delivered in, by spatial rank. A level axis of one and a single grey
# channel, so the encoders' `prepare_input` contract (which requires 'l') is met unchanged.
SAMPLE_AXES = {2: "lcxy", 3: "lcxyz"}


@dataclass(frozen=True)
class Preset:
    """Everything that distinguishes one Hub segmentation dataset from another."""

    repo: str
    spatial_rank: int
    encoding: str                    # see DECODERS
    image_key: str = "image"
    label_key: str = "label"
    shape_key: str | None = None     # required by the "raw" encoding
    group_key: str | None = None     # column identifying correlated rows, e.g. "crop_name"
    group_levels: int = 1            # leading "/"-separated components of it that name the group
    hub_split: str = "train"


def decode_image(row: dict[str, Any], preset: Preset) -> tuple[np.ndarray, np.ndarray]:
    """Columns holding encoded images (PNG bytes), as `datasets`' Image feature stores them."""
    from PIL import Image

    def load(value: Any) -> np.ndarray:
        if isinstance(value, dict):          # {"bytes": ..., "path": ...}
            return np.array(Image.open(io.BytesIO(value["bytes"])))
        return np.array(value)               # already decoded to PIL by `datasets`

    return load(row[preset.image_key]), load(row[preset.label_key])


def decode_raw(row: dict[str, Any], preset: Preset) -> tuple[np.ndarray, np.ndarray]:
    """Columns holding raw uint8 buffers plus a shape column, one byte per voxel."""
    if preset.shape_key is None:
        raise ValueError("the 'raw' encoding needs shape_key naming the column holding the shape")
    shape = tuple(int(s) for s in row[preset.shape_key])
    image = np.frombuffer(row[preset.image_key], dtype=np.uint8).reshape(shape)
    label = np.frombuffer(row[preset.label_key], dtype=np.uint8).reshape(shape)
    return image, label


DECODERS = {"image": decode_image, "raw": decode_raw}


def group_of(name: str, levels: int) -> str:
    """`jrc_hela-2/recon-1/crop28` with levels=1 -> `jrc_hela-2`; levels=2 -> the recon."""
    return "/".join(name.split("/")[:levels])


def random_crop(
    image: np.ndarray, label: np.ndarray, size: tuple[int, ...], generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Take the same random `size` window from both, zero-padding any axis that is too short.

    Published crops have no common shape -- CellMap's 2D slices alone come in dozens of sizes,
    from 200x200 to 1796x2400 -- so something has to make a batch stackable. Cropping rather than
    resizing keeps one output pixel equal to one input voxel, which matters because a resampled
    label is a fabricated one: interpolating between class indices 3 and 8 does not mean class 5.
    """
    padding, offsets = [], []
    for extent, want in zip(image.shape, size, strict=True):
        if extent < want:
            padding.append((0, want - extent))
            offsets.append(0)
        else:
            padding.append((0, 0))
            offsets.append(int(generator.integers(0, extent - want + 1)))

    if any(before or after for before, after in padding):
        image = np.pad(image, padding)
        label = np.pad(label, padding)
    window = tuple(slice(o, o + w) for o, w in zip(offsets, size, strict=True))
    return image[window], label[window]


class _Rows(data.Dataset):
    """Map-style view over the selected rows, cropping each to a fixed window.

    `samples_per_epoch` decouples an epoch from the row count, which Hub datasets disagree about
    by orders of magnitude -- CellMap ships 293 volumes and 366k planes of the same data. Without
    it an "epoch" means something different for each, and every schedule expressed in epochs
    silently changes meaning between them.
    """

    def __init__(self, rows: Any, owner: HuggingFaceSemanticSegmentation) -> None:
        self.rows = rows
        self.owner = owner

    def __len__(self) -> int:
        return self.owner.samples_per_epoch or len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # Seeded per index so a sample is reproducible and workers do not duplicate windows.
        generator = np.random.default_rng((self.owner.seed, index))
        row = self.rows[int(generator.integers(0, len(self.rows)))]
        image, label = DECODERS[self.owner.preset.encoding](row, self.owner.preset)

        image, label = random_crop(image, label, self.owner.patch_size, generator)
        if label.max() >= self.owner.num_classes:
            raise ValueError(
                f"label id {int(label.max())} exceeds num_classes={self.owner.num_classes}; "
                "raise it to cover every class in the dataset"
            )
        return {
            "img": torch.from_numpy(np.ascontiguousarray(image)).float().div_(255.0)[None, None],
            "label": torch.from_numpy(np.ascontiguousarray(label)).long()[None],
        }


@DataRegistry.register("hf_semantic_seg")
class HuggingFaceSemanticSegmentation(BaseDataset):
    """A Hub semantic segmentation dataset, described by a `preset` or spelled out inline.

    `hold_out_groups` names the groups the validation split owns; `split` picks which side of that
    line this instance reads. Both splits therefore come from one description, and the two cannot
    drift into overlapping.

    `row_filters` keeps only rows whose column matches, e.g. `{"axis": ["z"]}` to train a 2D model
    on z-planes alone. Left empty, every row is kept -- which for CellMap 2D means all three slice
    orientations, and that is what makes the encoder usable for orthoplane prediction later: it
    has seen each orientation, so averaging its three views averages three things it is equally
    qualified to say.
    """

    def __init__(
        self,
        patch_size: tuple[int, ...] | list[int],
        preset: str | None = None,
        repo: str | None = None,
        spatial_rank: int | None = None,
        encoding: str | None = None,
        image_key: str | None = None,
        label_key: str | None = None,
        shape_key: str | None = None,
        group_key: str | None = None,
        group_levels: int | None = None,
        hub_split: str | None = None,
        hold_out_groups: tuple[str, ...] | list[str] = (),
        split: str = "train",
        num_classes: int = 64,
        row_filters: dict[str, list[str]] | None = None,
        cache_dir: str | None = None,
        seed: int = 0,
        samples_per_epoch: int | None = None,
    ) -> None:
        super().__init__()
        self.preset = self._resolve_preset(
            preset, repo=repo, spatial_rank=spatial_rank, encoding=encoding,
            image_key=image_key, label_key=label_key, shape_key=shape_key,
            group_key=group_key, group_levels=group_levels, hub_split=hub_split,
        )
        if self.preset.encoding not in DECODERS:
            raise ValueError(
                f"unknown encoding {self.preset.encoding!r}; expected one of {sorted(DECODERS)}"
            )
        if self.preset.spatial_rank not in SAMPLE_AXES:
            raise ValueError(
                f"spatial_rank must be 2 or 3, got {self.preset.spatial_rank}"
            )
        if len(patch_size) != self.preset.spatial_rank:
            raise ValueError(
                f"{self.preset.repo} is {self.preset.spatial_rank}D, so patch_size needs "
                f"{self.preset.spatial_rank} entries, got {list(patch_size)}"
            )
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if split == "val" and not hold_out_groups:
            raise ValueError(
                "split='val' with no hold_out_groups would be empty. Name the groups to hold "
                "out; with group_key='crop_name' and group_levels=1 those are specimen names, "
                "e.g. ['jrc_hela-2', 'jrc_mus-liver']."
            )
        if hold_out_groups and self.preset.group_key is None:
            raise ValueError("hold_out_groups needs group_key naming the column to split on")

        self.patch_size = tuple(int(p) for p in patch_size)
        self.hold_out_groups = tuple(hold_out_groups)
        self.split = split
        self.num_classes = int(num_classes)
        self.row_filters = dict(row_filters or {})
        self.cache_dir = cache_dir
        self.seed = int(seed)
        self.samples_per_epoch = samples_per_epoch

    @staticmethod
    def _resolve_preset(name: str | None, **overrides: Any) -> Preset:
        """A named preset, then anything given explicitly on top of it."""
        given = {k: v for k, v in overrides.items() if v is not None}
        if name is not None:
            if name not in PRESETS:
                raise ValueError(f"unknown preset {name!r}; expected one of {sorted(PRESETS)}")
            base = PRESETS[name]
            return Preset(**{**base.__dict__, **given})
        missing = [k for k in ("repo", "spatial_rank", "encoding") if k not in given]
        if missing:
            raise ValueError(
                f"without a preset, {missing} must be given; presets available: {sorted(PRESETS)}"
            )
        return Preset(**given)

    def build_dataset(self) -> data.Dataset:
        try:
            import datasets
        except ImportError as error:
            raise ImportError(
                "Hub datasets are read with HuggingFace `datasets`, an optional dependency of "
                "mia-train. Install it with `pip install 'mia-train[cellmap]'`."
            ) from error

        rows = datasets.load_dataset(
            self.preset.repo, split=self.preset.hub_split, cache_dir=self.cache_dir
        )

        for column, allowed in self.row_filters.items():
            if column not in rows.column_names:
                raise ValueError(
                    f"row_filters names column {column!r}, which {self.preset.repo} does not "
                    f"have; its columns are {rows.column_names}"
                )
            values = rows[column]
            rows = rows.select([i for i, v in enumerate(values) if v in allowed])

        if self.preset.group_key is not None:
            names = rows[self.preset.group_key]
            held = set(self.hold_out_groups)
            groups = [group_of(n, self.preset.group_levels) for n in names]
            wanted = (lambda g: g in held) if self.split == "val" else (lambda g: g not in held)
            keep = [i for i, g in enumerate(groups) if wanted(g)]
            if not keep:
                raise ValueError(
                    f"no rows left for split={self.split!r} after holding out {sorted(held)}; "
                    f"the groups present are {sorted(set(groups))}"
                )
            rows = rows.select(keep)

        if len(rows) == 0:
            raise ValueError(f"{self.preset.repo} yielded no rows after filtering")
        return _Rows(rows, self)

    @property
    def sample_axes(self) -> str:
        return SAMPLE_AXES[self.preset.spatial_rank]

    @classmethod
    def resolve_settings(cls, **kwargs: Any) -> dict[str, Any]:
        """Expand the preset, so the run record says what was read rather than naming a shortcut."""
        preset = cls._resolve_preset(
            kwargs.get("preset"),
            **{k: kwargs.get(k) for k in
               ("repo", "spatial_rank", "encoding", "image_key", "label_key", "shape_key",
                "group_key", "group_levels", "hub_split")},
        )
        return {**kwargs, **preset.__dict__}


# Adding a Hub dataset is an entry here, nothing more.
PRESETS: dict[str, Preset] = {
    # 293 annotated EM volumes from the CellMap challenge; raw uint8 buffers plus a shape column.
    "cellmap_3d": Preset(
        repo="eminorhan/cellmap-3d",
        spatial_rank=3,
        encoding="raw",
        image_key="volume",
        shape_key="shape",
        group_key="crop_name",
    ),
    # Every x, y and z plane through those same volumes -- 366k PNGs.
    "cellmap_2d": Preset(
        repo="eminorhan/cellmap-2d",
        spatial_rank=2,
        encoding="image",
        group_key="crop_name",
    ),
}
