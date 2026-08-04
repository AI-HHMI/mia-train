from __future__ import annotations

from typing import Any

import torch.utils.data as data
from miao.config import MiaoConfig
from miao.dataset import VolumeDataset

from .base import BaseDataset
from .registry import DataRegistry


@DataRegistry.register("miao_volumes")
class MiaoVolumeDataset(BaseDataset):
    """Multi-scale OME-NGFF patches from `miao`, as a rank-aware mia-train dataset.

    Every key in the config's `[data]` section other than `name` is forwarded straight to
    `miao.MiaoConfig`, so the whole miao schema is reachable from a mia-train .toml and is
    captured in the run's `resolved_config.json` rather than hidden behind a path to a
    second file.

    Shape contract: with `output_axes = "lzyx"` a sample's `"img"` is `(L, Z, Y, X)` for L
    scale levels, so a collated batch is `(B, L, Z, Y, X)`. The level axis stays its own axis;
    what an encoder makes of it is the encoder's business, declared in `BaseModel.prepare_input`
    (`ViT3D` requires `L == 1`, since miao's levels are not pixel-aligned and so are not
    channels). `sample_axes` reports this order so an algorithm can pass it along.
    `output_axes` must contain `"l"`; miao rejects it otherwise.
    """

    def __init__(self, **miao_config: Any) -> None:
        super().__init__()
        # `MiaoConfig` leaves pydantic's `extra="ignore"` default in place, so a misspelled
        # `[data]` key would be dropped in silence and the run would train on miao's default
        # instead — e.g. `patch_sizes = [8, 8, 8]` yields miao's default patch size with no
        # warning. Reject unknown keys explicitly, since this section is user-authored.
        unknown = sorted(set(miao_config) - set(MiaoConfig.model_fields))
        if unknown:
            raise ValueError(
                f"unknown [data] key(s) for miao_volumes: {unknown}; "
                f"valid keys are {sorted(MiaoConfig.model_fields)}"
            )
        # Validated here rather than in build_dataset so a malformed volume list or axes
        # string fails at startup instead of on the first batch of a multi-hour job.
        self.config = MiaoConfig(**miao_config)

    @property
    def sample_axes(self) -> str | None:
        return str(self.config.output_axes)

    def build_dataset(self) -> data.Dataset:
        return VolumeDataset(self.config)
