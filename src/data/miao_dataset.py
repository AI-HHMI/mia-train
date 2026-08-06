from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.utils.data as data
from miao.config import MiaoConfig, load_config
from miao.dataset import VolumeDataset

from .base import BaseDataset
from .registry import DataRegistry

# Relative `config_path` values resolve against the repo, never the process's working directory.
# A submitted job's cwd is wherever it was launched from, which is routinely not the repo, and a
# path that silently means something different under `bsub` than it does interactively is the kind
# of thing that fails after the queue wait rather than at submission.
_REPO_ROOT = Path(__file__).resolve().parents[2]


@DataRegistry.register("miao_volumes")
class MiaoVolumeDataset(BaseDataset):
    """Multi-scale OME-NGFF patches from `miao`, as a rank-aware mia-train dataset.

    The dataset can be described two ways, and they compose:

    * **Inline.** Every `[data]` key other than `name` is forwarded straight to `miao.MiaoConfig`,
      so the whole miao schema is reachable from the .toml and lands in the run's
      `resolved_config.json` verbatim.
    * **By reference.** `config_path` points at a miao YAML — the format miao itself uses — which
      is loaded first. Inline keys then override what the file sets, so one shared dataset
      definition can be reused across runs and adjusted per run (a smaller `patch_size` for a
      smoke test, say) without copying it.

    Both are equally reproducible: `resolve_settings` expands a referenced file into plain values
    before the run record is written, so `resolved_config.json` describes the dataset in full
    either way and editing the YAML afterwards cannot rewrite history. The run directory also keeps
    a copy of the file itself. So the choice is only about convenience — inline when a dataset
    belongs to one run, `config_path` when several runs share one definition.

    Shape contract: with `output_axes = "lzyx"` a sample's `"img"` is `(L, Z, Y, X)` for L
    scale levels, so a collated batch is `(B, L, Z, Y, X)`. The level axis stays its own axis;
    what an encoder makes of it is the encoder's business, declared in `BaseModel.prepare_input`
    (`ViT3D` requires `L == 1`, since miao's levels are not pixel-aligned and so are not
    channels). `sample_axes` reports this order so an algorithm can pass it along.
    `output_axes` must contain `"l"`; miao rejects it otherwise.
    """

    def __init__(self, config_path: str | Path | None = None, **miao_config: Any) -> None:
        super().__init__()
        # Built from the same classmethod the engine uses for the run record, so the record and
        # the dataset cannot describe different things. Validated here rather than in
        # build_dataset so a malformed volume list or axes string fails at startup instead of on
        # the first batch of a multi-hour job.
        self.config = MiaoConfig(**self.resolve_settings(config_path=config_path, **miao_config))

    @classmethod
    def resolve_settings(
        cls, config_path: str | Path | None = None, **miao_config: Any
    ) -> dict[str, Any]:
        """The complete miao settings this `[data]` section amounts to, as plain values.

        Expanded rather than left as a path, so `resolved_config.json` describes the dataset in
        full: a referenced YAML can be edited afterwards, and a run record that kept only the
        reference would then misreport what it trained on. Expanding also fills in miao's own
        defaults, so the record survives a miao version that changes one.
        """
        # `MiaoConfig` leaves pydantic's `extra="ignore"` default in place, so a misspelled
        # `[data]` key would be dropped in silence and the run would train on miao's default
        # instead — e.g. `patch_sizes = [8, 8, 8]` yields miao's default patch size with no
        # warning. Reject unknown keys explicitly, since this section is user-authored.
        unknown = sorted(set(miao_config) - set(MiaoConfig.model_fields))
        if unknown:
            raise ValueError(
                f"unknown [data] key(s) for miao_volumes: {unknown}; "
                f"valid keys are {sorted(MiaoConfig.model_fields)} (plus 'config_path')"
            )

        settings: dict[str, Any] = {}
        if config_path is not None:
            settings.update(cls._from_yaml(config_path))
        settings.update(miao_config)
        # mode="json" so the result drops straight into resolved_config.json.
        return MiaoConfig(**settings).model_dump(mode="json")

    @classmethod
    def referenced_files(
        cls, config_path: str | Path | None = None, **miao_config: Any
    ) -> tuple[Path, ...]:
        """The referenced YAML, if any, so the run directory keeps a copy with its comments."""
        return () if config_path is None else (cls._resolve_path(config_path),)

    @staticmethod
    def _resolve_path(config_path: str | Path) -> Path:
        """Locate a referenced config, resolving a relative path against the repository."""
        resolved = Path(config_path)
        if not resolved.is_absolute():
            resolved = _REPO_ROOT / resolved
        if not resolved.is_file():
            raise FileNotFoundError(
                f"[data] config_path={str(config_path)!r} does not exist (looked at {resolved}). "
                "Relative paths resolve against the repository root, not the working directory."
            )
        return resolved

    @classmethod
    def _from_yaml(cls, config_path: str | Path) -> dict[str, Any]:
        """Read a miao YAML into plain settings that inline keys can then override."""
        # Parsed by miao itself rather than by a yaml call here, so the file is validated against
        # the schema that owns it and a bad one is reported against the file, not against the
        # merged result. `model_dump` then re-exposes it as plain settings to merge into.
        return load_config(cls._resolve_path(config_path)).model_dump()

    @property
    def sample_axes(self) -> str | None:
        return str(self.config.output_axes)

    def build_dataset(self) -> data.Dataset:
        return VolumeDataset(self.config)
