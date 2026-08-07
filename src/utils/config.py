from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from distributed.parallel_dims import ParallelDims
from engine.config import InitConfig, TrainerConfig

_COMPONENT_SECTIONS = ("model", "algorithm", "data")


@dataclass(frozen=True)
class ComponentConfig:
    """A registry lookup key plus the keyword arguments to construct it with."""

    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    """One fully resolved training run, assembled from a .toml file."""

    experiment_name: str
    model: ComponentConfig
    algorithm: ComponentConfig
    data: ComponentConfig
    trainer: TrainerConfig
    parallelism: ParallelDims = field(default_factory=ParallelDims)
    val_data: ComponentConfig | None = None
    init: InitConfig = field(default_factory=InitConfig)


def _component(raw: dict[str, Any], section: str) -> ComponentConfig:
    body = dict(raw[section])
    if "name" not in body:
        raise ValueError(f"[{section}] must set 'name' to select a registered implementation")
    return ComponentConfig(name=body.pop("name"), kwargs=body)


def _dataclass_from_section(cls: Any, body: dict[str, Any], section: str) -> Any:
    """Build a dataclass from a config section, naming unknown keys instead of raising TypeError."""
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(body) - valid)
    if unknown:
        raise ValueError(
            f"[{section}] has unknown key(s) {unknown}; valid keys are {sorted(valid)}"
        )
    return cls(**body)


def load_run_config(path: Path) -> RunConfig:
    """Parse a run configuration. Raises ValueError with the offending section named on any
    missing section, missing 'name', or unrecognized key."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    if "experiment_name" not in raw:
        raise ValueError("config must set a top-level 'experiment_name'")
    missing = [section for section in (*_COMPONENT_SECTIONS, "trainer") if section not in raw]
    if missing:
        raise ValueError(f"config is missing required section(s): {missing}")

    return RunConfig(
        experiment_name=raw["experiment_name"],
        model=_component(raw, "model"),
        algorithm=_component(raw, "algorithm"),
        data=_component(raw, "data"),
        trainer=_dataclass_from_section(TrainerConfig, dict(raw["trainer"]), "trainer"),
        parallelism=_dataclass_from_section(
            ParallelDims, dict(raw.get("parallelism", {})), "parallelism"
        ),
        val_data=_component(raw, "val_data") if "val_data" in raw else None,
        init=_dataclass_from_section(InitConfig, dict(raw.get("init", {})), "init"),
    )


def as_plain_dict(config: RunConfig) -> dict[str, Any]:
    """Fully resolved settings, including defaults absent from the source file."""
    return dataclasses.asdict(config)


def flatten_resolved(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested resolved settings as dotted paths, e.g. {"model.kwargs.embed_dim": 768}."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_resolved(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def diff_resolved(old: dict[str, Any], new: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Settings that differ between two resolved configs, as path -> (old, new).

    Used when resuming, to say plainly what changed since the run was started rather than
    letting a silent difference alter training half way through.
    """
    before, after = flatten_resolved(old), flatten_resolved(new)
    return {
        path: (before.get(path), after.get(path))
        for path in sorted(set(before) | set(after))
        if before.get(path) != after.get(path)
    }
