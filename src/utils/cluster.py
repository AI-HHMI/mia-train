from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_CONFIG_PATH = _REPO_ROOT / "configs" / "cluster" / "active.toml"
TEMPLATE_CONFIG_PATH = _REPO_ROOT / "configs" / "cluster" / "template.toml"


def load_cluster_config(path: Path | None = None) -> dict[str, Any]:
    """Read this machine's cluster configuration.

    Kept out of version control, so an absent file is a setup step the user has not done yet
    rather than a bug — say so plainly instead of failing deep inside a training run.
    """
    config_path = path or ACTIVE_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(
            f"no cluster configuration at {config_path}. Copy "
            f"{TEMPLATE_CONFIG_PATH} to active.toml and fill in this machine's paths."
        )
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def checkpoint_dir(path: Path | None = None) -> Path:
    """Root directory for run artifacts, from [environment].checkpoint_dir.

    Required to be absolute: run artifacts must land in the same place no matter which
    directory a job happened to be launched from.
    """
    config_path = path or ACTIVE_CONFIG_PATH
    config = load_cluster_config(config_path)
    value = config.get("environment", {}).get("checkpoint_dir")
    if value is None:
        raise ValueError(f"{config_path} must set [environment].checkpoint_dir")

    resolved = Path(value).expanduser()
    if not resolved.is_absolute():
        raise ValueError(
            f"[environment].checkpoint_dir in {config_path} must be an absolute path so run "
            f"artifacts do not depend on the working directory; got {value!r}"
        )
    return resolved
