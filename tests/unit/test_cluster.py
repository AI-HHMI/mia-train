from __future__ import annotations

import sys
from pathlib import Path

import pytest

from utils.cluster import checkpoint_dir, load_cluster_config


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "active.toml"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.unit
def test_loads_configuration_sections(tmp_path):
    path = _write(
        tmp_path,
        '[environment]\ncheckpoint_dir = "/scratch/runs"\n\n[scheduler]\nname = "lsf"\n',
    )
    config = load_cluster_config(path)
    assert config["environment"]["checkpoint_dir"] == "/scratch/runs"
    assert config["scheduler"]["name"] == "lsf"


@pytest.mark.unit
def test_absent_file_points_at_the_template(tmp_path):
    with pytest.raises(FileNotFoundError, match="template.toml"):
        load_cluster_config(tmp_path / "does_not_exist.toml")


@pytest.mark.unit
def test_checkpoint_dir_returns_configured_absolute_path(tmp_path):
    path = _write(tmp_path, '[environment]\ncheckpoint_dir = "/scratch/runs"\n')
    assert checkpoint_dir(path) == Path("/scratch/runs")


@pytest.mark.unit
def test_missing_checkpoint_dir_key_is_named(tmp_path):
    path = _write(tmp_path, '[environment]\ndataset_root = "/data"\n')
    with pytest.raises(ValueError, match="checkpoint_dir"):
        checkpoint_dir(path)


@pytest.mark.unit
def test_missing_environment_section_is_rejected(tmp_path):
    path = _write(tmp_path, '[scheduler]\nname = "lsf"\n')
    with pytest.raises(ValueError, match="checkpoint_dir"):
        checkpoint_dir(path)


@pytest.mark.unit
def test_relative_checkpoint_dir_is_rejected(tmp_path):
    path = _write(tmp_path, '[environment]\ncheckpoint_dir = "outputs"\n')
    with pytest.raises(ValueError, match="absolute"):
        checkpoint_dir(path)


@pytest.mark.unit
def test_home_relative_path_is_expanded_not_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", "/home/someone")
    path = _write(tmp_path, '[environment]\ncheckpoint_dir = "~/runs"\n')
    # expanduser must run before the absolute check, or a valid "~" path would be refused.
    assert checkpoint_dir(path) == Path("/home/someone/runs")


@pytest.mark.unit
def test_train_output_root_defaults_to_none(monkeypatch):
    import train

    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "run.toml"])
    args = train.parse_args()
    # None means engine.run.run() resolves the root from the cluster config instead.
    assert args.output_root is None
    assert args.config == Path("run.toml")
