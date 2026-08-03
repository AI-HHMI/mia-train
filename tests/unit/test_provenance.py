from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from utils.provenance import git_metadata, write_run_artifacts


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "tracked.txt").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "initial"], check=True)


@pytest.mark.unit
def test_git_metadata_is_none_outside_a_repository(tmp_path):
    assert git_metadata(tmp_path) is None


@pytest.mark.unit
def test_git_metadata_reports_clean_commit(tmp_path):
    _init_repo(tmp_path)
    metadata = git_metadata(tmp_path)

    assert metadata is not None
    assert len(metadata["commit"]) == 40
    assert metadata["dirty"] is False
    assert metadata["diff"] == ""


@pytest.mark.unit
def test_git_metadata_reports_dirty_diff(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "tracked.txt").write_text("modified\n", encoding="utf-8")
    metadata = git_metadata(tmp_path)

    assert metadata is not None
    assert metadata["dirty"] is True
    assert "modified" in metadata["diff"]


@pytest.mark.unit
def test_write_run_artifacts_records_config_and_resolved_settings(tmp_path):
    config_path = tmp_path / "run.toml"
    config_path.write_text('experiment_name = "demo"\n', encoding="utf-8")
    output_dir = tmp_path / "out"

    write_run_artifacts(output_dir, config_path, {"trainer": {"seed": 7}}, tmp_path)

    copied = (output_dir / "config.toml").read_text(encoding="utf-8")
    assert copied == 'experiment_name = "demo"\n'
    resolved = json.loads((output_dir / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["trainer"]["seed"] == 7


@pytest.mark.unit
def test_write_run_artifacts_notes_absent_repository(tmp_path):
    config_path = tmp_path / "run.toml"
    config_path.write_text("x = 1\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    write_run_artifacts(output_dir, config_path, {}, tmp_path)

    assert "no git repository" in (output_dir / "git_commit.txt").read_text(encoding="utf-8")
    assert not (output_dir / "dirty.patch").exists()


@pytest.mark.unit
def test_write_run_artifacts_saves_dirty_patch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")

    config_path = tmp_path / "run.toml"
    config_path.write_text("x = 1\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    write_run_artifacts(output_dir, config_path, {}, repo)

    assert "(dirty)" in (output_dir / "git_commit.txt").read_text(encoding="utf-8")
    assert "changed" in (output_dir / "dirty.patch").read_text(encoding="utf-8")
