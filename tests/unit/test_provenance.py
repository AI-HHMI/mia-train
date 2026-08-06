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


@pytest.mark.unit
def test_referenced_config_files_are_copied_in(tmp_path):
    # The resolved JSON holds the values; this keeps the file itself, whose comments explain why
    # the values are what they are. A dump preserves neither comments nor structure.
    config = tmp_path / "run.toml"
    config.write_text('experiment_name = "x"\n', encoding="utf-8")
    dataset = tmp_path / "shared_dataset.yaml"
    dataset.write_text("# why 33nm and not 32\npatch_size: [64, 64, 64]\n", encoding="utf-8")

    out = tmp_path / "run"
    write_run_artifacts(out, config, {"experiment_name": "x"}, tmp_path, (dataset,))

    copied = out / "shared_dataset.yaml"
    assert copied.is_file()
    assert "# why 33nm and not 32" in copied.read_text(), "comments must survive the copy"


@pytest.mark.unit
def test_the_copy_is_independent_of_the_original(tmp_path):
    config = tmp_path / "run.toml"
    config.write_text('experiment_name = "x"\n', encoding="utf-8")
    dataset = tmp_path / "shared.yaml"
    dataset.write_text("samples_per_epoch: 4\n", encoding="utf-8")

    out = tmp_path / "run"
    write_run_artifacts(out, config, {}, tmp_path, (dataset,))
    dataset.write_text("samples_per_epoch: 999\n", encoding="utf-8")

    assert "4" in (out / "shared.yaml").read_text()


@pytest.mark.unit
def test_two_different_files_with_one_name_are_refused(tmp_path):
    # Copying both under the same basename would silently drop one, and this directory's whole
    # job is to still be true in a year.
    config = tmp_path / "run.toml"
    config.write_text('experiment_name = "x"\n', encoding="utf-8")
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    first = tmp_path / "train" / "data.yaml"
    second = tmp_path / "val" / "data.yaml"
    first.write_text("samples_per_epoch: 1\n", encoding="utf-8")
    second.write_text("samples_per_epoch: 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="both named 'data.yaml'"):
        write_run_artifacts(tmp_path / "run", config, {}, tmp_path, (first, second))


@pytest.mark.unit
def test_the_same_file_referenced_twice_is_fine(tmp_path):
    # Train and validation sections naming one dataset file is ordinary, not a collision.
    config = tmp_path / "run.toml"
    config.write_text('experiment_name = "x"\n', encoding="utf-8")
    shared = tmp_path / "data.yaml"
    shared.write_text("samples_per_epoch: 1\n", encoding="utf-8")

    out = tmp_path / "run"
    write_run_artifacts(out, config, {}, tmp_path, (shared, shared))
    assert (out / "data.yaml").is_file()
