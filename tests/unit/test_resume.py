from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

from engine.run import RESUME_LATEST, _latest_run_dir, _report_resumed_config, resolve_output_dir
from utils.config import diff_resolved, flatten_resolved
from utils.provenance import write_run_artifacts


def _run_dir(root: Path, name: str, resolved: dict | None = None) -> Path:
    path = root / name
    path.mkdir(parents=True)
    if resolved is not None:
        (path / "resolved_config.json").write_text(json.dumps(resolved), encoding="utf-8")
    return path


@pytest.mark.unit
def test_flatten_resolved_uses_dotted_paths():
    flat = flatten_resolved({"model": {"kwargs": {"embed_dim": 768}}, "trainer": {"lr": 0.1}})
    assert flat == {"model.kwargs.embed_dim": 768, "trainer.lr": 0.1}


@pytest.mark.unit
def test_diff_resolved_reports_only_what_changed():
    old = {"trainer": {"lr": 0.1, "max_steps": 50}}
    new = {"trainer": {"lr": 0.1, "max_steps": 80}}
    assert diff_resolved(old, new) == {"trainer.max_steps": (50, 80)}


@pytest.mark.unit
def test_diff_resolved_reports_added_and_removed_keys():
    assert diff_resolved({"a": 1}, {"b": 2}) == {"a": (1, None), "b": (None, 2)}


@pytest.mark.unit
def test_diff_resolved_is_empty_for_identical_configs():
    config = {"trainer": {"lr": 0.1}, "model": {"kwargs": {"depth": 6}}}
    assert diff_resolved(config, config) == {}


@pytest.mark.unit
def test_latest_run_dir_is_none_when_nothing_exists(tmp_path):
    assert _latest_run_dir(tmp_path, "exp") is None
    assert _latest_run_dir(tmp_path / "not_created", "exp") is None


@pytest.mark.unit
def test_latest_run_dir_picks_the_newest_stamp(tmp_path):
    _run_dir(tmp_path, "exp_20260101_000000")
    newest = _run_dir(tmp_path, "exp_20260804_120000")
    _run_dir(tmp_path, "exp_20260315_235959")
    assert _latest_run_dir(tmp_path, "exp") == newest


@pytest.mark.unit
def test_latest_run_dir_ignores_other_experiments(tmp_path):
    _run_dir(tmp_path, "other_20260804_120000")
    mine = _run_dir(tmp_path, "exp_20260101_000000")
    assert _latest_run_dir(tmp_path, "exp") == mine


@pytest.mark.unit
def test_without_resume_a_fresh_directory_is_minted(tmp_path):
    existing = _run_dir(tmp_path, "exp_20260101_000000")
    chosen = resolve_output_dir(tmp_path, "exp")
    assert chosen != existing
    assert chosen.name.startswith("exp_")


@pytest.mark.unit
def test_resume_latest_reuses_the_newest_directory(tmp_path):
    _run_dir(tmp_path, "exp_20260101_000000")
    newest = _run_dir(tmp_path, "exp_20260804_120000")
    assert resolve_output_dir(tmp_path, "exp", "latest") == newest


@pytest.mark.unit
def test_resume_latest_falls_back_to_a_fresh_directory(tmp_path):
    # This is what lets one submission script serve the first launch and every resubmission.
    chosen = resolve_output_dir(tmp_path, "exp", "latest")
    assert chosen.name.startswith("exp_")
    assert not chosen.exists()


@pytest.mark.unit
def test_resume_accepts_an_explicit_directory(tmp_path):
    target = _run_dir(tmp_path, "exp_20260101_000000")
    assert resolve_output_dir(tmp_path, "exp", str(target)) == target


@pytest.mark.unit
def test_resume_rejects_a_directory_that_does_not_exist(tmp_path):
    with pytest.raises(ValueError, match="not an existing directory"):
        resolve_output_dir(tmp_path, "exp", str(tmp_path / "nope"))


@pytest.mark.unit
def test_a_fresh_run_has_nothing_to_compare(tmp_path):
    _report_resumed_config(tmp_path / "absent", {"trainer": {"lr": 0.1}})  # must not raise


@pytest.mark.unit
def test_identical_settings_report_nothing(tmp_path, capsys):
    resolved = {"trainer": {"lr": 0.1}}
    directory = _run_dir(tmp_path, "exp_1", resolved)
    _report_resumed_config(directory, resolved)
    assert capsys.readouterr().out == ""


@pytest.mark.unit
def test_a_changed_hyperparameter_warns_and_continues(tmp_path, capsys):
    directory = _run_dir(tmp_path, "exp_1", {"trainer": {"lr": 0.1, "max_steps": 50}})
    _report_resumed_config(directory, {"trainer": {"lr": 0.1, "max_steps": 80}})

    printed = capsys.readouterr().out
    assert "[resume]" in printed
    assert "trainer.max_steps" in printed
    assert "50" in printed and "80" in printed


@pytest.mark.unit
def test_a_changed_architecture_is_refused(tmp_path):
    # The checkpoint holds differently-shaped parameters, so fail here rather than deep in DCP.
    directory = _run_dir(tmp_path, "exp_1", {"model": {"kwargs": {"embed_dim": 384}}})
    with pytest.raises(ValueError, match="architecture changed"):
        _report_resumed_config(directory, {"model": {"kwargs": {"embed_dim": 256}}})


@pytest.mark.unit
def test_a_changed_algorithm_is_refused(tmp_path):
    directory = _run_dir(tmp_path, "exp_1", {"algorithm": {"name": "mae"}})
    with pytest.raises(ValueError, match="architecture changed"):
        _report_resumed_config(directory, {"algorithm": {"name": "simmim"}})


@pytest.mark.unit
def test_a_changed_algorithm_hyperparameter_only_warns(tmp_path, capsys):
    directory = _run_dir(
        tmp_path, "exp_1", {"algorithm": {"name": "mae", "kwargs": {"mask_ratio": 0.75}}}
    )
    _report_resumed_config(directory, {"algorithm": {"name": "mae", "kwargs": {"mask_ratio": 0.5}}})
    assert "mask_ratio" in capsys.readouterr().out


@pytest.mark.unit
def test_each_attempt_appends_to_the_attempts_log(tmp_path):
    # `git_commit.txt` and `resolved_config.json` describe only the attempt that wrote them, and
    # a resume overwrites both, so the code state of the attempt that produced the earlier steps
    # in the checkpoint would otherwise be lost.
    output_dir = tmp_path / "exp_20260101_000000"
    config_path = tmp_path / "config.toml"
    config_path.write_text('experiment_name = "exp"\n', encoding="utf-8")
    absent_repo = tmp_path / "not_a_repo"

    write_run_artifacts(output_dir, config_path, {"trainer": {"max_steps": 50}}, absent_repo)
    write_run_artifacts(output_dir, config_path, {"trainer": {"max_steps": 80}}, absent_repo)

    lines = (output_dir / "attempts.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        stamp, _, detail = line.partition("  ")
        datetime.fromisoformat(stamp)  # raises if the attempt is not stamped with a real time
        assert detail == "no git repository"

    # The single-attempt artifacts hold the newest attempt only, which is why the log is needed.
    stored = json.loads((output_dir / "resolved_config.json").read_text(encoding="utf-8"))
    assert stored["trainer"]["max_steps"] == 80


@pytest.mark.unit
def test_omitting_the_resume_flag_starts_a_fresh_run(monkeypatch):
    import train

    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "run.toml"])
    assert train.parse_args().resume is None


@pytest.mark.unit
def test_a_bare_resume_flag_means_latest(monkeypatch):
    # `--resume` with no value is what makes a submission script idempotent, so the flag's const
    # has to be the sentinel `resolve_output_dir` recognizes — not merely a truthy string.
    import train

    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "run.toml", "--resume"])
    assert train.parse_args().resume == RESUME_LATEST


@pytest.mark.unit
def test_the_resume_flag_accepts_a_run_directory(monkeypatch):
    import train

    argv = ["train.py", "--config", "run.toml", "--resume", "/runs/exp_1"]
    monkeypatch.setattr(sys, "argv", argv)
    assert train.parse_args().resume == "/runs/exp_1"
