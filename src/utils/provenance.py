from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _git(repo_dir: Path, *args: str) -> str | None:
    """Run a git command in `repo_dir`, or return None if git/the repo is unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def git_metadata(repo_dir: Path) -> dict[str, Any] | None:
    """Commit hash and uncommitted diff for `repo_dir`, or None when it is not a git repo.

    DESIGN.md requires this alongside every run so a result can be traced back to exact code.
    A missing repository is reported by the caller rather than failing the run.
    """
    commit = _git(repo_dir, "rev-parse", "HEAD")
    if commit is None:
        return None
    diff = _git(repo_dir, "diff", "HEAD") or ""
    return {"commit": commit.strip(), "dirty": bool(diff.strip()), "diff": diff}


def write_run_artifacts(
    output_dir: Path,
    config_path: Path,
    resolved_config: dict[str, Any],
    repo_dir: Path,
) -> None:
    """Record everything needed to reproduce this run: source config, resolved settings, code state.

    The source .toml is copied verbatim and the resolved settings are written as JSON, which
    captures defaults the source file left implicit. JSON is used because the standard library
    can read TOML but not write it, and core code must stay free of extra dependencies.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output_dir / f"config{config_path.suffix}")
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2, default=str), encoding="utf-8"
    )

    metadata = git_metadata(repo_dir)
    if metadata is None:
        (output_dir / "git_commit.txt").write_text(
            f"no git repository at {repo_dir}; run provenance unavailable\n", encoding="utf-8"
        )
        return

    (output_dir / "git_commit.txt").write_text(
        f"{metadata['commit']}{' (dirty)' if metadata['dirty'] else ''}\n", encoding="utf-8"
    )
    if metadata["dirty"]:
        (output_dir / "dirty.patch").write_text(metadata["diff"], encoding="utf-8")
