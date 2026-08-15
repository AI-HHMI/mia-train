"""Enforces the layout of `models/`, `layers/`, `algorithms/` and `experiments/`.

Three rules, the first two about being able to answer a question by listing a directory.

**`ls src/models/`** answers "what can I train?" -- every file there registers something a config
can name, plus the two that define the contract. Reusable pieces that no config ever names --
attention, transformer blocks, position encodings -- live in `src/layers/` instead.

**`ls src/algorithms/*.py`** answers "how can I train it?" -- likewise every top-level file there
registers a strategy. A strategy's own supporting code goes in a subpackage named after it
(`algorithms/dinov3/`, `algorithms/affinity/`), which keeps the top level a menu rather than a
pile. Note the subpackage cannot share a name with the module that registers the strategy, since
one would shadow the other -- hence `affinity_seg.py` beside `affinity/`.

**`experiments/` stays text.** Configs, submission scripts and write-ups belong in git; checkpoints,
TensorBoard events and predictions do not. That separation is what keeps the directory cheap to
carry: all four experiments together are ~135 KB of text, while the run artifacts they produced are
1.4 TB on `/nrs`, a ratio of about ten million to one. `[environment].checkpoint_dir` routes
artifacts out of the tree, and `utils.cluster` refuses a relative path so they cannot land here by
accident -- but a figure, a metrics dump or a small checkpoint copied in by hand would slip past
both, and git never forgets a blob.

Checked rather than merely documented because a boundary like this decays quietly. Nothing fails
when a helper is dropped into `models/` or `algorithms/`, or when a PNG is dropped into
`experiments/`; the package simply drifts back to a bag of modules over a few months.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import subprocess

import pytest

import components  # noqa: F401  (populates the registries as a real run would)
from algorithms.registry import AlgorithmRegistry
from models.registry import ModelRegistry

REPO = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO / "src"

# Generous against today's worst case -- the largest tracked file anywhere in the repo is a 25 KB
# source file, and the largest under `experiments/` an 11 KB README -- while still far below
# anything a run produces, the smallest of which is hundreds of megabytes. The point is to catch a
# category error, not to police prose, so a write-up can grow sixfold before this complains.
MAX_EXPERIMENT_FILE_BYTES = 64 * 1024

# base.py defines the ABC and registry.py the registry itself: the contract every registrable
# implementation satisfies, which is neither an implementation nor a reusable part.
MODEL_CONTRACT = {"base.py", "registry.py"}


def _modules_in(package: str) -> list[pathlib.Path]:
    """Every module in a package, including nested subpackages.

    Recursive because `layers/` is grouped into subpackages -- `common/` for blocks shared across
    architectures, `dinov3/` for one architecture's own -- and a non-recursive glob would quietly
    stop enforcing the boundary rules below on exactly the files most likely to break them.
    """
    return sorted(
        path
        for path in (SRC / package).rglob("*.py")
        if path.name != "__init__.py"
    )


def _module_name(path: pathlib.Path) -> str:
    """`src/layers/dinov3/block.py` -> `layers.dinov3.block`."""
    return ".".join(path.relative_to(SRC).with_suffix("").parts)


def _imported_names(path: pathlib.Path) -> set[str]:
    """Top-level packages this module imports, read statically so nothing has to be executed."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.unit
def test_every_model_module_registers_a_runnable_model():
    # The payoff of the split: `ls src/models/` is the menu of `[model].name` values.
    assert ModelRegistry.available(), (
        "no models registered; components.py may have stopped importing them"
    )

    for path in _modules_in("models"):
        if path.name in MODEL_CONTRACT:
            continue
        module = importlib.import_module(_module_name(path))
        found = [
            name
            for name in ModelRegistry.available()
            if ModelRegistry.get(name).__module__ == module.__name__
        ]
        assert found, (
            f"src/models/{path.name} registers no model. If it is a reusable building block "
            "rather than something a user names in a config, it belongs in src/layers/."
        )


@pytest.mark.unit
def test_every_top_level_algorithm_module_registers_a_strategy():
    """The payoff of the split: `ls src/algorithms/*.py` is the menu of `[algorithm].name` values.

    Only the top level -- a subpackage like `algorithms/dinov3/` holds that strategy's supporting
    code, which is exactly what should not appear in the menu.
    """
    assert AlgorithmRegistry.available(), (
        "no algorithms registered; components.py may have stopped importing them"
    )

    for path in sorted((SRC / "algorithms").glob("*.py")):
        if path.name in MODEL_CONTRACT or path.name == "__init__.py":
            continue
        module = importlib.import_module(_module_name(path))
        found = [
            name
            for name in AlgorithmRegistry.available()
            if AlgorithmRegistry.get(name).__module__ == module.__name__
        ]
        assert found, (
            f"src/algorithms/{path.name} registers no algorithm. If it is supporting code for one "
            "strategy, move it into that strategy's subpackage (e.g. algorithms/dinov3/)."
        )


@pytest.mark.unit
def test_algorithm_subpackages_hold_only_supporting_code():
    """The other direction: a strategy inside a subpackage would be invisible in the menu."""
    for name in AlgorithmRegistry.available():
        module = AlgorithmRegistry.get(name).__module__
        depth = len(module.split("."))
        assert depth == 2, (
            f"algorithm {name!r} is registered in {module}, which is nested; registrable "
            "strategies belong directly in src/algorithms/"
        )


@pytest.mark.unit
def test_no_layer_module_registers_a_model():
    for name in ModelRegistry.available():
        module = ModelRegistry.get(name).__module__
        assert not module.startswith("layers."), (
            f"model {name!r} is registered in {module}; registrable models belong in src/models/"
        )


@pytest.mark.unit
def test_layers_do_not_depend_on_models():
    # The dependency arrow points one way. A layer reaching back into `models/` would mean the two
    # packages are really one, and the split would stop carrying information.
    for path in _modules_in("layers"):
        assert "models" not in _imported_names(path), (
            f"src/{path.relative_to(SRC)} imports from models/; layers must not depend on models"
        )


@pytest.mark.unit
def test_layers_do_not_depend_on_algorithms_or_the_engine():
    # Same argument, one step further out: a building block that knows about training strategy or
    # the run loop is not a building block.
    forbidden = {"algorithms", "engine", "data", "evals"}
    for path in _modules_in("layers"):
        leaked = forbidden & _imported_names(path)
        assert not leaked, f"src/{path.relative_to(SRC)} imports {sorted(leaked)}"


# No test that layers/ is non-empty. It was tried and removed: a layer going missing already fails
# loudly at import (`ModuleNotFoundError: No module named 'layers.common.rope'` takes the whole
# suite down at collection), so such a test guards nothing while hardcoding filenames a rename
# would break.


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout


def _tracked_experiment_blobs() -> dict[str, int]:
    """Tracked paths under `experiments/` mapped to the size of the blob git holds for each.

    Asks git rather than walking the filesystem, because tracked is the only thing that matters
    here: an artifact left in a working tree costs nothing and `.gitignore` already covers the
    `outputs/` case, whereas a committed one is in history permanently.

    The size comes from the index rather than from `stat`, which is not the same thing twice over.
    A file staged for deletion, or moved out of the tree before the removal is staged, is still
    listed by `ls-files` and would raise `FileNotFoundError` on `stat`. And what costs the repo is
    the blob, not whatever the worktree happens to hold right now.
    """
    records = []
    for record in _git("ls-files", "-s", "-z", "--", "experiments").split("\0"):
        if not record:
            continue
        meta, path = record.split("\t", 1)
        records.append((meta.split()[1], path))

    if not records:
        return {}

    # Keyed by object id, and deliberately not built as a sha -> path mapping: identical files
    # share one blob, and this repo has three byte-identical copies of `nisb_base_256_aug.yaml`,
    # so a dict keyed the other way would silently drop two of the three paths.
    unique = sorted({sha for sha, _ in records})
    batch = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objectsize)"],
        cwd=REPO,
        input="\n".join(unique),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    sizes = {
        line.split()[0]: int(line.split()[1]) for line in batch.splitlines() if line.strip()
    }
    return {path: sizes[sha] for sha, path in records}


@pytest.mark.unit
def test_experiments_hold_no_large_files():
    """No tracked file under `experiments/` may be artifact-sized."""
    tracked = _tracked_experiment_blobs()
    # A guard that silently checks nothing is worse than no guard: if the pathspec ever stops
    # matching -- directory renamed, tests run from a tarball -- this says so instead of passing.
    assert tracked, "no tracked files under experiments/; this check is not looking at anything"

    oversized = {
        name: size for name, size in tracked.items() if size > MAX_EXPERIMENT_FILE_BYTES
    }
    assert not oversized, (
        f"tracked files under experiments/ exceed {MAX_EXPERIMENT_FILE_BYTES // 1024} KB: "
        f"{oversized}. Experiments hold configs, scripts and write-ups; artifacts belong under "
        "[environment].checkpoint_dir, which is outside the repo."
    )


@pytest.mark.unit
def test_experiments_hold_no_binary_files():
    """Size alone would let a small checkpoint or a thumbnail through.

    Enforced by extension rather than by sniffing bytes, so the failure names the offending kind
    of file and stays readable when it fires.
    """
    binary_suffixes = {
        ".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz", ".pkl",
        ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".tif", ".tiff",
        ".zip", ".gz", ".tar", ".h5", ".hdf5",
    }
    found = [
        name
        for name in _tracked_experiment_blobs()
        if pathlib.Path(name).suffix.lower() in binary_suffixes
        or ".zarr" in pathlib.Path(name).parts
        or pathlib.Path(name).name.startswith("events.out.tfevents")
    ]
    assert not found, (
        f"binary or artifact files tracked under experiments/: {found}. These belong beside the "
        "run that produced them, under [environment].checkpoint_dir."
    )
