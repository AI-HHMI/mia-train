"""Enforces the layout of `models/`, `layers/`, `algorithms/` and `prediction/`.

Three rules, the first two about being able to answer a question by listing a directory.

**`ls src/models/`** answers "what can I train?" -- every file there registers something a config
can name, plus the two that define the contract. Reusable pieces that no config ever names --
attention, transformer blocks, position encodings -- live in `src/layers/` instead.

**`ls src/algorithms/*.py`** answers "how can I train it?" -- likewise every top-level file there
registers a strategy. A strategy's own supporting code goes in a subpackage named after it
(`algorithms/dinov3/`, `algorithms/affinity/`), which keeps the top level a menu rather than a
pile. Note the subpackage cannot share a name with the module that registers the strategy, since
one would shadow the other -- hence `affinity_seg.py` beside `affinity/`.

**`src/prediction/` imports nothing of the repo's own, and nothing imports `predict.py` or
`train.py`.** The library every whole-volume prediction shares -- the lattice, the reads, the
predictor contract, the dense default -- sits below both the strategies and the entrypoint that
drives them, so a strategy can build a predictor on it at runtime. An entrypoint imported as a
library is either a circular import or, when it is the running script, a second copy of every
class it defines.

Checked rather than merely documented because a boundary like this decays quietly. Nothing fails
when a helper is dropped into `models/` or `algorithms/`; the package simply drifts back to a bag
of modules over a few months.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

import components  # noqa: F401  (populates the registries as a real run would)
from algorithms.registry import AlgorithmRegistry
from models.registry import ModelRegistry

REPO = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO / "src"

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
    forbidden = {"algorithms", "engine", "data"}
    for path in _modules_in("layers"):
        leaked = forbidden & _imported_names(path)
        assert not leaked, f"src/{path.relative_to(SRC)} imports {sorted(leaked)}"


ENTRYPOINTS = {"predict", "train"}


@pytest.mark.unit
def test_prediction_is_a_leaf_package():
    """`prediction/` may import numpy, torch and miao; nothing of this repo's own.

    It holds what every whole-volume prediction shares, and both `algorithms/` and `predict.py`
    import it at runtime. The moment it imports either of them back, the cycle that once kept this
    code inside the entrypoint -- reachable from a strategy only through a `TYPE_CHECKING` import
    -- is back.
    """
    forbidden = {"algorithms", "models", "layers", "engine", "data", "components"} | ENTRYPOINTS
    modules = _modules_in("prediction")
    assert modules, "src/prediction/ holds no modules; this check is not looking at anything"
    for path in modules:
        leaked = forbidden & _imported_names(path)
        assert not leaked, f"src/{path.relative_to(SRC)} imports {sorted(leaked)}"


@pytest.mark.unit
def test_nothing_imports_an_entrypoint():
    """`predict.py` and `train.py` are the top of the graph; no module may import them.

    Not even under `TYPE_CHECKING`: `_imported_names` reads every import statically, so a
    type-only import fails this too. Such an import hides the dependency instead of removing it --
    the shared code belongs in `prediction/`. A runtime import is worse: it either fails as a
    circular import or, when the entrypoint is the running script, executes it a second time under
    its module name and hands the importer different class objects from the ones `__main__` holds.
    """
    for path in sorted(SRC.rglob("*.py")):
        if path.parent == SRC and path.stem in ENTRYPOINTS:
            continue
        leaked = ENTRYPOINTS & _imported_names(path)
        assert not leaked, f"src/{path.relative_to(SRC)} imports entrypoint {sorted(leaked)}"


# No test that layers/ is non-empty. It was tried and removed: a layer going missing already fails
# loudly at import (`ModuleNotFoundError: No module named 'layers.common.rope'` takes the whole
# suite down at collection), so such a test guards nothing while hardcoding filenames a rename
# would break.
