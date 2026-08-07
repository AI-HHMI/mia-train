"""Enforces the split between `models/` and `layers/`.

The rule: **`src/models/` holds what a user can name in a config**, i.e. what is registered in
`ModelRegistry`, plus the two files that define the model contract. Everything reusable that is
never named in a config -- attention, transformer blocks, position encodings -- lives in
`src/layers/`.

Checked rather than merely documented because a boundary like this decays quietly. Nothing fails
when a block is dropped into `models/`; the package simply drifts back to a bag of modules over a
few months, and `ls src/models/` stops answering "what can I train?".
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

import components  # noqa: F401  (populates the registries as a real run would)
from models.registry import ModelRegistry

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"

# base.py defines BaseModel and registry.py the registry itself: the contract every registrable
# model implements, not models and not reusable layers.
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
