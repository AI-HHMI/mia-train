"""`[init].target` decides whether a checkpoint reaches the algorithm's own parameters.

The default, `"model"`, loads into the bare encoder before the algorithm wraps it, so a strategy's
head keeps the initialisation it was built with. `"algorithm"` loads into the algorithm instead,
which is what makes a *true* warm start possible -- continuing a previous run's whole model,
head included, in a new run with a fresh optimizer and step counter.

These tests pin the routing rather than the copying: `load_pretrained` is exercised by
`test_pretrained.py`, and what is easy to break here is *where* it gets called from.
"""

from __future__ import annotations

import pytest

from engine.config import InitConfig


class TestInitTargetValidation:
    def test_defaults_to_model(self) -> None:
        assert InitConfig().target == "model"

    def test_accepts_algorithm(self) -> None:
        assert InitConfig(path="ckpt", target="algorithm").target == "algorithm"

    def test_rejects_unknown_target(self) -> None:
        with pytest.raises(ValueError, match="must be 'model' or 'algorithm'"):
            InitConfig(path="ckpt", target="nonsense")

    def test_target_without_path_is_refused(self) -> None:
        """Same rule the other options follow: a load option with nothing to load is a typo.

        Without this, `target = "algorithm"` and no `path` would start from scratch in silence --
        exactly the failure the surrounding check exists to prevent.
        """
        with pytest.raises(ValueError, match="no 'path'"):
            InitConfig(target="algorithm")


class TestInitTargetRouting:
    """Which of the two load sites fires, for each target."""

    def _config(self, monkeypatch, target: str):
        from engine import run as run_module

        calls: list[str] = []
        monkeypatch.setattr(
            run_module, "load_pretrained", lambda module, path, **kw: calls.append(path)
        )
        return calls, run_module

    def test_model_target_loads_in_prepare_model(self, monkeypatch) -> None:
        calls, run_module = self._config(monkeypatch, "model")

        class Cfg:
            class model:  # noqa: N801
                name = "_stub"
                kwargs: dict = {}

            class lora:  # noqa: N801
                @staticmethod
                def enabled() -> bool:
                    return False

            init = InitConfig(path="ckpt", prefix="model.")

        monkeypatch.setattr(
            run_module.ModelRegistry, "build", staticmethod(lambda name, **kw: object())
        )
        run_module.prepare_model(Cfg)
        assert calls == ["ckpt"], "target='model' must load inside prepare_model"

    def test_algorithm_target_skips_prepare_model(self, monkeypatch) -> None:
        calls, run_module = self._config(monkeypatch, "algorithm")

        class Cfg:
            class model:  # noqa: N801
                name = "_stub"
                kwargs: dict = {}

            class lora:  # noqa: N801
                @staticmethod
                def enabled() -> bool:
                    return False

            init = InitConfig(path="ckpt", target="algorithm")

        monkeypatch.setattr(
            run_module.ModelRegistry, "build", staticmethod(lambda name, **kw: object())
        )
        run_module.prepare_model(Cfg)
        assert calls == [], (
            "target='algorithm' must NOT load in prepare_model -- the algorithm's own parameters "
            "do not exist yet, so loading here would silently drop them"
        )
