from __future__ import annotations

import pytest

from utils.registry import Registry


class _Base:
    pass


class _Concrete(_Base):
    pass


@pytest.mark.unit
def test_register_get_build_available():
    registry = Registry(_Base)
    registry.register("concrete")(_Concrete)

    assert registry.available() == ["concrete"]
    assert registry.get("concrete") is _Concrete
    assert isinstance(registry.build("concrete"), _Concrete)


@pytest.mark.unit
def test_register_rejects_duplicate_name():
    registry = Registry(_Base)
    registry.register("concrete")(_Concrete)
    with pytest.raises(ValueError):
        registry.register("concrete")(_Concrete)


@pytest.mark.unit
def test_register_rejects_non_subclass():
    registry = Registry(_Base)
    with pytest.raises(TypeError):
        registry.register("not-a-base")(object)


@pytest.mark.unit
def test_get_unknown_name_raises_key_error():
    registry = Registry(_Base)
    with pytest.raises(KeyError):
        registry.get("does-not-exist")


@pytest.mark.unit
def test_two_registry_instances_are_independent():
    registry_a = Registry(_Base)
    registry_b = Registry(_Base)
    registry_a.register("concrete")(_Concrete)

    assert registry_a.available() == ["concrete"]
    assert registry_b.available() == []
