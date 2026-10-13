from __future__ import annotations

import pytest
import torch
import torch.utils.data as data

from data.base import BaseDataset
from data.registry import DataRegistry


class _InMemoryDataset(data.Dataset):
    def __init__(self, n: int = 10) -> None:
        self._items = [torch.tensor([float(i)]) for i in range(n)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._items[index]


class _DummyDataset(BaseDataset):
    def __init__(self, n: int = 10) -> None:
        super().__init__()
        self._n = n
        self.build_calls = 0

    def build_dataset(self) -> data.Dataset:
        self.build_calls += 1
        return _InMemoryDataset(self._n)


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    monkeypatch.setattr(DataRegistry, "_registry", {})


@pytest.mark.unit
def test_base_dataset_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseDataset()


@pytest.mark.unit
def test_dataset_property_is_built_lazily_and_cached():
    dataset = _DummyDataset()
    assert dataset.build_calls == 0
    _ = dataset.dataset
    _ = dataset.dataset
    assert dataset.build_calls == 1


@pytest.mark.unit
def test_build_dataloader_partitions_across_ranks_without_overlap():
    dataset = _DummyDataset(n=10)
    loader_rank0 = dataset.build_dataloader(
        batch_size=1, rank=0, world_size=2, shuffle=False, drop_last=False
    )
    loader_rank1 = dataset.build_dataloader(
        batch_size=1, rank=1, world_size=2, shuffle=False, drop_last=False
    )

    seen_rank0 = {int(batch.item()) for batch in loader_rank0}
    seen_rank1 = {int(batch.item()) for batch in loader_rank1}

    assert seen_rank0.isdisjoint(seen_rank1)
    assert seen_rank0 | seen_rank1 == set(range(10))
    assert dataset.build_calls == 1


@pytest.mark.unit
def test_shuffled_order_follows_the_seed():
    """The shuffled order follows `seed`, and for one seed the ranks still split it without
    overlap: every rank draws the same permutation and takes its own share of it."""
    dataset = _DummyDataset(n=10)

    def order(seed: int, rank: int = 0, world_size: int = 1) -> list[int]:
        loader = dataset.build_dataloader(
            batch_size=1, rank=rank, world_size=world_size, drop_last=False, seed=seed
        )
        return [int(batch.item()) for batch in loader]

    assert order(0) == order(0)
    assert order(0) != order(1)
    rank0, rank1 = order(1, rank=0, world_size=2), order(1, rank=1, world_size=2)
    assert set(rank0).isdisjoint(rank1)
    assert set(rank0) | set(rank1) == set(range(10))


@pytest.mark.unit
def test_registry_register_and_build():
    DataRegistry.register("dummy")(_DummyDataset)

    assert DataRegistry.available() == ["dummy"]
    dataset = DataRegistry.build("dummy", n=4)
    assert isinstance(dataset, _DummyDataset)


@pytest.mark.unit
def test_registry_rejects_duplicate_name():
    DataRegistry.register("dummy")(_DummyDataset)
    with pytest.raises(ValueError):
        DataRegistry.register("dummy")(_DummyDataset)


@pytest.mark.unit
def test_registry_rejects_non_dataset_subclass():
    with pytest.raises(TypeError):
        DataRegistry.register("not-a-dataset")(object)


@pytest.mark.unit
def test_registry_unknown_name_raises_key_error():
    with pytest.raises(KeyError):
        DataRegistry.get("does-not-exist")
