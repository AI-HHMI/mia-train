from __future__ import annotations

import abc
from typing import Any

import torch.utils.data as data


class BaseDataset(abc.ABC):
    """Wraps a data source (e.g. a miao VolumeDataset) into a rank-aware DataLoader."""

    def __init__(self) -> None:
        self._dataset: data.Dataset | None = None

    @abc.abstractmethod
    def build_dataset(self) -> data.Dataset:
        """Construct the underlying map-style dataset (e.g. a miao.VolumeDataset)."""

    @property
    def dataset(self) -> data.Dataset:
        if self._dataset is None:
            self._dataset = self.build_dataset()
        return self._dataset

    @property
    def sample_axes(self) -> str | None:
        """Axis order of one sample, if the source defines one (e.g. miao's "lcxyz").

        Lets an algorithm adopt the data's layout instead of having it restated in config,
        where the two could drift apart. `None` means the source makes no such promise.
        """
        return None

    def build_dataloader(
        self,
        *,
        batch_size: int,
        rank: int,
        world_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = True,
    ) -> data.DataLoader:
        # torch's `DistributedSampler.__init__` stub takes an unparameterized
        # `Dataset` for its `dataset` arg, so mypy can't infer the sampler's type
        # parameter from the call; annotate it explicitly instead.
        sampler: data.DistributedSampler[Any] = data.DistributedSampler(
            self.dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        return data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=drop_last,
        )
