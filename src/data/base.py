from __future__ import annotations

import abc
from pathlib import Path
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

    @classmethod
    def resolve_settings(cls, **kwargs: Any) -> dict[str, Any]:
        """What a `[data]` section actually amounts to, for the run record.

        Called on the class, without building anything, so the engine can write a complete
        `resolved_config.json` before any data is touched. The default is the section as written,
        which is already complete for a dataset configured entirely inline.

        A dataset that draws settings from somewhere else -- another file, an environment, a
        service -- overrides this to return what it resolved to. Otherwise the run record would
        preserve only a reference, and the reference can change afterwards while the record still
        claims to describe the run.
        """
        return dict(kwargs)

    @classmethod
    def referenced_files(cls, **kwargs: Any) -> tuple[Path, ...]:
        """Files a `[data]` section points at, to be copied into the run directory.

        `resolve_settings` already captures every value, so this is for the file itself: comments
        and structure that explain *why* the values are what they are, which no dump preserves.
        """
        return ()

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
