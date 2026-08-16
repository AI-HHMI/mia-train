from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import torch.utils.data as data

from .augment import AugmentedDataset


class BaseDataset(abc.ABC):
    """Wraps a data source (e.g. a miao VolumeDataset) into a rank-aware DataLoader."""

    def __init__(self) -> None:
        self._dataset: data.Dataset | None = None
        self._transform: Any = None

    def attach_transform(self, transform: Any) -> None:
        """Apply `transform` to every sample this dataset yields.

        Lives here rather than in each dataset's constructor so augmentation is available to any
        data source without each one plumbing it through, and so the engine can attach it to the
        training set alone. Must be called before the dataloader is built; a dataset already
        materialised would otherwise keep handing out untransformed samples.
        """
        if self._dataset is not None:
            raise RuntimeError(
                "attach_transform was called after the dataset was built, so the samples already "
                "being served would not go through it"
            )
        self._transform = transform

    @abc.abstractmethod
    def build_dataset(self) -> data.Dataset:
        """Construct the underlying map-style dataset (e.g. a miao.VolumeDataset)."""

    @property
    def dataset(self) -> data.Dataset:
        if self._dataset is None:
            source = self.build_dataset()
            self._dataset = (
                source if self._transform is None else AugmentedDataset(source, self._transform)
            )
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
            # A volumetric crop is large enough that how it reaches the GPU is a real cost, not a
            # detail. At 256^3 one sample is 134 MB (a float32 image beside an int32 label), and
            # out of pageable memory the driver cannot DMA it directly: it stages through an
            # internal bounce buffer at roughly a fifth of the link's rate, and `.to(non_blocking=
            # True)` silently degrades to a blocking copy because there is no way to know when a
            # pageable page may be reused. Measured in a trace of this repo's affinity fine-tune,
            # that copy was 21.7 ms of every 377 ms step, spent as `Memcpy HtoD (Pageable ->
            # Device)`. Pinning makes the same transfer a direct DMA and lets it overlap the
            # previous step's compute.
            #
            # Conditional on there being workers to pin in: the pinning thread only exists on the
            # multi-process path, and asking for it with `num_workers=0` adds a synchronous copy
            # to the main process for no benefit.
            pin_memory=num_workers > 0,
            # Without this the iterator is torn down and rebuilt every time the loader is
            # exhausted, which for a step-based run is far more often than it sounds: with
            # `samples_per_epoch` 1000 over 8 ranks at batch 1, an epoch is 125 steps, so a 100k
            # step run forks its workers 800 times. Each respawn re-imports torch and re-opens
            # every zarr store in the dataset. The same trace caught one such boundary costing
            # ~1.9 s -- `data_wait_frac` for that logging window read 0.33 against a 0.06 steady
            # state, or about 15 ms amortized over every step of the run.
            persistent_workers=num_workers > 0,
        )
