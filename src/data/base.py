from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import torch
import torch.utils.data as data

from .augment import TransformedDataset


def _single_threaded_worker(worker_id: int) -> None:
    """Hold each dataloader worker to one thread.

    A worker inherits `OMP_NUM_THREADS` from the job, and this repo's submission scripts set it to
    4 so that the *training* process does not thread to the node's full core count. Workers then
    inherit that too, and the arithmetic stops working: eight ranks of six workers at four threads
    each is 192 threads on a 96-core node, against a real demand of about seven cores.

    That oversubscription is not free, because the training loop is launch-bound rather than
    compute-bound -- a profile of the affinity fine-tune shows the host inside `backward` for
    218 ms of a 325 ms step, barely able to issue kernels fast enough to keep the device fed. CPU
    taken from that process is CPU the GPU waits for, and the measurement that isolates it is
    stark: moving the connected-components pass into the workers left GPU *busy* time unchanged
    (235.5 ms against 237.2 ms for a run that skipped the work entirely) while GPU *idle* rose
    16.5 ms per step. The device was not doing more work; the host had less time to feed it.

    One thread rather than a tunable count, because `num_workers` is already the knob for how much
    parallelism the input pipeline gets, and per-worker threading on top of it double-counts: two
    settings whose product has to stay under the core budget, where one would do.
    """
    torch.set_num_threads(1)


class BaseDataset(abc.ABC):
    """Wraps a data source (e.g. a miao VolumeDataset) into a rank-aware DataLoader."""

    def __init__(self) -> None:
        self._dataset: data.Dataset | None = None
        self._transforms: tuple[Any, ...] = ()

    def attach_transform(self, transform: Any) -> None:
        """Apply `transform` to every sample this dataset yields, after any already attached.

        Lives here rather than in each dataset's constructor so a transform is available to any
        data source without each one plumbing it through, and so the engine can decide which
        datasets receive which. Must be called before the dataloader is built; a dataset already
        materialised would otherwise keep handing out untransformed samples.

        Transforms compose rather than replace, because two unrelated callers attach them: the
        engine attaches augmentation to the training set, and an algorithm attaches per-sample
        preprocessing it needs done in the workers. Replacing would mean whichever ran second
        silently discarded the other, and the symptom -- a run that trains without the
        augmentation its config asked for -- is invisible in every metric.
        """
        if self._dataset is not None:
            raise RuntimeError(
                "attach_transform was called after the dataset was built, so the samples already "
                "being served would not go through it"
            )
        self._transforms = (*self._transforms, transform)

    @abc.abstractmethod
    def build_dataset(self) -> data.Dataset:
        """Construct the underlying map-style dataset (e.g. a miao.VolumeDataset)."""

    @property
    def dataset(self) -> data.Dataset:
        if self._dataset is None:
            source = self.build_dataset()
            self._dataset = (
                TransformedDataset(source, self._transforms) if self._transforms else source
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
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
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
            worker_init_fn=_single_threaded_worker if num_workers > 0 else None,
            # Defaults off; see `TrainerConfig.persistent_workers` for the measurement. Briefly:
            # respawning workers each epoch costs ~4 ms per step amortized, and keeping them alive
            # costs ~68 ms, because a process that never restarts never gives back the memory a
            # 134 MB sample and a 256^3 components pass churn through.
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
