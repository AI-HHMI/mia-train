"""Moving the connected-components pass into the dataloader's workers must not change targets.

The whole justification for the `affinity` extra is that `cc3d` and the repo's own
`relabel_connected` produce the *same partition* of a crop into components -- neither promises
particular ids, and `affinities_from_labels` only ever evaluates `a == b` and `a > 0`, so the
grouping is the entire contract. If that ever stopped holding, an affinity run would train
against subtly different targets depending on whether an optional dependency happened to be
installed, and nothing in the loss would say so.

The `cc3d` tests skip when the extra is absent rather than failing, since `affinity_seg` is
deliberately correct without it.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from algorithms.affinity.targets import (
    SplitDisconnectedLabels,
    affinities_from_labels,
    affinity_offsets,
    cc3d_available,
    relabel_connected,
    relabel_connected_cc3d,
)
from data.augment import TransformedDataset
from data.base import BaseDataset

needs_cc3d = pytest.mark.skipif(not cc3d_available(), reason="needs the 'affinity' extra (cc3d)")


def _same_partition(reference: torch.Tensor, candidate: torch.Tensor) -> bool:
    """Do two labelings group the voxels identically, whatever ids they used to say so?"""
    _, left = torch.unique(reference, return_inverse=True)
    right_ids, right = torch.unique(candidate, return_inverse=True)
    joint = torch.unique(left * len(right_ids) + right)
    return len(torch.unique(left)) == len(right_ids) == len(joint)


@pytest.mark.unit
@needs_cc3d
@pytest.mark.parametrize("seed", range(6))
def test_cc3d_and_torch_agree_on_random_volumes(seed: int) -> None:
    """The property the extra rests on, on the same volumes the torch version is pinned against."""
    torch.manual_seed(seed)
    labels = (torch.rand(8, 8, 8) < 0.45).long() * torch.randint(1, 4, (8, 8, 8))
    labels[torch.rand(8, 8, 8) < 0.1] = -1  # ignore voxels, which cc3d has no label space for

    reference, candidate = relabel_connected(labels), relabel_connected_cc3d(labels)
    foreground = labels > 0
    assert _same_partition(reference[foreground], candidate[foreground])
    assert torch.equal(reference[~foreground], candidate[~foreground]), "background and ignore"
    assert bool((candidate[foreground] > 0).all()), "components must be positive ids"


@pytest.mark.unit
@needs_cc3d
def test_cc3d_splits_a_reentering_object() -> None:
    """The case the pass exists for: one id, two disconnected runs inside the crop."""
    labels = torch.zeros(1, 1, 8, dtype=torch.long)
    labels[0, 0, :2] = 5
    labels[0, 0, 5:] = 5
    out = relabel_connected_cc3d(labels)
    assert torch.unique(out[labels > 0]).numel() == 2
    assert out[0, 0, 0] == out[0, 0, 1]
    assert out[0, 0, 0] != out[0, 0, 5]


@pytest.mark.unit
@needs_cc3d
def test_cc3d_does_not_merge_touching_different_instances() -> None:
    labels = torch.zeros(1, 1, 4, dtype=torch.long)
    labels[0, 0, 1] = 3
    labels[0, 0, 2] = 4
    out = relabel_connected_cc3d(labels)
    assert out[0, 0, 1] != out[0, 0, 2]


@pytest.mark.unit
@needs_cc3d
def test_split_targets_match_whichever_side_computed_them() -> None:
    """End to end: the affinity target is the same whether the split ran in a worker or on device.

    This is the assertion that actually matters -- the two relabelings may differ in ids, and what
    must not differ is the tensor the loss sees.
    """
    torch.manual_seed(0)
    labels = (torch.rand(6, 6, 6) < 0.5).long() * torch.randint(1, 3, (6, 6, 6))
    offsets = affinity_offsets(3, long_range=2)

    on_device, _ = affinities_from_labels(relabel_connected(labels).unsqueeze(0), offsets)
    in_worker, _ = affinities_from_labels(relabel_connected_cc3d(labels).unsqueeze(0), offsets)
    assert torch.equal(on_device, in_worker)


@pytest.mark.unit
@needs_cc3d
def test_transform_handles_the_level_axis_and_keeps_the_dtype() -> None:
    """A sample's label is (L, X, Y, Z) and int32; both must survive, or the H2D cost changes."""
    labels = torch.zeros(1, 1, 1, 8, dtype=torch.int32)
    labels[0, 0, 0, :2] = 5
    labels[0, 0, 0, 5:] = 5

    out = SplitDisconnectedLabels()({"img": torch.zeros(1), "label": labels})["label"]
    assert out.shape == labels.shape
    assert out.dtype == torch.int32
    assert torch.unique(out[labels > 0]).numel() == 2


@pytest.mark.unit
def test_transform_refuses_a_sample_with_no_labels() -> None:
    with pytest.raises(KeyError, match="split into connected components"):
        SplitDisconnectedLabels()({"img": torch.zeros(1)})


# --------------------------------------------------------------- the dataset transform chain


class _Dataset(BaseDataset):
    def build_dataset(self) -> Any:
        class _Source(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return 2

            def __getitem__(self, index: int) -> dict[str, Any]:
                return {"seen": []}

        return _Source()


@pytest.mark.unit
def test_transforms_compose_in_attachment_order() -> None:
    """Two callers attach transforms; neither may discard the other's.

    Order is part of the contract: augmentation is attached first and can sever an object, so a
    components pass attached afterwards has to see the augmented volume, not the original.
    """

    def tag(name: str) -> Any:
        def apply(sample: dict[str, Any]) -> dict[str, Any]:
            return {"seen": [*sample["seen"], name]}

        return apply

    dataset = _Dataset()
    dataset.attach_transform(tag("augment"))
    dataset.attach_transform(tag("split"))
    assert dataset.dataset[0]["seen"] == ["augment", "split"]


@pytest.mark.unit
def test_a_dataset_with_no_transforms_is_left_alone() -> None:
    dataset = _Dataset()
    assert dataset.dataset[0] == {"seen": []}


@pytest.mark.unit
def test_attaching_after_the_dataset_is_built_is_refused() -> None:
    dataset = _Dataset()
    _ = dataset.dataset
    with pytest.raises(RuntimeError, match="after the dataset was built"):
        dataset.attach_transform(lambda sample: sample)


# ------------------------------------------------------------------ the engine does the wiring


@pytest.mark.unit
def test_build_trainer_attaches_the_transform_to_train_and_validation(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring that makes all of the above reach a real run.

    Validation as well as training, which is the part worth pinning: `[augment]` is deliberately
    training-only, so the obvious reading of the surrounding code is that anything attached to a
    dataset is. Target construction is the opposite -- a validation loss built from unsplit labels
    is not comparable to a training loss built from split ones, and the discrepancy would look
    like a generalisation gap rather than a bug.

    The stand-in components go into the registries through `monkeypatch.setitem` rather than
    `register`, which has no inverse. Three other tests sweep the whole registry and build
    everything in it -- `test_engine_mfu`, `test_engine_optimizer`, `test_package_layout` -- so a
    test that leaves entries behind fails them instead of itself, at a distance.
    """
    import torch.nn as nn
    import torch.utils.data as data

    from algorithms.base import BaseAlgorithm
    from algorithms.registry import AlgorithmRegistry
    from data.registry import DataRegistry
    from distributed.parallel_dims import ParallelDims
    from engine.config import TrainerConfig
    from engine.run import build_trainer
    from models.base import BaseModel
    from models.registry import ModelRegistry
    from utils.config import ComponentConfig, RunConfig

    marker = SplitDisconnectedLabels("label")

    class _Model(BaseModel):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

        def flops(self, input_shape: tuple[int, ...]) -> int:
            return 0

    class _Algorithm(BaseAlgorithm):
        def sample_transform(self) -> Any:
            return marker

        def training_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"loss": self.model(batch).pow(2).mean()}

        def validation_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"loss": self.model(batch).pow(2).mean()}

    class _Data(BaseDataset):
        def build_dataset(self) -> data.Dataset:
            class _Source(data.Dataset):
                def __len__(self) -> int:
                    return 8

                def __getitem__(self, index: int) -> dict[str, Any]:
                    return {"label": torch.zeros(1, 2, 1, 1, dtype=torch.int32)}

            return _Source()

    monkeypatch.setitem(ModelRegistry._registry, "_split_probe_model", _Model)
    monkeypatch.setitem(AlgorithmRegistry._registry, "_split_probe_algorithm", _Algorithm)
    monkeypatch.setitem(DataRegistry._registry, "_split_probe_data", _Data)

    config = RunConfig(
        experiment_name="split_probe",
        model=ComponentConfig(name="_split_probe_model"),
        algorithm=ComponentConfig(name="_split_probe_algorithm"),
        data=ComponentConfig(name="_split_probe_data"),
        val_data=ComponentConfig(name="_split_probe_data"),
        trainer=TrainerConfig(max_steps=1, batch_size=2, measure_mfu=False),
        parallelism=ParallelDims(),
    )
    trainer = build_trainer(config, tmp_path)

    assert trainer.val_loader is not None, "the config named a val_data section"
    for loader in (trainer.train_loader, trainer.val_loader):
        assert isinstance(loader.dataset, TransformedDataset)
        assert marker in loader.dataset.transforms


@pytest.mark.unit
def test_workers_are_held_to_one_thread() -> None:
    """The loop is launch-bound, so worker threads compete with the process feeding the GPU.

    `num_workers` is the pipeline's parallelism knob; letting each worker also inherit
    `OMP_NUM_THREADS` multiplies the two into a thread count far past the node's cores.
    """
    from data.base import _single_threaded_worker

    before = torch.get_num_threads()
    try:
        _single_threaded_worker(0)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)


@pytest.mark.unit
def test_a_single_process_loader_gets_no_worker_hooks() -> None:
    """With no workers there is nothing to initialise, and pinning would only add a copy."""
    loader = _Dataset().build_dataloader(batch_size=1, rank=0, world_size=1, num_workers=0)
    assert loader.worker_init_fn is None
    assert loader.pin_memory is False


@pytest.mark.unit
def test_workers_do_not_persist_across_epochs_by_default() -> None:
    """Measured, not assumed: long-lived workers cost ~68 ms/step against ~4 ms saved.

    A volumetric sample is 134 MB and the affinity components pass allocates a 256^3 array per
    call, and a worker process that never restarts never gives that back. Recycling them at each
    epoch boundary is the cheaper end of the trade by a wide margin -- see
    `TrainerConfig.persistent_workers` for the numbers. Pinned here because the default reads as
    a pessimisation to anyone who has not seen them.
    """
    from engine.config import TrainerConfig

    assert TrainerConfig(max_steps=10, batch_size=1).persistent_workers is False

    loader = _Dataset().build_dataloader(batch_size=1, rank=0, world_size=1, num_workers=2)
    assert loader.persistent_workers is False
    assert loader.prefetch_factor == 2


@pytest.mark.unit
def test_prefetch_factor_must_be_at_least_one() -> None:
    from engine.config import TrainerConfig

    with pytest.raises(ValueError, match="prefetch_factor"):
        TrainerConfig(max_steps=10, batch_size=1, prefetch_factor=0)
