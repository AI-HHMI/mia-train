from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from algorithms.registry import AlgorithmRegistry
from data.registry import DataRegistry
from distributed.setup import destroy_distributed, device_type, init_distributed
from models.registry import ModelRegistry
from utils.cluster import checkpoint_dir
from utils.config import RunConfig, as_plain_dict, load_run_config
from utils.provenance import write_run_artifacts

from .trainer import Trainer

_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_output_dir(output_root: Path, experiment_name: str) -> Path:
    """Pick this run's artifact directory: output_root/<experiment_name>_<timestamp>.

    Every rank must agree on the path, so rank 0's choice is broadcast — independently formatted
    timestamps can straddle a second boundary and silently split a run across two directories.
    """
    name = [f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"]
    if dist.is_initialized():
        dist.broadcast_object_list(name, src=0)
    return output_root / name[0]


def build_trainer(
    config: RunConfig,
    output_dir: Path,
    mesh: DeviceMesh | None = None,
    device: torch.device | None = None,
) -> Trainer:
    """Instantiate model, algorithm, and datasets from the registries named in the config."""
    model = ModelRegistry.build(config.model.name, **config.model.kwargs)
    algorithm = AlgorithmRegistry.build(config.algorithm.name, model, **config.algorithm.kwargs)
    train_dataset = DataRegistry.build(config.data.name, **config.data.kwargs)
    val_dataset = (
        DataRegistry.build(config.val_data.name, **config.val_data.kwargs)
        if config.val_data is not None
        else None
    )
    return Trainer(
        algorithm=algorithm,
        train_dataset=train_dataset,
        config=config.trainer,
        output_dir=output_dir,
        dims=config.parallelism,
        mesh=mesh,
        val_dataset=val_dataset,
        device=device,
    )


def run(config_path: Path, output_root: Path | None = None) -> Path:
    """Execute one training run end to end. Returns the run's artifact directory.

    `output_root` defaults to [environment].checkpoint_dir from configs/cluster/active.toml, an
    absolute path, so every run lands in the same place regardless of the launching directory.
    """
    config = load_run_config(config_path)
    root = output_root or checkpoint_dir()
    rank, _world_size, local_rank = init_distributed()
    try:
        mesh = config.parallelism.build_mesh(device_type())
        device = (
            torch.device("cuda", local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        output_dir = resolve_output_dir(root, config.experiment_name)
        if rank == 0:
            write_run_artifacts(output_dir, config_path, as_plain_dict(config), _REPO_ROOT)
        dist.barrier()  # no rank may write checkpoints before rank 0 creates the directory

        build_trainer(config, output_dir, mesh=mesh, device=device).train()
    finally:
        destroy_distributed()
    return output_dir
