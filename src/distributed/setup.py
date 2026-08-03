from __future__ import annotations

import os

import torch
import torch.distributed as dist


def default_backend() -> str:
    """NCCL when CUDA is present, else Gloo."""
    return "nccl" if torch.cuda.is_available() else "gloo"


def device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def init_distributed(backend: str | None = None) -> tuple[int, int, int]:
    """Initialize the process group from torchrun's environment.

    Returns (rank, world_size, local_rank). Binds the process to its local GPU when CUDA is
    available, so `torch.cuda.current_device()` is correct for every later collective.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend or default_backend())
    return dist.get_rank(), dist.get_world_size(), local_rank


def destroy_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
