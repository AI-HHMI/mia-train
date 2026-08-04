from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def move_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move every tensor in a collated batch to `device`.

    Batches are whatever a dataset yields — miao returns a dict of tensors alongside a nested
    metadata dict — so the structure is walked rather than assumed to be a bare tensor. Values
    that are not tensors or containers (strings, ints) are passed through untouched.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, Mapping):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(value, device) for value in batch)
    return batch
