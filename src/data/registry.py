from __future__ import annotations

from utils.registry import Registry

from .base import BaseDataset

DataRegistry: Registry[BaseDataset] = Registry(BaseDataset)
