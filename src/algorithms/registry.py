from __future__ import annotations

from utils.registry import Registry

from .base import BaseAlgorithm

AlgorithmRegistry: Registry[BaseAlgorithm] = Registry(BaseAlgorithm)
