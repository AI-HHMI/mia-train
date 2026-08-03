from __future__ import annotations

from utils.registry import Registry

from .base import BaseModel

ModelRegistry: Registry[BaseModel] = Registry(BaseModel)
