from __future__ import annotations

from utils.registry import Registry

from .base import BaseEvalTask

EvalRegistry: Registry[BaseEvalTask] = Registry(BaseEvalTask)
