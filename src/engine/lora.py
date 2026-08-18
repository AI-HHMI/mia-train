"""Freeze a pretrained encoder and adapt it through low-rank deltas instead.

The same division of labour as `engine.activation_checkpoint`: the architecture declares what can be
adapted (`BaseModel.lora_target_groups`) and what must keep training whatever the config says
(`BaseModel.lora_required_trainable`), and this decides whether to honour it. That keeps `[lora]`
one section meaning the same thing on every architecture, instead of a flag per model.

Applied to the bare *model*, before the algorithm wraps it. A pretraining objective's own parameters
-- SimMIM's linear head, an affinity decoder -- are randomly initialised and have no prior to
protect, so they must train at full rank; doing this to the model alone makes that structural rather
than a name filter that could drift as strategies are added.

Ordering, all of which `engine.run.build_trainer` satisfies by calling this where it does:

  - **Before `load_pretrained`.** Promotion preserves every parameter name, so an adapted model
    reads a plain checkpoint with no special casing -- which makes this look like a free choice. It
    is not. A *second* stage of a LoRA arm loads a checkpoint that already carries
    `lora_a`/`lora_b`/`lora_scaling`, and a model not yet adapted has nowhere to put them:
    `load_pretrained` refuses 288 homeless tensors, which is the correct behaviour and a dead run.
    See `engine.run.prepare_model`, which owns the order.
  - **Before activation checkpointing.** `checkpoint_wrapper` replaces a block with a wrapper
    holding it as a child, so promoting afterwards would have to see through it.
  - **Before `parallelize_algorithm`.** `fully_shard` converts parameters to DTensors and registers
    hooks against the parameter set it found; adding a Parameter to a sharded module afterwards is
    not supported.
  - **Before `build_optimizer`.** `build_param_groups` skips parameters that do not require grad, so
    the freeze has to be in place or the whole encoder joins the optimizer and the adapters are the
    only thing that would have been needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch.nn as nn

from layers.common.layer_scale import LayerScale
from layers.common.lora import adapted_modules, promote
from layers.common.rms_norm import RMSNorm
from models.base import BaseModel

from .config import LoRAConfig
from .optimizer import is_stem

logger = logging.getLogger(__name__)

# Modules whose parameters `train_norms` opens. Decided by type rather than by name because the
# naming here is not reliable: `muvit3d` holds a LayerNorm inside an `nn.Sequential`, so its gain is
# called `patch_proj.0.1.weight` with no "norm" anywhere in it -- the same reason
# `engine.optimizer.wd_scale` decides by rank instead of by name.
_NORM_TYPES = (nn.LayerNorm, nn.GroupNorm, RMSNorm, LayerScale)


@dataclass
class LoRAReport:
    """What `apply_lora` did, for logging and for tests to assert on."""

    adapted: list[str] = field(default_factory=list)
    trainable: list[str] = field(default_factory=list)
    frozen: list[str] = field(default_factory=list)
    adapter_params: int = 0
    trainable_params: int = 0
    total_params: int = 0

    def summary(self) -> str:
        fraction = 100.0 * self.trainable_params / max(1, self.total_params)
        return (
            f"{len(self.adapted)} adapters, {self.adapter_params / 1e6:.2f}M adapter params; "
            f"{len(self.trainable)} of {len(self.trainable) + len(self.frozen)} tensors trainable "
            f"({self.trainable_params / 1e6:.2f}M of {self.total_params / 1e6:.1f}M params, "
            f"{fraction:.2f}%)"
        )


def _target_modules(model: BaseModel, config: LoRAConfig) -> list[nn.Linear]:
    """The Linears the named groups resolve to, deduplicated, in a stable order."""
    groups = model.lora_target_groups()
    if not groups:
        raise ValueError(
            f"[lora] is enabled, but {type(model).__name__} declares no lora_target_groups(), so "
            "there is nothing to adapt and the run would train a fully frozen encoder. Implement "
            "it on the model, or drop the [lora] section."
        )
    unknown = sorted(set(config.targets) - set(groups))
    if unknown:
        raise ValueError(
            f"[lora].targets names {unknown}, which {type(model).__name__} does not offer; its "
            f"groups are {sorted(groups)}"
        )

    modules: list[nn.Linear] = []
    seen: set[int] = set()
    for name in config.targets:
        group = groups[name]
        if not group:
            raise ValueError(
                f"[lora].targets names {name!r}, which {type(model).__name__} declares but leaves "
                "empty on this configuration, so it would adapt nothing"
            )
        for module in group:
            # A Linear may legitimately appear in two groups; adapting it twice would stack a
            # second delta on the first, which `promote` refuses -- dedupe before it has to.
            if id(module) not in seen:
                seen.add(id(module))
                modules.append(module)
    return modules


def _trainable_names(model: BaseModel, config: LoRAConfig) -> set[str]:
    """Every parameter name that keeps training: adapters, required, and the `train_*` sets."""
    keep = {name for name, _ in model.named_parameters() if name.endswith((".lora_a", ".lora_b"))}

    required = model.lora_required_trainable()
    known = dict(model.named_parameters())
    missing = sorted(name for name in required if name not in known)
    if missing:
        # A declaration that stopped matching -- a renamed parameter, a differently configured
        # model -- would otherwise silently freeze the very thing it exists to protect.
        raise ValueError(
            f"{type(model).__name__}.lora_required_trainable() names {missing}, which is not a "
            "parameter of this model. Update the declaration to match the architecture."
        )
    keep |= set(required)

    if config.train_norms:
        for prefix, module in model.named_modules():
            if isinstance(module, _NORM_TYPES):
                keep |= {
                    f"{prefix}.{name}" if prefix else name
                    for name, _ in module.named_parameters(recurse=False)
                }

    for name, _ in model.named_parameters():
        if config.train_stem and is_stem(name):
            keep.add(name)
        if config.train_tokens and (name.endswith("_token") or "tokens" in name):
            keep.add(name)
    return keep


def apply_lora(model: BaseModel, config: LoRAConfig) -> LoRAReport:
    """Adapt and freeze `model` in place. Returns what was done.

    Not called at all when `config.enabled()` is false; the caller decides, so that a run without a
    `[lora]` section is byte-identical to one from before this existed.
    """
    for module in _target_modules(model, config):
        promote(module, rank=config.rank, alpha=config.alpha)

    keep = _trainable_names(model, config)
    report = LoRAReport(adapted=[name for name, _ in adapted_modules(model)])
    for name, parameter in model.named_parameters():
        trainable = name in keep
        parameter.requires_grad_(trainable)
        report.total_params += parameter.numel()
        if trainable:
            report.trainable.append(name)
            report.trainable_params += parameter.numel()
        else:
            report.frozen.append(name)
        if name.endswith((".lora_a", ".lora_b")):
            report.adapter_params += parameter.numel()

    if not report.trainable:
        raise ValueError(
            "[lora] left no parameter trainable, so the run would compute a constant function of "
            "its input. This needs every train_* switch off and a model declaring no adapters."
        )
    logger.info("lora: %s", report.summary())
    return report
