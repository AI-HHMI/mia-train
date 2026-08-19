"""Initialise a model from weights trained somewhere else.

Distinct from resuming: `engine.checkpoint.CheckpointManager` restores a run's own model *and*
optimizer *and* step so training continues exactly, and only from that run's directory. This
starts a *new* run from someone else's weights -- an earlier pretraining run here, or a released
checkpoint like DINOv3's -- taking the model and nothing else.

Two things this deliberately makes noisy rather than convenient:

  - **Nothing loaded is an error.** A mistyped path prefix that matches no parameter would leave
    the model at its random initialisation and train without complaint, and the loss curve of a
    from-scratch run looks entirely reasonable. That failure is silent and expensive, so it raises.
  - **Every decision is reported.** The summary names what was copied, what was reshaped to fit,
    what stayed at its initial value, and what the checkpoint offered that the model had no use
    for. A pretrained run whose encoder silently kept 3 of its 40 tensors is not worth debugging
    a week later.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)

# Wrappers a checkpoint's real state dict is commonly buried under.
_NESTING_KEYS = ("model", "state_dict", "teacher", "student")


@dataclass
class LoadReport:
    """What `load_pretrained` did with each tensor, for logging and for tests to assert on."""

    copied: list[str] = field(default_factory=list)
    inflated: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    kept_initial: list[str] = field(default_factory=list)
    unused: list[str] = field(default_factory=list)
    mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = field(default_factory=list)
    # Checkpoint keys `prefix` removed before any matching happened.
    #
    # These are invisible to every other category: the filter runs first, so a key it drops is
    # never a candidate to be copied and never counts as `unused` either. That is how an [init]
    # can look completely healthy -- "370 copied, 0 unused" -- while silently discarding a trained
    # head, which is exactly what happened to a pseudo-labelling arm meant to warm-start from the
    # model that produced its labels. Recorded so the count appears in the run's log.
    filtered_by_prefix: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"{len(self.copied)} copied",
            f"{len(self.inflated)} inflated 2D->3D",
            f"{len(self.merged)} LoRA adapters merged",
            f"{len(self.skipped)} skipped",
            f"{len(self.kept_initial)} kept at initial value",
            f"{len(self.unused)} unused from checkpoint",
        ]
        if self.mismatched:
            parts.append(f"{len(self.mismatched)} shape mismatches")
        # Appended only when it happened, so a load without a prefix reads exactly as before.
        if self.filtered_by_prefix:
            parts.append(f"{len(self.filtered_by_prefix)} filtered out by prefix")
        return ", ".join(parts)


def _unwrap(state: Any) -> dict[str, torch.Tensor]:
    """Dig the tensor mapping out of whatever the checkpoint file wrapped it in.

    A named wrapper wins over loose tensors alongside it. A run saved here holds
    `{"model": ..., "optim": ..., "train_state": {"step": tensor}}`, and taking whichever level
    happens to contain a tensor first would come back with the step counter and drop the weights.
    """
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint is a {type(state).__name__}, not a state dict")

    for key in _NESTING_KEYS:
        inner = state.get(key)
        if isinstance(inner, dict) and any(isinstance(v, torch.Tensor) for v in inner.values()):
            return {k: v for k, v in inner.items() if isinstance(v, torch.Tensor)}

    tensors = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    if tensors:
        return tensors
    raise ValueError(
        "checkpoint contains no tensors at the top level or under any of "
        f"{list(_NESTING_KEYS)}; is this a state dict?"
    )


def read_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a state dict from a `.pth`/`.pt` file or a PyTorch Distributed Checkpoint directory.

    A run saved by this repo is a DCP *directory*, written sharded; a released checkpoint is a
    single file. Both arrive here as one flat name -> tensor mapping.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")

    if path.is_dir():
        # DCP stores one shard per rank, so it is converted through torch's own reader rather
        # than reassembled here.
        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

        with tempfile.TemporaryDirectory() as scratch:
            flattened = Path(scratch) / "converted.pt"
            dcp_to_torch_save(str(path), str(flattened))
            return _unwrap(torch.load(flattened, map_location="cpu", weights_only=False))

    return _unwrap(torch.load(path, map_location="cpu", weights_only=False))


def inflate_2d_to_3d(weight: torch.Tensor, target: torch.Size) -> torch.Tensor:
    """A 2D patch-embedding kernel (E, C, kh, kw) -> a 3D one (E, C', kd, kh, kw).

    Two independent adjustments, in the order the reference implementation makes them:

    *Channels.* A checkpoint trained on RGB has three input channels where volumetric microscopy
    has one. Averaging over them is what DINOv3's own hub loader does, and it means a grey input
    produces the response the 2D model would have given the same image.

    *Depth.* The kernel is repeated along the new axis and divided by its length, so a volume that
    is constant in z reproduces the 2D model's output on that slice exactly. Dividing matters: a
    plain repeat would multiply every activation by the patch depth and push the first layer far
    outside the range the rest of the pretrained stack expects.
    """
    if weight.ndim != 4 or len(target) != 5:
        raise ValueError(
            f"inflation goes from a 4D conv kernel to a 5D one, got {tuple(weight.shape)} -> "
            f"{tuple(target)}"
        )
    out_channels, in_channels, height, width = weight.shape
    target_out, target_in, depth, target_height, target_width = target

    if (out_channels, height, width) != (target_out, target_height, target_width):
        raise ValueError(
            f"cannot inflate a {tuple(weight.shape)} kernel into {tuple(target)}: the output "
            "channels and the in-plane kernel must already match. Configure the model with the "
            "checkpoint's patch size, or resize the kernel yourself first."
        )

    if in_channels != target_in:
        if target_in == 1:
            weight = weight.mean(dim=1, keepdim=True)
        elif in_channels == 1:
            weight = weight.expand(-1, target_in, -1, -1) / target_in
        else:
            raise ValueError(
                f"cannot map {in_channels} checkpoint input channels onto {target_in}; only "
                "averaging to 1 channel or spreading 1 channel over several is defined"
            )

    # (E, C, kh, kw) -> (E, C, kd, kh, kw), each z-slice carrying 1/kd of the original kernel.
    return (weight.unsqueeze(2).expand(-1, -1, depth, -1, -1) / depth).contiguous()


def merge_lora_tensors(
    checkpoint: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Fold every LoRA adapter in a state dict into the weight it adapts. Returns (state, merged).

    An adapted layer stores `<layer>.weight` alongside `<layer>.lora_a`, `<layer>.lora_b` and
    `<layer>.lora_scaling`; the layer it is equivalent to stores `weight + scaling * B @ A` and
    nothing else. This performs that substitution on the *checkpoint*, before any of it reaches a
    model, which is what makes an adapted run's output loadable by a config that knows nothing about
    LoRA -- and, when the destination model is itself adapted, what re-bases the adaptation into the
    frozen prior so the new adapter starts from zero again.

    The scaling is read from the checkpoint rather than from the loading run's `[lora].alpha`,
    because those are not the same number: `alpha` may have changed since the adapter was trained,
    and using the current one would rescale a delta by the ratio with nothing to indicate it. That
    is the whole reason `LoRAMixin` registers it as a persistent buffer.
    """
    merged: dict[str, torch.Tensor] = dict(checkpoint)
    folded = []
    for key in sorted(checkpoint):
        if not key.endswith(".lora_a"):
            continue
        layer = key[: -len(".lora_a")]
        names = (f"{layer}.lora_a", f"{layer}.lora_b", f"{layer}.lora_scaling", f"{layer}.weight")
        missing = [name for name in names if name not in merged]
        if missing:
            # An adapter is three tensors plus the weight they modify; any subset is a checkpoint
            # that was assembled by hand or truncated, and folding what is left would produce a
            # weight that is neither the base nor the adapted one.
            raise ValueError(
                f"cannot merge the LoRA adapter at {layer!r}: the checkpoint is missing {missing}. "
                "A complete adapter is lora_a, lora_b, lora_scaling and the weight they adapt."
            )
        lora_a = merged.pop(f"{layer}.lora_a")
        lora_b = merged.pop(f"{layer}.lora_b")
        lora_scaling = merged.pop(f"{layer}.lora_scaling")
        base = merged[f"{layer}.weight"]
        merged[f"{layer}.weight"] = base + (
            lora_scaling.to(base.dtype) * (lora_b.to(base.dtype) @ lora_a.to(base.dtype))
        )
        folded.append(layer)
    return merged, folded


def load_pretrained(
    model: nn.Module,
    path: str | Path,
    *,
    prefix: str = "",
    inflate: bool = False,
    skip: Sequence[str] = (),
    strict: bool = True,
    allow_unused: bool = False,
    merge_lora: bool = False,
) -> LoadReport:
    """Copy what fits from the checkpoint at `path` into `model`, in place.

    `prefix` is stripped from the checkpoint's keys before matching, which is how an encoder is
    lifted out of a checkpoint that stored a whole algorithm: a masked-autoencoding run saves its
    encoder under `model.` alongside its decoder, and `prefix="model."` takes the former and
    ignores the latter.

    `inflate` allows a 2D patch embedding to be reshaped into a 3D one. Nothing else is reshaped,
    so a transformer block whose width disagrees is reported rather than quietly truncated.

    `allow_unused` permits the checkpoint to carry tensors this model has no home for. Left off,
    that is an error, because the usual cause is an architecture that disagrees: a released DINOv3
    ViT has LayerScale and a masked key bias, and a model configured without them loads perfectly,
    reports nothing wrong, and computes a different function from the one that was trained. Turn
    it on when the checkpoint genuinely holds more than you want -- a full algorithm when you only
    need its encoder.

    `skip` names key prefixes the model should keep its own values for. The case this exists for
    is a buffer the model *derives* rather than learns: DINOv3's `rope_embed.periods` is computed
    from `base` in `init_weights`, and its length depends on how many axes the rotary embedding
    splits across -- 2D and 3D disagree. Transferring it would be meaningless and not transferring
    it costs nothing, so it is skipped rather than treated as a mismatch.
    """
    checkpoint = read_state_dict(path)
    filtered_by_prefix: list[str] = []
    if prefix:
        filtered_by_prefix = sorted(key for key in checkpoint if not key.startswith(prefix))
        checkpoint = {
            key[len(prefix) :]: value for key, value in checkpoint.items() if key.startswith(prefix)
        }
    # After the prefix strip, so the adapter keys are named as the model names them, and before
    # matching, so the folded weights are what gets copied and the adapter tensors never appear as
    # `unused`.
    merged_layers: list[str] = []
    if merge_lora:
        checkpoint, merged_layers = merge_lora_tensors(checkpoint)
        if not merged_layers:
            raise ValueError(
                f"merge_lora is set but {path} holds no LoRA adapter, so nothing would be merged "
                "and the option is silently doing nothing. Drop it, or point at a checkpoint from "
                "a run whose config had a [lora] section."
            )

    # Parameters and buffers together: a model's rotary period tables and masked-bias masks are
    # buffers, and a checkpoint carries them alongside the weights.
    destinations: dict[str, torch.Tensor] = dict(model.named_parameters())
    destinations.update(model.named_buffers())

    report = LoadReport(merged=merged_layers, filtered_by_prefix=filtered_by_prefix)
    with torch.no_grad():
        for name, destination in destinations.items():
            source = checkpoint.pop(name, None)
            if any(name.startswith(ignored) for ignored in skip):
                report.skipped.append(name)
                continue
            if source is None:
                report.kept_initial.append(name)
                continue
            if source.shape == destination.shape:
                destination.copy_(source)
                report.copied.append(name)
            elif inflate and source.ndim == 4 and destination.ndim == 5:
                destination.copy_(inflate_2d_to_3d(source, destination.shape).to(destination.dtype))
                report.inflated.append(name)
            else:
                report.mismatched.append(
                    (name, tuple(source.shape), tuple(destination.shape))
                )
    report.unused = sorted(
        key for key in checkpoint if not any(key.startswith(ignored) for ignored in skip)
    )

    logger.info("loaded pretrained weights from %s: %s", path, report.summary())
    if report.inflated:
        logger.info("inflated 2D->3D: %s", ", ".join(report.inflated))
    if report.kept_initial:
        logger.info("kept at initial value: %s", ", ".join(report.kept_initial))

    if not strict:
        return report

    if report.mismatched:
        details = "\n".join(
            f"  {name}: checkpoint {source} vs model {destination}"
            for name, source, destination in report.mismatched
        )
        raise ValueError(
            f"{len(report.mismatched)} tensor(s) in {path} do not fit this model:\n{details}\n"
            f"{_mismatch_hint(report)}"
        )
    # Checked before the leftovers: if *nothing* matched, every checkpoint key is unused, and
    # "36 tensors have no home" would bury the real problem, which is that none of them do.
    if not report.copied and not report.inflated:
        raise ValueError(
            f"nothing in {path} matched this model's parameters, so it would have trained from "
            f"scratch. The checkpoint offers keys like {sorted(checkpoint)[:5]}; the model wants "
            f"{sorted(destinations)[:5]}. Set `prefix` if the checkpoint stores the encoder under "
            "one."
        )
    if report.unused and not allow_unused:
        raise ValueError(
            f"{len(report.unused)} tensor(s) in {path} have no home in this model, so it is "
            f"configured differently from the one that was trained: {report.unused[:6]}"
            f"{' ...' if len(report.unused) > 6 else ''}\n{_unused_hint(report)}"
        )
    _log_load(model, path, report)
    return report


def _log_load(model: nn.Module, path: Path, report: LoadReport) -> None:
    """Put what `[init]` actually did into the run's log, on the primary rank.

    A successful load used to say nothing at all, which made the one failure mode that raises no
    error impossible to notice after the fact: a `prefix` narrow enough to drop tensors you meant
    to keep. Nothing is wrong from `load_pretrained`'s point of view -- every key it was shown
    found a home -- so it returns quietly and the run trains on a partly-initialised model.

    The categories that mean "something did not arrive" are therefore named with examples, not
    just counted. `filtered_by_prefix` is the one worth reading first: a nonzero count next to a
    `prefix` is either deliberate (lifting an encoder out of an algorithm checkpoint) or the bug.
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    def examples(names: list[str], limit: int = 4) -> str:
        return ", ".join(names[:limit]) + (" ..." if len(names) > limit else "")

    print(f"[init] {type(model).__name__} <- {path}", flush=True)
    print(f"[init]   {report.summary()}", flush=True)
    for label, names in (
        ("filtered out by prefix", report.filtered_by_prefix),
        ("skipped", report.skipped),
        ("kept at initial value", report.kept_initial),
        ("unused from checkpoint", report.unused),
    ):
        if names:
            print(f"[init]   {len(names)} {label}: {examples(names)}", flush=True)


def _unused_hint(report: LoadReport) -> str:
    """Name the setting that would consume the leftovers, when it is one we recognise."""
    missing = []
    if any("ls1.gamma" in key or "ls2.gamma" in key for key in report.unused):
        missing.append("layerscale_init (the released DINOv3 ViTs use 1.0e-05)")
    if any("bias_mask" in key for key in report.unused):
        missing.append("mask_k_bias = true")
    if any("storage_tokens" in key for key in report.unused):
        missing.append("n_storage_tokens (the released DINOv3 ViTs use 4)")
    if missing:
        return "The model appears to be missing: " + "; ".join(missing) + "."
    return (
        "Set allow_unused = true if the checkpoint is expected to hold more than this model "
        "needs, or `prefix` to select the part you want."
    )


def _mismatch_hint(report: LoadReport) -> str:
    """Turn the most common mismatch into the sentence that actually fixes it."""
    for name, _, _ in report.mismatched:
        if "rope_embed" in name:
            return (
                "rope_embed buffers are derived from `base` in the model's own init_weights, not "
                "learned, and 2D and 3D split their channels differently -- so their lengths "
                'disagree and there is nothing to transfer. Add skip = ["rope_embed."] to the '
                "[init] section; the model will keep its own, which loses nothing."
            )
        if name.startswith("patch_embed."):
            return (
                "The patch embedding does not fit. Set inflate_2d_to_3d = true to lift a 2D "
                "kernel into 3D, and configure the model with the checkpoint's patch size -- the "
                "in-plane kernel has to match already."
            )
    return (
        "Check that the model's embed_dim, depth, num_heads and patch_size match the checkpoint's."
    )
