"""Gradient-norm clipping when a run's gradients do not all live on one device mesh.

`torch.nn.utils.clip_grad_norm_` takes one norm over every gradient at once, and the way it does
that is `torch.stack` over the per-tensor norms. Under FSDP alone that is fine -- every gradient is
a DTensor on the data-parallel mesh. Add tensor parallelism and it is not: the projections the plan
covers become DTensors on the 2D `(dp, tp)` mesh, while everything the plan deliberately leaves
replicated -- the patch embedding, the learned tokens, the output norms, and every parameter the
*algorithm* owns rather than the model -- stays on the 1D data-parallel mesh. Stacking the two
raises:

    All operands in aten.stack.default must have the same mesh, but got
    DeviceMesh((dp_shard=2, tp=2)) and DeviceMesh((dp_shard=2))

at the first optimizer step, which is to say that `tp > 1` and `grad_clip_norm` were mutually
exclusive until this existed.

The fix is to take the norm per mesh, materialize each to a plain scalar, and combine those --
`||g||_p` over a partition is the p-norm of the parts' p-norms, so nothing is approximated. A run
whose gradients *are* all on one mesh is handed straight to torch, so the overwhelmingly common
case keeps the exact code path it has always had rather than a reimplementation of it.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate


def clip_grad_norm_(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    norm_type: float = 2.0,
    foreach: bool | None = None,
) -> torch.Tensor:
    """Clip gradients to `max_norm` in aggregate. Returns the total norm before clipping.

    Signature-compatible with `torch.nn.utils.clip_grad_norm_` for the arguments this repo uses.
    The return value is a plain tensor rather than a `_NormPartial` DTensor when the multi-mesh
    path runs, which is what a caller wanting to log it would have had to reduce anyway.
    """
    parameters = list(parameters)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return torch.tensor(0.0)

    by_mesh: dict[DeviceMesh | None, list[torch.Tensor]] = {}
    for gradient in gradients:
        mesh = gradient.device_mesh if isinstance(gradient, DTensor) else None
        by_mesh.setdefault(mesh, []).append(gradient)

    if len(by_mesh) == 1:
        return torch.nn.utils.clip_grad_norm_(
            parameters, max_norm, norm_type, foreach=foreach
        )

    norms = []
    for group in by_mesh.values():
        total = torch.nn.utils.get_total_norm(group, norm_type, False, foreach)
        # `full_tensor` on a `_NormPartial` reduces with the norm's own combination rule, not a
        # sum, so this is the group's true norm rather than this rank's share of it.
        norms.append(total.full_tensor() if isinstance(total, DTensor) else total)
    total_norm = torch.linalg.vector_norm(torch.stack(norms), norm_type)

    coefficient = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for mesh, group in by_mesh.items():
        # Wrapped per mesh rather than left as a plain scalar: an in-place foreach over DTensors
        # needs its operand on the same mesh, and a `float()` here would block the host on the
        # device every step to read a number the device is about to use.
        scale = (
            DTensor.from_local(coefficient, mesh, [Replicate()] * mesh.ndim)
            if mesh is not None
            else coefficient
        )
        torch._foreach_mul_(group, scale)
    return total_norm
