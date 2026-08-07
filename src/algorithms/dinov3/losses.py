"""The three loss terms DINOv3 trains with by default.

Ported from the DINOv3 reference implementation. All three are pure torch.

The shared idea: a teacher network turns each view into a distribution over a large bank of
prototypes, and the student is trained to reproduce it from a *different* view. Nothing labels
the prototypes; what carries the signal is that two views of the same scene must agree while
different scenes must not collapse onto the same answer.

  - `DINOLoss` compares CLS tokens -- one distribution per view, so it is about the image as a
    whole.
  - `IBOTPatchLoss` compares *masked patch* tokens, so the student must infer what it cannot see
    from what it can. This is what makes the patch features good enough for dense tasks.
  - `KoLeoLoss` is a regularizer on the student's own batch, pushing embeddings apart so they
    spread over the sphere rather than clumping.

Collapse is prevented by Sinkhorn-Knopp: rather than centering the teacher's logits, its
distribution is projected onto one whose prototypes are used equally often across the batch.

Everything upcasts to fp32 before the log/softmax. The reference runs these under bf16 parameters
with no autocast, and a bf16 log-softmax over 65536 prototypes loses far too much precision.

**Distributed.** The reference all-reduces inside Sinkhorn, so the normalization spans the global
batch. Here the process group is consulted only if one is initialised, and every reduction is a
no-op on one process -- so single-process numerics are identical to the distributed path run on
one rank, which is what makes the tests below a valid oracle.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """Sum across processes, in place, or leave alone when running on one."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor


def sinkhorn_knopp(
    logits: torch.Tensor, temperature: float, n_iterations: int = 3, batch_size: float | None = None
) -> torch.Tensor:
    """Teacher logits (B, K) -> an assignment (B, K) whose rows sum to 1.

    Optimal-transport centering: instead of subtracting a running mean from the teacher's logits,
    the whole batch is projected onto a distribution that uses every prototype about equally.
    Rows are normalized across processes, columns only locally -- each process holds a slice of
    the batch, so a column sum is already complete, while a row spans the global batch.

    No eps and no clamping, matching the reference: `exp(logits / temperature)` can overflow if a
    logit is large relative to the temperature.
    """
    Q = torch.exp(logits.float() / temperature).t()  # (K, B_local)
    prototypes, local_batch = Q.shape
    total = float(local_batch * _world_size()) if batch_size is None else float(batch_size)

    Q /= _all_reduce(torch.sum(Q))
    for _ in range(n_iterations):
        Q /= _all_reduce(torch.sum(Q, dim=1, keepdim=True))
        Q /= prototypes
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= total
    Q *= total
    return Q.t()


class DINOLoss(nn.Module):
    """Cross-entropy between a student's CLS distribution and a teacher's, across views."""

    def __init__(self, out_dim: int, student_temp: float = 0.1) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.student_temp = student_temp

    def sinkhorn_knopp_teacher(
        self, teacher_logits: torch.Tensor, teacher_temp: float, n_iterations: int = 3
    ) -> torch.Tensor:
        return sinkhorn_knopp(teacher_logits, teacher_temp, n_iterations)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        ignore_diagonal: bool = False,
    ) -> torch.Tensor:
        """(S, B, K) student logits against (T, B, K) teacher distributions -> scalar.

        `ignore_diagonal` drops the terms pairing a student view with the *same* view's teacher
        output. Between the two global crops that pairing is the trivial one -- predicting a view
        from itself -- so the reference leaves it out of the global term.
        """
        log_probs = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)
        n_student, batch, _ = log_probs.shape
        n_teacher = teacher_probs.shape[0]

        if not ignore_diagonal:
            total = -torch.einsum("sbk,tbk->", log_probs, teacher_probs)
            return total / (batch * n_student * n_teacher)

        pairwise = -torch.einsum("sbk,tbk->st", log_probs, teacher_probs)
        diagonal = min(n_student, n_teacher)
        pairwise = torch.diagonal_scatter(
            pairwise, torch.zeros(diagonal, device=pairwise.device, dtype=pairwise.dtype)
        )
        return pairwise.sum() / (batch * n_student * n_teacher - batch * diagonal)


class IBOTPatchLoss(nn.Module):
    """The same cross-entropy, over patch tokens the student was not allowed to see."""

    def __init__(self, patch_out_dim: int, student_temp: float = 0.1) -> None:
        super().__init__()
        self.patch_out_dim = patch_out_dim
        self.student_temp = student_temp

    def sinkhorn_knopp_teacher(
        self,
        teacher_logits: torch.Tensor,
        teacher_temp: float,
        n_masked_patches: int | None = None,
        n_iterations: int = 3,
    ) -> torch.Tensor:
        """Unlike the CLS case the batch size is the number of *masked patches*, which varies.

        The reference all-reduces that count in place with no `is_initialized` guard, so it raises
        outside a process group; here it is reduced only when there is one.
        """
        if n_masked_patches is None:
            n_masked_patches = teacher_logits.shape[0]
        total = torch.tensor(
            float(n_masked_patches), device=teacher_logits.device, dtype=torch.float32
        )
        _all_reduce(total)
        return sinkhorn_knopp(
            teacher_logits, teacher_temp, n_iterations, batch_size=float(total.item())
        )

    def forward_masked(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        student_masks_flat: torch.Tensor,
        n_masked_patches: int | None = None,
        masks_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gathered masked tokens (M, K) against (M, K) -> scalar.

        Each patch is weighted by one over the number of masked patches in *its own* view, so a
        heavily masked view does not dominate, and the sum is divided by the number of views
        rather than by M -- which is what makes the term comparable across steps even though the
        mask count changes every batch.
        """
        loss = torch.sum(
            teacher_probs * F.log_softmax(student_logits.float() / self.student_temp, dim=-1),
            dim=-1,
        )
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        return -(loss * masks_weight).sum() / student_masks_flat.shape[0]


class KoLeoLoss(nn.Module):
    """Push each embedding away from its nearest neighbour in the batch.

    "Kozachenko-Leonenko" differential-entropy estimator: maximising the log distance to the
    nearest neighbour spreads the batch over the sphere. Without it the CLS features are free to
    crowd into a small region while still satisfying the cross-entropy, and the representation
    degrades even though the loss looks healthy.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.pdist = nn.PairwiseDistance(2, eps=eps)

    def _pairwise_nearest(self, x: torch.Tensor) -> torch.Tensor:
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        # Blank the diagonal so a point is not its own nearest neighbour. The strided view walks
        # the diagonal of the flattened matrix.
        dots.view(-1)[:: n + 1].fill_(-1)
        return torch.argmax(dots, dim=1)

    def forward(self, student_output: torch.Tensor) -> torch.Tensor:
        """(B, D) embeddings -> scalar."""
        # Explicitly outside autocast: the distances here are small and a half-precision log of
        # them loses the differences that carry the gradient.
        with torch.autocast("cuda", enabled=False):
            x = F.normalize(student_output.float(), eps=self.eps, p=2, dim=-1)
            nearest = self._pairwise_nearest(x)
            distances = self.pdist(x, x[nearest])
            return -torch.log(distances + self.eps).mean()
