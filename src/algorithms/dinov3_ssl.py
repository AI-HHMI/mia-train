"""DINOv3 self-supervised pretraining.

Ported from the DINOv3 reference implementation, minus the Gram-anchoring term (off by default
upstream, and in its full form it needs a third frozen backbone with its own crops).

The shape of it. Two networks with identical architecture: a *student* trained by gradient
descent, and a *teacher* that is only ever an exponential moving average of the student. Each
sample is cropped into two wide "global" views and several narrow "local" ones. The teacher sees
only the global views; the student sees everything, with some of its patches masked out. Three
losses then say what agreement means:

  - **DINO** -- the student's view-level distribution must match the teacher's for a *different*
    view of the same sample.
  - **iBOT** -- for patches the student had masked, its patch-level distribution must match what
    the teacher saw at those same positions unmasked.
  - **KoLeo** -- a regularizer spreading the batch over the sphere.

Nothing labels anything. What stops the whole thing collapsing to a constant is Sinkhorn-Knopp
centering of the teacher's output, which forces its prototypes to be used about equally often.

One algorithm serves both ranks. Everything above operates on tokens, so the only rank-aware part
is how views and masks are made -- see `algorithms.dinov3.multicrop`. It does require a
DINOv3-family encoder, since it needs `forward_features(x, masks)` and the mask-token
substitution that goes with it; `ViT3D` and `MuViT3D` do not have those and are not usable here.
"""

from __future__ import annotations

import copy
import math
from typing import Any, cast

import torch
import torch.nn as nn

from data.base import BaseDataset
from layers.dinov3.head import DINOHead
from models.base import BaseModel

from .base import BaseAlgorithm
from .dinov3.losses import DINOLoss, IBOTPatchLoss, KoLeoLoss
from .dinov3.multicrop import (
    AugmentationConfig,
    block_mask,
    photometric,
    random_resized_crop,
)
from .registry import AlgorithmRegistry


@AlgorithmRegistry.register("dinov3")
class DINOv3(BaseAlgorithm):
    """Teacher-student self-distillation with a patch-level objective.

    `total_steps` is needed because the teacher's momentum and temperature are scheduled over the
    whole run, not per epoch. It is stated here rather than read from the trainer because an
    algorithm is handed only its model and dataset, and a schedule that silently ran on the wrong
    horizon would train without any error.
    """

    _step: torch.Tensor

    def __init__(
        self,
        model: BaseModel,
        dataset: BaseDataset | None = None,
        total_steps: int = 100_000,
        input_axes: str | None = None,
        input_key: str = "img",
        # crops
        global_crop_size: int = 64,
        local_crop_size: int = 32,
        n_local_crops: int = 8,
        global_crop_scale: tuple[float, float] = (0.32, 1.0),
        local_crop_scale: tuple[float, float] = (0.05, 0.32),
        # heads
        head_n_prototypes: int = 65536,
        head_bottleneck_dim: int = 256,
        head_hidden_dim: int = 2048,
        head_nlayers: int = 3,
        # objective
        student_temp: float = 0.1,
        teacher_temp: float = 0.07,
        warmup_teacher_temp: float = 0.04,
        warmup_teacher_temp_steps: int = 30_000,
        momentum_teacher: float = 0.992,
        final_momentum_teacher: float = 1.0,
        dino_loss_weight: float = 1.0,
        ibot_loss_weight: float = 1.0,
        koleo_loss_weight: float = 0.1,
        global_ignore_diagonal: bool = True,
        # masking
        mask_ratio_min_max: tuple[float, float] = (0.1, 0.5),
        mask_sample_probability: float = 0.5,
        augmentation: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model, dataset)
        for name, value in (("head_n_prototypes", head_n_prototypes), ("total_steps", total_steps)):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if not hasattr(model, "forward_features"):
            raise TypeError(
                f"{type(model).__name__} has no forward_features, so it cannot be trained with "
                "DINOv3, which needs the encoder's own masked forward. Use a dinov3_vit or "
                "dinov3_vit3d encoder."
            )

        self.input_axes = self._resolve_input_axes(input_axes, dataset)
        self.input_key = input_key
        self.total_steps = total_steps
        self.n_local_crops = n_local_crops
        self.global_crop_scale: tuple[float, float] = tuple(global_crop_scale)  # type: ignore[assignment]
        self.local_crop_scale: tuple[float, float] = tuple(local_crop_scale)  # type: ignore[assignment]
        self.mask_ratio_min_max: tuple[float, float] = tuple(mask_ratio_min_max)  # type: ignore[assignment]
        self.mask_sample_probability = mask_sample_probability
        self.augmentation = AugmentationConfig(**(augmentation or {}))

        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.warmup_teacher_temp_steps = warmup_teacher_temp_steps
        self.momentum_teacher = momentum_teacher
        self.final_momentum_teacher = final_momentum_teacher
        self.dino_loss_weight = dino_loss_weight
        self.ibot_loss_weight = ibot_loss_weight
        self.koleo_loss_weight = koleo_loss_weight
        self.global_ignore_diagonal = global_ignore_diagonal

        rank = self._spatial_rank(model)
        self.spatial_rank = rank
        self.global_crop_size = (global_crop_size,) * rank
        self.local_crop_size = (local_crop_size,) * rank
        patch_size = cast(Any, model).patch_size
        patch = patch_size if isinstance(patch_size, int) else int(patch_size[0])
        self.global_grid = tuple(size // patch for size in self.global_crop_size)

        embed_dim = int(model.embed_dim)  # type: ignore[arg-type]
        head = lambda: DINOHead(  # noqa: E731
            embed_dim, head_n_prototypes, nlayers=head_nlayers,
            hidden_dim=head_hidden_dim, bottleneck_dim=head_bottleneck_dim,
        )
        dino_head, ibot_head = head(), head()
        dino_head.init_weights()
        ibot_head.init_weights()
        self.student = nn.ModuleDict(
            {"backbone": model, "dino_head": dino_head, "ibot_head": ibot_head}
        )

        # The teacher starts where the student is or the first steps just distil noise. 
        # `update_teacher` is its only writer, its parameters are detached from autograd entirely.
        self.teacher = copy.deepcopy(self.student)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

        self.dino_loss = DINOLoss(head_n_prototypes, student_temp=student_temp)
        self.ibot_loss = IBOTPatchLoss(head_n_prototypes, student_temp=student_temp)
        self.koleo_loss = KoLeoLoss()

        # A buffer, so the schedules survive a checkpoint-resume: the trainer restores model state
        # but never tells an algorithm what step it is on.
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    def _part(self, network: nn.ModuleDict, name: str) -> Any:
        """A ModuleDict lookup that keeps its static type.

        `network[name]` widens to `Module | Tensor` for a type checker, and `Module.__getattr__`
        widens again, so every call site would be an error even though the value is always a
        Module with the methods being called. Typed as Any to say that once, here.
        """
        member = network[name]
        assert isinstance(member, nn.Module)
        return cast(Any, member)

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _spatial_rank(model: nn.Module) -> int:
        from models.dinov3_vit import DinoVisionTransformer

        return 2 if isinstance(model, DinoVisionTransformer) else 3

    @staticmethod
    def _resolve_input_axes(input_axes: str | None, dataset: BaseDataset | None) -> str:
        from_dataset = dataset.sample_axes if dataset is not None else None
        if input_axes is not None and from_dataset is not None and input_axes != from_dataset:
            raise ValueError(
                f"input_axes={input_axes!r} contradicts the dataset's sample_axes="
                f"{from_dataset!r}; remove input_axes and let the dataset declare the layout"
            )
        axes = input_axes or from_dataset
        if axes is None:
            raise ValueError(
                "no axis order available: pass input_axes, or use a dataset that declares "
                "sample_axes"
            )
        return axes

    # ------------------------------------------------------------------ schedules

    @property
    def step(self) -> int:
        return int(self._step.item())

    def current_teacher_temp(self) -> float:
        """Warm up linearly, then hold.

        A cold teacher early on produces near-one-hot targets the student cannot yet match, and
        training diverges; the warmup keeps the target soft until the student is worth distilling.
        """
        if self.warmup_teacher_temp_steps <= 0:
            return self.teacher_temp
        progress = min(1.0, self.step / self.warmup_teacher_temp_steps)
        return self.warmup_teacher_temp + progress * (self.teacher_temp - self.warmup_teacher_temp)

    def current_momentum(self) -> float:
        """Cosine from `momentum_teacher` to `final_momentum_teacher` over the run.

        Ending at 1.0 freezes the teacher, so the target stops moving as the student converges.
        """
        progress = min(1.0, self.step / max(1, self.total_steps))
        span = self.final_momentum_teacher - self.momentum_teacher
        return self.final_momentum_teacher - span * (1 + math.cos(math.pi * progress)) / 2

    @torch.no_grad()
    def update_teacher(self) -> None:
        """One EMA step over parameters. Buffers are left alone, following the reference."""
        momentum = self.current_momentum()
        student: list[torch.Tensor] = list(self.student.parameters())
        teacher: list[torch.Tensor] = list(self.teacher.parameters())
        torch._foreach_mul_(teacher, momentum)
        torch._foreach_add_(teacher, student, alpha=1 - momentum)

    def train(self, mode: bool = True) -> DINOv3:
        """The teacher stays in eval whatever the algorithm is set to.

        It has no dropout or norm statistics to update -- it is a frozen average -- and letting
        `train()` propagate into it would start updating batch statistics that nothing corrects.
        """
        super().train(mode)
        self.teacher.eval()
        return self

    # ------------------------------------------------------------------ views

    def _make_views(self, volumes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One crop per sample -> (2B, C, *global), (nB, C, *local), crop-major.

        Crop-major stacking (all samples' crop 0, then all samples' crop 1) is the reference's
        layout, and `unflatten(0, (n_crops, B))` downstream depends on it.
        """
        globals_ = [
            photometric(
                random_resized_crop(volumes, self.global_crop_size, self.global_crop_scale,
                                    flip_prob=self.augmentation.flip_prob,
                                    rotate_prob=self.augmentation.rotate_prob),
                self.augmentation,
            )
            for _ in range(2)
        ]
        locals_ = [
            photometric(
                random_resized_crop(volumes, self.local_crop_size, self.local_crop_scale,
                                    flip_prob=self.augmentation.flip_prob,
                                    rotate_prob=self.augmentation.rotate_prob),
                self.augmentation,
            )
            for _ in range(self.n_local_crops)
        ]
        return torch.cat(globals_, dim=0), torch.cat(locals_, dim=0)

    # ------------------------------------------------------------------ step

    def _step_impl(self, batch: Any, train: bool) -> dict[str, torch.Tensor]:
        backbone = self._part(self.student, "backbone")
        volumes = backbone.prepare_input(batch[self.input_key], self.input_axes)
        global_views, local_views = self._make_views(volumes)
        batch_size = volumes.shape[0]
        n_global = 2 * batch_size

        masks = block_mask(
            self.global_grid, n_global, self.mask_ratio_min_max,
            sample_probability=self.mask_sample_probability, device=volumes.device,
        )
        teacher_temp = self.current_teacher_temp()

        # --- teacher: global views only, unmasked, no grad ---
        with torch.no_grad():
            teacher_out = self._part(self.teacher, "backbone").forward_features(global_views)
            teacher_cls = teacher_out["x_norm_clstoken"]
            teacher_patches = teacher_out["x_norm_patchtokens"]
            teacher_cls_logits = self._part(self.teacher, "dino_head")(teacher_cls)
            teacher_cls_probs = self.dino_loss.sinkhorn_knopp_teacher(
                teacher_cls_logits, teacher_temp
            ).unflatten(0, (2, batch_size))

            masked_teacher = teacher_patches.flatten(0, 1)[masks.flatten()]
            n_masked = int(masks.sum().item())
            if n_masked > 0:
                teacher_patch_probs = self.ibot_loss.sinkhorn_knopp_teacher(
                    self._part(self.teacher, "ibot_head")(masked_teacher), teacher_temp,
                    n_masked_patches=n_masked,
                )

        # --- student: global views masked, local views whole ---
        student_global = backbone.forward_features(global_views, masks)
        student_local = backbone.forward_features(local_views)

        student_head = self._part(self.student, "dino_head")
        student_global_cls = student_head(student_global["x_norm_clstoken"])
        student_local_cls = student_head(student_local["x_norm_clstoken"])

        # The reference weights the global and local terms by how many crop pairs each contributes,
        # so adding local crops does not silently reweight the objective.
        global_terms = 2 if self.global_ignore_diagonal else 4
        local_terms = 2 * self.n_local_crops
        total_terms = global_terms + local_terms

        dino_global = self.dino_loss(
            student_global_cls.unflatten(0, (2, batch_size)),
            teacher_cls_probs,
            ignore_diagonal=self.global_ignore_diagonal,
        )
        dino_local = self.dino_loss(
            student_local_cls.unflatten(0, (self.n_local_crops, batch_size)), teacher_cls_probs
        )
        loss = self.dino_loss_weight * (
            dino_global * (global_terms / total_terms) + dino_local * (local_terms / total_terms)
        )
        metrics = {"dino_global": dino_global.detach(), "dino_local": dino_local.detach()}

        if n_masked > 0:
            masked_student = self._part(self.student, "ibot_head")(
                student_global["x_norm_patchtokens"].flatten(0, 1)[masks.flatten()]
            )
            ibot = self.ibot_loss.forward_masked(
                masked_student, teacher_patch_probs, masks, n_masked_patches=n_masked
            )
            loss = loss + self.ibot_loss_weight * ibot
            metrics["ibot"] = ibot.detach()

        if self.koleo_loss_weight > 0:
            # On pre-head CLS features, and per global view: the regularizer is about the geometry
            # of the representation, not of the prototype logits.
            pre_head = student_global["x_norm_clstoken"].unflatten(0, (2, batch_size))
            koleo = (self.koleo_loss(pre_head[0]) + self.koleo_loss(pre_head[1])) / 2
            loss = loss + self.koleo_loss_weight * 2 * koleo
            metrics["koleo"] = koleo.detach()

        if train:
            self._step += 1
        metrics["loss"] = loss
        metrics["teacher_temp"] = torch.tensor(teacher_temp, device=loss.device)
        metrics["teacher_momentum"] = torch.tensor(self.current_momentum(), device=loss.device)
        metrics["masked_fraction"] = masks.float().mean()
        return metrics

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._step_impl(batch, train=True)

    def validation_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._step_impl(batch, train=False)
