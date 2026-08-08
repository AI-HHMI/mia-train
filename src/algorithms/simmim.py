"""SimMIM: masked image modelling by predicting the raw voxels you hid.

From "SimMIM: A Simple Framework for Masked Image Modeling" (Xie et al.). Mask a large fraction of
the patches, let the encoder see the whole grid with a learned mask token standing in for what was
removed, and predict the original voxel values at the masked positions with one linear layer. The
loss is an L1 on those positions and nothing else.

Why this and not masked autoencoding. MAE *drops* masked tokens and encodes only the visible ones,
which is what makes it cheap, but it also means the encoder must accept a scattered subset of the
grid positioned by per-token coordinates. The DINOv3 backbones cannot: they derive their rotary
embedding from the patch grid and apply it to a contiguous suffix. SimMIM keeps every token and
substitutes a mask token instead -- exactly what `prepare_tokens_with_masks` already does -- so it
runs on those backbones unchanged.

Why this and not DINOv3. Roughly an order of magnitude less compute and memory: one view instead
of ten, one network instead of a student and a teacher, and a `patch_volume`-wide linear head
instead of four heads over 65536 prototypes. It learns less than DINOv3 does, but it runs on a
volume budget where DINOv3 does not.

One algorithm serves both ranks. Masking and patchifying are the only rank-aware parts and both
are written generically over the spatial axes.
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.base import BaseDataset
from models.base import BaseModel

from .base import BaseAlgorithm
from .registry import AlgorithmRegistry


def patchify(volumes: torch.Tensor, patch_size: tuple[int, ...]) -> torch.Tensor:
    """(B, C, *spatial) -> (B, N, C * prod(patch_size)), in the encoder's patch order.

    Row-major over the patch grid, channels ahead of the within-patch offsets, matching what the
    patch-embedding convolution produces -- so token `n` of the prediction lines up with token `n`
    of the encoder output without any further bookkeeping.
    """
    batch, channels = volumes.shape[0], volumes.shape[1]
    spatial = volumes.shape[2:]
    rank = len(patch_size)
    if len(spatial) != rank:
        raise ValueError(f"input has {len(spatial)} spatial axes but patch_size has {rank}")
    for extent, patch in zip(spatial, patch_size, strict=True):
        if extent % patch != 0:
            raise ValueError(
                f"input {tuple(spatial)} is not divisible by patch_size {tuple(patch_size)}; a "
                "partial patch would silently crop the target"
            )

    grid = [extent // patch for extent, patch in zip(spatial, patch_size, strict=True)]
    interleaved: list[int] = [batch, channels]
    for count, patch in zip(grid, patch_size, strict=True):
        interleaved += [count, patch]
    x = volumes.reshape(interleaved)

    # (B, C, g0, p0, g1, p1, ...) -> (B, g0, g1, ..., C, p0, p1, ...)
    permutation = [0] + [2 + 2 * axis for axis in range(rank)]
    permutation += [1] + [3 + 2 * axis for axis in range(rank)]
    x = x.permute(permutation)
    return x.reshape(batch, math.prod(grid), channels * math.prod(patch_size))


def random_mask(
    grid: tuple[int, ...],
    batch: int,
    ratio: float,
    *,
    granularity: int = 1,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """A random mask over the patch grid -> (batch, prod(grid)) bool.

    `granularity` masks in square groups of that many patches per axis rather than one at a time.
    SimMIM's central finding is that this matters: masking single 16-voxel patches leaves every
    hidden patch surrounded by visible neighbours, so the task is solved by local interpolation and
    the encoder learns nothing much. A coarser unit forces prediction from context.

    Every sample gets exactly the same *number* of masked units, drawn independently -- the ratio
    is a rate, not an average over the batch.
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"mask ratio must be in (0, 1), got {ratio}")
    if granularity < 1:
        raise ValueError(f"granularity must be at least 1 patch, got {granularity}")

    coarse = tuple(max(1, math.ceil(extent / granularity)) for extent in grid)
    units = math.prod(coarse)
    n_masked = max(1, int(round(units * ratio)))

    # Randomly permute each row and take the first n_masked -- a uniform sample without
    # replacement, done batched.
    order = torch.rand(batch, units, device=device, generator=generator).argsort(dim=1)
    flat = torch.zeros(batch, units, dtype=torch.bool, device=device)
    flat.scatter_(1, order[:, :n_masked], True)

    mask = flat.reshape(batch, *coarse)
    for axis in range(len(grid)):
        mask = mask.repeat_interleave(granularity, dim=axis + 1)
    # `ceil` above may have overshot; trim back to the real grid.
    mask = mask[(slice(None), *(slice(0, extent) for extent in grid))]
    return mask.reshape(batch, math.prod(grid))


@AlgorithmRegistry.register("simmim")
class SimMIM(BaseAlgorithm):
    """Predict the masked voxels from what is left, with one linear head.

    `mask_granularity` is in *patches*, not voxels: at patch size 16 and granularity 2, the unit
    that gets hidden is 32 voxels across, which is the setting the paper reports for 16-pixel
    patches.
    """

    def __init__(
        self,
        model: BaseModel,
        dataset: BaseDataset | None = None,
        input_axes: str | None = None,
        input_key: str = "img",
        mask_ratio: float = 0.6,
        mask_granularity: int = 2,
        norm_pix_loss: bool = False,
    ) -> None:
        super().__init__(model, dataset)
        if not hasattr(model, "forward_features"):
            raise TypeError(
                f"{type(model).__name__} has no forward_features, so it cannot be trained with "
                "SimMIM, which needs the encoder to substitute a mask token for hidden patches. "
                "Use a dinov3_vit or dinov3_vit3d encoder."
            )
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")

        self.input_axes = self._resolve_input_axes(input_axes, dataset)
        self.input_key = input_key
        self.mask_ratio = mask_ratio
        self.mask_granularity = mask_granularity
        self.norm_pix_loss = norm_pix_loss
        self.encoder = model

        backbone = cast(Any, model)
        rank = 2 if backbone.__class__.__name__ == "DinoVisionTransformer" else 3
        self.spatial_rank = rank
        patch = backbone.patch_size
        self.patch_size: tuple[int, ...] = (
            (patch,) * rank if isinstance(patch, int) else tuple(patch)
        )
        self.in_chans = int(backbone.in_chans)

        # One linear layer, as in the paper. A heavier decoder makes the pretext task easier and
        # the learned representation worse, because the encoder can offload work onto it.
        self.head = nn.Linear(
            int(backbone.embed_dim), self.in_chans * math.prod(self.patch_size)
        )

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

    def _step(self, batch: Any) -> dict[str, torch.Tensor]:
        encoder = cast(Any, self.encoder)
        volumes = encoder.prepare_input(batch[self.input_key], self.input_axes)
        spatial = volumes.shape[2:]
        grid = tuple(
            extent // patch for extent, patch in zip(spatial, self.patch_size, strict=True)
        )

        mask = random_mask(
            grid, volumes.shape[0], self.mask_ratio,
            granularity=self.mask_granularity, device=volumes.device,
        )
        out = encoder.forward_features(volumes, mask)
        assert isinstance(out, dict)  # a single tensor in gives the dict form back
        prediction = self.head(out["x_norm_patchtokens"])

        target = patchify(volumes, self.patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            variance = target.var(dim=-1, keepdim=True)
            target = (target - mean) / torch.sqrt(variance + 1e-6)

        # L1, as in the paper, and only where the encoder was blind. Scoring visible patches would
        # reward copying the input straight through the residual stream.
        per_patch = F.l1_loss(prediction, target, reduction="none").mean(dim=-1)
        denominator = mask.sum().clamp_min(1.0)
        loss = (per_patch * mask).sum() / denominator

        with torch.no_grad():
            visible = (per_patch * ~mask).sum() / (~mask).sum().clamp_min(1.0)
        return {
            "loss": loss,
            "masked_fraction": mask.float().mean(),
            # Reconstruction error on the patches the encoder *could* see. It should sit well
            # below the masked error; if the two converge, the model is not using the mask token
            # and the task has stopped being predictive.
            "visible_l1": visible,
        }

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._step(batch)

    def validation_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._step(batch)
