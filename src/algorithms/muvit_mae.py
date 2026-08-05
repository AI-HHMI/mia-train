"""Multi-resolution masked autoencoding, as described in the MuViT paper (section 3.3).

A separate algorithm from `mae` rather than a branch inside it, because three of its four moving
parts genuinely differ:

  - **Masking** weights the levels by a Dirichlet draw instead of treating all tokens alike, so a
    step might hide almost all of the fine level and little of the coarse one. That is the point:
    it forces the encoder to reconstruct detail from context and vice versa, which uniform masking
    never asks for.
  - **Decoding** uses one small decoder per resolution level, and its first layer cross-attends to
    every visible token from *all* levels. That cross-attention is where information crosses scale.
  - **The loss** averages per level, not over all masked patches at once. Since the Dirichlet draw
    deliberately leaves levels with very different numbers of masked patches, a global mean would
    silently weight whichever level happened to be masked most.
  - Only the reconstruction target is shared, and that now lives on the model as `patchify`.

Unifying the two would mean giving `ViT3D` world coordinates and rotary attention it does not have,
so the single- and multi-resolution paths stay separate deliberately.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from data.base import BaseDataset
from layers.attention import CrossAttention, SelfAttention
from layers.rope import AxialRotaryEmbedding
from models.muvit import SPATIAL_RANK, MuViT3D

from .base import BaseAlgorithm
from .registry import AlgorithmRegistry


class MuViTDecoderLayer(nn.Module):
    """Pre-norm decoder layer: self-attention, optional cross-attention, feed-forward.

    Queries and context keys are rotated by the *same* rotary module, which is what makes the
    cross-attention geometric: two separate schedules would put the two sets of coordinates on
    different frequency scales, and the displacement between a masked patch and a visible one would
    no longer mean anything.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attention_backend: str = "auto",
        rotary_base: float = 10000.0,
        with_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.rotary = AxialRotaryEmbedding(dim // num_heads, SPATIAL_RANK, base=rotary_base)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, backend=attention_backend)

        if with_cross_attention:
            self.norm_query = nn.LayerNorm(dim)
            self.norm_context = nn.LayerNorm(dim)
            self.cross_attn: CrossAttention | None = CrossAttention(
                dim, num_heads, backend=attention_backend
            )
        else:
            self.cross_attn = None

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        context: torch.Tensor | None = None,
        context_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rope = self.rotary(coords)
        x = x + self.attn(self.norm1(x), rope=rope)

        if self.cross_attn is not None:
            if context is None or context_coords is None:
                raise ValueError(
                    "this decoder layer has cross-attention, so it needs both `context` and "
                    "`context_coords`; without the coordinates the visible tokens would carry no "
                    "position and the cross-scale geometry would be lost"
                )
            x = x + self.cross_attn(
                self.norm_query(x),
                self.norm_context(context),
                rope=rope,
                context_rope=self.rotary(context_coords),
            )

        return x + self.mlp(self.norm2(x))


class MuViTLevelDecoder(nn.Module):
    """The lightweight decoder for one resolution level.

    Groups the level's layers, final norm and pixel projection into one module so that everything
    belonging to a level lives together, rather than being spread across parallel lists that have
    to be indexed in lockstep.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float,
        patch_volume: int,
        attention_backend: str = "auto",
        rotary_base: float = 10000.0,
    ) -> None:
        super().__init__()
        # Cross-attention in the first layer only, per the paper: that is the one point where this
        # level's masked patches read from the visible tokens of every other level.
        self.layers = nn.ModuleList(
            MuViTDecoderLayer(
                dim, num_heads, mlp_ratio, attention_backend, rotary_base,
                with_cross_attention=(index == 0),
            )
            for index in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch_volume)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        context: torch.Tensor,
        context_coords: torch.Tensor,
    ) -> torch.Tensor:
        """(B, P, dim) -> (B, P, patch_volume)."""
        for layer in self.layers:
            x = layer(x, coords, context=context, context_coords=context_coords)
        return self.head(self.norm(x))


@AlgorithmRegistry.register("muvit_mae")
class MuViTMAE(BaseAlgorithm):
    """Masked autoencoding across several true resolutions of the same scene.

    Requires a `bbox` in the batch. miao supplies one per sample -- `(B, L, 2, 3)`, the low and high
    world-coordinate corner of every level -- and it is not optional here on purpose: the paper
    measures a substantial performance loss from feeding wrong coordinates, and wrong coordinates
    are still perfectly well-shaped, so a silent fall back to a guessed geometry would be
    undetectable from the loss curve.
    """

    def __init__(
        self,
        model: MuViT3D,
        dataset: BaseDataset | None = None,
        input_axes: str | None = None,
        input_key: str = "img",
        bbox_key: str = "bbox",
        mask_ratio: float = 0.75,
        dirichlet_alpha: float = 0.5,
        decoder_embed_dim: int = 384,
        decoder_depth: int = 2,
        decoder_num_heads: int = 6,
        decoder_mlp_ratio: float = 4.0,
        norm_pix_loss: bool = True,
    ) -> None:
        super().__init__(model, dataset)
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
        if dirichlet_alpha <= 0.0:
            raise ValueError(f"dirichlet_alpha must be positive, got {dirichlet_alpha}")
        if decoder_depth < 1:
            raise ValueError(f"decoder_depth must be at least 1, got {decoder_depth}")

        self.input_axes = self._resolve_input_axes(input_axes, dataset)
        self.input_key = input_key
        self.bbox_key = bbox_key
        self.mask_ratio = mask_ratio
        self.dirichlet_alpha = dirichlet_alpha
        self.norm_pix_loss = norm_pix_loss
        self.encoder: MuViT3D = model

        levels = model.num_levels
        self.to_decoder = nn.Linear(model.embed_dim, decoder_embed_dim)
        # One mask token per level: what "unknown here" should look like differs between a patch of
        # fine detail and a patch of wide-field context.
        self.mask_tokens = nn.Parameter(torch.zeros(levels, 1, decoder_embed_dim))

        # One decoder per level, each cross-attending to the visible tokens of every level.
        self.decoders = nn.ModuleList(
            MuViTLevelDecoder(
                decoder_embed_dim,
                decoder_num_heads,
                decoder_depth,
                decoder_mlp_ratio,
                model.patch_volume,
                model.attention_backend,
            )
            for _ in range(levels)
        )
        nn.init.trunc_normal_(self.mask_tokens, std=0.02)

    @staticmethod
    def _resolve_input_axes(input_axes: str | None, dataset: BaseDataset | None) -> str:
        """Settle on the sample axis order, preferring the dataset's own answer."""
        from_dataset = dataset.sample_axes if dataset is not None else None
        if input_axes is not None and from_dataset is not None and input_axes != from_dataset:
            raise ValueError(
                f"input_axes={input_axes!r} contradicts the dataset's sample_axes="
                f"{from_dataset!r}; drop input_axes and let the dataset define the layout"
            )
        resolved = input_axes or from_dataset
        if resolved is None:
            raise ValueError(
                "cannot determine the sample axis order: either pass input_axes explicitly or "
                "use a dataset that exposes sample_axes"
            )
        return resolved

    def _dirichlet_masking(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick visible tokens, weighting levels by a Dirichlet draw.

        Returns `(keep_indices, mask)` where `keep_indices` is (B, K) into the joint sequence and
        `mask` is (B, N) with 1 marking a hidden token, which is where the loss applies.

        The draw is per sample rather than one per step as in the reference implementation: the
        stated purpose is to expose the model to diverse cross-scale configurations, and a
        per-step draw gives every sample in the batch the same one. It costs nothing extra.
        """
        levels = self.encoder.num_levels
        per_level = self.encoder.patches_per_level
        total = self.encoder.num_patches
        keep = max(1, int(round(total * (1.0 - self.mask_ratio))))

        concentration = torch.full((levels,), self.dirichlet_alpha, device=device)
        level_weight = torch.distributions.Dirichlet(concentration).sample((batch_size,))
        # A near-zero weight is meaningful -- it means "hide this level almost entirely" -- but an
        # exactly-zero one would divide to NaN below rather than simply never being chosen.
        level_weight = level_weight.clamp_min(torch.finfo(level_weight.dtype).tiny)
        token_weight = level_weight.repeat_interleave(per_level, dim=1)  # (B, N)

        # Weighted sampling without replacement, vectorised over the batch: draw u ~ U(0,1) per
        # token and rank by log(u)/w. Ranking by that is equivalent to ranking by u**(1/w), the
        # standard exponential-race construction, and avoids a Python loop over the batch.
        uniform = torch.rand(batch_size, total, device=device)
        keys = uniform.clamp_min(torch.finfo(uniform.dtype).tiny).log() / token_weight
        order = keys.argsort(dim=1, descending=True)

        keep_indices = order[:, :keep]
        mask = torch.ones(batch_size, total, device=device)
        mask.scatter_(1, keep_indices, 0.0)
        return keep_indices, mask

    def _decode(
        self,
        latent: torch.Tensor,
        coords: torch.Tensor,
        keep_indices: torch.Tensor,
        visible_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct every patch of every level -> (B, N, patch_volume)."""
        batch_size = latent.shape[0]
        per_level = self.encoder.patches_per_level
        context = self.to_decoder(latent)
        width = context.shape[-1]

        # Start from each level's mask token everywhere, then drop the encoded visible tokens back
        # into the positions they came from. Scattering keeps this a single indexing operation
        # rather than per-level bookkeeping that would have to stay in step with the masking.
        # Cast to the context's dtype: under autocast `to_decoder` returns bf16 while the mask
        # tokens stay fp32 as parameters do, and scatter refuses to mix the two.
        background = torch.cat(
            [self.mask_tokens[index].expand(batch_size, per_level, width)
             for index in range(self.encoder.num_levels)],
            dim=1,
        ).to(context.dtype)
        tokens = background.scatter(
            1, keep_indices.unsqueeze(-1).expand(-1, -1, width), context
        )

        predictions = []
        for index, decoder in enumerate(self.decoders):
            span = slice(index * per_level, (index + 1) * per_level)
            predictions.append(
                decoder(tokens[:, span], coords[:, span], context, visible_coords)
            )
        return torch.cat(predictions, dim=1)

    def _loss(self, batch: Any) -> dict[str, torch.Tensor]:
        if self.bbox_key not in batch:
            raise KeyError(
                f"MuViTMAE needs world coordinates under batch[{self.bbox_key!r}], but the batch "
                f"has only {sorted(batch)}. miao's VolumeDataset supplies them; a dataset that "
                "does not cannot position tokens across resolution levels, which is what this "
                "algorithm is for."
            )

        volumes = self.encoder.prepare_input(batch[self.input_key], self.input_axes)
        bbox = batch[self.bbox_key]
        tokens, coords = self.encoder.embed(volumes, bbox)

        keep_indices, mask = self._dirichlet_masking(tokens.shape[0], tokens.device)
        gather_tokens = keep_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        gather_coords = keep_indices.unsqueeze(-1).expand(-1, -1, SPATIAL_RANK)
        visible_tokens = torch.gather(tokens, 1, gather_tokens)
        # Coordinates are gathered with the same indices as the tokens; a mismatch here would
        # attach every visible token to another patch's position and still train.
        visible_coords = torch.gather(coords, 1, gather_coords)

        latent = self.encoder.encode(visible_tokens, visible_coords)
        prediction = self._decode(latent, coords, keep_indices, visible_coords)

        target = self.encoder.patchify(volumes)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / torch.sqrt(var + 1e-6)

        per_patch = (prediction - target.to(prediction.dtype)).pow(2).mean(dim=-1)  # (B, N)

        # Per level, then across levels, per the paper. The Dirichlet draw makes the number of
        # masked patches differ a lot between levels, so a single mean over all masked patches
        # would quietly weight the level that happened to be hidden most.
        levels, per_level = self.encoder.num_levels, self.encoder.patches_per_level
        shaped_mask = mask.reshape(-1, levels, per_level)
        hidden_per_level = shaped_mask.sum(dim=-1)  # (B, L)
        level_loss = (per_patch.reshape(-1, levels, per_level) * shaped_mask).sum(
            dim=-1
        ) / hidden_per_level.clamp_min(1.0)

        # A level can end up with nothing hidden when the Dirichlet draw gives it the whole visible
        # budget. Averaging over levels that did contribute avoids counting that as a zero loss.
        contributed = (hidden_per_level > 0).to(level_loss.dtype)
        loss = (level_loss * contributed).sum() / contributed.sum().clamp_min(1.0)

        return {
            "loss": loss,
            "masked_fraction": mask.mean().detach(),
            # How lopsided the draw was: the share of visible tokens taken by the finest level.
            "finest_visible_share": (
                (1.0 - shaped_mask[:, 0].mean(dim=-1)).mean().detach()
            ),
        }

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._loss(batch)

    def validation_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._loss(batch)
