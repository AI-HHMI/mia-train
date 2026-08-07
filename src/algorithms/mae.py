from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from data.base import BaseDataset
from layers.common.blocks import TransformerBlock
from models.vit import SPATIAL_RANK, ViT3D

from .base import BaseAlgorithm
from .registry import AlgorithmRegistry


@AlgorithmRegistry.register("mae")
class MAE(BaseAlgorithm):
    """Masked autoencoding: hide most patches, reconstruct them from the rest.

    Scale handling belongs to the encoder, not here: this algorithm passes the dataset-shaped
    batch to `model.prepare_input`, so a single-resolution ViT rejects a multi-scale batch while
    a multi-scale encoder consumes the level axis itself. Masked autoencoding works either way.

    The axis order is taken from the dataset (`BaseDataset.sample_axes`) and handed to the model;
    `input_axes` is only for datasets that declare none.

    The decoder lives here, not on the model: it exists only to serve the pretraining objective
    and is discarded before any downstream use.
    """

    def __init__(
        self,
        model: ViT3D,
        dataset: BaseDataset | None = None,
        input_axes: str | None = None,
        input_key: str = "img",
        mask_ratio: float = 0.75,
        decoder_embed_dim: int = 192,
        decoder_depth: int = 2,
        decoder_num_heads: int = 6,
        norm_pix_loss: bool = True,
    ) -> None:
        super().__init__(model, dataset)
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")

        self.input_axes = self._resolve_input_axes(input_axes, dataset)
        self.input_key = input_key
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.encoder: ViT3D = model

        self.decoder_embed = nn.Linear(model.embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList(
            # Same attention kernel as the encoder, rather than a second knob to keep in sync. The
            # decoder is positioned by the same patch coordinates as the encoder, so it needs no
            # position table of its own.
            TransformerBlock(
                decoder_embed_dim, decoder_num_heads, 4.0, model.attention_backend,
                spatial_rank=SPATIAL_RANK,
            )
            for _ in range(decoder_depth)
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_head = nn.Linear(decoder_embed_dim, model.patch_volume)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    @staticmethod
    def _resolve_input_axes(input_axes: str | None, dataset: BaseDataset | None) -> str:
        """Settle on the sample axis order, preferring the dataset's own answer.

        An explicit `input_axes` that contradicts the dataset is rejected rather than silently
        winning: a rank-preserving disagreement (say "lcxyz" vs "clzyx") would strip the wrong
        axis and quietly train on nonsense.
        """
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
        # What a valid order looks like is the encoder's business, not this algorithm's, so it
        # is left to `model.prepare_input` to accept or reject.
        return resolved

    def _random_masking(
        self, batch_size: int, num_tokens: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose which tokens stay visible. Returns (keep_indices, mask, restore_indices).

        `mask` is 1 for removed patches, which is exactly where the loss is applied. Indices are
        returned rather than the gathered tokens because the caller has to gather the tokens *and*
        their coordinates with the same indices, and doing both at the call site keeps that pairing
        visible instead of splitting it across two functions.
        """
        keep = max(1, int(round(num_tokens * (1.0 - self.mask_ratio))))
        shuffle = torch.argsort(torch.rand(batch_size, num_tokens, device=device), dim=1)
        restore = torch.argsort(shuffle, dim=1)

        keep_idx = shuffle[:, :keep]
        mask = torch.ones(batch_size, num_tokens, device=device)
        mask.scatter_(1, keep_idx, 0.0)
        return keep_idx, mask, restore

    def _decode(
        self, visible: torch.Tensor, restore: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """Re-insert mask tokens, decode, and predict every patch -> (B, N, patch_volume).

        `coords` are the full patch grid, in grid order: the decoder sees every position, so unlike
        the encoder it needs no gathering -- but it is restored to grid order first, so the
        coordinates line up with the tokens.
        """
        x = self.decoder_embed(visible)
        b, n = restore.shape
        pad = self.mask_token.expand(b, n - x.shape[1], -1).to(x.dtype)
        x = torch.cat([x, pad], dim=1)
        x = torch.gather(x, 1, restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        for block in self.decoder_blocks:
            x = block(x, coords)
        return self.decoder_head(self.decoder_norm(x))

    def _loss(self, batch: Any) -> dict[str, torch.Tensor]:
        # The encoder decides what a dataset-shaped batch means for it, including how many scale
        # levels it accepts; this algorithm only says which axis order the data came in.
        volumes = self.encoder.prepare_input(batch[self.input_key], self.input_axes)
        tokens, coords = self.encoder.embed(volumes)

        keep_idx, mask, restore = self._random_masking(
            tokens.shape[0], tokens.shape[1], tokens.device
        )
        visible = torch.gather(
            tokens, 1, keep_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        )
        # The same indices for the coordinates: a mismatch here would attach every visible token to
        # another patch's position and still train.
        visible_coords = torch.gather(
            coords, 1, keep_idx.unsqueeze(-1).expand(-1, -1, SPATIAL_RANK)
        )

        latent = self.encoder.encode(visible, visible_coords)
        prediction = self._decode(latent, restore, coords)

        target = self.encoder.patchify(volumes)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / torch.sqrt(var + 1e-6)

        per_patch = (prediction - target).pow(2).mean(dim=-1)
        # Only the hidden patches count: scoring visible ones would reward copying the input.
        loss = (per_patch * mask).sum() / mask.sum().clamp_min(1.0)
        return {"loss": loss, "masked_fraction": mask.mean().detach()}

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._loss(batch)

    def validation_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._loss(batch)
