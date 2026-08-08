"""MuViT: a transformer over several true resolutions of the same scene at once.

From "MuViT: Multi-Resolution Vision Transformers for Learning Across Scales in Microscopy"
(Dominguez Mantes, La Manno, Weigert). Implemented from the paper rather than ported, so every
building block here is plain `torch.nn`.

The idea in one paragraph. A plain ViT sees one crop at one resolution. MuViT takes L crops of the
same scene that share a pixel size but differ in field of view -- level `l` is `l`x downsampled, so
it covers `l`x more of the scene in the same number of pixels -- embeds each level with its own
projection, and concatenates all of their tokens into a single sequence that one transformer
attends over jointly. What keeps that sequence geometrically coherent is that every token carries
its **world coordinate**: the position of its patch in the pixel frame of the finest level. Those
coordinates drive rotary position embeddings, so two patches covering the same physical place get
the same rotation whether they came from the fine level or the coarse one, and attention can relate
wide-field context to high-resolution detail without any explicit alignment step.

That is the whole trick, and it is why the coordinates are not decoration: with wrong coordinates
this degenerates into a model holding several unrelated crops. The paper measures exactly that and
finds a substantial drop.

Deliberate departures from the reference implementation, which are visible in the parameter names
and mean its published checkpoints cannot be loaded here:
  - `torch.nn` only -- no `einops`, no `x_transformers`.
  - Level embeddings initialise at std 0.02, this repo's convention for learned embeddings, rather
    than unit-scale `randn`, which starts comparable in magnitude to the normalised token itself.
  - A final `LayerNorm`, as in `ViT3D`: a pre-norm stack otherwise hands downstream code the
    unnormalised residual stream.
  - Only joint attention over all levels, the architecture the paper evaluates. The reference also
    carries level-restricted masking modes for ablations.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from layers.common.blocks import TransformerBlock

from .base import BaseModel
from .registry import ModelRegistry

SPATIAL_RANK = 3


@ModelRegistry.register("muvit3d")
class MuViT3D(BaseModel):
    """Multi-resolution 3D ViT: one transformer over the tokens of every scale level.

    `levels` are downsampling factors relative to the finest level, so `(1, 4, 16)` means the
    native resolution plus 4x and 16x downsampled views. They set the physical extent each level
    covers, which is what the default world coordinates are built from, so they must be ordered
    the same way as the dataset's scale levels.

    Unlike `ViT3D`, which requires a single level, this consumes the level axis itself -- see
    `prepare_input`.
    """

    def __init__(
        self,
        levels: tuple[int, ...] = (1, 4, 16),
        img_size: tuple[int, int, int] = (64, 64, 64),
        patch_size: tuple[int, int, int] = (8, 8, 8),
        in_channels: int = 1,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        attention_backend: str = "auto",
        rotary_base: float = 10000.0,
    ) -> None:
        super().__init__()
        levels = tuple(levels)
        img_size = tuple(img_size)  # type: ignore[assignment]
        patch_size = tuple(patch_size)  # type: ignore[assignment]

        if not levels:
            raise ValueError("levels must name at least one scale level")
        if len(set(levels)) != len(levels):
            raise ValueError(
                f"levels must be distinct, got {levels}; two levels at the same scale would "
                "occupy the same world coordinates and be told apart only by their embeddings"
            )
        if any(level <= 0 for level in levels):
            raise ValueError(f"levels must be positive downsampling factors, got {levels}")
        if list(levels) != sorted(levels):
            raise ValueError(
                f"levels must be ordered finest first, got {levels}. miao sorts a sample's scale "
                "levels fine-to-coarse, and the per-level projections are indexed positionally, so "
                "a different order here would pair each level's data with another level's weights "
                "-- which trains without complaint."
            )
        if len(img_size) != SPATIAL_RANK or len(patch_size) != SPATIAL_RANK:
            raise ValueError(
                f"img_size and patch_size must be {SPATIAL_RANK}D, got {img_size} {patch_size}"
            )
        for size, patch in zip(img_size, patch_size, strict=True):
            if size % patch != 0:
                raise ValueError(
                    f"img_size {img_size} must be divisible by patch_size {patch_size} on every "
                    "axis; a partial patch would silently crop the volume"
                )

        self.levels = levels
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.attention_backend = attention_backend
        self.grid_size = tuple(s // p for s, p in zip(img_size, patch_size, strict=True))

        # One projection per level, not one shared: the same pixel pattern means something
        # different at 1x and at 16x, so each level gets to map its patches its own way.
        self.patch_proj = nn.ModuleList(
            nn.Sequential(nn.Linear(self.patch_volume, embed_dim), nn.LayerNorm(embed_dim))
            for _ in levels
        )
        # Tells the encoder which resolution a token came from; world coordinates alone cannot,
        # since a fine and a coarse patch can sit at the same place.
        self.level_embed = nn.Parameter(torch.zeros(len(levels), 1, embed_dim))
        nn.init.trunc_normal_(self.level_embed, std=0.02)

        self.blocks = nn.ModuleList(
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio, attention_backend,
                spatial_rank=SPATIAL_RANK, rotary_base=rotary_base,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def patches_per_level(self) -> int:
        return int(math.prod(self.grid_size))

    @property
    def num_patches(self) -> int:
        """Length of the joint token sequence: every level's patches, concatenated."""
        return self.patches_per_level * self.num_levels

    @property
    def patch_volume(self) -> int:
        """Number of values in one patch, i.e. the width of a per-patch reconstruction."""
        return int(math.prod(self.patch_size)) * self.in_channels

    def checkpointable_modules(self) -> tuple[nn.Module, ...]:
        """The transformer blocks: repeated, sequence-length-sized, and cheap to rerun."""
        return tuple(self.blocks)

    def prepare_input(self, batch: torch.Tensor, axes: str) -> torch.Tensor:
        """(B, *axes) -> (B, L, C, D, H, W). Multi-scale: the level axis is consumed here.

        This is the counterpart of `ViT3D.prepare_input`, which rejects anything with more than
        one level. MuViT wants the level axis, so what it enforces instead is that the batch
        carries exactly the levels it was configured for -- the per-level projections and level
        embeddings are indexed positionally, so a batch with a different number of levels, or the
        same number in a different order, would be silently mismatched with them.
        """
        if "l" not in axes:
            raise ValueError(f"axis order must contain 'l' (scale level), got {axes!r}")

        remainder = axes.replace("l", "", 1)
        if not (len(remainder) == 3 or (len(remainder) == 4 and remainder[0] == "c")):
            raise ValueError(
                f"axis order {axes!r} is not usable by a 3D encoder: after the level axis it must "
                'be three spatial axes, optionally preceded by \'c\' (e.g. "lzyx" or "lcxyz"), '
                f"got {remainder!r}"
            )

        expected_dims = len(axes) + 1
        if batch.dim() != expected_dims:
            raise ValueError(
                f"axis order {axes!r} implies a {expected_dims}-D batch (batch + {len(axes)} "
                f"axes), got {tuple(batch.shape)}; it must match the dataset's output_axes"
            )

        level_dim = axes.index("l") + 1
        levels = batch.shape[level_dim]
        if levels != self.num_levels:
            raise ValueError(
                f"{type(self).__name__} is configured for {self.num_levels} scale levels "
                f"{self.levels}, but this batch carries {levels} on axis 'l' (shape "
                f"{tuple(batch.shape)}). Configure the dataset with one resolution per level, in "
                "the same order as `levels`."
            )

        # Move the level axis to position 1 and make the channel axis explicit, so the rest of
        # this class can assume (B, L, C, D, H, W) whatever order the dataset declared.
        volumes = batch.movedim(level_dim, 1)
        if volumes.dim() == 5:  # axis order declared no channel; add a singleton
            volumes = volumes.unsqueeze(2)
        return volumes

    def default_bbox(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """World-coordinate extents for concentric crops -> (B, L, 2, 3).

        Each level is centred on the origin and spans its downsampling factor times the finest
        level's extent, which is the geometry you get when every level is cropped about the same
        point. Datasets that place their levels off-centre must pass their own boxes: the paper
        finds that feeding wrong coordinates costs a substantial amount of performance, and nothing
        downstream can detect it, since wrong coordinates are still perfectly well-shaped.
        """
        # Half-extent in units of finest-level pixels. The 0.5 puts the outermost patch centres on
        # the edge pixel centres rather than half a pixel beyond them.
        half = torch.tensor(
            [[level * (size / 2 - 0.5) for size in self.img_size] for level in self.levels],
            dtype=torch.float32,
            device=device,
        )  # (L, 3)
        bbox = torch.stack((-half, half), dim=1)  # (L, 2, 3)
        return bbox.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def world_coords(self, bbox: torch.Tensor) -> torch.Tensor:
        """Patch-centre coordinates in the shared world frame -> (B, L * patches_per_level, 3).

        Each level's patch grid is mapped linearly onto that level's box, so a coarse level's
        patches spread over a proportionally wider span. Coordinates from different levels are
        therefore directly comparable, which is the property the rotary embedding needs.

        Coordinates come out relative to the finest level's centre, for a numerical reason rather
        than a modelling one. Datasets report boxes in absolute volume coordinates -- miao gives
        nanometres, which reach the millions in a large volume -- and rotary angles are that
        coordinate times a frequency. float32 keeps about seven digits total, so a large absolute
        offset spends them on magnitude and leaves too few for the phase differences between
        neighbouring patches that carry the actual signal. Subtracting one reference point per
        sample is exactly what rotary attention already ignores (it sees only differences, which
        `tests/unit/test_muvit.py` pins), so this discards nothing while keeping the arithmetic in
        a range where the differences survive.

        One reference for the whole sample, not one per level: subtracting each level's own centre
        would collapse every level onto the origin and destroy the cross-level geometry that is the
        entire point of the architecture.

        Units are whatever the dataset uses. Only ratios matter to rotary attention, and the
        frequencies are learnable, so the scale is absorbed during training.
        """
        batch_size = bbox.shape[0]
        if bbox.shape[1:] != (self.num_levels, 2, SPATIAL_RANK):
            raise ValueError(
                f"bbox must have shape (B, {self.num_levels}, 2, {SPATIAL_RANK}) giving the low "
                f"and high world-coordinate corner of each level, got {tuple(bbox.shape)}"
            )

        bbox = bbox.to(torch.float32)
        # The finest level defines the world frame, following the paper, and is also the level whose
        # box is known most precisely. (B, 3) -> (B, 1, 1, 3) to broadcast over levels and corners.
        origin = ((bbox[:, 0, 0, :] + bbox[:, 0, 1, :]) / 2).reshape(batch_size, 1, 1, SPATIAL_RANK)
        bbox = bbox - origin

        # linspace rather than arange/(n-1): it puts a single-patch axis at the box's low corner
        # instead of dividing by zero.
        axis_fractions = [
            torch.linspace(0.0, 1.0, count, device=bbox.device, dtype=torch.float32)
            for count in self.grid_size
        ]
        grid = torch.stack(torch.meshgrid(*axis_fractions, indexing="ij"), dim=-1)
        fractions = grid.reshape(1, 1, -1, SPATIAL_RANK)  # (1, 1, patches, 3)

        low = bbox[:, :, 0, :].unsqueeze(2)  # (B, L, 1, 3)
        high = bbox[:, :, 1, :].unsqueeze(2)
        coords = low + fractions * (high - low)  # (B, L, patches, 3)
        return coords.reshape(batch_size, -1, SPATIAL_RANK)

    def patchify(self, volumes: torch.Tensor) -> torch.Tensor:
        """(B, L, C, D, H, W) -> (B, L * patches_per_level, patch_volume).

        Level-major, matching the token order out of `embed`, so a reconstruction target lines up
        with the predictions without further bookkeeping.
        """
        pd, ph, pw = self.patch_size
        gd, gh, gw = self.grid_size
        batch, levels, channels = volumes.shape[0], volumes.shape[1], volumes.shape[2]
        x = volumes.reshape(batch, levels, channels, gd, pd, gh, ph, gw, pw)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6, 8)
        return x.reshape(batch, levels * gd * gh * gw, channels * pd * ph * pw)

    def embed(
        self, volumes: torch.Tensor, bbox: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, L, C, D, H, W) -> tokens (B, N, embed_dim) and their coordinates (B, N, 3).

        Coordinates come back alongside the tokens because they are not a property of the model
        but of each individual token, and anything that reorders or drops tokens -- masked
        autoencoding above all -- has to carry them along in step. Returning them makes that
        impossible to forget.
        """
        expected = (self.num_levels, self.in_channels, *self.img_size)
        if volumes.shape[1:] != expected:
            raise ValueError(
                f"expected input (B, {', '.join(map(str, expected))}), got {tuple(volumes.shape)}"
            )

        if bbox is None:
            bbox = self.default_bbox(volumes.shape[0], volumes.device)
        coords = self.world_coords(bbox.to(volumes.device))

        patches = self.patchify(volumes)  # (B, L * patches, patch_volume)
        per_level = patches.reshape(
            volumes.shape[0], self.num_levels, self.patches_per_level, self.patch_volume
        )
        tokens = torch.cat(
            [
                projection(per_level[:, index]) + self.level_embed[index]
                for index, projection in enumerate(self.patch_proj)
            ],
            dim=1,
        )
        return tokens, coords

    def encode(self, tokens: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Run the transformer over any number of tokens -> (B, N, embed_dim).

        The token count is free -- masked autoencoding passes a visible subset -- but every token
        must bring its coordinate, so `coords` is required rather than optional.
        """
        if coords.shape[:2] != tokens.shape[:2]:
            raise ValueError(
                f"every token needs a coordinate: got {tokens.shape[1]} tokens but "
                f"{coords.shape[1]} coordinates"
            )
        for block in self.blocks:
            tokens = block(tokens, coords)
        return self.norm(tokens)

    def forward(self, x: torch.Tensor, bbox: torch.Tensor | None = None) -> torch.Tensor:
        """Mean-pooled embedding over all levels' tokens, for downstream heads."""
        tokens, coords = self.embed(x, bbox)
        return self.encode(tokens, coords).mean(dim=1)

    def extra_forward_methods(self) -> tuple[str, ...]:
        """`embed` and `encode` are called directly by masked autoencoding, not through forward."""
        return ("embed", "encode")

    def flops(self, input_shape: tuple[int, ...]) -> int:
        """Rough forward FLOPs for one sample: patch projections, attention, and MLPs.

        Attention is quadratic in the *joint* sequence, so adding a level costs more than the
        tokens it contributes -- that cross-level term is the point of the architecture.
        """
        n = self.num_patches
        d = self.embed_dim
        depth = len(self.blocks)
        hidden = int(d * self.mlp_ratio)
        patch_proj = 2 * n * self.patch_volume * d
        per_block = 2 * (4 * n * d * d) + 2 * (2 * n * n * d) + 2 * (2 * n * d * hidden)
        return int(patch_proj + depth * per_block)
