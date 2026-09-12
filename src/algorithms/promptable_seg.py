"""Promptable instance segmentation: given a click, a box or a class, return one object.

The other supervised strategies here answer a question about every voxel at once -- which class is
this (`semantic_seg`), does this voxel belong with its neighbour (`affinity_seg`). This one answers
a question about *one object*, asked by pointing at it, and returns a mask for it. That is Segment
Anything's promptable segmentation task, and what makes it worth having on volumes is that instance
identity in EM/ExM is unbounded and arbitrary: there is no fixed vocabulary to predict into, and a
crop can hold thousands of objects, so a model that segments the one you ask for composes into
proofreading, sparse annotation and whole-volume mask generation in ways a dense predictor does not.

**Why the prompt encoder and mask decoder live in the algorithm rather than in the model.** They
ship with the trained system rather than being discarded like a pretraining decoder, which by the
letter of `DESIGN.md` argues for `models/`. Four engine mechanisms read `algorithm.model` and take
it to be the backbone: `Trainer._backbone_parameters` (so `freeze_backbone_steps` would freeze this
head during its own warm-up, which is the opposite of the intent), `engine.optimizer._depth` (so
`layerwise_lr_decay` would give head tensors an encoder's depth), `[lora].targets`, and every
`[init].prefix` that warm-starts a DINOv3 checkpoint. Keeping the backbone bare leaves all four
correct with no core change, which is also what `MAE`, `MuViTMAE` and `AffinitySegmentation` do.
The reusable parts are in `layers/common/` so they are not private to this file.

**Any backbone with patch tokens will do.** The only thing asked of the model is
`BaseModel.patch_features` and `prepare_input` -- the same contract the other dense strategies use
-- plus its `embed_dim` and `patch_size`. A neck decouples the decoder's width from the encoder's,
so one decoder recipe serves a ViT-B and a 7B alike.

**Masks come out at `patch_size / mask_upscale`, not at voxel resolution.** In 2D the final
upsample is nearly free; in 3D it multiplies the largest tensor in the step by the cube of the
stride -- 16 masks of a 256-cube is 12.9 GB at full resolution against 200 MB at stride 4. The
reference itself trains at stride 4 (256-pixel masks for a 1024-pixel image); the difference is
that here the targets are area-pooled to meet the prediction rather than the prediction being
upsampled to meet the targets. `mask_upscale` is the knob, and the right value is a measurement:
these objects are thin where natural-image objects are not.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from data.base import BaseDataset
from layers.common.norms import ChannelLayerNorm
from layers.common.prompt import (
    BACKGROUND,
    BOX_FAR,
    BOX_NEAR,
    FOREGROUND,
    PAD,
    PromptEncoder3D,
)
from layers.common.rope import patch_grid_coords, voxel_coords
from models.base import BaseModel

from .base import BaseAlgorithm
from .promptable.amg import PREFER, TILE_MERGE, PromptGridPredictor
from .promptable.decoder import MaskDecoder3D
from .promptable.losses import best_of, dice_loss, focal_loss, mask_iou, voxel_mask_iou
from .promptable.targets import (
    BOXES_KEY,
    IDS_KEY,
    OFFOBJECT_KEY,
    POINTS_KEY,
    SPLIT_LABEL_KEY,
    VALID_KEY,
    PromptTargets,
    pooled_known,
    pooled_masks,
)
from .registry import AlgorithmRegistry

SPATIAL_RANK = 3
#: What `PromptTargets` attaches to a sample and `_step` reads back, in one place.
_OBJECT_KEYS = (SPLIT_LABEL_KEY, IDS_KEY, POINTS_KEY, BOXES_KEY, VALID_KEY, OFFOBJECT_KEY)


@AlgorithmRegistry.register("promptable_seg")
class PromptableSegmentation(BaseAlgorithm):
    """Train a backbone to answer "segment *this*" from a point, a box or a class name.

    `masks_per_sample` objects are drawn per crop and decoded against one encoder pass, which is
    the amortisation the architecture exists for: the encoder is the expensive half and it does not
    depend on the prompt.

    `rounds` simulates an interactive session. Round 0 gets a single foreground point or, with
    probability `box_prob`, the object's noised bounding box. Each later round adds one point drawn
    from the region where the previous prediction and the target disagree -- foreground if the
    prediction missed, background if it overreached -- and feeds the previous round's raw logits
    back as a dense prompt. The reference uses 11 rounds because its decoder is under 1% of its
    encoder's compute; in 3D the mask head runs at voxel scale and that ratio does not hold, so the
    default is 3 and the count is a knob to be measured.

    Round 0 predicts `num_multimask_outputs` candidates and is scored on whichever is best; later
    rounds predict one. A single click genuinely is ambiguous -- a point inside a mitochondrion
    could mean the cristae, the organelle or the cell -- and averaging those answers trains the
    model to predict their union, an object that is none of them.

    `num_classes` enables semantic prompting against a closed ontology (CellMap's `classes.csv` and
    the corpus's label-group names, ~35 atomic classes). It is an embedding table rather than a
    text encoder because the corpus carries no captions or descriptions to train one from; a text
    encoder producing tokens in the same space would slot in behind the same interface.

    This strategy deliberately implements no `logits()` / `prediction_kind`: `predict.py`'s dense
    protocol is a fixed number of channels per voxel, and what this model emits depends on what it
    was asked. Whole-volume inference is instead a search over prompts, provided through
    `volume_predictor()`, which `predict.py` drives through the same command as every other
    strategy.
    """

    def __init__(
        self,
        model: BaseModel,
        dataset: BaseDataset | None = None,
        input_axes: str | None = None,
        input_key: str = "img",
        label_key: str = "label",
        prompt_dim: int = 256,
        decoder_depth: int = 2,
        decoder_heads: int = 8,
        decoder_mlp_ratio: float = 4.0,
        num_multimask_outputs: int = 3,
        mask_upscale: int = 4,
        mask_feature_dim: int = 32,
        mask_refine_depth: int = 2,
        mask_upscale_hidden: int | None = None,
        masks_per_sample: int = 16,
        min_object_voxels: int = 64,
        offobject_prob: float = 0.0,
        boundary_prob: float = 0.0,
        boundary_radius: int = 1,
        rounds: int = 3,
        box_prob: float = 0.5,
        box_noise: float = 0.1,
        box_noise_max: int = 20,
        num_classes: int = 0,
        focal_weight: float = 20.0,
        dice_weight: float = 1.0,
        iou_weight: float = 1.0,
        attention_backend: str = "auto",
        points_per_side: int = 12,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.5,
        stability_thresh: float = 0.8,
        stability_delta: float = 1.0,
        nms_iou: float = 0.7,
        min_mask_voxels: int = 512,
        max_mask_fraction: float = 0.95,
        prefer: str = "whole",
        containment_thresh: float = 0.8,
        merge_fraction: float = 0.5,
        tile_merge: str = "canvas",
        edge_discard: bool = False,
        skip_claimed_clicks: bool = False,
        propagate_min_coverage: float = 0.8,
        consistency_clicks: int = 0,
        consistency_thresh: float = 0.5,
        consistency_pick: str = "top",
        split_tiled_wholes: bool = False,
        tiled_cover_thresh: float = 0.8,
        agree_thresh: float = 0.5,
        min_support: int = 1,
    ) -> None:
        super().__init__(model, dataset)
        if rounds < 1:
            raise ValueError(f"rounds must be at least 1, got {rounds}")
        if not 0.0 <= box_prob <= 1.0:
            raise ValueError(f"box_prob must be a probability, got {box_prob}")
        if prefer not in PREFER:
            raise ValueError(f"prefer must be one of {PREFER}, got {prefer!r}")
        if tile_merge not in TILE_MERGE:
            raise ValueError(f"tile_merge must be one of {TILE_MERGE}, got {tile_merge!r}")

        self.input_axes = self._resolve_input_axes(input_axes, dataset)
        self.input_key = input_key
        self.label_key = label_key
        self.masks_per_sample = masks_per_sample
        self.min_object_voxels = min_object_voxels
        self.rounds = rounds
        self.box_prob = box_prob
        self.box_noise = box_noise
        self.box_noise_max = box_noise_max
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.encoder = model
        # Whole-volume mask generation, none of which touches training. Constructor arguments
        # rather than constants because every one of them is decided *after* training, against a
        # score on a held-out volume (`predict.py --override algorithm.<name>=...`), and the
        # reference's values assume a far better model than a first run produces.
        self.amg_settings: dict[str, Any] = {
            "points_per_side": points_per_side,
            "points_per_batch": points_per_batch,
            "pred_iou_thresh": pred_iou_thresh,
            "stability_thresh": stability_thresh,
            "stability_delta": stability_delta,
            "nms_iou": nms_iou,
            "min_mask_voxels": min_mask_voxels,
            "max_mask_fraction": max_mask_fraction,
            "prefer": prefer,
            "containment_thresh": containment_thresh,
            "merge_fraction": merge_fraction,
            "tile_merge": tile_merge,
            "edge_discard": edge_discard,
            "skip_claimed_clicks": skip_claimed_clicks,
            "propagate_min_coverage": propagate_min_coverage,
            "consistency_clicks": consistency_clicks,
            "consistency_thresh": consistency_thresh,
            "consistency_pick": consistency_pick,
            "split_tiled_wholes": split_tiled_wholes,
            "tiled_cover_thresh": tiled_cover_thresh,
            "agree_thresh": agree_thresh,
            "min_support": min_support,
        }

        patch = cast(Any, model).patch_size
        self.patch_size: tuple[int, ...] = (
            (patch,) * SPATIAL_RANK if isinstance(patch, int) else tuple(patch)
        )
        if any(step % mask_upscale for step in self.patch_size):
            raise ValueError(
                f"mask_upscale {mask_upscale} does not divide the encoder's patch size "
                f"{self.patch_size}; the decoder expands each token into a whole number of output "
                "blocks, so the mask stride would not be an integer"
            )
        self.mask_upscale = mask_upscale
        #: Voxels per output element, per axis. The targets are pooled to exactly this.
        self.mask_stride = tuple(step // mask_upscale for step in self.patch_size)

        embed_dim: int = model.embed_dim  # type: ignore[assignment]
        # Decouples the decoder's width from the encoder's, so the same decoder recipe serves a
        # ViT-B and a 7B. Bias-free because a normalization follows each convolution.
        self.neck = nn.Sequential(
            nn.Conv3d(embed_dim, prompt_dim, kernel_size=1, bias=False),
            ChannelLayerNorm(prompt_dim),
            nn.Conv3d(prompt_dim, prompt_dim, kernel_size=3, padding=1, bias=False),
            ChannelLayerNorm(prompt_dim),
        )
        self.prompt_encoder = PromptEncoder3D(
            prompt_dim, mask_downscale=mask_upscale, num_classes=num_classes
        )
        # `mask_refine_depth` and `mask_upscale_hidden` reach the sub-pixel expansion's own
        # knobs. The refinement convolutions are the only layers that run *at* mask resolution --
        # everything else is decided on the patch grid -- so they are where extra capacity for
        # fine spatial detail would go, and their count sets the receptive field across the
        # seams between independently decoded blocks (`2 * depth + 1`). Exposed rather than fixed
        # so the trade can be measured; the defaults are the decoder's own.
        self.decoder = MaskDecoder3D(
            prompt_dim,
            num_heads=decoder_heads,
            depth=decoder_depth,
            mlp_ratio=decoder_mlp_ratio,
            num_multimask_outputs=num_multimask_outputs,
            upscale=mask_upscale,
            mask_feature_dim=mask_feature_dim,
            upscale_hidden=mask_upscale_hidden,
            refine_depth=mask_refine_depth,
            attention_backend=attention_backend,
        )

        # Set by `sample_transform` when the engine moves object selection into the dataloader's
        # workers. Until then `_objects` runs the same transform itself, so an algorithm driven
        # without the engine -- a test, a notebook -- still gets targets rather than a KeyError.
        self._targets = PromptTargets(
            label_key=label_key,
            masks_per_sample=masks_per_sample,
            min_object_voxels=min_object_voxels,
            offobject_prob=offobject_prob,
            boundary_prob=boundary_prob,
            boundary_radius=boundary_radius,
        )
        self._delegated = False

    def sample_transform(self) -> PromptTargets:
        self._delegated = True
        return self._targets

    def volume_predictor(self) -> PromptGridPredictor:
        """Segment everything: a grid of clicks per tile, reconciled into one labelling.

        The reason this strategy does not implement `logits()`: its whole-volume output is a search
        over prompts yielding a set of masks, not a fixed number of channels per voxel, and
        `predict.py` drives it through this rather than through the dense path. See
        `promptable/amg.py` for the procedure and the three places it departs from the reference.
        """
        return PromptGridPredictor(self, **self.amg_settings)

    def checkpointable_modules(self) -> tuple[nn.Module, ...]:
        """The expansion to mask resolution, which is where this strategy's memory goes.

        Everything before it runs on the patch grid; everything inside it carries
        `mask_feature_dim` channels over the upscaled volume, once per prompt. The same division
        `affinity_seg` makes, for the same reason: a checkpointed region stores its own inputs, and
        the patch grid is thousands of times smaller than the volume.
        """
        return (self.decoder.upscaler,)

    @staticmethod
    def _resolve_input_axes(input_axes: str | None, dataset: BaseDataset | None) -> str:
        """Settle on the sample axis order, preferring the dataset's own answer.

        Mirrors `AffinitySegmentation._resolve_input_axes` but imposes no order on the spatial
        axes. Affinity targets are defined against a published x,y,z channel convention and would
        be silently transposed under any other; a prompt is a position in whatever frame the crop
        came in, and the labels arrive in the same frame, so nothing here can be transposed
        relative to anything else.
        """
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

    def _objects(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """The drawn objects for this batch, computed in the workers or here as a fallback."""
        if SPLIT_LABEL_KEY in batch:
            return {
                key: batch[key]
                for key in _OBJECT_KEYS
            }
        if self.label_key not in batch:
            raise KeyError(
                f"batch has no {self.label_key!r} key, so there is nothing to prompt for. Set "
                "`label_key` on the dataset's volumes so miao reads the instance segmentation "
                f"alongside the image; got keys {sorted(batch)}"
            )
        # Same transform, run here rather than in a worker. A placement difference, not a fallback
        # to different behaviour -- it is the same object the engine would have attached.
        samples = [
            self._targets({self.label_key: batch[self.label_key][index]})
            for index in range(batch[self.label_key].shape[0])
        ]
        return {
            key: torch.stack([sample[key] for sample in samples])
            for key in _OBJECT_KEYS
        }

    def _initial_prompt(
        self,
        points: torch.Tensor,
        boxes: torch.Tensor,
        extent: tuple[int, ...],
        offobject: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Round 0's prompt: a click, or a noised box -> `(P, 2, 3)` coords and `(P, 2)` labels.

        Two slots either way, so the batch is rectangular: a click occupies one and pads the other.
        The reference's noise profile -- a per-coordinate normal with a standard deviation of 10%
        of that side, capped at 20 voxels -- is what keeps a box prompt from being usable only when
        it is a *tight* box, which is the case a detector rarely delivers.
        """
        prompts = points.shape[0]
        device = points.device
        # Never a box for an off-object prompt: there is no object to draw one around, and the
        # grid this prompt stands in for only ever clicks.
        use_box = torch.rand(prompts, device=device) < self.box_prob
        if offobject is not None:
            use_box = use_box & ~offobject

        # Independent noise per corner per axis, not one offset applied to both corners: a shared
        # offset only ever *translates* the box, so the model would never see a box that is too
        # large or too small for its object -- which is the failure mode a detector's box actually
        # has. The standard deviation is 10% of that axis's side length, capped at 20 voxels, which
        # is the reference's profile.
        sides = (boxes[:, 1] - boxes[:, 0]).float()
        corners = torch.stack([boxes[:, 0].float(), boxes[:, 1].float() - 1], dim=1)
        noise = torch.randn_like(corners) * (self.box_noise * sides).unsqueeze(1)
        corners = corners + noise.clamp(-self.box_noise_max, self.box_noise_max)
        limit = torch.tensor(extent, device=device, dtype=corners.dtype) - 1
        corners = corners.clamp(torch.zeros_like(limit), limit)

        click = torch.stack([points.float(), torch.zeros_like(points.float())], dim=1)
        coords = torch.where(use_box.reshape(-1, 1, 1), corners, click)

        labels = torch.where(
            use_box.reshape(-1, 1),
            torch.tensor([BOX_NEAR, BOX_FAR], device=device).expand(prompts, 2),
            torch.tensor([FOREGROUND, PAD], device=device).expand(prompts, 2),
        )
        return voxel_coords(coords, extent), labels

    def _correction(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        extent: tuple[int, ...],
        known: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One point per prompt from where the prediction and the target disagree.

        `known` (`(P, 1, *grid)`, the labelled fraction of each cell) keeps the click out of
        unlabelled space: a mask reaching into voxels nobody labelled is not a spill, and an
        unlabelled voxel is not a miss.

        `logits` and `target` are at the decoder's own stride, so the point is drawn on the mask
        grid and reported at the centre of the block it lands in. That is a coarser click than the
        reference's, which draws at pixel resolution -- but the coordinate is continuous and the
        prompt only has to fall inside the right region, and drawing at voxel resolution would mean
        expanding every prediction by `stride^3` to find the error.

        A prompt whose prediction already agrees everywhere contributes a padding token: there is
        no correction to make, and inventing one would teach the model to distrust a correct mask.
        """
        predicted = logits.squeeze(1) > 0
        actual = target.squeeze(1) > 0.5
        missed, spilled = actual & ~predicted, predicted & ~actual
        if known is not None:
            labelled = known.squeeze(1) > 0.5
            missed, spilled = missed & labelled, spilled & labelled
        wrong = (missed | spilled).flatten(1)

        key = torch.rand(wrong.shape, device=wrong.device).masked_fill(~wrong, float("inf"))
        chosen = key.argmin(dim=1)
        has_error = wrong.any(dim=1)

        blocks = tuple(predicted.shape[1:])
        indices, remaining = [], chosen
        for size in reversed(blocks):
            indices.append(remaining % size)
            remaining = remaining // size
        block = torch.stack(list(reversed(indices)), dim=-1)
        stride = torch.tensor(self.mask_stride, device=block.device)
        centre = block * stride + (stride - 1).float() / 2

        # Foreground where the prediction missed the object, background where it spilled out of it.
        is_missed = missed.flatten(1).gather(1, chosen.unsqueeze(1)).squeeze(1)
        labels = torch.where(is_missed, FOREGROUND, BACKGROUND)
        labels = torch.where(has_error, labels, torch.full_like(labels, PAD))
        return voxel_coords(centre, extent).unsqueeze(1), labels.unsqueeze(1)

    def encode(
        self, volumes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
        """`(B, C, *spatial)` volumes -> image embedding `(B, N, D)`, its coordinates, its grid.

        The prompt-independent half of the model, and the expensive one: run once per volume and
        reused by every prompt against it. `patch_features` is the only thing asked of the
        encoder, so any backbone implementing it serves. The coordinates come back with a leading
        axis of 1 because every prompt reads the same crop -- they broadcast over the prompt batch
        instead of being copied per prompt.

        Public because whole-volume prediction (`volume_predictor`) drives it directly, once per
        tile, and then decodes a few thousand prompts against the result.
        """
        with torch.profiler.record_function("encoder"):
            tokens, grid = self.encoder.patch_features(volumes)
        image = self.neck(
            tokens.transpose(1, 2).reshape(tokens.shape[0], -1, *grid)
        ).flatten(2).transpose(1, 2)
        extent = tuple(volumes.shape[-SPATIAL_RANK:])
        image_coords = patch_grid_coords(
            grid, self.patch_size, extent, device=image.device
        ).unsqueeze(0)
        return image, image_coords, grid

    def decode_points(
        self,
        image: torch.Tensor,
        image_coords: torch.Tensor,
        grid: tuple[int, ...],
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        multimask: bool,
        mask_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Point prompts against an embedding -> mask logits `(P, K, *mask grid)` and IoUs `(P, K)`.

        `image` is `(P, N, D)`, already expanded to the prompt batch; `point_coords` are in the
        [-1, 1] frame `rope.voxel_coords` produces and `point_labels` use `layers.common.prompt`'s
        vocabulary. Exactly what one interactive round does in training, so a prompt at inference
        is decoded by the same path it was trained through.
        """
        sparse, sparse_coords, dense = self.prompt_encoder(
            point_coords, point_labels, grid, mask_input=mask_input
        )
        with torch.profiler.record_function("decoder"):
            return self.decoder(
                image, image_coords, grid, sparse, sparse_coords, dense, multimask=multimask
            )

    def _step(self, batch: Any) -> dict[str, torch.Tensor]:
        objects = self._objects(batch)
        volumes = self.encoder.prepare_input(batch[self.input_key], self.input_axes)
        labels = objects[SPLIT_LABEL_KEY]
        extent = tuple(volumes.shape[-SPATIAL_RANK:])
        if tuple(labels.shape[1:]) != extent:
            raise ValueError(
                f"label crop {tuple(labels.shape[1:])} does not match the image crop {extent} it "
                "must be co-registered with"
            )

        image, image_coords, grid = self.encode(volumes)
        batch_size, masks = objects[IDS_KEY].shape
        prompts = batch_size * masks
        image = image.repeat_interleave(masks, dim=0)

        cells = [extent[axis] // self.mask_stride[axis] for axis in range(SPATIAL_RANK)]
        target = pooled_masks(labels, objects[IDS_KEY], self.mask_stride)
        target = target.reshape(prompts, 1, *cells)
        # The labelled fraction of every cell, per prompt. A cell's target is its object's share
        # of the LABELLED voxels, and every loss below weights cells by `known`, so unlabelled
        # voxels (-1: a pseudo-label's unclaimed space) train nothing: neither object nor
        # background.
        known = pooled_known(labels, self.mask_stride).repeat_interleave(masks, dim=0)
        target = torch.where(
            known > 0, (target / known.clamp_min(1e-6)).clamp_max(1.0), torch.zeros_like(target)
        )
        valid = objects[VALID_KEY].reshape(prompts)
        offobject = objects[OFFOBJECT_KEY].reshape(prompts)
        # Two weightings. The mask losses see object prompts only: an off-object prompt has no
        # object to segment, and pulling the decoder towards an empty mask there is the collapse
        # every mask model starts in. The IoU head sees both -- for an off-object prompt every
        # candidate's achieved IoU is 0, which is exactly what it has to learn to say.
        mask_weight = valid.float()
        head_weight = (valid | offobject).float()

        coords, point_labels = self._initial_prompt(
            objects[POINTS_KEY].reshape(prompts, SPATIAL_RANK),
            objects[BOXES_KEY].reshape(prompts, 2, SPATIAL_RANK),
            extent,
            offobject,
        )

        mask_input: torch.Tensor | None = None
        totals: dict[str, torch.Tensor] = {}
        for index in range(self.rounds):
            # Only the first round is ambiguous: by the second the model has been told where it
            # went wrong, so three answers would be three copies and the min-reduction would give
            # the winner a third of the gradient it should have.
            logits, scores = self.decode_points(
                image, image_coords, grid, coords, point_labels,
                multimask=index == 0, mask_input=mask_input,
            )

            candidates = logits.shape[1]
            expanded = target.expand(-1, candidates, *(-1,) * SPATIAL_RANK)
            cell_weight = known.expand(-1, candidates, *(-1,) * SPATIAL_RANK)
            per_candidate = (
                self.focal_weight * focal_loss(logits, expanded, weight=cell_weight)
                + self.dice_weight * dice_loss(logits, expanded, weight=cell_weight)
            )
            chosen, which = best_of(per_candidate)
            achieved = mask_iou(logits, expanded, weight=cell_weight)
            iou_loss = (scores - achieved.detach()).square().mean(dim=1)

            weight = mask_weight
            denominator = weight.sum().clamp_min(1.0)
            loss = (chosen * weight).sum() / denominator + self.iou_weight * (
                (iou_loss * head_weight).sum() / head_weight.sum().clamp_min(1.0)
            )

            picked = achieved.gather(1, which.unsqueeze(1)).squeeze(1)
            # The same masks scored at voxel resolution rather than on the mask grid. Reported
            # beside `*_iou` rather than replacing it, because they answer different questions and
            # their difference IS the quantisation ceiling: `*_iou` is what the loss optimises,
            # `*_voxel_iou` is what a consumer of the mask gets. Only the second is comparable
            # across `mask_upscale`.
            at_voxels = voxel_mask_iou(logits, expanded, weight=cell_weight)
            stage = "first" if index == 0 else "final"
            totals[f"loss_round_{index}"] = loss.detach()
            totals[f"{stage}_iou"] = (picked * weight).sum() / denominator
            oracle = achieved.max(dim=1).values
            totals[f"{stage}_oracle_iou"] = (oracle * weight).sum() / denominator
            totals[f"{stage}_voxel_iou"] = (
                at_voxels.gather(1, which.unsqueeze(1)).squeeze(1) * weight
            ).sum() / denominator
            totals[f"{stage}_iou_error"] = (
                (scores - achieved).abs().mean(dim=1) * weight
            ).sum() / denominator
            totals["loss"] = loss if "loss" not in totals else totals["loss"] + loss
            if index == 0:
                # What the mask generator's gate sees for a click on nothing: the best predicted
                # IoU among the candidates. The recipe exists to drive this to 0; if it stays
                # near the object prompts' scores the head has not learned the distinction.
                off = offobject.float()
                totals["offobject_pred_iou"] = (
                    scores.max(dim=1).values * off
                ).sum() / off.sum().clamp_min(1.0)

            if index + 1 < self.rounds:
                # The mask carried forward is the model's own best guess, unthresholded: the
                # reference feeds logits rather than a binary mask so the next round can see how
                # confident the last one was, which is most of what makes refinement work.
                best = logits.gather(
                    1,
                    which.reshape(-1, 1, *(1,) * SPATIAL_RANK).expand(-1, 1, *logits.shape[2:]),
                )
                new_coords, new_labels = self._correction(best.detach(), target, extent, known)
                coords = torch.cat([coords, new_coords], dim=1)
                point_labels = torch.cat([point_labels, new_labels], dim=1)
                mask_input = best.detach()

        totals["loss"] = totals["loss"] / self.rounds
        totals["valid_fraction"] = valid.float().mean()
        totals["offobject_fraction"] = offobject.float().mean()
        return totals

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._step(batch)

    def validation_step(self, batch: Any) -> dict[str, torch.Tensor]:
        return self._step(batch)
