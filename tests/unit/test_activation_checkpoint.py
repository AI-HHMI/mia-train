"""Unit tests for activation checkpointing.

The property that matters is that it changes *nothing* observable. Recomputing a region in
backward instead of storing it is a memory-for-compute trade and must leave the loss, the
gradients, and the saved checkpoint byte-identical -- otherwise a run that enables it to fit on a
GPU is no longer the run it was compared against.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from algorithms.affinity_seg import AffinitySegmentation
from algorithms.base import BaseAlgorithm
from engine.activation_checkpoint import (
    apply_activation_checkpointing,
    checkpointable_targets,
)
from engine.optimizer import parameter_depth
from models.dinov3_vit import DinoVisionTransformer
from models.dinov3_vit3d import DinoVisionTransformer3D
from models.muvit import MuViT3D
from models.vit import ViT3D

CROP = 16
PATCH = 8
DEPTH = 3

# The wrapper interposes a module, so a parameter's path gains a segment. Everything that reads
# parameter *names* has to keep working through it; this is what those tests strip.
WRAPPER_SEGMENT = "._checkpoint_wrapped_module"


def _model(**overrides: Any) -> DinoVisionTransformer3D:
    kwargs: dict[str, Any] = dict(
        img_size=CROP, patch_size=PATCH, in_chans=1, embed_dim=32, depth=DEPTH, num_heads=2,
        n_storage_tokens=4, layerscale_init=1.0e-05, mask_k_bias=True,
        pos_embed_rope_type="superposition",
    )
    kwargs.update(overrides)
    return DinoVisionTransformer3D(**kwargs)


def _algorithm(**overrides: Any) -> AffinitySegmentation:
    kwargs: dict[str, Any] = dict(input_axes="lcxyz", decoder_hidden_dim=8, long_range=4)
    kwargs.update(overrides)
    return AffinitySegmentation(_model(), **kwargs)


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "img": torch.rand(2, 1, 1, CROP, CROP, CROP),
        "label": torch.randint(0, 3, (2, 1, CROP, CROP, CROP)),
    }


def _loss_and_grads(
    algorithm: BaseAlgorithm, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One step's loss and gradients, keyed by name with any wrapper segment removed."""
    torch.manual_seed(0)
    algorithm.zero_grad(set_to_none=True)
    loss = algorithm.training_step(batch)["loss"]
    loss.backward()
    return loss.detach(), {
        name.replace(WRAPPER_SEGMENT, ""): parameter.grad.clone()
        for name, parameter in algorithm.named_parameters()
        if parameter.grad is not None
    }


# ------------------------------------------------------------------ what gets wrapped


@pytest.mark.unit
def test_collects_from_both_the_model_and_the_algorithm():
    # An algorithm's own dense head is as much a candidate as the encoder's blocks, and the
    # engine sees one tree, so the walk has to reach both.
    algorithm = _algorithm()
    targets = checkpointable_targets(algorithm)

    assert len(targets) == DEPTH + 1
    assert all(block in targets for block in algorithm.model.blocks)
    assert isinstance(algorithm, AffinitySegmentation)
    assert algorithm.decoder_out in targets


@pytest.mark.unit
def test_wraps_exactly_the_declared_modules():
    algorithm = _algorithm()
    declared = {id(target) for target in checkpointable_targets(algorithm)}

    assert apply_activation_checkpointing(algorithm) == len(declared)

    wrapped = {
        name for name, _ in algorithm.named_modules() if name.endswith("_checkpoint_wrapped_module")
    }
    assert len(wrapped) == len(declared)


@pytest.mark.unit
def test_enabling_it_where_nothing_is_declared_is_an_error():
    # Silently doing nothing is the bad outcome: the run OOMs with the setting apparently on.
    class Bare(BaseAlgorithm):
        def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
            return {"loss": self.model(batch)}

        def validation_step(self, batch: Any) -> dict[str, torch.Tensor]:
            return self.training_step(batch)

    with pytest.raises(ValueError, match="declares checkpointable_modules"):
        apply_activation_checkpointing(Bare(nn.Linear(2, 2)))


@pytest.mark.unit
def test_every_registered_transformer_declares_its_blocks():
    # A model that grows blocks but forgets the hook cannot be checkpointed, and finds out only
    # when someone tries to train it at a size that needs it.
    models = [
        _model(),
        DinoVisionTransformer(img_size=CROP, patch_size=PATCH, embed_dim=32, depth=DEPTH,
                              num_heads=2),
        ViT3D(img_size=(CROP,) * 3, patch_size=(PATCH,) * 3, in_channels=1, embed_dim=32,
              depth=DEPTH, num_heads=2),
        MuViT3D(levels=(1, 4), img_size=(CROP,) * 3, patch_size=(PATCH,) * 3, in_channels=1,
                embed_dim=32, depth=DEPTH, num_heads=2),
    ]
    for model in models:
        blocks = model.checkpointable_modules()
        assert len(blocks) == DEPTH, f"{type(model).__name__} declared {len(blocks)}"


# ------------------------------------------------------------------ observational equivalence


@pytest.mark.unit
def test_loss_and_gradients_are_unchanged():
    batch = _batch()
    reference = _algorithm()
    checkpointed = _algorithm()
    checkpointed.load_state_dict(reference.state_dict())

    expected_loss, expected_grads = _loss_and_grads(reference, batch)
    apply_activation_checkpointing(checkpointed)
    actual_loss, actual_grads = _loss_and_grads(checkpointed, batch)

    assert torch.equal(expected_loss, actual_loss)
    assert set(actual_grads) == set(expected_grads)
    for name, expected in expected_grads.items():
        assert torch.equal(expected, actual_grads[name]), name


@pytest.mark.unit
def test_state_dict_keys_are_unchanged():
    # The wrapper renames parameters but registers hooks that strip it from the state dict. If
    # that ever stopped holding, every checkpoint written with the setting on would be unloadable
    # by a run with it off -- including the fine-tune that reads a pretraining run's encoder.
    reference = _algorithm()
    checkpointed = _algorithm()
    apply_activation_checkpointing(checkpointed)

    assert list(checkpointed.state_dict()) == list(reference.state_dict())


@pytest.mark.unit
def test_a_checkpointed_run_can_be_loaded_by_a_plain_one():
    checkpointed = _algorithm()
    apply_activation_checkpointing(checkpointed)
    for parameter in checkpointed.parameters():
        torch.nn.init.normal_(parameter)

    plain = _algorithm()
    plain.load_state_dict(checkpointed.state_dict())

    batch = _batch()
    with torch.no_grad():
        assert torch.equal(
            plain.validation_step(batch)["loss"], checkpointed.validation_step(batch)["loss"]
        )


@pytest.mark.unit
def test_layerwise_lr_decay_still_reads_the_block_index():
    # `build_param_groups` derives a parameter's depth from its name. The wrapper segment lands
    # between the block index and the rest, so a regex anchored too tightly would silently give
    # every block the same learning rate.
    algorithm = _algorithm()
    apply_activation_checkpointing(algorithm)

    depths = {
        parameter_depth(name, DEPTH, backbone_scoped=True)
        for name, _ in algorithm.named_parameters()
        if ".blocks." in name
    }
    assert len(depths) == DEPTH


@pytest.mark.unit
def test_no_recomputation_under_no_grad():
    # Recomputation exists to serve a backward pass; validation would pay for it and get nothing.
    algorithm = _algorithm()
    apply_activation_checkpointing(algorithm)
    batch = _batch()

    with torch.no_grad():
        metrics = algorithm.validation_step(batch)

    assert not metrics["loss"].requires_grad
