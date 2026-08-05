from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.utils.data as data

from algorithms.mae import MAE
from algorithms.registry import AlgorithmRegistry
from data.base import BaseDataset
from models.vit import ViT3D


def _algorithm(**overrides: Any) -> MAE:
    encoder = ViT3D(
        img_size=(16, 16, 16), patch_size=(8, 8, 8), in_channels=1,
        embed_dim=32, depth=2, num_heads=4,
    )
    # Annotated because a heterogeneous dict literal infers `dict[str, object]`, which cannot be
    # unpacked into MAE's typed signature.
    kwargs: dict[str, Any] = dict(
        input_axes="lzyx", decoder_embed_dim=16, decoder_depth=1, decoder_num_heads=2
    )
    kwargs.update(overrides)
    return MAE(encoder, **kwargs)


class _AxesDataset(BaseDataset):
    """A dataset that exists only to declare (or withhold) a sample axis order."""

    def __init__(self, sample_axes: str | None) -> None:
        super().__init__()
        self._sample_axes = sample_axes

    def build_dataset(self) -> data.Dataset:
        return data.TensorDataset(torch.zeros(1, 1, 16, 16, 16))

    @property
    def sample_axes(self) -> str | None:
        return self._sample_axes


@pytest.mark.unit
def test_registered_under_its_config_name():
    assert AlgorithmRegistry.get("mae") is MAE


@pytest.mark.unit
@pytest.mark.parametrize("bad_ratio", [0.0, 1.0, -0.1, 1.5])
def test_rejects_out_of_range_mask_ratio(bad_ratio):
    with pytest.raises(ValueError, match="mask_ratio"):
        _algorithm(mask_ratio=bad_ratio)


@pytest.mark.unit
def test_takes_the_axis_order_from_the_dataset():
    # The data's own layout wins, so it never has to be restated in the algorithm's config.
    algorithm = _algorithm(input_axes=None, dataset=_AxesDataset("lcxyz"))
    assert algorithm.input_axes == "lcxyz"


@pytest.mark.unit
def test_accepts_input_axes_that_agrees_with_the_dataset():
    algorithm = _algorithm(input_axes="lcxyz", dataset=_AxesDataset("lcxyz"))
    assert algorithm.input_axes == "lcxyz"


@pytest.mark.unit
def test_rejects_input_axes_that_contradicts_the_dataset():
    # "lcxyz" and "clzyx" have the same rank, so letting input_axes win would strip the wrong
    # axis and train on nonsense without any shape error.
    with pytest.raises(ValueError, match="contradicts"):
        _algorithm(input_axes="clzyx", dataset=_AxesDataset("lcxyz"))


@pytest.mark.unit
def test_rejects_an_undeterminable_axis_order():
    with pytest.raises(ValueError, match="cannot determine"):
        _algorithm(input_axes=None, dataset=_AxesDataset(None))


@pytest.mark.unit
def test_masks_the_configured_fraction_of_patches():
    algorithm = _algorithm(mask_ratio=0.75)
    keep_idx, mask, restore = algorithm._random_masking(4, 8, torch.device("cpu"))

    assert keep_idx.shape == (4, 2)  # 8 patches, keep 25%
    assert torch.equal(mask.sum(dim=1), torch.full((4,), 6.0))
    assert restore.shape == (4, 8)
    # Indices and mask must agree: a kept token is never marked hidden.
    for row in range(4):
        assert (mask[row, keep_idx[row]] == 0).all()


@pytest.mark.unit
def test_patchify_round_trips_exactly():
    algorithm = _algorithm()
    volumes = torch.randn(2, 1, 16, 16, 16)
    patches = algorithm.encoder.patchify(volumes)
    assert patches.shape == (2, 8, 512)

    # Invert the reshape/permute and demand the original back: unlike a sum, this fails if the
    # permutation is wrong.
    gd, gh, gw = algorithm.encoder.grid_size
    pd, ph, pw = algorithm.encoder.patch_size
    restored = patches.reshape(2, gd, gh, gw, 1, pd, ph, pw)
    restored = restored.permute(0, 4, 1, 5, 2, 6, 3, 7)
    restored = restored.reshape(2, 1, gd * pd, gh * ph, gw * pw)
    assert torch.allclose(restored, volumes)


@pytest.mark.unit
def test_training_step_returns_a_differentiable_loss():
    algorithm = _algorithm()
    out = algorithm.training_step({"img": torch.randn(2, 1, 16, 16, 16)})

    assert out["loss"].requires_grad
    assert out["loss"].item() > 0.0
    assert out["masked_fraction"].item() == pytest.approx(0.75)
    out["loss"].backward()
    assert algorithm.encoder.patch_embed.weight.grad is not None


@pytest.mark.unit
def test_validation_step_matches_training_step_shape():
    algorithm = _algorithm()
    out = algorithm.validation_step({"img": torch.randn(1, 1, 16, 16, 16)})
    assert set(out) == {"loss", "masked_fraction"}


@pytest.mark.unit
def test_overfits_a_single_batch():
    torch.manual_seed(0)
    algorithm = _algorithm(mask_ratio=0.5, norm_pix_loss=False)
    batch = {"img": torch.randn(2, 1, 16, 16, 16)}
    optimizer = torch.optim.AdamW(algorithm.parameters(), lr=1e-3)

    first = algorithm.training_step(batch)["loss"].item()
    for _ in range(60):
        loss = algorithm.training_step(batch)["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert algorithm.training_step(batch)["loss"].item() < first


@pytest.mark.unit
def test_loss_ignores_visible_patches():
    torch.manual_seed(0)
    algorithm = _algorithm(mask_ratio=0.5, norm_pix_loss=False)
    volumes = torch.randn(2, 1, 16, 16, 16)
    encoder_input = algorithm.encoder.prepare_input(volumes, algorithm.input_axes)
    tokens, coords = algorithm.encoder.embed(encoder_input)
    keep_idx, mask, restore = algorithm._random_masking(
        tokens.shape[0], tokens.shape[1], tokens.device
    )
    gather = keep_idx.unsqueeze(-1)
    visible = torch.gather(tokens, 1, gather.expand(-1, -1, tokens.shape[-1]))
    visible_coords = torch.gather(coords, 1, gather.expand(-1, -1, 3))
    target = algorithm.encoder.patchify(encoder_input)

    latent = algorithm.encoder.encode(visible, visible_coords)
    prediction = algorithm._decode(latent, restore, coords)
    per_patch = (prediction - target).pow(2).mean(dim=-1)
    baseline = (per_patch * mask).sum() / mask.sum()

    # Corrupt the prediction only where mask == 0, i.e. the visible patches.
    corrupted = prediction + (1.0 - mask).unsqueeze(-1) * 1000.0
    per_patch_corrupted = (corrupted - target).pow(2).mean(dim=-1)
    after = (per_patch_corrupted * mask).sum() / mask.sum()

    assert torch.allclose(baseline, after), "visible patches must not contribute to the loss"


def _algorithm_with_backend(backend: str) -> MAE:
    encoder = ViT3D(
        img_size=(16, 16, 16), patch_size=(8, 8, 8), in_channels=1,
        embed_dim=32, depth=2, num_heads=4, attention_backend=backend,
    )
    return MAE(
        encoder, input_axes="lzyx", decoder_embed_dim=16, decoder_depth=2, decoder_num_heads=2
    )


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["sdpa", "auto"])
def test_decoder_uses_the_encoders_attention_backend(backend):
    # The decoder takes its kernel from the encoder instead of exposing a second knob, so nothing
    # can leave the two halves of the model on different kernels. Without this, a decoder pinned
    # to the default would still train and still pass every other test.
    algorithm = _algorithm_with_backend(backend)
    assert [block.attn.backend for block in algorithm.encoder.blocks] == [backend] * 2
    assert [block.attn.backend for block in algorithm.decoder_blocks] == [backend] * 2


@pytest.mark.unit
def test_no_part_of_the_model_uses_torch_multihead_attention():
    # Encoder and decoder together: the swappable kernel is only swappable if every attention
    # call goes through our module.
    algorithm = _algorithm_with_backend("sdpa")
    assert not any(isinstance(m, nn.MultiheadAttention) for m in algorithm.modules())


@pytest.mark.unit
def test_visible_coordinates_stay_in_step_with_visible_tokens():
    # Position now lives in attention, so the encoder is handed coordinates alongside the surviving
    # tokens. Gathering the two with different indices attaches every visible token to another
    # patch's position: shapes still line up, the loss still falls, and the geometry is nonsense.
    algorithm = _algorithm()
    volumes = torch.randn(1, 1, 16, 16, 16)

    keep_idx = torch.tensor([[3, 6]])
    mask = torch.ones(1, 8)
    mask[0, keep_idx[0]] = 0.0
    restore = torch.arange(8).reshape(1, 8)
    algorithm._random_masking = lambda *args: (keep_idx, mask, restore)

    seen: dict[str, torch.Tensor] = {}
    real_encode = algorithm.encoder.encode

    def spy(tokens, coords):
        seen["tokens"], seen["coords"] = tokens, coords
        return real_encode(tokens, coords)

    algorithm.encoder.encode = spy
    algorithm({"img": volumes})

    prepared = algorithm.encoder.prepare_input(volumes, algorithm.input_axes)
    all_tokens, all_coords = algorithm.encoder.embed(prepared)
    assert torch.allclose(seen["tokens"], all_tokens[:, keep_idx[0]], atol=1e-5)
    assert torch.allclose(seen["coords"], all_coords[:, keep_idx[0]], atol=1e-5)


@pytest.mark.unit
def test_decoder_is_positioned_by_the_full_patch_grid():
    # The decoder sees every position, so it needs no gathering -- but it does need coordinates in
    # grid order, matching the tokens after they are restored from the shuffle.
    algorithm = _algorithm()
    assert not any("decoder_pos_embed" in name for name in dict(algorithm.named_parameters()))

    seen: dict[str, torch.Tensor] = {}
    real_block = algorithm.decoder_blocks[0].forward

    def spy(x, coords):
        seen["coords"] = coords
        return real_block(x, coords)

    algorithm.decoder_blocks[0].forward = spy
    algorithm({"img": torch.randn(1, 1, 16, 16, 16)})

    expected = algorithm.encoder.patch_coords(1, torch.device("cpu"))
    assert torch.equal(seen["coords"], expected)
