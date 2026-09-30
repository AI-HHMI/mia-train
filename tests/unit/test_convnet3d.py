"""Unit tests for `convnet3d`, the dense-convolution encoder of large_inputs phase 3."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

import models.convnet3d as convnet3d
from algorithms.affinity_seg import AffinitySegmentation
from engine.activation_checkpoint import apply_activation_checkpointing
from models.convnet3d import Conv3x3, ConvNet3D, PatchMerge
from models.registry import ModelRegistry

CROP = 32
SMALL = dict(img_size=CROP, widths=(8, 16), depths=(1, 2))  # two stages: output stride 8


@pytest.mark.unit
def test_registered_under_its_config_name():
    assert ModelRegistry.get("convnet3d") is ConvNet3D


@pytest.mark.unit
@pytest.mark.parametrize("block", ["basic", "fused"])
def test_patch_features_are_the_last_stage_in_grid_order(block):
    model = ConvNet3D(block=block, **SMALL)
    x = torch.randn(2, 1, 32, 48, 16)
    tokens, grid = model.patch_features(x)
    assert (model.patch_size, model.embed_dim, model.num_patches) == (8, 16, 4**3)
    assert grid == (4, 6, 2)
    # Token (i, j, k) is the last stage's position (i, j, k): row-major, as a head folds it.
    torch.testing.assert_close(tokens.view(2, *grid, 16), model(x)[-1])


@pytest.mark.unit
def test_pyramid_features_are_every_stage_at_its_stride():
    model = ConvNet3D(img_size=CROP, widths=(8, 16, 24), depths=(1, 2, 1))
    x = torch.randn(1, 1, 32, 48, 64)
    maps, grid = model.pyramid_features(x)
    assert model.pyramid_strides == (4, 8, 16) and model.pyramid_dims == (8, 16, 24)
    assert [tuple(m.shape) for m in maps] == [
        (1, width, 32 // stride, 48 // stride, 64 // stride)
        for width, stride in zip(model.pyramid_dims, model.pyramid_strides, strict=True)
    ]
    tokens, token_grid = model.patch_features(x)
    assert grid == token_grid
    torch.testing.assert_close(maps[-1].permute(0, 2, 3, 4, 1).reshape(1, -1, 24), tokens)


@pytest.mark.unit
def test_patch_merge_is_a_convolution_with_kernel_equal_to_stride():
    """Space-to-depth plus a matmul: each output position reads exactly its own 2^3 block."""
    merge = PatchMerge(3, 5, 2)
    x = torch.randn(1, 4, 6, 8, 3)  # (B, D, H, W, C)
    weight = merge.proj.weight.view(5, 2, 2, 2, 3).permute(0, 4, 1, 2, 3)
    expected = F.conv3d(x.permute(0, 4, 1, 2, 3), weight, merge.proj.bias, stride=2)
    torch.testing.assert_close(merge(x), expected.permute(0, 2, 3, 4, 1))


@pytest.mark.unit
@pytest.mark.parametrize(("cin", "cout"), [(4, 4), (2, 6), (6, 2)])
@pytest.mark.parametrize("planes", [1, 2, 3])
def test_slabs_along_depth_are_exact(monkeypatch, cin, cout, planes):
    """Past 2^31 elements the convolution runs in D slabs with halos; each slab size must agree."""
    conv = Conv3x3(cin, cout).double()
    x = torch.randn(1, 11, 5, 4, cin, dtype=torch.float64)
    whole = conv(x)
    plane = 5 * 4 * max(cin, cout)
    monkeypatch.setattr(convnet3d, "INDEX_LIMIT", (planes + 2) * plane + 1)
    torch.testing.assert_close(conv(x), whole)


@pytest.mark.unit
def test_a_plane_too_large_to_slab_is_refused(monkeypatch):
    conv = Conv3x3(4, 4)
    monkeypatch.setattr(convnet3d, "INDEX_LIMIT", 2 * 5 * 4 * 4)
    with pytest.raises(ValueError, match="no slab along D"):
        conv(torch.randn(1, 6, 5, 4, 4))


@pytest.mark.unit
@pytest.mark.parametrize("block", ["basic", "fused"])
def test_flops_are_the_convolutions_and_matmuls_torch_counts(block):
    model = ConvNet3D(block=block, **SMALL)
    with FlopCounterMode(display=False) as counter:
        model(torch.randn(1, 1, CROP, CROP, CROP))
    assert model.flops((1, CROP, CROP, CROP)) == counter.get_total_flops()


@pytest.mark.unit
def test_flops_refuse_a_crop_forward_would_refuse():
    model = ConvNet3D(**SMALL)
    with pytest.raises(ValueError, match="multiple of the output stride"):
        model.flops((1, CROP, CROP, CROP + 4))
    with pytest.raises(ValueError, match="in_chans"):
        model.flops((2, CROP, CROP, CROP))


@pytest.mark.unit
def test_every_block_is_a_checkpoint_and_fsdp_unit():
    model = ConvNet3D(**SMALL)
    blocks = list(model.blocks)
    assert list(model.checkpointable_modules()) == blocks == list(model.fsdp_units())


@pytest.mark.unit
def test_parameters_are_named_the_way_layerwise_decay_reads_depth():
    """`patch_embed` is the stem, and `blocks.<i>` is one flat list across the stages, with each
    stage's downsampling inside its first block: the names `engine.optimizer` reads depth from."""
    model = ConvNet3D(img_size=CROP, widths=(8, 16), depths=(2, 3))
    assert len(model.blocks) == 5
    assert [block.downsample is not None for block in model.blocks] == [
        False, False, True, False, False,
    ]
    names = [name for name, _ in model.named_parameters()]
    assert all(
        name.startswith(("patch_embed.", "blocks.", "norm.")) for name in names
    ), names


@pytest.mark.unit
def test_prepare_input_is_the_single_scale_contract():
    model = ConvNet3D(**SMALL)
    assert model.prepare_input(torch.randn(2, 1, 1, CROP, CROP, CROP), "lcxyz").shape == (
        2, 1, CROP, CROP, CROP,
    )
    with pytest.raises(ValueError, match="single-scale"):
        model.prepare_input(torch.randn(2, 2, 1, CROP, CROP, CROP), "lcxyz")


@pytest.mark.unit
def test_configuration_errors_are_named():
    with pytest.raises(ValueError, match="block must be one of"):
        ConvNet3D(block="depthwise", **SMALL)
    with pytest.raises(ValueError, match="same, nonzero number of stages"):
        ConvNet3D(img_size=CROP, widths=(8, 16), depths=(1,))
    with pytest.raises(ValueError, match="multiple of the output stride"):
        ConvNet3D(img_size=36, widths=(8, 16), depths=(1, 1))
    with pytest.raises(ValueError, match="every stage needs a block"):
        ConvNet3D(img_size=CROP, widths=(8, 16), depths=(1, 0))


@pytest.mark.unit
@pytest.mark.parametrize("block", ["basic", "fused"])
def test_the_subpixel_head_trains_through_it_with_checkpointing(block):
    """The affinity head folds these tokens as it folds a ViT's, and gradients reach the stem."""
    model = ConvNet3D(block=block, **SMALL)
    algorithm = AffinitySegmentation(
        model, input_axes="lcxyz", long_range=4, decoder="subpixel",
        decoder_hidden_dim=8, decoder_readout_dim=4, decoder_zero_init_output=False,
    )
    apply_activation_checkpointing(algorithm)
    torch.manual_seed(0)
    batch = {
        "img": torch.rand(1, 1, 1, CROP, CROP, CROP),
        "label": torch.randint(0, 3, (1, 1, CROP, CROP, CROP)),
    }
    metrics = algorithm.training_step(batch)
    metrics["loss"].backward()
    assert torch.isfinite(metrics["loss"])
    stem = model.patch_embed[0].proj.weight
    assert stem.grad is not None and stem.grad.abs().sum() > 0
