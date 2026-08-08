"""Unit tests for initialising a model from weights trained elsewhere."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.dinov3_vit import DinoVisionTransformer
from models.dinov3_vit3d import DinoVisionTransformer3D
from utils.pretrained import inflate_2d_to_3d, load_pretrained, read_state_dict

DINO = dict(img_size=32, patch_size=8, embed_dim=96, depth=1, num_heads=4,
            pos_embed_rope_dtype="fp32")


def _released_checkpoint(tmp_path, **overrides):
    """A file structurally identical to a published DINOv3 release.

    The port kept upstream's attribute names, so a 2D model's own state dict has exactly the key
    names Meta ships -- which is what makes this a faithful stand-in without needing the (licence
    gated) weights themselves.
    """
    kwargs = {**DINO, "in_chans": 3, **overrides}
    model = DinoVisionTransformer(**kwargs)
    path = tmp_path / "dinov3_pretrain.pth"
    torch.save(model.state_dict(), path)
    return path, model


# ---------------------------------------------------------------- inflation


@pytest.mark.unit
def test_inflation_spreads_the_kernel_over_depth_and_normalises():
    weight = torch.randn(4, 1, 8, 8)
    out = inflate_2d_to_3d(weight, torch.Size((4, 1, 8, 8, 8)))

    assert out.shape == (4, 1, 8, 8, 8)
    # Every z-slice carries 1/depth of the original, so summing over z returns it exactly.
    assert torch.allclose(out.sum(dim=2), weight, atol=1e-6)
    assert torch.allclose(out[:, :, 0], weight / 8, atol=1e-6)


@pytest.mark.unit
def test_inflation_averages_rgb_down_to_one_channel():
    weight = torch.randn(4, 3, 8, 8)
    out = inflate_2d_to_3d(weight, torch.Size((4, 1, 8, 8, 8)))
    assert torch.allclose(out.sum(dim=2), weight.mean(dim=1, keepdim=True), atol=1e-6)


@pytest.mark.unit
def test_inflation_rejects_a_different_in_plane_kernel():
    with pytest.raises(ValueError, match="in-plane kernel must already match"):
        inflate_2d_to_3d(torch.randn(4, 1, 8, 8), torch.Size((4, 1, 16, 16, 16)))


@pytest.mark.unit
def test_inflation_rejects_an_undefined_channel_mapping():
    with pytest.raises(ValueError, match="input channels"):
        inflate_2d_to_3d(torch.randn(4, 3, 8, 8), torch.Size((4, 2, 8, 8, 8)))


@pytest.mark.unit
def test_inflated_patch_embedding_reproduces_the_2d_response(tmp_path):
    """The property the 1/depth normalisation exists for.

    A volume that is constant along z is the same picture on every slice, so its patch embedding
    should equal the 2D model's on that picture. A plain repeat would scale it by the patch depth
    and push the rest of the pretrained stack out of range.
    """
    torch.manual_seed(0)
    path, model_2d = _released_checkpoint(tmp_path, in_chans=1)
    model_3d = DinoVisionTransformer3D(in_chans=1, **DINO)
    load_pretrained(model_3d, path, inflate=True, skip=["rope_embed."])

    image = torch.rand(1, 1, 32, 32)
    volume = image.unsqueeze(2).expand(-1, -1, 32, -1, -1).contiguous()
    with torch.no_grad():
        flat = model_2d.eval().patch_embed(image)
        solid = model_3d.eval().patch_embed(volume)

    for z in range(solid.shape[1]):
        assert torch.allclose(solid[0, z], flat[0], atol=1e-5)


# ---------------------------------------------------------------- reading


@pytest.mark.unit
def test_reads_a_plain_state_dict_file(tmp_path):
    path, model = _released_checkpoint(tmp_path)
    assert set(read_state_dict(path)) == set(model.state_dict())


@pytest.mark.unit
@pytest.mark.parametrize("wrapper", ["model", "state_dict", "teacher"])
def test_digs_the_state_dict_out_of_a_wrapper(tmp_path, wrapper):
    path = tmp_path / "wrapped.pth"
    torch.save({wrapper: {"a": torch.zeros(2)}, "epoch": 7}, path)
    assert set(read_state_dict(path)) == {"a"}


@pytest.mark.unit
def test_rejects_a_file_with_no_tensors(tmp_path):
    path = tmp_path / "empty.pth"
    torch.save({"epoch": 7, "notes": "hello"}, path)
    with pytest.raises(ValueError, match="no tensors"):
        read_state_dict(path)


@pytest.mark.unit
def test_missing_checkpoint_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="no checkpoint at"):
        read_state_dict(tmp_path / "absent.pth")


# ---------------------------------------------------------------- loading


@pytest.mark.unit
def test_a_2d_release_transfers_into_the_3d_model(tmp_path):
    path, _ = _released_checkpoint(tmp_path, n_storage_tokens=4)
    model = DinoVisionTransformer3D(in_chans=1, n_storage_tokens=4, **DINO)

    report = load_pretrained(model, path, inflate=True, skip=["rope_embed."])

    assert report.inflated == ["patch_embed.proj.weight"]
    assert not report.mismatched and not report.unused
    # Every transformer weight has to have moved -- that is the point of pretraining.
    blocks = {name for name, _ in model.named_parameters() if name.startswith("blocks.")}
    assert blocks <= set(report.copied)
    assert "cls_token" in report.copied and "storage_tokens" in report.copied


@pytest.mark.unit
@pytest.mark.parametrize("rope", ["vanilla", "superposition"])
def test_both_rope_types_accept_a_2d_checkpoint(rope, tmp_path):
    """`rope_embed` buffers are derived from `base`, not learned, so neither type needs them."""
    path, _ = _released_checkpoint(tmp_path)
    model = DinoVisionTransformer3D(in_chans=1, pos_embed_rope_type=rope, **DINO)
    report = load_pretrained(model, path, inflate=True, skip=["rope_embed."])
    assert not report.mismatched
    assert torch.isfinite(model.eval()(torch.randn(1, 1, 32, 32, 32))).all()


@pytest.mark.unit
def test_superposition_depth_gate_stays_closed_after_a_2d_load(tmp_path):
    """A 2D checkpoint carries no depth information, so the gate must start at zero -- which makes
    the loaded model behave exactly as its 2D self until it learns otherwise."""
    path, _ = _released_checkpoint(tmp_path)
    model = DinoVisionTransformer3D(in_chans=1, pos_embed_rope_type="superposition", **DINO)
    load_pretrained(model, path, inflate=True, skip=["rope_embed."])
    assert model.rope_embed.depth_scale.detach().item() == 0.0


@pytest.mark.unit
def test_rope_mismatch_without_skip_says_what_to_add(tmp_path):
    path, _ = _released_checkpoint(tmp_path)
    model = DinoVisionTransformer3D(in_chans=1, **DINO)
    with pytest.raises(ValueError, match=r'skip = \["rope_embed."\]'):
        load_pretrained(model, path, inflate=True)


@pytest.mark.unit
def test_patch_embed_mismatch_without_inflate_says_what_to_set(tmp_path):
    path, _ = _released_checkpoint(tmp_path)
    model = DinoVisionTransformer3D(in_chans=1, **DINO)
    with pytest.raises(ValueError, match="inflate_2d_to_3d = true"):
        load_pretrained(model, path, skip=["rope_embed."])


@pytest.mark.unit
def test_a_checkpoint_matching_nothing_is_an_error_not_a_silent_scratch_run(tmp_path):
    """The failure this guards against: a wrong prefix leaves the model at random init, and the
    resulting loss curve looks entirely normal."""
    path = tmp_path / "unrelated.pth"
    torch.save({"totally.different.key": torch.zeros(3)}, path)
    model = DinoVisionTransformer(in_chans=3, **DINO)
    with pytest.raises(ValueError, match="would have trained from scratch"):
        load_pretrained(model, path)


@pytest.mark.unit
def test_prefix_lifts_an_encoder_out_of_a_whole_algorithm_checkpoint(tmp_path):
    """A pretraining run checkpoints the algorithm, so the encoder sits under `model.` beside a
    decoder that a downstream task has no use for."""
    encoder = DinoVisionTransformer3D(in_chans=1, **DINO)
    algorithm = nn.Module()
    algorithm.model = encoder
    algorithm.decoder_head = nn.Linear(96, 512)  # the SSL objective's own, discarded here
    path = tmp_path / "ssl_run.pth"
    torch.save(algorithm.state_dict(), path)

    fresh = DinoVisionTransformer3D(in_chans=1, **DINO)
    report = load_pretrained(fresh, path, prefix="model.")

    assert not report.mismatched and not report.kept_initial
    assert all(not k.startswith("decoder") for k in report.copied)
    for name, tensor in fresh.state_dict().items():
        assert torch.equal(tensor, encoder.state_dict()[name]), name


@pytest.mark.unit
def test_non_strict_reports_mismatches_instead_of_raising(tmp_path):
    path, _ = _released_checkpoint(tmp_path)
    model = DinoVisionTransformer3D(in_chans=1, **DINO)
    report = load_pretrained(model, path, strict=False)

    assert report.mismatched, "the mismatches are still reported"
    assert report.copied, "and what did fit was still loaded"


@pytest.mark.unit
def test_reads_back_a_distributed_checkpoint_directory(tmp_path):
    """A run saved by this repo is a DCP directory holding the whole algorithm.

    Laid out as `CheckpointManager` writes it: the algorithm's state under "model", the optimizer
    beside it, and the step counter as a loose tensor -- which must not be mistaken for the
    weights.
    """
    import torch.distributed.checkpoint as dcp

    encoder = DinoVisionTransformer3D(in_chans=1, **DINO)
    algorithm = nn.Module()
    algorithm.model = encoder
    algorithm.decoder_head = nn.Linear(96, 512)

    dcp.save(
        {
            "model": algorithm.state_dict(),
            "optim": {"state": {}},
            "train_state": {"step": torch.tensor(10)},
        },
        checkpoint_id=str(tmp_path / "step_10"),
    )

    fresh = DinoVisionTransformer3D(in_chans=1, **DINO)
    report = load_pretrained(fresh, tmp_path / "step_10", prefix="model.")

    assert not report.mismatched and not report.kept_initial
    assert all(not k.startswith("decoder") for k in report.copied)
    for name, tensor in fresh.state_dict().items():
        assert torch.equal(tensor, encoder.state_dict()[name]), name


@pytest.mark.unit
def test_leftover_checkpoint_tensors_are_an_error_by_default(tmp_path):
    """The silent failure this exists for.

    A released DINOv3 ViT has LayerScale and a masked key bias. A model configured without them
    loads every tensor it recognises, reports nothing wrong, and computes a different function
    from the one that was trained.
    """
    path, _ = _released_checkpoint(tmp_path, layerscale_init=1e-5, mask_k_bias=True)
    plain = DinoVisionTransformer(in_chans=3, **DINO)  # no layerscale, no masked bias

    with pytest.raises(ValueError, match="no home in this model"):
        load_pretrained(plain, path)


@pytest.mark.unit
def test_the_leftovers_name_the_setting_that_would_consume_them(tmp_path):
    path, _ = _released_checkpoint(tmp_path, layerscale_init=1e-5, mask_k_bias=True)
    plain = DinoVisionTransformer(in_chans=3, **DINO)

    with pytest.raises(ValueError, match="layerscale_init"):
        load_pretrained(plain, path)


@pytest.mark.unit
def test_a_correctly_configured_model_consumes_the_whole_checkpoint(tmp_path):
    path, _ = _released_checkpoint(tmp_path, layerscale_init=1e-5, mask_k_bias=True)
    matching = DinoVisionTransformer(
        in_chans=3, layerscale_init=1e-5, mask_k_bias=True, **DINO
    )
    report = load_pretrained(matching, path)
    assert not report.unused and not report.kept_initial and not report.mismatched


@pytest.mark.unit
def test_allow_unused_permits_a_checkpoint_that_holds_more(tmp_path):
    """Legitimate when the checkpoint really does carry more than this model wants."""
    path, _ = _released_checkpoint(tmp_path, layerscale_init=1e-5, mask_k_bias=True)
    plain = DinoVisionTransformer(in_chans=3, **DINO)

    report = load_pretrained(plain, path, allow_unused=True)
    assert report.unused and report.copied
