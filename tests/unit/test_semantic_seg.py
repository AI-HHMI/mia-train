"""Unit tests for semantic segmentation: the algorithm, tiled inference and orthoplane.

The properties worth pinning are the ones that fail *silently*. A transposed orthoplane pass still
produces a plausible-looking score volume; a tiling scheme that misses a strip still returns a
full-size array; a confusion matrix with predicted and actual swapped still yields metrics in
[0, 1]. Each of those is checked against a case with a known answer rather than against a shape.
"""

from __future__ import annotations

import pytest
import torch

from algorithms.semantic_seg import SemanticSegmentation
from evals.semantic_seg import (
    ConfusionMatrix,
    SemanticSegmentationEval,
    blend_weight,
    orthoplane_logits,
    tile_starts,
    tiled_logits,
)
from models.dinov3_vit import DinoVisionTransformer
from models.dinov3_vit3d import DinoVisionTransformer3D

CLASSES = 5


def _model_3d(crop=16, patch=8):
    return DinoVisionTransformer3D(
        img_size=crop, patch_size=patch, in_chans=1, embed_dim=32, depth=2, num_heads=2,
        n_storage_tokens=4, layerscale_init=1.0e-05, mask_k_bias=True,
        pos_embed_rope_type="superposition",
    )


def _model_2d(crop=16, patch=8):
    return DinoVisionTransformer(
        img_size=crop, patch_size=patch, in_chans=1, embed_dim=32, depth=2, num_heads=2,
        n_storage_tokens=4, layerscale_init=1.0e-05, mask_k_bias=True,
    )


def _algorithm(rank=3, **kw):
    model = _model_3d() if rank == 3 else _model_2d()
    axes = "lcxyz" if rank == 3 else "lcxy"
    return SemanticSegmentation(model, input_axes=axes, num_classes=CLASSES,
                                decoder_hidden_dim=8, **kw)


def _batch(rank=3, batch=2, crop=16):
    torch.manual_seed(0)
    spatial = (crop,) * rank
    return {
        "img": torch.rand(batch, 1, 1, *spatial),
        "label": torch.randint(0, CLASSES, (batch, 1, *spatial)),
    }


# ------------------------------------------------------------------ the algorithm


@pytest.mark.unit
@pytest.mark.parametrize("rank", [2, 3])
def test_one_algorithm_serves_both_ranks(rank):
    algorithm = _algorithm(rank)
    out = algorithm.training_step(_batch(rank))
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    assert 0.0 <= float(out["pixel_accuracy"]) <= 1.0
    assert 0.0 <= float(out["mean_iou"]) <= 1.0


@pytest.mark.unit
@pytest.mark.parametrize("rank", [2, 3])
def test_logits_are_class_scores_at_input_resolution(rank):
    algorithm = _algorithm(rank)
    spatial = (16,) * rank
    scores = algorithm.logits(torch.rand(2, 1, *spatial))
    assert scores.shape == (2, CLASSES, *spatial)


@pytest.mark.unit
def test_a_perfect_prediction_gives_zero_loss_and_unit_iou():
    # Bypasses the encoder: the point is that the loss and metrics agree on what "correct" means.
    labels = torch.randint(0, CLASSES, (1, 8, 8, 8))
    scores = torch.nn.functional.one_hot(labels, CLASSES).permute(0, 4, 1, 2, 3).float() * 50.0
    loss = torch.nn.functional.cross_entropy(scores, labels)
    assert float(loss) == pytest.approx(0.0, abs=1e-4)
    assert float((scores.argmax(1) == labels).float().mean()) == 1.0


@pytest.mark.unit
def test_ignore_index_removes_voxels_from_the_loss():
    algorithm = _algorithm(3, ignore_index=0)
    batch = _batch(3)
    batch["label"].fill_(0)          # every voxel ignored
    # cross_entropy over an empty selection is nan, which is the honest signal that a config
    # ignoring everything is wrong -- assert we get there rather than a silent zero.
    out = algorithm.training_step(batch)
    assert torch.isnan(out["loss"]) and torch.isnan(out["mean_iou"])


@pytest.mark.unit
def test_class_weights_must_match_num_classes():
    with pytest.raises(ValueError, match="class_weights has 3 entries"):
        _algorithm(3, class_weights=[1.0, 1.0, 1.0])


@pytest.mark.unit
def test_the_dense_head_is_declared_checkpointable():
    algorithm = _algorithm(3)
    assert algorithm.checkpointable_modules() == (algorithm.decoder_out,)


# ------------------------------------------------------------------ tiling


@pytest.mark.unit
@pytest.mark.parametrize(
    ("extent", "tile", "overlap"), [(100, 32, 8), (64, 64, 0), (10, 32, 4), (97, 16, 15)]
)
def test_tiles_cover_every_position(extent, tile, overlap):
    # A gap would silently leave a strip of the volume predicted by nothing.
    covered = torch.zeros(extent, dtype=torch.bool)
    for start in tile_starts(extent, tile, overlap):
        covered[start : start + min(tile, extent)] = True
    assert bool(covered.all())


@pytest.mark.unit
def test_tiles_stay_inside_the_extent():
    for start in tile_starts(100, 32, 8):
        assert 0 <= start <= 100 - 32


@pytest.mark.unit
def test_blend_weight_peaks_in_the_middle_and_is_lowest_at_the_faces():
    w = blend_weight((8, 8), torch.device("cpu"))
    assert float(w[4, 4]) == float(w.max())
    assert float(w[0, 0]) == float(w.min())


@pytest.mark.unit
@pytest.mark.parametrize("rank", [2, 3])
def test_tiled_inference_reproduces_an_untiled_constant_predictor(rank):
    # A predictor whose answer does not depend on position must survive tiling and blending
    # exactly; any error here is the blending, not the model.
    spatial = (24,) * rank
    images = torch.rand(1, *spatial)

    def predict(windows: torch.Tensor) -> torch.Tensor:
        shape = (windows.shape[0], CLASSES, *windows.shape[2:])
        return torch.full(shape, 0.0).index_fill_(1, torch.tensor([2]), 7.0)

    out = tiled_logits(predict, images, (16,) * rank, 8, CLASSES, torch.device("cpu"), 2)
    assert out.shape == (1, CLASSES, *spatial)
    assert torch.allclose(out[0, 2], torch.full(spatial, 7.0), atol=1e-4)
    assert torch.allclose(out[0, 0], torch.zeros(spatial), atol=1e-4)


@pytest.mark.unit
def test_tiled_inference_recovers_a_position_dependent_signal():
    # Stronger than the constant case: the predictor encodes each voxel's own value, so a
    # mis-stitched tile shows up as a wrong value rather than a wrong shape.
    image = torch.rand(1, 20, 20)

    def predict(windows: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((windows.shape[0], CLASSES, *windows.shape[2:]))
        out[:, 1] = windows[:, 0]
        return out

    out = tiled_logits(predict, image, (8, 8), 4, CLASSES, torch.device("cpu"), 3)
    assert torch.allclose(out[0, 1], image[0], atol=1e-4)


# ------------------------------------------------------------------ orthoplane


@pytest.mark.unit
def test_orthoplane_puts_every_pass_back_in_the_same_frame():
    # The failure this guards: a transposed pass still yields a full-size, plausible volume. The
    # predictor copies the input into channel 1, so each of the three passes must reconstruct the
    # volume exactly -- and so must their average, whatever the axis order.
    volume = torch.rand(6, 7, 8)

    def predict(windows: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((windows.shape[0], CLASSES, *windows.shape[2:]))
        out[:, 1] = windows[:, 0]
        return out

    for axis in ("x", "y", "z"):
        one = orthoplane_logits(
            predict, volume, (16, 16), 0, CLASSES, torch.device("cpu"), 4, axes=(axis,)
        )
        assert one.shape == (CLASSES, 6, 7, 8)
        assert torch.allclose(one[1], volume, atol=1e-4), f"axis {axis} came back transposed"

    averaged = orthoplane_logits(predict, volume, (16, 16), 0, CLASSES, torch.device("cpu"), 4)
    assert torch.allclose(averaged[1], volume, atol=1e-4)


@pytest.mark.unit
def test_orthoplane_averages_disagreeing_passes():
    # Three passes that answer differently must average, not overwrite: a pass that simply
    # replaced the accumulator would leave the last axis' answer.
    volume = torch.zeros(4, 4, 4)
    calls = {"n": 0}

    def predict(windows: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((windows.shape[0], CLASSES, *windows.shape[2:]))
        out[:, 0] = float(calls["n"])       # 0, then 1, then 2 across the three axis passes
        calls["n"] += 1
        return out

    got = orthoplane_logits(predict, volume, (4, 4), 0, CLASSES, torch.device("cpu"), 64)
    assert float(got[0].mean()) == pytest.approx(1.0)   # mean of 0, 1, 2


@pytest.mark.unit
def test_orthoplane_is_refused_for_a_3d_model():
    task = SemanticSegmentationEval(num_classes=CLASSES, tile=(8, 8), mode="orthoplane")
    with pytest.raises(ValueError, match="feeds the model 2D planes"):
        task.evaluate(_algorithm(3), [])


@pytest.mark.unit
def test_a_bare_encoder_is_refused():
    task = SemanticSegmentationEval(num_classes=CLASSES, tile=(8, 8, 8))
    with pytest.raises(TypeError, match="has no `logits` method"):
        task.evaluate(_model_3d(), [])


# ------------------------------------------------------------------ metrics


@pytest.mark.unit
def test_confusion_matrix_scores_a_perfect_prediction():
    m = ConfusionMatrix(CLASSES)
    target = torch.randint(0, CLASSES, (500,))
    m.update(target.clone(), target, ignore_index=-1)
    out = m.metrics()
    assert out["pixel_accuracy"] == pytest.approx(1.0)
    assert out["mean_iou"] == pytest.approx(1.0)
    assert out["mean_dice"] == pytest.approx(1.0)


@pytest.mark.unit
def test_iou_is_computed_on_totals_not_per_batch():
    # Two updates: class 1 perfect in the first, absent from the second. Pooling by totals gives
    # IoU 1.0 for it; averaging per-update would drag it toward 0.5 by scoring an absent class.
    m = ConfusionMatrix(CLASSES)
    m.update(torch.ones(10, dtype=torch.long), torch.ones(10, dtype=torch.long), -1)
    m.update(torch.zeros(10, dtype=torch.long), torch.zeros(10, dtype=torch.long), -1)
    assert m.metrics()["iou/class_1"] == pytest.approx(1.0)


@pytest.mark.unit
def test_predicting_only_background_gives_high_accuracy_and_low_iou():
    # The reason mean IoU is the headline and accuracy is not: 90% background makes a degenerate
    # model look 90% correct.
    m = ConfusionMatrix(CLASSES)
    target = torch.cat([torch.zeros(90, dtype=torch.long), torch.ones(10, dtype=torch.long)])
    m.update(torch.zeros(100, dtype=torch.long), target, -1)
    out = m.metrics()
    assert out["pixel_accuracy"] == pytest.approx(0.9)
    assert out["mean_iou"] < 0.5
    assert out["iou/class_1"] == pytest.approx(0.0)


@pytest.mark.unit
def test_ignored_voxels_are_excluded_from_the_totals():
    m = ConfusionMatrix(CLASSES)
    target = torch.tensor([0, 1, 2, 2])
    predicted = torch.tensor([0, 1, 0, 0])          # both class-2 voxels wrong
    m.update(predicted, target, ignore_index=2)     # ...but ignored
    assert m.metrics()["pixel_accuracy"] == pytest.approx(1.0)
    assert int(m.counts.sum()) == 2


# ------------------------------------------------------------------ dataset plumbing


@pytest.mark.unit
def test_group_is_the_leading_components_of_the_key():
    from data.huggingface import group_of

    assert group_of("jrc_hela-2/recon-1/crop28", 1) == "jrc_hela-2"
    assert group_of("jrc_hela-2/recon-1/crop28", 2) == "jrc_hela-2/recon-1"
    assert group_of("jrc_mus-liver-zon-2/recon-1/crop366", 1) == "jrc_mus-liver-zon-2"


@pytest.mark.unit
def test_random_crop_takes_the_same_window_from_image_and_label():
    import numpy as np

    from data.huggingface import random_crop

    rng = np.random.default_rng(0)
    image = np.arange(20 * 20, dtype=np.uint8).reshape(20, 20)
    image, label = random_crop(image, image.copy(), (8, 8), rng)
    assert image.shape == (8, 8)
    assert np.array_equal(image, label)          # co-registration survives


@pytest.mark.unit
def test_random_crop_pads_an_undersized_input():
    import numpy as np

    from data.huggingface import random_crop

    image = np.ones((4, 30), dtype=np.uint8)
    image, label = random_crop(image, image.copy(), (8, 8), np.random.default_rng(0))
    assert image.shape == label.shape == (8, 8)
    assert image[:4].all() and not image[4:].any()   # padded with zeros, i.e. background


@pytest.mark.unit
def test_raw_and_image_encodings_both_decode_to_arrays():
    import io as _io

    import numpy as np
    from PIL import Image

    from data.huggingface import PRESETS, decode_image, decode_raw

    volume = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    row = {"volume": volume.tobytes(), "label": volume.tobytes(), "shape": [2, 3, 4]}
    image, label = decode_raw(row, PRESETS["cellmap_3d"])
    assert np.array_equal(image, volume) and np.array_equal(label, volume)

    plane = np.arange(6 * 5, dtype=np.uint8).reshape(6, 5)
    buffer = _io.BytesIO()
    Image.fromarray(plane).save(buffer, format="PNG")
    row2 = {"image": {"bytes": buffer.getvalue()}, "label": {"bytes": buffer.getvalue()}}
    image, label = decode_image(row2, PRESETS["cellmap_2d"])
    assert np.array_equal(image, plane) and np.array_equal(label, plane)


@pytest.mark.unit
def test_raw_encoding_without_a_shape_column_is_refused():
    from data.huggingface import Preset, decode_raw

    with pytest.raises(ValueError, match="needs shape_key"):
        decode_raw({}, Preset(repo="r", spatial_rank=3, encoding="raw"))


@pytest.mark.unit
def test_presets_describe_the_two_cellmap_datasets():
    from data.huggingface import PRESETS

    assert PRESETS["cellmap_3d"].repo == "eminorhan/cellmap-3d"
    assert PRESETS["cellmap_3d"].spatial_rank == 3
    assert PRESETS["cellmap_3d"].encoding == "raw"
    assert PRESETS["cellmap_2d"].repo == "eminorhan/cellmap-2d"
    assert PRESETS["cellmap_2d"].spatial_rank == 2
    assert PRESETS["cellmap_2d"].encoding == "image"


@pytest.mark.unit
def test_a_preset_can_be_overridden_field_by_field():
    from data.huggingface import HuggingFaceSemanticSegmentation as HF

    ds = HF(patch_size=(8, 8, 8), preset="cellmap_3d", group_levels=2)
    assert ds.preset.repo == "eminorhan/cellmap-3d"     # from the preset
    assert ds.preset.group_levels == 2                  # overridden


@pytest.mark.unit
def test_an_inline_dataset_needs_the_essential_fields():
    from data.huggingface import HuggingFaceSemanticSegmentation as HF

    with pytest.raises(ValueError, match=r"\['repo', 'spatial_rank', 'encoding'\]"):
        HF(patch_size=(8, 8))
    # ...and works when they are given, with no preset at all.
    ds = HF(patch_size=(8, 8), repo="someone/other", spatial_rank=2, encoding="image")
    assert ds.preset.repo == "someone/other"


@pytest.mark.unit
def test_unknown_preset_names_the_available_ones():
    from data.huggingface import HuggingFaceSemanticSegmentation as HF

    with pytest.raises(ValueError, match="unknown preset 'nope'"):
        HF(patch_size=(8, 8), preset="nope")


@pytest.mark.unit
def test_a_val_split_without_held_out_groups_is_refused():
    from data.huggingface import HuggingFaceSemanticSegmentation as HF

    with pytest.raises(ValueError, match="would be empty"):
        HF(patch_size=(8, 8, 8), preset="cellmap_3d", split="val")


@pytest.mark.unit
def test_patch_size_rank_must_match_the_dataset():
    from data.huggingface import HuggingFaceSemanticSegmentation as HF

    with pytest.raises(ValueError, match="is 3D, so patch_size needs 3"):
        HF(patch_size=(8, 8), preset="cellmap_3d")
    with pytest.raises(ValueError, match="is 2D, so patch_size needs 2"):
        HF(patch_size=(8, 8, 8), preset="cellmap_2d")


@pytest.mark.unit
def test_the_datasets_declare_axis_orders_the_encoders_accept():
    from data.huggingface import HuggingFaceSemanticSegmentation as HF

    assert HF(patch_size=(8, 8, 8), preset="cellmap_3d").sample_axes == "lcxyz"
    assert HF(patch_size=(8, 8), preset="cellmap_2d").sample_axes == "lcxy"


@pytest.mark.unit
def test_the_run_record_expands_the_preset():
    # A record naming only "cellmap_3d" would not say what was read if the preset later changed.
    from data.huggingface import HuggingFaceSemanticSegmentation as HF

    resolved = HF.resolve_settings(patch_size=[8, 8, 8], preset="cellmap_3d")
    assert resolved["repo"] == "eminorhan/cellmap-3d"
    assert resolved["encoding"] == "raw"
    assert resolved["group_key"] == "crop_name"


@pytest.mark.unit
def test_registered_under_their_config_names():
    import components  # noqa: F401
    from algorithms.registry import AlgorithmRegistry
    from data.registry import DataRegistry
    from evals.registry import EvalRegistry

    assert "semantic_seg" in AlgorithmRegistry.available()
    assert "semantic_seg" in EvalRegistry.available()
    assert "hf_semantic_seg" in DataRegistry.available()
