"""Labels past the window: both targets are built on the labels around it, then cropped to it.

miao's `label_margin` returns the label larger than the image by a margin on each side. What has to
hold: the window's targets are the whole label box's targets at the same voxels -- affinities whose
partner lies past a face are supervised, descriptors see their whole window -- whether the head is
decoded whole or in slabs; a margin too narrow for either target, or off the descriptors' strided
grid, is refused. Without a margin nothing changes, which every other test here exercises.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from algorithms.affinity.lsd import lsd_from_labels
from algorithms.affinity.targets import affinities_from_labels
from algorithms.affinity_seg import AffinitySegmentation
from models.vit import ViT3D

CROP = 32
PATCH = 8
MARGIN = 8
PIXEL_SIZE = (8.0, 8.0, 8.0)
SIGMA = 16.0  # nm: 2 voxels; at downsample 2 a radius of 3 strided cells, so a halo of 6 voxels
WINDOW = (slice(None), slice(None), *(slice(MARGIN, MARGIN + CROP),) * 3)


def _algorithm(seed: int = 0, **overrides: Any) -> AffinitySegmentation:
    torch.manual_seed(seed)
    encoder = ViT3D(
        img_size=(CROP, CROP, CROP), patch_size=(PATCH, PATCH, PATCH), in_channels=1,
        embed_dim=32, depth=1, num_heads=4,
    )
    kwargs: dict[str, Any] = dict(
        input_axes="lcxyz", decoder="subpixel", long_range=4, decoder_hidden_dim=8,
        decoder_readout_dim=4, decoder_refine_depth=2, decoder_zero_init_output=False,
        split_disconnected=False, lsd_sigma=SIGMA, lsd_downsample=2,
    )
    kwargs.update(overrides)
    return AffinitySegmentation(encoder, **kwargs)


def _labels(margin: int = MARGIN, low: int = -1, batch_size: int = 2) -> torch.Tensor:
    """Blocky ids (B, X, Y, Z) over the window plus `margin` per side; -1 marks ignored voxels."""
    torch.manual_seed(1)
    extent = CROP + 2 * margin
    coarse = torch.randint(low, 6, (batch_size, *(-(-extent // 4),) * 3))
    for axis in (1, 2, 3):
        coarse = coarse.repeat_interleave(4, dim=axis)
    return coarse[:, :extent, :extent, :extent]


def _batch(labels: torch.Tensor) -> dict[str, torch.Tensor]:
    torch.manual_seed(2)
    return {
        "img": torch.rand(labels.shape[0], 1, 1, CROP, CROP, CROP),
        "label": labels[:, None],
        "pixel_size": torch.tensor(PIXEL_SIZE).expand(labels.shape[0], 1, 3).clone(),
    }


@pytest.mark.unit
def test_the_targets_are_the_whole_label_boxs_in_the_window():
    algorithm = _algorithm()
    labels = _labels()
    sigma = algorithm._sigma_voxels(_batch(labels))
    halo = algorithm._lsd_halo(sigma)
    assert halo == (6, 6, 6)
    target, mask, lsd_target, lsd_mask = algorithm._window_targets(
        labels, (MARGIN,) * 3, (CROP,) * 3, (0, CROP), sigma, halo
    )
    whole_target, whole_mask = affinities_from_labels(labels, algorithm.offsets, -1)
    whole_lsd, whole_lsd_mask = lsd_from_labels(labels, sigma, 2, -1, "ignore")
    assert torch.equal(target, whole_target.float()[WINDOW])
    assert torch.equal(mask, whole_mask[WINDOW])
    assert torch.equal(lsd_mask, whole_lsd_mask[WINDOW])
    assert torch.allclose(lsd_target, whole_lsd[WINDOW], atol=1e-6)


@pytest.mark.unit
def test_a_margin_supervises_the_window_faces():
    """No ignored voxels, so every affinity in the window has a partner in the margin."""
    algorithm = _algorithm(lsd_sigma=None)
    labels = _labels(low=0)
    with_margin = algorithm.training_step(_batch(labels))
    without = algorithm.training_step(_batch(labels[(slice(None), *WINDOW[2:])]))
    assert float(with_margin["masked_fraction"]) == 1.0
    assert float(without["masked_fraction"]) < 1.0


@pytest.mark.unit
@pytest.mark.parametrize("chunks", [2, 4])
def test_chunked_metrics_match_the_undivided_decode(chunks):
    batch = _batch(_labels())
    whole = _algorithm().training_step(batch)
    split = _algorithm(decode_chunks=chunks).training_step(batch)
    assert set(whole) == set(split)
    for name, expected in whole.items():
        assert float(split[name]) == pytest.approx(float(expected), rel=1e-5, abs=1e-6), name


@pytest.mark.unit
@pytest.mark.parametrize(
    ("margin", "overrides", "match"),
    [
        (4, {}, "narrower than"),                   # the descriptors read 6 past the window
        (2, {"lsd_sigma": None}, "narrower than"),  # the affinities 4
        (7, {}, "multiple of lsd_downsample"),
    ],
)
def test_a_margin_that_cannot_hold_the_targets_is_refused(margin, overrides, match):
    labels = torch.zeros((1, *(CROP + 2 * margin,) * 3), dtype=torch.int64)
    with pytest.raises(ValueError, match=match):
        _algorithm(**overrides).training_step(_batch(labels))


@pytest.mark.unit
def test_an_affinity_only_margin_needs_only_long_range():
    labels = torch.zeros((1, *(CROP + 8,) * 3), dtype=torch.int64)
    _algorithm(lsd_sigma=None).training_step(_batch(labels))


@pytest.mark.unit
def test_labels_must_cover_the_image_evenly():
    labels = torch.zeros((1, CROP + 3, CROP + 4, CROP + 4), dtype=torch.int64)
    with pytest.raises(ValueError, match="does not match the image crop"):
        _algorithm(lsd_sigma=None).training_step(_batch(labels))
