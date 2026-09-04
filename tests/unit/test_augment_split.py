"""Splitting `[augment]` across the worker and the device must not change what it means.

The move exists for speed -- intensity and noise cost a dataloader worker 165 ms per 256^3 volume
against 0.19 ms batched on the device -- so every test here is about the thing speed could quietly
cost: that the two halves partition the operations rather than duplicating or dropping them, and
that a batched implementation still draws per sample.
"""

from __future__ import annotations

import pytest
import torch

from data.augment import (
    APPLY_PROB,
    BatchPhotometric,
    VolumeAugmentation,
    split_augmentation,
)

SETTINGS = dict(
    rotate="inplane",
    drop_slice_prob=0.05,
    shift_slice_prob=0.05,
    shift_magnitude=10,
    intensity=True,
    mul_intensity=0.1,
    add_intensity=0.1,
    noise_scale=0.5,
)


def _batch(count: int = 8, size: int = 8) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(count, 1, 1, size, size, size)


# ------------------------------------------------------------------ the partition


@pytest.mark.unit
def test_split_leaves_photometric_only_to_the_device_half():
    """The whole point: applying intensity or noise in both halves would augment twice.

    Twice is not an error anyone would see -- no exception, no shape change, just a run trained on
    distributions the config never asked for. So the worker's half is asserted inert, not assumed.
    """
    geometric, photometric = split_augmentation(sample_axes="lcxyz", **SETTINGS)

    assert geometric.intensity is False
    assert geometric.noise_scale == 0.0
    assert photometric.intensity is True
    assert photometric.noise_scale == 0.5


@pytest.mark.unit
def test_split_keeps_the_geometric_settings_in_the_worker():
    geometric, _ = split_augmentation(sample_axes="lcxyz", **SETTINGS)

    assert geometric.rotate == "inplane"
    assert geometric.drop_slice_prob == 0.05
    assert geometric.shift_slice_prob == 0.05
    assert geometric.shift_magnitude == 10


@pytest.mark.unit
def test_worker_half_does_not_change_intensity_statistics():
    """A volume through the geometric half alone keeps its intensity distribution.

    `rotate` and the section operations permute and blank voxels; none of them rescales or adds
    noise, so the mean over many samples must not move the way `intensity_jitter` moves it.
    """
    geometric, _ = split_augmentation(
        sample_axes="lcxyz", rotate="none", drop_slice_prob=0.0, shift_slice_prob=0.0,
        shift_magnitude=0, intensity=True, mul_intensity=0.5, add_intensity=0.5, noise_scale=1.0,
    )
    image = torch.rand(1, 1, 8, 8, 8)
    out = geometric({"img": image})["img"]

    torch.testing.assert_close(out, image), "the worker half must leave photometric alone"


# ------------------------------------------------------------------ per-sample draws


@pytest.mark.unit
def test_draws_are_per_sample_not_per_batch():
    """One draw broadcast over a batch would be a weaker augmentation wearing the same config.

    Every sample here is identical, so any spread in the output can only come from the draws.
    """
    torch.manual_seed(0)
    one = torch.rand(1, 1, 1, 8, 8, 8)
    batch = one.repeat(16, 1, 1, 1, 1, 1)

    out = BatchPhotometric(intensity=True, mul_intensity=0.5, add_intensity=0.5, noise_scale=0.0)(
        {"img": batch}
    )["img"]
    shifts = (out - batch).mean(dim=(1, 2, 3, 4, 5))

    assert shifts.max() - shifts.min() > 1e-3, "identical samples came out identically transformed"


@pytest.mark.unit
def test_noise_deviation_is_drawn_per_sample():
    torch.manual_seed(0)
    one = torch.rand(1, 1, 1, 8, 8, 8)
    batch = one.repeat(32, 1, 1, 1, 1, 1)

    out = BatchPhotometric(intensity=False, noise_scale=1.0)({"img": batch})["img"]
    spread = (out - batch).std(dim=(1, 2, 3, 4, 5))

    assert spread.max() - spread.min() > 1e-3


# ------------------------------------------------------------------ the coin


@pytest.mark.unit
def test_each_operation_still_reaches_about_half_the_samples():
    """`APPLY_PROB` gates each operation before its own parameters apply.

    A batched implementation has no single answer for a batch, so the gate has to become a
    per-sample mask. Dropping it would apply noise to everything -- twice the augmentation the
    reference specifies, and a config that no longer means what it did.
    """
    torch.manual_seed(0)
    batch = torch.rand(512, 1, 1, 4, 4, 4)
    out = BatchPhotometric(intensity=False, noise_scale=1.0)({"img": batch})["img"]

    touched = ((out - batch).abs().amax(dim=(1, 2, 3, 4, 5)) > 0).float().mean().item()
    assert abs(touched - APPLY_PROB) < 0.1, (
        f"noise reached {touched:.0%} of samples, expected ~{APPLY_PROB:.0%}"
    )


# ------------------------------------------------------------------ scope


@pytest.mark.unit
def test_labels_are_never_touched():
    torch.manual_seed(0)
    labels = torch.randint(0, 5, (4, 1, 8, 8, 8))
    out = BatchPhotometric(intensity=True, noise_scale=1.0)(
        {"img": torch.rand(4, 1, 1, 8, 8, 8), "label": labels}
    )

    torch.testing.assert_close(out["label"], labels)


@pytest.mark.unit
def test_disabled_photometric_is_an_identity():
    batch = {"img": _batch()}
    photometric = BatchPhotometric(intensity=False, noise_scale=0.0)

    assert photometric.enabled() is False
    torch.testing.assert_close(photometric(batch)["img"], batch["img"])


@pytest.mark.unit
def test_a_sample_without_the_image_key_is_left_alone():
    """Rather than raising: the device half sees whatever the algorithm's batch carries."""
    other = torch.rand(2, 3)
    out = BatchPhotometric(intensity=True, noise_scale=1.0)({"something_else": other})

    torch.testing.assert_close(out["something_else"], other)


@pytest.mark.unit
def test_rejects_a_negative_noise_scale():
    with pytest.raises(ValueError, match="noise_scale"):
        BatchPhotometric(noise_scale=-1.0)


# ------------------------------------------------------------------ equivalence


@pytest.mark.unit
def test_device_half_matches_the_per_sample_implementation_in_distribution():
    """The batched form is a second implementation, so its output distribution is pinned.

    Not equality: both draw randomly and neither promises particular numbers. What must agree is
    what the draws are *distributed* like -- the mean shift `intensity_jitter` produces and the
    deviation `additive_noise` adds, whose expectation is `scale/2` because the deviation is
    itself uniform on `[0, scale]`.
    """
    per_sample = VolumeAugmentation(
        sample_axes="lcxyz", rotate="none", drop_slice_prob=0.0, shift_slice_prob=0.0,
        intensity=True, mul_intensity=0.1, add_intensity=0.1, noise_scale=0.5,
    )
    torch.manual_seed(0)
    image = torch.rand(1, 1, 16, 16, 16)
    reference = torch.stack(
        [(per_sample({"img": image})["img"] - image).std() for _ in range(600)]
    )

    torch.manual_seed(0)
    batched = BatchPhotometric(
        intensity=True, mul_intensity=0.1, add_intensity=0.1, noise_scale=0.5
    )
    stacked = image.repeat(600, 1, 1, 1, 1, 1)
    candidate = (batched({"img": stacked})["img"] - stacked).std(dim=(1, 2, 3, 4, 5))

    assert abs(reference.mean() - candidate.mean()) < 0.05, (
        f"per-sample {reference.mean():.4f} vs batched {candidate.mean():.4f}"
    )
