"""Unit tests for volumetric EM augmentations."""

from __future__ import annotations

import pytest
import torch

from data.augment import AugmentedDataset, VolumeAugmentation
from engine.config import AugmentConfig

SHAPE = (1, 1, 8, 10, 12)   # l, c, x, y, z -- deliberately non-cubic
LABEL_SHAPE = (1, 8, 10, 12)


def _sample() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "img": torch.rand(SHAPE),
        "label": torch.arange(float(torch.tensor(LABEL_SHAPE).prod())).reshape(LABEL_SHAPE),
    }


@pytest.mark.unit
def test_disabled_by_default_leaves_the_sample_alone() -> None:
    original = _sample()
    out = VolumeAugmentation()(dict(original))
    assert torch.equal(out["img"], original["img"])
    assert torch.equal(out["label"], original["label"])


@pytest.mark.unit
def test_does_not_mutate_the_input_sample() -> None:
    """A dataset may hand out a view of something it caches, so writes must not escape."""
    original = _sample()
    before = original["img"].clone()
    VolumeAugmentation(intensity=True, noise_scale=1.0)(original)
    assert torch.equal(original["img"], before)


@pytest.mark.unit
def test_intensity_is_affine_in_the_image_and_leaves_labels_untouched() -> None:
    sample = _sample()
    torch.manual_seed(3)
    out = VolumeAugmentation(intensity=True, mul_intensity=0.1, add_intensity=0.1)(sample)
    if torch.equal(out["img"], sample["img"]):
        pytest.skip("the 50% gate declined this draw")
    # One scale and one offset for the whole volume: differences between voxels scale by exactly
    # the same factor, so a ratio of differences is constant.
    d_in = sample["img"].flatten()[1:] - sample["img"].flatten()[:-1]
    d_out = out["img"].flatten()[1:] - out["img"].flatten()[:-1]
    ratio = d_out[d_in.abs() > 1e-6] / d_in[d_in.abs() > 1e-6]
    assert ratio.std() < 1e-5
    assert torch.equal(out["label"], sample["label"])


@pytest.mark.unit
def test_noise_touches_only_the_image() -> None:
    sample = _sample()
    torch.manual_seed(1)
    out = VolumeAugmentation(noise_scale=0.5)(sample)
    assert torch.equal(out["label"], sample["label"]), "noise on a class index is meaningless"


@pytest.mark.unit
def test_dropped_sections_blank_the_image_but_keep_the_labels() -> None:
    """A lost section removes the picture, not the neuron that runs through it."""
    sample = _sample()
    out = VolumeAugmentation(drop_slice_prob=1.0)(sample)
    zeroed = (out["img"] == 0).all(dim=1)[0]     # over the channel axis
    planes = [int(zeroed.all(dim=d1).all(dim=d2).sum()) for d1, d2 in ((0, 0), (0, 1), (1, 1))]
    assert max(planes) > 0, "with probability 1 an entire axis of sections should be blanked"
    assert torch.equal(out["label"], sample["label"])


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(12))
def test_slice_shift_moves_image_and_labels_together(seed: int) -> None:
    """The divergence from the reference: a shifted section must keep its labels.

    Runs over several seeds because the shifted axis is drawn at random, and an axis-arithmetic
    error shows up only for the axes it gets wrong.
    """
    torch.manual_seed(seed)
    # A label field equal to the image lets one comparison check both moved identically.
    base = torch.rand(SHAPE)
    sample = {"img": base.clone(), "label": base.clone()[:, 0]}
    out = VolumeAugmentation(shift_slice_prob=1.0, shift_magnitude=3)(sample)
    assert torch.equal(out["img"][:, 0], out["label"]), (
        "image and labels ended up shifted differently, so the targets no longer describe the "
        "image -- the axis arithmetic is wrong for this axis"
    )


@pytest.mark.unit
def test_slice_shift_is_a_permutation_of_the_volume() -> None:
    """Rolling relocates voxels; it must not create or destroy any."""
    sample = _sample()
    out = VolumeAugmentation(shift_slice_prob=1.0, shift_magnitude=4)(sample)
    assert torch.equal(out["img"].flatten().sort().values, sample["img"].flatten().sort().values)


@pytest.mark.unit
def test_slice_shift_actually_changes_something() -> None:
    torch.manual_seed(5)
    sample = _sample()
    changed = False
    for _ in range(20):
        out = VolumeAugmentation(shift_slice_prob=1.0, shift_magnitude=4)(sample)
        changed = changed or not torch.equal(out["img"], sample["img"])
    assert changed, "shift_slice_prob=1 never moved anything"


@pytest.mark.unit
def test_a_missing_image_key_is_an_error_not_a_silent_no_op() -> None:
    with pytest.raises(KeyError, match="no image key"):
        VolumeAugmentation(intensity=True)({"volume": torch.rand(SHAPE)})


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"drop_slice_prob": 1.5}, "probability"),
        ({"shift_slice_prob": -0.1}, "probability"),
        ({"shift_magnitude": -1}, "voxels"),
        ({"noise_scale": -1.0}, "noise_scale"),
        ({"image_keys": ()}, "image_keys is empty"),
    ],
)
def test_rejects_nonsense_settings(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        VolumeAugmentation(**kwargs)


@pytest.mark.unit
def test_wrapper_applies_the_transform_and_preserves_length() -> None:
    class _Source(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 3

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {"img": torch.zeros(SHAPE)}

    wrapped = AugmentedDataset(_Source(), VolumeAugmentation(intensity=True, add_intensity=1.0))
    assert len(wrapped) == 3
    torch.manual_seed(0)
    assert any(not torch.equal(wrapped[i]["img"], torch.zeros(SHAPE)) for i in range(3))


# ---------------------------------------------------------------- config


@pytest.mark.unit
def test_config_is_off_by_default() -> None:
    assert not AugmentConfig().enabled()


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"drop_slice_prob": 0.05},
        {"shift_slice_prob": 0.05},
        {"intensity": True},
        {"noise_scale": 0.5},
    ],
)
def test_any_single_setting_enables_augmentation(kwargs) -> None:
    assert AugmentConfig(**kwargs).enabled()


@pytest.mark.unit
def test_a_shift_probability_without_magnitude_is_not_enabled() -> None:
    """Zero magnitude makes the shift a no-op, so it should not count as augmentation."""
    assert not AugmentConfig(shift_slice_prob=0.05, shift_magnitude=0).enabled()


@pytest.mark.unit
def test_config_fields_match_the_augmentation_signature() -> None:
    """The engine passes the config through by `asdict`, so the names have to line up."""
    import dataclasses
    import inspect

    fields = {f.name for f in dataclasses.fields(AugmentConfig())}
    parameters = set(inspect.signature(VolumeAugmentation.__init__).parameters) - {"self"}
    assert fields <= parameters, fields - parameters


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(12))
def test_slice_shift_never_rolls_the_level_axis(seed: int) -> None:
    """Shifting must stay inside a section's own plane.

    The two in-plane axes are always the *final* two dimensions of the indexed view, whichever
    section axis was drawn. Computing them from the section axis instead is off by one for two of
    the three axes, and lands on the scale-level axis -- which a single-level sample hides
    completely, because rolling a length-1 axis does nothing. So this uses two levels with
    disjoint value ranges: any mixing between them means the roll left the plane.
    """
    torch.manual_seed(seed)
    levels = torch.stack([torch.rand(1, 8, 10, 12), 100.0 + torch.rand(1, 8, 10, 12)])
    out = VolumeAugmentation(shift_slice_prob=1.0, shift_magnitude=3)({"img": levels.clone()})

    assert out["img"][0].max() < 50.0, "level 1 values leaked into level 0"
    assert out["img"][1].min() > 50.0, "level 0 values leaked into level 1"
