"""The `[augment]` recipe: composition, ordering, and the axis arithmetic it derives.

The operations are miao's and are tested there. What is this repo's, and what these tests cover,
is the composition: which operations see the labels, what order they run in, and -- the part with
a silent failure mode -- working out where each tensor's spatial axes are. An image and its labels
disagree about that under a channel-last `output_axes`, because miao returns labels without the
image's channel axis, and transforming them on different axes does not raise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from data.augment import APPLY_PROB, ROTATIONS, VolumeAugmentation
from engine.config import AugmentConfig

AXES = "lcxyz"
SHAPE = (1, 1, 8, 8, 8)          # l, c, x, y, z -- cubic, since rotation permutes the axes
LABEL_SHAPE = (1, 8, 8, 8)


def _sample(axes: str = AXES) -> dict[str, Any]:
    """A miao-shaped sample: image with a channel axis, label without."""
    torch.manual_seed(0)
    sizes = {"l": 1, "c": 1}
    image = torch.rand(tuple(sizes.get(a, 8) for a in axes))
    label = torch.rand(tuple(sizes.get(a, 8) for a in axes.replace("c", "")))
    return {"img": image, "label": label, "pixel_size": [[9.0, 9.0, 20.0]]}


def _until_applied(augmentation: VolumeAugmentation, sample, tries: int = 20):
    """The first output whose `APPLY_PROB` gate fired.

    Every operation sits behind a coin flip, so drawing once and asserting on the result asserts
    on the seed. Sweeping fixed seeds until the gate opens tests the operation instead, and stays
    deterministic.
    """
    for seed in range(tries):
        np.random.seed(seed)
        out = augmentation(sample)
        if not torch.equal(out["img"], sample["img"]):
            return out
    raise AssertionError(f"the {APPLY_PROB} apply gate declined all {tries} draws")


# ---------------------------------------------------------------- composition


@pytest.mark.unit
def test_disabled_by_default_leaves_the_sample_alone() -> None:
    sample = _sample()
    out = VolumeAugmentation(sample_axes=AXES)(dict(sample))
    assert torch.equal(out["img"], sample["img"])
    assert torch.equal(out["label"], sample["label"])


@pytest.mark.unit
def test_photometric_operations_never_touch_the_labels() -> None:
    """A class index is not a quantity that can be brightened or have noise added to it."""
    sample = _sample()
    for kwargs in ({"intensity": True}, {"noise_scale": 0.5}, {"drop_slice_prob": 1.0}):
        out = _until_applied(VolumeAugmentation(sample_axes=AXES, **kwargs), sample)
        assert torch.equal(out["label"], sample["label"]), f"{kwargs} altered the label"


@pytest.mark.unit
def test_does_not_mutate_the_input_sample() -> None:
    """A dataset may hand out a view of something it caches, so writes must not escape."""
    sample = _sample()
    before = sample["img"].clone()
    _until_applied(VolumeAugmentation(sample_axes=AXES, intensity=True, noise_scale=1.0), sample)
    assert torch.equal(sample["img"], before)


@pytest.mark.unit
def test_a_missing_image_key_is_an_error_not_a_silent_no_op() -> None:
    with pytest.raises(KeyError, match="no image key"):
        VolumeAugmentation(sample_axes=AXES, intensity=True)({"volume": torch.rand(SHAPE)})


@pytest.mark.unit
def test_an_empty_label_sentinel_is_skipped_not_rotated() -> None:
    """A volume without labels yields miao's empty tensor, which has no axes to transform."""
    sample = _sample() | {"label": torch.empty(0)}
    out = VolumeAugmentation(sample_axes=AXES, rotate="inplane")(sample)
    assert out["label"].numel() == 0


# ---------------------------------------------------------------- axis arithmetic


@pytest.mark.unit
@pytest.mark.parametrize("axes", ["lcxyz", "lczyx", "lzyxc", "lxyzc"])
@pytest.mark.parametrize("op", [{"rotate": "inplane"}, {"shift_slice_prob": 1.0}])
def test_image_and_label_stay_aligned_under_any_layout(axes, op) -> None:
    """The silent failure this composition exists to prevent.

    miao returns labels without the image's channel axis, so a channel-last layout puts the
    image's spatial axes one position earlier than the label's. Transforming both on one set of
    offsets moves them apart, and nothing raises.
    """
    sizes = {"l": 1, "c": 1}
    base = torch.zeros(tuple(sizes.get(a, 8) for a in axes))
    label = torch.zeros(tuple(sizes.get(a, 8) for a in axes.replace("c", "")))
    base[tuple(0 if a in "lc" else 2 for a in axes)] = 1.0
    label[tuple(0 if a == "l" else 2 for a in axes.replace("c", ""))] = 1.0
    # pixel_size follows output_axes spatial order, so z's coarse entry moves with the layout.
    # Hardcoding [9, 9, 20] would be right for "lcxyz" and wrong for "lczyx" -- the same
    # fixed-order assumption these tests exist to catch.
    spatial = [a for a in axes if a in "zyx"]
    sample = {
        "img": base, "label": label,
        "pixel_size": [[20.0 if a == "z" else 9.0 for a in spatial]],
    }

    augmentation = VolumeAugmentation(sample_axes=axes, shift_magnitude=2, **op)
    for seed in range(12):
        np.random.seed(seed)
        out = augmentation(sample)
        img_at = [(out["img"] == 1.0).nonzero()[0].tolist()[axes.index(a)] for a in "zyx"]
        lbl_axes = axes.replace("c", "")
        lbl_at = [(out["label"] == 1.0).nonzero()[0].tolist()[lbl_axes.index(a)] for a in "zyx"]
        assert img_at == lbl_at, (
            f"{op} on {axes!r} moved image and label to different voxels, so the targets no "
            "longer describe the image"
        )


@pytest.mark.unit
def test_inplane_never_permutes_the_sectioning_axis() -> None:
    """At 9x9x20 nm, exchanging z with x would relabel a 20 nm neighbour as a 9 nm one."""
    z = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 1, 8).expand(SHAPE).contiguous()
    augmentation = VolumeAugmentation(sample_axes=AXES, rotate="inplane")
    for seed in range(30):
        np.random.seed(seed)
        out = augmentation({"img": z.clone(), "pixel_size": [[9.0, 9.0, 20.0]]})["img"]
        spread = out[0, 0].reshape(-1, 8).std(dim=0)
        assert torch.allclose(spread, torch.zeros(8), atol=1e-6), "z was permuted away"


@pytest.mark.unit
def test_full_rotation_refuses_anisotropic_voxels() -> None:
    """miao asserts the condition against the sample's own pixel_size, so a mismatch fails."""
    with pytest.raises(AssertionError, match="isotropic"):
        VolumeAugmentation(sample_axes=AXES, rotate="full")(_sample())


# ---------------------------------------------------------------- configuration


@pytest.mark.unit
def test_geometric_operations_need_a_declared_layout() -> None:
    """Without an axis order there is no way to know which axes are spatial."""
    for kwargs in ({"rotate": "inplane"}, {"shift_slice_prob": 0.5}):
        with pytest.raises(ValueError, match="declares no sample_axes"):
            VolumeAugmentation(**kwargs)


@pytest.mark.unit
def test_photometric_operations_do_not_need_one() -> None:
    """Brightness and noise do not care where the axes are, so they must not demand a layout."""
    out = _until_applied(VolumeAugmentation(intensity=True), {"img": torch.rand(SHAPE)})
    assert out["img"].shape == SHAPE


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rotate": "sideways"}, "rotate must be one of"),
        ({"drop_slice_prob": 1.5}, "probability"),
        ({"shift_slice_prob": -0.1}, "probability"),
        ({"shift_magnitude": -1}, "voxels"),
        ({"noise_scale": -1.0}, "noise_scale"),
        ({"image_keys": ()}, "image_keys is empty"),
    ],
)
def test_rejects_nonsense_settings(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        VolumeAugmentation(sample_axes=AXES, **kwargs)


@pytest.mark.unit
def test_config_is_off_by_default() -> None:
    assert not AugmentConfig().enabled()
    assert AugmentConfig().rotate in ROTATIONS


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [{"rotate": "inplane"}, {"drop_slice_prob": 0.05}, {"shift_slice_prob": 0.05},
     {"intensity": True}, {"noise_scale": 0.5}],
)
def test_any_single_setting_enables_augmentation(kwargs) -> None:
    assert AugmentConfig(**kwargs).enabled()


@pytest.mark.unit
def test_a_shift_probability_without_magnitude_is_not_enabled() -> None:
    assert not AugmentConfig(shift_slice_prob=0.05, shift_magnitude=0).enabled()


@pytest.mark.unit
def test_every_config_field_reaches_the_recipe() -> None:
    """`engine.run` splats `[augment]` into the recipe, so the two must agree field for field.

    A field added to the config and not accepted here would be written into the run record and
    then never applied -- an augmentation the run claims to have used and did not.
    """
    import dataclasses
    import inspect

    config_fields = {f.name for f in dataclasses.fields(AugmentConfig)}
    accepted = set(inspect.signature(VolumeAugmentation.__init__).parameters) - {"self"}
    assert not config_fields - accepted, f"{sorted(config_fields - accepted)} never reaches it"
