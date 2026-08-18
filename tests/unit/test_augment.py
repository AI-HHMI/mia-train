"""The in-house augmentation recipe: operations, composition, and the axis arithmetic it derives.

These operations used to be imported from `miao`. They are implemented here now, so the tests
cover both halves: that each operation does what it claims, and that the recipe composes them in
the right order onto the right keys.

The property with a silent failure mode gets the most attention. A sample's spatial axes are
wherever `sample_axes` says, and an image and its labels disagree about that whenever the channel
axis trails the spatial ones -- labels carry no channel. Transforming the two on one set of
offsets moves them apart and raises nothing, so several tests below drive a single marker voxel
through an operation and assert it lands in the same place in both.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from data.augment import (
    APPLY_PROB,
    ROTATIONS,
    VolumeAugmentation,
    additive_noise,
    drop_sections,
    intensity_jitter,
    rot90,
    shift_sections,
    spatial_dims,
)
from engine.config import AugmentConfig

AXES = "lcxyz"
SHAPE = (1, 1, 8, 8, 8)
# Every layout mia-train's datasets can produce, plus the awkward ones. 3D miao configs use
# "lcxyz"; the Hub presets use "lcxyz" and "lcxy"; the rest are legal `output_axes` that put the
# channel or the level axis somewhere the trailing-three assumption does not survive.
LAYOUTS_3D = ["lcxyz", "lczyx", "lxyz", "lzyxc", "lxyzc", "zyxl"]
LAYOUTS_2D = ["lcxy", "lcyx", "lxyc"]


def _sample(axes: str = AXES, marker: bool = False) -> dict[str, Any]:
    """A dataset-shaped sample: image with a channel axis, label without."""
    torch.manual_seed(0)
    sizes = {"l": 1, "c": 1}
    label_axes = axes.replace("c", "")
    if marker:
        image = torch.zeros(tuple(sizes.get(a, 8) for a in axes))
        label = torch.zeros(tuple(sizes.get(a, 8) for a in label_axes))
        image[tuple(0 if a in "lc" else 2 for a in axes)] = 1.0
        label[tuple(0 if a == "l" else 2 for a in label_axes)] = 1.0
    else:
        image = torch.rand(tuple(sizes.get(a, 8) for a in axes))
        label = torch.rand(tuple(sizes.get(a, 8) for a in label_axes))
    spatial = [a for a in axes if a in "xyz"]
    return {
        "img": image, "label": label,
        # pixel_size follows the layout's own spatial order, so z's coarse entry moves with it.
        "pixel_size": [[20.0 if a == "z" else 9.0 for a in spatial]],
    }


def _marker(tensor: torch.Tensor, axes: str) -> list[int]:
    """Where the marker sits, spatial axes only, so tensors of different rank compare."""
    index = (tensor == 1.0).nonzero()[0].tolist()
    return [index[axes.index(a)] for a in axes if a in "xyz"]


def _until_applied(augmentation: VolumeAugmentation, sample, tries: int = 20):
    """The first output whose `APPLY_PROB` gate fired.

    Every operation sits behind a coin flip, so drawing once and asserting on the result asserts
    on the seed. Sweeping fixed seeds until the gate opens tests the operation instead.
    """
    for seed in range(tries):
        torch.manual_seed(seed)
        out = augmentation(sample)
        if not torch.equal(out["img"], sample["img"]):
            return out
    raise AssertionError(f"the {APPLY_PROB} apply gate declined all {tries} draws")


# ---------------------------------------------------------------- the operations


class TestRot90:
    @pytest.mark.parametrize("axes", LAYOUTS_3D + LAYOUTS_2D)
    def test_is_a_bijection_of_the_volume(self, axes):
        """A signed permutation relocates voxels; it must not create or destroy any."""
        sample = _sample(axes)
        for seed in range(8):
            torch.manual_seed(seed)
            (out,) = rot90([sample["img"]], [spatial_dims(axes)])
            assert torch.equal(out.flatten().sort().values, sample["img"].flatten().sort().values)

    def test_offers_the_whole_group_in_3d(self):
        """6 orderings x 8 flips, and every one reachable."""
        img = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(4, 4, 4)
        seen = set()
        for seed in range(4000):
            torch.manual_seed(seed)
            seen.add(rot90([img], [(-3, -2, -1)])[0].numpy().tobytes())
        assert len(seen) == 48

    def test_offers_the_dihedral_group_in_2d(self):
        """2 orderings x 4 flips: rank-generic, so a plane gets its own 8."""
        img = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4)
        seen = set()
        for seed in range(500):
            torch.manual_seed(seed)
            seen.add(rot90([img], [(-2, -1)])[0].numpy().tobytes())
        assert len(seen) == 8

    def test_a_fixed_slot_halves_the_orderings(self):
        """Holding one axis out of the permutation leaves 2 orderings x 8 flips = 16."""
        img = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(4, 4, 4)
        seen = set()
        for seed in range(4000):
            torch.manual_seed(seed)
            seen.add(rot90([img], [(-3, -2, -1)], fixed_slot=0)[0].numpy().tobytes())
        assert len(seen) == 16

    def test_a_fixed_slot_is_never_permuted(self):
        """The property the subgroup exists for: the held axis keeps its variation."""
        ramp = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1).expand(6, 6, 6).contiguous()
        for seed in range(50):
            torch.manual_seed(seed)
            (out,) = rot90([ramp], [(-3, -2, -1)], fixed_slot=0)
            assert torch.allclose(out.reshape(6, -1).std(dim=1), torch.zeros(6), atol=1e-6)

    def test_a_non_cubic_permutation_is_refused(self):
        with pytest.raises(ValueError, match="equal-sized"):
            for seed in range(50):          # sweep until a draw actually permutes
                torch.manual_seed(seed)
                rot90([torch.rand(4, 5, 6)], [(-3, -2, -1)])


class TestShiftSections:
    @pytest.mark.parametrize("axes", LAYOUTS_3D + LAYOUTS_2D)
    def test_is_a_permutation_of_the_volume(self, axes):
        sample = _sample(axes)
        for seed in range(8):
            torch.manual_seed(seed)
            (out,) = shift_sections([sample["img"]], [spatial_dims(axes)], prob=1.0, magnitude=3)
            assert torch.equal(out.flatten().sort().values, sample["img"].flatten().sort().values)

    def test_zero_magnitude_is_the_identity(self):
        img = torch.rand(4, 4, 4)
        torch.manual_seed(0)
        (out,) = shift_sections([img], [(-3, -2, -1)], prob=1.0, magnitude=0)
        assert torch.equal(out, img)

    def test_does_not_mutate_its_input(self):
        img = torch.rand(6, 6, 6)
        before = img.clone()
        torch.manual_seed(0)
        shift_sections([img], [(-3, -2, -1)], prob=1.0, magnitude=2)
        assert torch.equal(img, before)


class TestDropSections:
    def test_certain_probability_blanks_everything(self):
        torch.manual_seed(0)
        out = drop_sections(torch.ones(4, 4, 4), (-3, -2, -1), prob=1.0)
        assert torch.equal(out, torch.zeros(4, 4, 4))

    def test_zero_probability_is_the_identity(self):
        img = torch.rand(4, 4, 4)
        torch.manual_seed(0)
        assert torch.equal(drop_sections(img, (-3, -2, -1), prob=0.0), img)

    def test_blanks_whole_sections_not_parts(self):
        torch.manual_seed(0)
        out = drop_sections(torch.ones(8, 8, 8), (-3, -2, -1), prob=0.5)
        per_axis = [out.movedim(d, 0).reshape(8, -1) for d in range(3)]
        assert any(
            all(float(s[i].min()) == float(s[i].max()) for i in range(8)) for s in per_axis
        ), "a drop must blank a whole section"

    def test_does_not_mutate_its_input(self):
        img = torch.ones(6, 6, 6)
        before = img.clone()
        torch.manual_seed(0)
        drop_sections(img, (-3, -2, -1), prob=1.0)
        assert torch.equal(img, before)


class TestPhotometric:
    def test_intensity_jitter_is_exactly_affine(self):
        """Two scalars for the whole volume, so a fit leaves no residual."""
        img = torch.rand(16, 16, 16)
        torch.manual_seed(0)
        out = intensity_jitter(img, mul=0.1, add=0.1)
        d_in, d_out = img.flatten().diff(), out.flatten().diff()
        ratio = d_out[d_in.abs() > 1e-6] / d_in[d_in.abs() > 1e-6]
        assert float(ratio.std()) < 1e-5

    def test_additive_noise_is_zero_mean_and_within_scale(self):
        torch.manual_seed(0)
        out = additive_noise(torch.zeros(48, 48, 48), scale=0.5)
        assert abs(float(out.mean())) < 0.01
        assert float(out.std()) <= 0.5 + 1e-3, "the drawn deviation must not exceed `scale`"

    @pytest.mark.parametrize("op", [lambda i: intensity_jitter(i, 0.1, 0.1),
                                    lambda i: additive_noise(i, 0.5)])
    def test_does_not_mutate_its_input(self, op):
        img = torch.rand(6, 6, 6)
        before = img.clone()
        torch.manual_seed(0)
        op(img)
        assert torch.equal(img, before)


# ---------------------------------------------------------------- axis arithmetic


@pytest.mark.unit
@pytest.mark.parametrize(("axes", "expected"), [
    ("lcxyz", (-3, -2, -1)), ("lczyx", (-3, -2, -1)), ("lxyz", (-3, -2, -1)),
    ("lzyxc", (-4, -3, -2)), ("zyxl", (-4, -3, -2)), ("lcxy", (-2, -1)),
])
def test_spatial_dims_follow_the_layout(axes, expected) -> None:
    assert spatial_dims(axes) == expected


@pytest.mark.unit
@pytest.mark.parametrize("axes", LAYOUTS_3D + LAYOUTS_2D)
def test_slot_i_is_the_i_th_spatial_axis(axes) -> None:
    """The invariant every operation indexes against."""
    resolved = "".join(axes[offset % len(axes)] for offset in spatial_dims(axes))
    assert resolved == "".join(a for a in axes if a in "xyz")


@pytest.mark.unit
@pytest.mark.parametrize("axes", LAYOUTS_3D + LAYOUTS_2D)
@pytest.mark.parametrize("op", [{"rotate": "full"}, {"shift_slice_prob": 1.0}])
def test_image_and_label_stay_aligned_under_any_layout(axes, op) -> None:
    """The silent failure the per-key offsets exist to prevent."""
    sample = _sample(axes, marker=True)
    sample["pixel_size"] = [[9.0] * len(spatial_dims(axes))]      # isotropic, so "full" is legal
    augmentation = VolumeAugmentation(sample_axes=axes, shift_magnitude=2, **op)
    for seed in range(12):
        torch.manual_seed(seed)
        out = augmentation(sample)
        assert _marker(out["img"], axes) == _marker(out["label"], axes.replace("c", "")), (
            f"{op} on {axes!r} moved image and label to different voxels"
        )


@pytest.mark.unit
@pytest.mark.parametrize("axes", LAYOUTS_3D)
def test_inplane_never_permutes_the_sectioning_axis(axes) -> None:
    """At 9x9x20 nm, exchanging z with x relabels a 20 nm neighbour as a 9 nm one."""
    z_at = axes.index("z")
    shape = tuple({"l": 1, "c": 1}.get(a, 6) for a in axes)
    ramp = torch.arange(6, dtype=torch.float32)
    img = ramp.reshape([6 if i == z_at else 1 for i in range(len(axes))]).expand(shape)
    spatial = [a for a in axes if a in "xyz"]
    sample = {"img": img.contiguous(),
              "pixel_size": [[20.0 if a == "z" else 9.0 for a in spatial]]}

    augmentation = VolumeAugmentation(sample_axes=axes, rotate="inplane")
    for seed in range(30):
        torch.manual_seed(seed)
        out = augmentation(sample)["img"]
        assert torch.allclose(
            out.movedim(z_at, 0).reshape(6, -1).std(dim=1), torch.zeros(6), atol=1e-6
        ), f"z was permuted away under {axes!r}"


@pytest.mark.unit
def test_full_rotation_refuses_anisotropic_voxels() -> None:
    with pytest.raises(ValueError, match="share a voxel size"):
        VolumeAugmentation(sample_axes=AXES, rotate="full")(_sample())


@pytest.mark.unit
def test_inplane_accepts_anisotropy_on_the_axis_it_holds_fixed() -> None:
    """The weaker condition the subgroup needs, and the reason it exists."""
    VolumeAugmentation(sample_axes=AXES, rotate="inplane")(_sample())


@pytest.mark.unit
def test_a_dataset_without_pixel_size_skips_the_voxel_check() -> None:
    """Hub datasets report no voxel size; the check is theirs to make, not ours to invent."""
    torch.manual_seed(0)
    out = VolumeAugmentation(sample_axes=AXES, rotate="full")({"img": torch.rand(SHAPE)})
    assert out["img"].shape == SHAPE


# ---------------------------------------------------------------- composition


@pytest.mark.unit
def test_disabled_by_default_leaves_the_sample_alone() -> None:
    sample = _sample()
    out = VolumeAugmentation(sample_axes=AXES)(dict(sample))
    assert torch.equal(out["img"], sample["img"])
    assert torch.equal(out["label"], sample["label"])


@pytest.mark.unit
def test_photometric_operations_never_touch_the_labels() -> None:
    sample = _sample()
    for kwargs in ({"intensity": True}, {"noise_scale": 0.5}, {"drop_slice_prob": 1.0}):
        out = _until_applied(VolumeAugmentation(sample_axes=AXES, **kwargs), sample)
        assert torch.equal(out["label"], sample["label"]), f"{kwargs} altered the label"


@pytest.mark.unit
def test_does_not_mutate_the_input_sample() -> None:
    sample = _sample()
    before = sample["img"].clone()
    _until_applied(VolumeAugmentation(sample_axes=AXES, intensity=True, noise_scale=1.0), sample)
    assert torch.equal(sample["img"], before)


@pytest.mark.unit
def test_an_empty_label_sentinel_is_skipped_not_rotated() -> None:
    """A volume without labels yields an empty tensor, which has no axes to transform."""
    sample = _sample() | {"label": torch.empty(0)}
    out = VolumeAugmentation(sample_axes=AXES, rotate="inplane")(sample)
    assert out["label"].numel() == 0


@pytest.mark.unit
def test_a_missing_image_key_is_an_error_not_a_silent_no_op() -> None:
    with pytest.raises(KeyError, match="no image key"):
        VolumeAugmentation(sample_axes=AXES, intensity=True)({"volume": torch.rand(SHAPE)})


# ---------------------------------------------------------------- configuration


@pytest.mark.unit
def test_geometric_operations_need_a_declared_layout() -> None:
    for kwargs in ({"rotate": "inplane"}, {"shift_slice_prob": 0.5}):
        with pytest.raises(ValueError, match="declares no sample_axes"):
            VolumeAugmentation(**kwargs)


@pytest.mark.unit
def test_photometric_operations_do_not_need_one() -> None:
    out = _until_applied(VolumeAugmentation(intensity=True), {"img": torch.rand(SHAPE)})
    assert out["img"].shape == SHAPE


@pytest.mark.unit
def test_inplane_on_a_plane_is_refused_with_a_way_forward() -> None:
    """2D data has no sectioning axis to hold out; `full` is the right subgroup there."""
    with pytest.raises(ValueError, match="rotate='full'"):
        VolumeAugmentation(sample_axes="lcxy", rotate="inplane")


@pytest.mark.unit
@pytest.mark.parametrize(("kwargs", "match"), [
    ({"rotate": "sideways"}, "rotate must be one of"),
    ({"drop_slice_prob": 1.5}, "probability"),
    ({"shift_slice_prob": -0.1}, "probability"),
    ({"shift_magnitude": -1}, "voxels"),
    ({"noise_scale": -1.0}, "noise_scale"),
    ({"image_keys": ()}, "image_keys is empty"),
])
def test_rejects_nonsense_settings(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        VolumeAugmentation(sample_axes=AXES, **kwargs)


@pytest.mark.unit
def test_config_is_off_by_default() -> None:
    assert not AugmentConfig().enabled()
    assert AugmentConfig().rotate in ROTATIONS


@pytest.mark.unit
@pytest.mark.parametrize("kwargs", [
    {"rotate": "inplane"}, {"drop_slice_prob": 0.05}, {"shift_slice_prob": 0.05},
    {"intensity": True}, {"noise_scale": 0.5},
])
def test_any_single_setting_enables_augmentation(kwargs) -> None:
    assert AugmentConfig(**kwargs).enabled()


@pytest.mark.unit
def test_a_shift_probability_without_magnitude_is_not_enabled() -> None:
    assert not AugmentConfig(shift_slice_prob=0.05, shift_magnitude=0).enabled()


@pytest.mark.unit
def test_every_config_field_reaches_the_recipe() -> None:
    """`engine.run` splats `[augment]` into the recipe, so the two must agree field for field."""
    import dataclasses
    import inspect

    config_fields = {f.name for f in dataclasses.fields(AugmentConfig)}
    accepted = set(inspect.signature(VolumeAugmentation.__init__).parameters) - {"self"}
    assert not config_fields - accepted, f"{sorted(config_fields - accepted)} never reaches it"
