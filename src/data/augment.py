"""Turning an `[augment]` section into one callable, from `miao.augment`'s primitives.

The operations themselves live in `miao`, which is where an augmentation belongs: it transforms a
sample and knows nothing about models, losses or parallelism. What is here is the part `miao`
deliberately does not take on -- composition. Its functions are pure, ungated and single-purpose
by design, and it declined to grow a config language for assembling them, on the grounds that a
recipe is imperative code and YAML is a poor place to write imperative code. So the recipe lives
in whatever consumes miao, and for this repo that is here.

Three decisions this file owns, none of which the config states:

  * **Order.** Geometric operations run first and receive image and labels together, so they stay
    registered; photometric ones run last and receive only the image. Section drops sit with the
    photometric group because they are image-only, though they model an artefact rather than a
    photometric effect.
  * **Gating.** Each operation sits behind its own coin flip at `APPLY_PROB`, which is the BANIS
    reference's structure: a per-section probability of 0.05 therefore reaches roughly 2.5% of
    sections overall, so the reference's numbers can be used here as written. miao's functions are
    ungated, so this is the only place that can be expressed.
  * **Axis layouts.** `miao`'s geometric functions need the position of z/y/x in each tensor they
    are given, and an image and its labels do not agree on that under every `output_axes` -- a
    label carries no channel axis, so a channel-last layout puts the image's spatial axes one
    position earlier. Getting this wrong does not raise; it rotates or rolls the two apart. The
    dataset already declares its layout, so the offsets are derived from it rather than configured.

Randomness comes from the global `numpy` generator, which `DataLoader` reseeds per worker and per
epoch. `miao`'s functions take the generator as an argument, and this is the choice that gets
multi-worker streams right without a `worker_init_fn`: a `default_rng()` held on this object would
be inherited by fork and every worker would draw the identical stream.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from miao.augment import (
    additive_noise,
    drop_sections,
    intensity_jitter,
    rot90inplane,
    rot90isocube,
    shift_sections,
    spatial_dims_for,
)

# Rotation subgroups, by what they may exchange. "inplane" excludes the sectioning axis, which is
# what anisotropic EM needs; "full" is the 48-element group and requires cubic voxels.
ROTATIONS = ("none", "inplane", "full")
# The axis serial sectioning runs along, and so the one "inplane" never permutes. Named rather
# than configured: it is a property of how the data was acquired, and the dataset already declares
# its axis order, so a second setting could only disagree with it.
SECTION_AXIS = "z"
# Each operation is gated by a coin flip before its own parameters apply. Copied from the
# reference rather than folded into the probabilities, so its numbers can be used as written.
APPLY_PROB = 0.5


class VolumeAugmentation:
    """One sample in, one augmented sample out, composing `miao.augment` in a fixed order.

    Every field mirrors one on `AugmentConfig`. `sample_axes` is the dataset's declared layout
    (e.g. `"lcxyz"`), needed only to locate the spatial axes; a dataset that declares none can
    still use the photometric operations, which do not care where the axes are.
    """

    def __init__(
        self,
        rotate: str = "none",
        drop_slice_prob: float = 0.0,
        shift_slice_prob: float = 0.0,
        shift_magnitude: int = 10,
        intensity: bool = False,
        mul_intensity: float = 0.1,
        add_intensity: float = 0.1,
        noise_scale: float = 0.0,
        sample_axes: str | None = None,
        image_keys: tuple[str, ...] = ("img",),
        label_keys: tuple[str, ...] = ("label",),
    ) -> None:
        if rotate not in ROTATIONS:
            raise ValueError(f"rotate must be one of {ROTATIONS}, got {rotate!r}")
        for name, value in (
            ("drop_slice_prob", drop_slice_prob),
            ("shift_slice_prob", shift_slice_prob),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a probability in [0, 1], got {value}")
        if shift_magnitude < 0:
            raise ValueError(f"shift_magnitude must be >= 0 voxels, got {shift_magnitude}")
        if noise_scale < 0.0:
            raise ValueError(f"noise_scale must be >= 0, got {noise_scale}")
        if not image_keys:
            raise ValueError("image_keys is empty, so no augmentation could ever apply")

        geometric = rotate != "none" or (shift_slice_prob > 0 and shift_magnitude > 0)
        if geometric and sample_axes is None:
            raise ValueError(
                "rotation and section shifting move the spatial axes, so they need the dataset's "
                "axis order to know where those axes are, but this dataset declares no "
                "sample_axes. Use a dataset that declares its layout, or configure only the "
                "photometric operations (intensity, noise), which do not depend on it."
            )
        if rotate == "inplane" and sample_axes is not None and SECTION_AXIS not in sample_axes:
            raise ValueError(
                f"rotate='inplane' holds the sectioning axis {SECTION_AXIS!r} fixed, but the "
                f"dataset's axis order {sample_axes!r} has no {SECTION_AXIS!r} among its axes."
            )

        self.rotate = rotate
        self.drop_slice_prob = drop_slice_prob
        self.shift_slice_prob = shift_slice_prob
        self.shift_magnitude = shift_magnitude
        self.intensity = intensity
        self.mul_intensity = mul_intensity
        self.add_intensity = add_intensity
        self.noise_scale = noise_scale
        self.sample_axes = sample_axes
        self.image_keys = tuple(image_keys)
        self.label_keys = tuple(label_keys)

    def _dims_for(self, key: str) -> tuple[int, int, int]:
        """Where the spatial axes sit in the tensor under `key`.

        Labels are the reason this is per key rather than one value for the sample: miao returns
        them without the image's channel axis, so under a channel-last `output_axes` the image's
        spatial axes sit one position earlier than the label's. Passing one offset for both
        transforms them differently, which no error reports.
        """
        assert self.sample_axes is not None  # guarded in __init__ for the ops that need it
        axes = self.sample_axes if key in self.image_keys else self.sample_axes.replace("c", "")
        return spatial_dims_for(axes)

    def _section_slot(self) -> int:
        """Which `spatial_dims` slot holds the sectioning axis.

        A slot index, not an axis position, because that is what `rot90inplane` takes -- and it
        moves with the layout, since `spatial_dims` lists the axes in the order `output_axes`
        gives them. `"lczyx"` puts z first and `"lcxyz"` puts it last, so hard-coding either is
        wrong for the other, and wrong quietly: the run holds the wrong axis fixed and permutes
        the sectioning axis into the image plane, which anisotropic data cannot survive.
        """
        assert self.sample_axes is not None
        return [axis for axis in self.sample_axes if axis in "zyx"].index(SECTION_AXIS)

    @staticmethod
    def _coin(rng: Any, probability: float = APPLY_PROB) -> bool:
        return bool(rng.random() < probability)

    def _present(self, sample: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
        # A volume without labels yields miao's empty sentinel, which has no spatial axes to
        # transform; treating it as a tensor is what the geometric functions assert against.
        return [
            key for key in keys
            if key in sample and getattr(sample[key], "numel", lambda: 1)()
        ]

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        # The global numpy generator; `DataLoader` reseeds it per worker and per epoch.
        rng = np.random

        images = self._present(sample, self.image_keys)
        if not images:
            raise KeyError(
                f"no image key {self.image_keys} in the sample, which carries {sorted(sample)}; "
                "augmentation would silently do nothing"
            )
        geometric = images + self._present(sample, self.label_keys)
        sample = dict(sample)

        # Geometric first, image and labels together so one draw moves both.
        if self.rotate != "none":
            spin = rot90isocube if self.rotate == "full" else rot90inplane
            extra = {} if self.rotate == "full" else {"fixed_axis": self._section_slot()}
            rotated = spin(
                rng,
                *(sample[key] for key in geometric),
                spatial_dims=[self._dims_for(key) for key in geometric],
                pixel_size=sample.get("pixel_size"),
                **extra,
            )
            sample.update(zip(geometric, rotated, strict=True))

        if self.shift_slice_prob > 0 and self.shift_magnitude > 0 and self._coin(rng):
            shifted = shift_sections(
                rng,
                *(sample[key] for key in geometric),
                prob=self.shift_slice_prob,
                magnitude=self.shift_magnitude,
                spatial_dims=[self._dims_for(key) for key in geometric],
            )
            sample.update(zip(geometric, shifted, strict=True))

        # Photometric and image-only from here. `drop_sections` is among them because it touches
        # the image alone -- a label must survive a lost section, or the model learns that a gap
        # in the picture is a boundary in the specimen.
        if self.drop_slice_prob > 0 and self._coin(rng):
            for key in images:
                sample[key] = drop_sections(
                    rng, sample[key], prob=self.drop_slice_prob,
                    spatial_dims=self._dims_for(key),
                )

        if self.intensity and self._coin(rng):
            scale = (1.0 - self.mul_intensity, 1.0 + self.mul_intensity)
            shift = (-self.add_intensity, self.add_intensity)
            for key in images:
                sample[key] = intensity_jitter(rng, sample[key], scale=scale, shift=shift)

        if self.noise_scale > 0 and self._coin(rng):
            for key in images:
                sample[key] = additive_noise(rng, sample[key], scale=self.noise_scale)

        return sample
