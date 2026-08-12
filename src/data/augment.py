"""Volumetric augmentations for electron-microscopy training data.

These reproduce the augmentation recipe the BANIS NISB baselines train with, which is where the
specific operations and their default magnitudes come from. Two of them model artefacts particular
to serial-section EM rather than generic image noise:

  * **Dropped sections.** A section can be lost or unusable, leaving a blank plane through the
    volume. The label is deliberately *not* blanked with it: the neuron still passes through, and
    the model has to carry an object across the gap rather than treat it as a boundary.
  * **Section shift.** Sections are imaged separately and aligned afterwards, so alignment is
    imperfect and a plane can sit offset from its neighbours.

The reference implementation shifts the image and leaves the labels in place, which desynchronises
them by up to `shift_magnitude` voxels. Here both move together: an image-only shift trains the
model against targets that no longer describe the image it is shown, and boundary localisation is
the first thing that costs. That is a deliberate divergence from the reference.

Magnitudes are meaningful only against a known intensity range. Both this repo and the reference
present images in [0, 1], so the reference defaults transfer unchanged -- note that
`noise_scale = 0.5` is severe, a standard deviation of up to half the dynamic range, and is
applied to only half of samples.

Everything runs on the sample, inside the dataloader's workers, so it costs no GPU time and
parallelises over `num_workers`. Randomness comes from torch's global generator, which
`DataLoader` already seeds differently per worker and per epoch.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.utils.data as data

SPATIAL_RANK = 3
# Each operation is gated by a coin flip before its own parameters apply, so a per-slice
# probability of 0.05 reaches roughly 2.5% of slices overall. Copied from the reference rather than
# folded into the probabilities, so its numbers can be used here as written.
APPLY_PROB = 0.5


class VolumeAugmentation:
    """Photometric and section-artefact augmentation of one sample.

    Applies to the trailing three axes, which is where the spatial dimensions sit in every layout
    this repo produces (`lcxyz` for images, `lxyz` for labels), so no axis string is needed.

    `image_keys` receive everything; `label_keys` receive only the geometric operations, since a
    class index is not a quantity that can be brightened or have noise added to it.
    """

    def __init__(
        self,
        drop_slice_prob: float = 0.0,
        shift_slice_prob: float = 0.0,
        shift_magnitude: int = 10,
        intensity: bool = False,
        mul_intensity: float = 0.1,
        add_intensity: float = 0.1,
        noise_scale: float = 0.0,
        image_keys: tuple[str, ...] = ("img",),
        label_keys: tuple[str, ...] = ("label",),
    ) -> None:
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

        self.drop_slice_prob = drop_slice_prob
        self.shift_slice_prob = shift_slice_prob
        self.shift_magnitude = shift_magnitude
        self.intensity = intensity
        self.mul_intensity = mul_intensity
        self.add_intensity = add_intensity
        self.noise_scale = noise_scale
        self.image_keys = tuple(image_keys)
        self.label_keys = tuple(label_keys)

    @staticmethod
    def _coin(probability: float = APPLY_PROB) -> bool:
        return bool(torch.rand(()).item() < probability)

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        return float(torch.empty(()).uniform_(low, high).item())

    def _present(self, sample: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
        return [key for key in keys if key in sample]

    def _drop_slices(self, sample: dict[str, Any], images: list[str]) -> None:
        """Blank whole sections of the image, leaving the labels intact."""
        axis = int(torch.randint(-SPATIAL_RANK, 0, ()).item())
        extent = sample[images[0]].shape[axis]
        drop = torch.rand(extent) < self.drop_slice_prob
        if not bool(drop.any()):
            return
        index = torch.nonzero(drop, as_tuple=False).squeeze(1)
        for key in images:
            sample[key].index_fill_(axis, index, 0.0)

    def _shift_slices(self, sample: dict[str, Any], keys: list[str]) -> None:
        """Offset individual sections within their own plane, image and labels together."""
        axis = int(torch.randint(-SPATIAL_RANK, 0, ()).item())
        extent = sample[keys[0]].shape[axis]
        selected = torch.rand(extent) < self.shift_slice_prob
        for position in torch.nonzero(selected, as_tuple=False).squeeze(1).tolist():
            # One pair of shifts, drawn once and applied to every tensor, so the image and its
            # labels stay registered to each other.
            shifts = [
                int(torch.randint(-self.shift_magnitude, self.shift_magnitude + 1, ()).item())
                for _ in range(SPATIAL_RANK - 1)
            ]
            for key in keys:
                # `movedim` gives a view with the shifted axis at the front; indexing it away
                # leaves the other two spatial axes as the final two dimensions, in their original
                # relative order -- whichever axis was chosen. So the in-plane axes are always
                # (-2, -1) here, and computing them from `axis` only agrees for `axis == -1`.
                view = sample[key].movedim(axis, 0)[position]
                view.copy_(torch.roll(view, shifts=shifts, dims=(-2, -1)))

    def _jitter_intensity(self, sample: dict[str, Any], images: list[str]) -> None:
        scale = self._uniform(1.0 - self.mul_intensity, 1.0 + self.mul_intensity)
        offset = self._uniform(-self.add_intensity, self.add_intensity)
        for key in images:
            sample[key].mul_(scale).add_(offset)

    def _add_noise(self, sample: dict[str, Any], images: list[str]) -> None:
        # The reference draws the standard deviation itself uniformly, so most affected samples
        # get much less than `noise_scale`.
        deviation = self._uniform(0.0, 1.0) * self.noise_scale
        for key in images:
            sample[key].add_(torch.randn_like(sample[key]) * deviation)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample = dict(sample)
        images = self._present(sample, self.image_keys)
        if not images:
            raise KeyError(
                f"no image key {self.image_keys} in the sample, which carries {sorted(sample)}; "
                "augmentation would silently do nothing"
            )
        # Cloned because every operation writes in place, and a dataset is free to hand out a view
        # of something it caches.
        for key in images + self._present(sample, self.label_keys):
            sample[key] = sample[key].clone()

        if self.drop_slice_prob > 0 and self._coin():
            self._drop_slices(sample, images)
        if self.shift_slice_prob > 0 and self.shift_magnitude > 0 and self._coin():
            self._shift_slices(sample, images + self._present(sample, self.label_keys))
        if self.intensity and self._coin():
            self._jitter_intensity(sample, images)
        if self.noise_scale > 0 and self._coin():
            self._add_noise(sample, images)
        return sample


class AugmentedDataset(data.Dataset):
    """A map-style dataset with a transform applied to each sample as it is read."""

    def __init__(self, source: data.Dataset, transform: VolumeAugmentation) -> None:
        self.source = source
        self.transform = transform

    def __len__(self) -> int:
        return len(self.source)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.transform(self.source[index])
