"""Volumetric augmentation for microscopy training data, implemented here rather than upstream.

These reproduce the augmentation recipe the BANIS NISB baselines train with, which is where the
operations and their default magnitudes come from. Two model artefacts particular to
serial-section EM rather than generic image noise:

  * **Dropped sections.** A section can be lost or unusable, leaving a blank plane through the
    volume. The label is deliberately *not* blanked with it: the object still passes through, and
    the model has to carry it across the gap rather than treat it as a boundary.
  * **Section shift.** Sections are imaged separately and aligned afterwards, so alignment is
    imperfect and a plane can sit offset from its neighbours. Both the image and its labels move
    together here, which is a deliberate divergence from the reference implementations: an
    image-only shift trains the model against targets that no longer describe what it is shown,
    and boundary localisation is the first thing that costs.

**Why in-house.** `miao` also has augmentations, and depending on them was tried and reverted.
Its API is young and has changed shape repeatedly -- config-driven, then callable-driven, with the
set of operations and their signatures still moving -- and a training repo that tracks it inherits
every change. Owning ~200 lines of tensor manipulation is the cheaper side of that trade. It also
removes a compatibility question that would otherwise be permanent: these run on any sample dict
that declares its axis order, so the same recipe serves `miao_volumes` and `hf_semantic_seg`
without either dataset's conventions leaking in here.

**Rank-generic.** Everything below works for 2D and 3D, deriving the spatial rank from the
dataset's declared axis order rather than assuming three axes. That is what lets a 2D CellMap
preset (`"lcxy"`) and a 3D volume (`"lcxyz"`) share one implementation.

**Axis layouts are derived, never assumed.** A sample's spatial axes are wherever `sample_axes`
says they are, and an image and its labels do not agree on that: `miao` returns labels without the
image's channel axis, so a channel-last layout puts the image's spatial axes one position earlier.
Transforming the two on one set of offsets moves them apart, and nothing raises -- so every
operation is told each tensor's offsets separately.

Randomness comes from torch's global generator, which `DataLoader` reseeds per worker and per
epoch. Everything runs on one sample inside the worker processes, so it costs no GPU time and
parallelises over `num_workers`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

# Axis characters that name a spatial dimension. Everything else in a layout string -- `l` for
# scale level, `c` for channel -- is carried along untouched by every operation here.
SPATIAL_AXES = "xyz"
# The axis serial sectioning runs along, and so the one `rotate = "inplane"` never permutes.
# Named rather than configured: it is a property of how the data was acquired, and the dataset
# already declares its axis order, so a second setting could only disagree with it.
SECTION_AXIS = "z"
# Rotation subgroups, by what they may exchange.
ROTATIONS = ("none", "inplane", "full")
# Each operation is gated by a coin flip before its own parameters apply, so a per-section
# probability of 0.05 reaches roughly 2.5% of sections overall. Copied from the reference rather
# than folded into the probabilities, so its numbers can be used here as written.
APPLY_PROB = 0.5


def spatial_dims(axes: str) -> tuple[int, ...]:
    """Negative offsets of the spatial axes in a tensor laid out as `axes`.

    Listed **in the order they appear in `axes`**, not in a fixed x/y/z order, because everything
    indexes them positionally: a permutation reorders slots, `pixel_size` follows the dataset's
    own axis order, and the sectioning axis is named by its slot. Normalising to a fixed order
    would agree with those only for layouts that happen to list the axes that way.

    Counted from the end so one value serves tensors with different numbers of leading axes --
    `"lcxyz"` and `"lxyz"` both give `(-3, -2, -1)`. That breaks down when the spatial axes are
    not trailing, which is why each tensor gets its own: `"lzyxc"` gives the image `(-4, -3, -2)`
    while its label, carrying no channel axis, keeps `(-3, -2, -1)`.
    """
    return tuple(i - len(axes) for i, axis in enumerate(axes) if axis in SPATIAL_AXES)


def _resolve(tensor: torch.Tensor, dims: Sequence[int]) -> list[int]:
    """`dims` as non-negative axis indices for this tensor."""
    return [d % tensor.ndim for d in dims]


def rot90(
    tensors: Sequence[torch.Tensor],
    dims: Sequence[Sequence[int]],
    fixed_slot: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """One axis-aligned rotation/flip, drawn once and applied to every tensor.

    A signed permutation of the spatial axes: `rank!` orderings times `2**rank` flips, so 48 for a
    cube and 8 for a plane. `fixed_slot` excludes one axis from the *permutation* (it may still
    flip), leaving 16 of the 48 in 3D -- the subgroup anisotropic data admits, since exchanging
    axes of unequal voxel size produces object shapes that do not occur in the specimen.

    `dims` gives each tensor its own spatial offsets, which is what keeps an image and its labels
    together under layouts where they disagree. Drawing once and applying many times is the point:
    a per-tensor draw would tear them apart.
    """
    rank = len(dims[0])
    free = [slot for slot in range(rank) if slot != fixed_slot]
    # Permute only the free slots among themselves; the fixed one maps to itself.
    order = [free[i] for i in torch.randperm(len(free)).tolist()]
    perm = list(range(rank))
    for slot, source in zip(free, order, strict=True):
        perm[slot] = source
    flips = (torch.rand(rank) < 0.5).tolist()

    out = []
    for tensor, tensor_dims in zip(tensors, dims, strict=True):
        absolute = _resolve(tensor, tensor_dims)
        sizes = [tensor.shape[a] for a in absolute]
        moved = [slot for slot in range(rank) if perm[slot] != slot]
        if len({sizes[slot] for slot in moved}) > 1:
            raise ValueError(
                f"rotation exchanges spatial slots {moved}, whose extents are "
                f"{[sizes[s] for s in moved]} in a {tuple(tensor.shape)} tensor. A permutation "
                "has to preserve shape, so those axes must be equal-sized."
            )
        flip_axes = [absolute[slot] for slot in range(rank) if flips[slot]]
        if flip_axes:
            tensor = torch.flip(tensor, dims=flip_axes)
        # Place the axis currently at slot perm[i] into slot i; identity everywhere else.
        full = list(range(tensor.ndim))
        for slot, axis in enumerate(absolute):
            full[axis] = absolute[perm[slot]]
        out.append(tensor.permute(full).contiguous())
    return tuple(out)


def shift_sections(
    tensors: Sequence[torch.Tensor],
    dims: Sequence[Sequence[int]],
    prob: float,
    magnitude: int,
) -> tuple[torch.Tensor, ...]:
    """Offset individual sections within their own plane, every tensor together.

    One axis is drawn per call; each section along it is selected with probability `prob` and
    rolled by an independent draw in `[-magnitude, magnitude]` on each remaining spatial axis.

    The in-plane axes are computed, not assumed to be the trailing ones. After the chosen axis is
    moved to the front and indexed away, an axis that sat at `a` lands at `a` if it preceded the
    one removed and at `a - 1` otherwise -- which for a channel-last layout is the difference
    between rolling the image in (x, channel) and rolling it in (y, x).
    """
    rank = len(dims[0])
    slot = int(torch.randint(rank, ()).item())
    extent = tensors[0].shape[_resolve(tensors[0], dims[0])[slot]]
    selected = torch.nonzero(torch.rand(extent) < prob, as_tuple=False).squeeze(1).tolist()
    if not selected or magnitude == 0:
        return tuple(tensors)

    # One draw per section, shared by every tensor, so image and labels stay registered.
    shifts = {
        index: tuple(
            int(torch.randint(-magnitude, magnitude + 1, ()).item()) for _ in range(rank - 1)
        )
        for index in selected
    }

    out = []
    for tensor, tensor_dims in zip(tensors, dims, strict=True):
        absolute = _resolve(tensor, tensor_dims)
        axis = absolute[slot]
        in_plane = tuple(a if a < axis else a - 1 for a in absolute if a != axis)
        moved = tensor.clone().movedim(axis, 0)
        for index, shift in shifts.items():
            section = moved[index]
            section.copy_(torch.roll(section, shifts=shift, dims=in_plane))
        out.append(moved.movedim(0, axis))
    return tuple(out)


def drop_sections(image: torch.Tensor, dims: Sequence[int], prob: float) -> torch.Tensor:
    """Blank whole sections of an image, each drawn independently with probability `prob`.

    Image-only, and the asymmetry is the point: an object still passes through a lost section, so
    blanking its label would teach the model that a gap in the picture is a boundary in the
    specimen.
    """
    absolute = _resolve(image, dims)
    axis = absolute[int(torch.randint(len(absolute), ()).item())]
    dropped = torch.nonzero(torch.rand(image.shape[axis]) < prob, as_tuple=False).squeeze(1)
    if dropped.numel() == 0:
        return image
    return image.clone().index_fill_(axis, dropped, 0.0)


def intensity_jitter(image: torch.Tensor, mul: float, add: float) -> torch.Tensor:
    """Rescale and shift intensities: `image * U(1-mul, 1+mul) + U(-add, add)`.

    Two scalars for the whole volume, so it is a monotone affine map -- voxel ordering survives and
    nothing is destroyed, only the meaning of the intensity units changes. That models staining,
    illumination and exposure differences, which is a different thing from `additive_noise`.
    """
    scale = float(torch.empty(()).uniform_(1.0 - mul, 1.0 + mul).item())
    offset = float(torch.empty(()).uniform_(-add, add).item())
    return image * scale + offset


def additive_noise(image: torch.Tensor, scale: float) -> torch.Tensor:
    """Add zero-mean Gaussian noise whose standard deviation is drawn in `[0, scale]`.

    The deviation is itself random, following the reference, so most affected samples get
    considerably less than `scale` -- it is a maximum, not a typical value. Unlike
    `intensity_jitter` this is per voxel and uncorrelated with the signal, so it lowers
    signal-to-noise rather than rescaling it.
    """
    deviation = float(torch.empty(()).uniform_(0.0, 1.0).item()) * scale
    return image + torch.randn_like(image) * deviation


class VolumeAugmentation:
    """One sample in, one augmented sample out, applying the operations above in a fixed order.

    Every field mirrors one on `AugmentConfig`. `sample_axes` is the dataset's declared layout
    (`"lcxyz"`, `"lcxy"`, ...), used to locate the spatial axes; a dataset that declares none can
    still use the photometric operations, which do not care where the axes are.

    Order is geometric first, receiving image and labels together so they stay registered, then
    photometric, receiving only the image. Section drops sit with the photometric group because
    they are image-only, though they model an artefact rather than a photometric effect.
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

        needs_axes = rotate != "none" or (shift_slice_prob > 0 and shift_magnitude > 0)
        if needs_axes and sample_axes is None:
            raise ValueError(
                "rotation and section shifting move the spatial axes, so they need the dataset's "
                "axis order to know where those axes are, but this dataset declares no "
                "sample_axes. Use a dataset that declares its layout, or configure only the "
                "photometric operations (intensity, noise), which do not depend on it."
            )
        if rotate == "inplane" and sample_axes is not None:
            if SECTION_AXIS not in sample_axes:
                raise ValueError(
                    f"rotate='inplane' holds the sectioning axis {SECTION_AXIS!r} fixed, but the "
                    f"dataset's axis order {sample_axes!r} has none. In-plane data has no axis to "
                    "hold out; use rotate='full', whose transforms are all valid there."
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

    def _dims_for(self, key: str) -> tuple[int, ...]:
        """Where the spatial axes sit in the tensor under `key`.

        Per key rather than one value for the sample: labels carry no channel axis, so under a
        channel-last layout the image's spatial axes sit one position earlier than the label's.
        """
        assert self.sample_axes is not None  # guarded in __init__ for the operations that need it
        axes = self.sample_axes if key in self.image_keys else self.sample_axes.replace("c", "")
        return spatial_dims(axes)

    def _section_slot(self) -> int:
        """Which spatial slot holds the sectioning axis.

        A slot index, because that is what `rot90` excludes -- and it moves with the layout, since
        the slots follow the order the axes appear in. `"lczyx"` puts z first and `"lcxyz"` puts it
        last, so hard-coding either is wrong for the other, and wrong quietly.
        """
        assert self.sample_axes is not None
        return [a for a in self.sample_axes if a in SPATIAL_AXES].index(SECTION_AXIS)

    def _check_voxels(self, sample: dict[str, Any]) -> None:
        """Reject a rotation the voxel sizes do not admit, when the dataset reports them.

        `pixel_size` is miao's, in the same spatial order as the axes; datasets that do not report
        it (the Hub ones) skip the check and the caller carries the responsibility. Permuting two
        axes of unequal voxel size relabels a neighbour relationship at one scale as one at
        another, which produces object shapes that do not occur in the data.
        """
        pixel_size = sample.get("pixel_size")
        if pixel_size is None:
            return
        rank = len(self._dims_for(self.image_keys[0]))
        sizes = torch.as_tensor(pixel_size, dtype=torch.float64).reshape(-1, rank)
        exchanged = [
            slot for slot in range(sizes.shape[1])
            if self.rotate == "full" or slot != self._section_slot()
        ]
        held = sizes[:, exchanged]
        if not torch.allclose(held, held[:, :1], rtol=1e-6, atol=1e-9):
            raise ValueError(
                f"rotate={self.rotate!r} exchanges spatial slots {exchanged}, so those axes must "
                f"share a voxel size, but pixel_size={sizes.tolist()} does not. Use "
                "rotate='inplane' to hold the sectioning axis out of the permutation."
            )

    @staticmethod
    def _coin(probability: float = APPLY_PROB) -> bool:
        return bool(torch.rand(()).item() < probability)

    def _present(self, sample: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
        # A volume without labels yields an empty sentinel, which has no spatial axes to transform.
        return [
            key for key in keys
            if key in sample and getattr(sample[key], "numel", lambda: 1)()
        ]

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        images = self._present(sample, self.image_keys)
        if not images:
            raise KeyError(
                f"no image key {self.image_keys} in the sample, which carries {sorted(sample)}; "
                "augmentation would silently do nothing"
            )
        geometric = images + self._present(sample, self.label_keys)
        sample = dict(sample)

        if self.rotate != "none":
            self._check_voxels(sample)
            rotated = rot90(
                [sample[key] for key in geometric],
                [self._dims_for(key) for key in geometric],
                fixed_slot=None if self.rotate == "full" else self._section_slot(),
            )
            sample.update(zip(geometric, rotated, strict=True))

        if self.shift_slice_prob > 0 and self.shift_magnitude > 0 and self._coin():
            shifted = shift_sections(
                [sample[key] for key in geometric],
                [self._dims_for(key) for key in geometric],
                prob=self.shift_slice_prob,
                magnitude=self.shift_magnitude,
            )
            sample.update(zip(geometric, shifted, strict=True))

        if self.drop_slice_prob > 0 and self._coin():
            for key in images:
                sample[key] = drop_sections(sample[key], self._dims_for(key), self.drop_slice_prob)

        if self.intensity and self._coin():
            for key in images:
                sample[key] = intensity_jitter(sample[key], self.mul_intensity, self.add_intensity)

        if self.noise_scale > 0 and self._coin():
            for key in images:
                sample[key] = additive_noise(sample[key], self.noise_scale)

        return sample
