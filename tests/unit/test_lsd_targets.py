"""The local shape descriptor target: the paper's definition, the reference's normalisation.

Three independent oracles, because each catches a different way of being wrong while looking
right:

  * a brute-force evaluation of Eq. 3-7 of the paper, one voxel at a time, for the *definition*;
  * a port of funkelab's `LsdExtractor` arithmetic (coordinate grids, `E[x^2] - E[x]^2`,
    strided downsample and nearest upsample), for the *conventions* -- the 0.5 offsets, the
    sigma normalisation, the variance floor, the background zeroing, the clipping;
  * and equivariance under axis permutations and flips, for the *axis order* of the channels --
    the failure this repo's affinity task has already had once, which trains perfectly well and
    scores as nonsense.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

import algorithms.affinity.lsd as lsd_module
from algorithms.affinity.lsd import (
    TRUNCATE,
    VARIANCE_FLOOR,
    channel_groups,
    kernel_radius,
    lsd_channels,
    lsd_from_labels,
    moment_kernels,
)

# ------------------------------------------------------------------------------ fixtures


def _blobby_labels(shape: tuple[int, ...], ids: int, block: int, seed: int) -> torch.Tensor:
    """Objects several voxels across: a coarse random field repeated `block` times per axis.

    Salt-and-pepper labels would make every variance sit on the floor, where the two floor
    conventions differ and nothing else is being tested.
    """
    generator = torch.Generator().manual_seed(seed)
    coarse = tuple(-(-extent // block) for extent in shape)
    field = torch.randint(0, ids, coarse, generator=generator)
    for axis in range(len(shape)):
        field = field.repeat_interleave(block, dim=axis)
    return field[tuple(slice(0, extent) for extent in shape)]


def _gaussian_1d(sigma: float) -> np.ndarray:
    radius = kernel_radius(sigma)
    u = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-0.5 * (u / sigma) ** 2)
    return g / g.sum()


# ------------------------------------------------------------------------------ oracles


def brute_force(labels: np.ndarray, sigma: list[float]) -> np.ndarray:
    """Eq. 3-7, evaluated per voxel over its truncated Gaussian window, then normalised."""
    rank = labels.ndim
    radii = [kernel_radius(s) for s in sigma]
    kernels = [_gaussian_1d(s) for s in sigma]
    offsets = np.meshgrid(*[np.arange(-r, r + 1) for r in radii], indexing="ij")
    weight = np.ones_like(offsets[0], dtype=np.float64)
    for axis in range(rank):
        weight = weight * kernels[axis][offsets[axis] + radii[axis]]
    padded = np.pad(labels, [(r, r) for r in radii], constant_values=-1_000_000)

    out = np.zeros((lsd_channels(rank), *labels.shape), dtype=np.float64)
    for voxel in itertools.product(*[range(e) for e in labels.shape]):
        label = labels[voxel]
        if label <= 0:
            continue
        window = padded[tuple(slice(v, v + 2 * r + 1) for v, r in zip(voxel, radii, strict=True))]
        inside = (window == label) * weight
        count = inside.sum()
        mean = np.array([(inside * offsets[a]).sum() for a in range(rank)]) / count
        second = np.array(
            [[(inside * offsets[a] * offsets[b]).sum() for b in range(rank)] for a in range(rank)]
        ) / count
        covariance = second - np.outer(mean, mean)
        variance = np.maximum(np.diag(covariance), VARIANCE_FLOOR)
        pearson = [
            covariance[a, b] / np.sqrt(variance[a] * variance[b])
            for a in range(rank) for b in range(a + 1, rank)
        ]
        channels = (
            [mean[a] / sigma[a] * 0.5 + 0.5 for a in range(rank)]
            + [variance[a] / sigma[a] ** 2 for a in range(rank)]
            + [p * 0.5 + 0.5 for p in pearson]
            + [count]
        )
        out[(slice(None), *voxel)] = np.clip(channels, 0.0, 1.0)
    return out


def reference_port(
    segmentation: np.ndarray, sigma_world: list[float], voxel_size: list[float], downsample: int = 1
) -> np.ndarray:
    """funkelab's `LsdExtractor.get_descriptors` (3D, `gaussian` mode, all components), verbatim
    in structure: absolute coordinate grids in world units, one label at a time, the reference's
    normalisation and its background handling. In float64 so that only the conventions differ.
    """
    from scipy.ndimage import gaussian_filter

    df = downsample
    shape = segmentation.shape
    sub_shape = tuple(s // df for s in shape)
    sub_voxel = tuple(v * df for v in voxel_size)
    sub_sigma_voxel = tuple(s / v for s, v in zip(sigma_world, sub_voxel, strict=True))
    coords = np.array(
        np.meshgrid(
            *[np.arange(0, sub_shape[d] * sub_voxel[d], sub_voxel[d]) for d in range(3)],
            indexing="ij",
        ),
        dtype=np.float64,
    )

    def aggregate(array: np.ndarray) -> np.ndarray:
        return gaussian_filter(array, sigma=sub_sigma_voxel, mode="constant", cval=0.0,
                               truncate=TRUNCATE)

    descriptors = np.zeros((10, *shape), dtype=np.float64)
    for label in np.unique(segmentation):
        if label == 0:
            continue
        mask = (segmentation == label).astype(np.float64)
        sub_mask = mask[::df, ::df, ::df]
        masked_coords = coords * sub_mask
        count = aggregate(sub_mask)
        count[count == 0] = 1
        mean = np.array([aggregate(masked_coords[d]) for d in range(3)]) / count
        mean_offset = mean - coords
        outer = np.einsum("i...,j...->ij...", masked_coords, masked_coords).reshape((9, *sub_shape))
        entries = [0, 4, 8, 1, 2, 5]
        covariance = np.array([aggregate(outer[d]) for d in entries]) / count
        covariance -= np.einsum("i...,j...->ij...", mean, mean).reshape((9, *sub_shape))[entries]
        variance, pearson = covariance[[0, 1, 2]], covariance[[3, 4, 5]]
        variance[variance < 1e-3] = 1e-3
        pearson[0] /= np.sqrt(variance[0] * variance[1])
        pearson[1] /= np.sqrt(variance[0] * variance[2])
        pearson[2] /= np.sqrt(variance[1] * variance[2])
        for d in range(3):
            variance[d] /= sigma_world[d] ** 2
        sub = np.concatenate([mean_offset, variance, pearson, count[None]])
        for axis in range(3):
            sub = np.repeat(sub, df, axis=axis + 1)
        descriptors += sub * mask
    max_distance = np.asarray(sigma_world, dtype=np.float64)[:, None, None, None]
    descriptors[[0, 1, 2]] = descriptors[[0, 1, 2]] / max_distance * 0.5 + 0.5
    descriptors[[6, 7, 8]] = descriptors[[6, 7, 8]] * 0.5 + 0.5
    descriptors[[0, 1, 2, 6, 7, 8]] *= segmentation != 0
    return np.clip(descriptors, 0.0, 1.0)


# ------------------------------------------------------------------------------ definition


@pytest.mark.unit
def test_channel_layout():
    assert lsd_channels(3) == 10 and lsd_channels(2) == 6
    groups = channel_groups(3)
    assert [groups[k] for k in ("offset", "variance", "pearson", "size")] == [
        slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 10),
    ]


@pytest.mark.unit
def test_kernels_are_scipys_truncated_gaussian_and_its_moments():
    g, ug, uug = moment_kernels(2.0, torch.device("cpu"), torch.float64)
    expected = _gaussian_1d(2.0)
    assert g.numel() == 2 * kernel_radius(2.0) + 1 == 13
    assert np.allclose(g.numpy(), expected)
    u = np.arange(-6, 7)
    assert np.allclose(ug.numpy(), u * expected) and np.allclose(uug.numpy(), u * u * expected)


@pytest.mark.unit
def test_matches_a_brute_force_evaluation_of_the_definition():
    labels = _blobby_labels((9, 8, 7), ids=4, block=3, seed=0)
    sigma = [1.5, 2.0, 1.2]
    target, mask = lsd_from_labels(labels[None], sigma)
    expected = brute_force(labels.numpy(), sigma)
    assert target.shape == (1, 10, 9, 8, 7) and mask.shape == (1, 1, 9, 8, 7)
    assert torch.equal(mask[0, 0], labels > 0)
    worst = np.abs(target[0].numpy() - expected).max()
    assert np.allclose(target[0].numpy(), expected, atol=1e-5), worst


@pytest.mark.unit
def test_2d_labels_get_six_channels_matching_the_definition():
    labels = _blobby_labels((11, 9), ids=3, block=3, seed=1)
    target, mask = lsd_from_labels(labels[None], [2.0, 1.5])
    assert target.shape == (1, 6, 11, 9)
    assert np.allclose(target[0].numpy(), brute_force(labels.numpy(), [2.0, 1.5]), atol=1e-5)


@pytest.mark.unit
def test_interior_of_a_large_object_reads_as_the_reference_constants():
    """Deep inside an object: zero offset (0.5), full size (1), Pearson 0 (0.5), and a variance of
    the truncated Gaussian's own, just under sigma squared."""
    target, _ = lsd_from_labels(torch.ones(1, 32, 32, 32, dtype=torch.long), [3.0, 3.0, 3.0])
    centre = target[0, :, 16, 16, 16]
    assert torch.allclose(centre[0:3], torch.full((3,), 0.5), atol=1e-6)
    assert torch.allclose(centre[6:9], torch.full((3,), 0.5), atol=1e-6)
    assert centre[9] == pytest.approx(1.0, abs=1e-6)
    g = _gaussian_1d(3.0)
    truncated_variance = float((g * np.arange(-9, 10) ** 2).sum()) / 9.0
    assert torch.allclose(centre[3:6], torch.full((3,), truncated_variance), atol=1e-6)
    assert 0.95 < truncated_variance < 1.0


@pytest.mark.unit
def test_offset_points_from_a_face_voxel_into_the_object():
    """On the x-low face of a slab the offset along x is positive (> 0.5); on the x-high face it is
    negative (< 0.5); along y and z, with the slab uniform, it is zero (0.5)."""
    labels = torch.zeros(1, 24, 16, 16, dtype=torch.long)
    labels[:, 6:18] = 1
    target, _ = lsd_from_labels(labels, [2.0, 2.0, 2.0])
    low, high = target[0, :, 6, 8, 8], target[0, :, 17, 8, 8]
    assert low[0] > 0.6 and high[0] < 0.4
    assert torch.allclose(low[1:3], torch.tensor([0.5, 0.5]), atol=1e-6)
    assert low[9] < 0.7  # half the window hangs outside the slab


# ------------------------------------------------------------------------------ reference


@pytest.mark.unit
@pytest.mark.parametrize("downsample", [1, 2])
def test_matches_a_port_of_the_funkelab_reference(downsample):
    """Same numbers as `lsd.train.local_shape_descriptor`, given the paper's parameters in world
    units on anisotropic voxels -- including its strided downsample and nearest upsample."""
    pytest.importorskip("scipy")
    labels = _blobby_labels((16, 12, 10), ids=5, block=4, seed=2)
    voxel_size, sigma_world = [9.0, 9.0, 20.0], [30.0, 30.0, 30.0]
    sigma_voxels = [s / v for s, v in zip(sigma_world, voxel_size, strict=True)]

    target, mask = lsd_from_labels(labels[None], sigma_voxels, downsample=downsample)
    expected = reference_port(labels.numpy(), sigma_world, voxel_size, downsample)

    # Every foreground voxel of these blocky objects reaches its coarse cell's window, so the
    # descriptor is defined everywhere the reference defines one.
    assert torch.equal(mask[0, 0], labels > 0)

    # The one convention that differs is the variance floor: the reference's is 1e-3 in world
    # units squared, which on nanometre-scaled data never triggers, this module's is 1e-3 voxels
    # squared on the computation grid. They part ways only where an object is one coarse voxel
    # thick along some axis -- here the two-voxel tail block the crop's odd extents leave along z
    # under downsample 2 -- and there the reference's Pearson is a ratio of two numbers at
    # rounding level. Those voxels are excluded and counted, and everything else must agree.
    groups = channel_groups(3)
    broadcast = (slice(None), None, None, None)
    variance_world = expected[groups["variance"]] * np.asarray(sigma_world)[broadcast] ** 2
    floor_world = VARIANCE_FLOOR * (np.asarray(voxel_size) * downsample)[broadcast] ** 2
    degenerate = (variance_world < floor_world).any(axis=0) & (labels.numpy() > 0)
    assert degenerate.mean() < 0.05
    difference = np.abs(target[0].numpy() - expected)[:, ~degenerate]
    assert difference.max() < 2e-5, difference.max()


# ------------------------------------------------------------------------------ axis order


def _transform(field: torch.Tensor, perm: tuple[int, ...], flips: tuple[bool, ...]) -> torch.Tensor:
    """Permute and flip the spatial axes of a `(*spatial)` or `(C, *spatial)` tensor."""
    lead = field.ndim - len(perm)
    out = field.permute(*range(lead), *[lead + p for p in perm])
    axes = [lead + a for a, flip in enumerate(flips) if flip]
    return torch.flip(out, axes) if axes else out


@pytest.mark.unit
@pytest.mark.parametrize("perm", list(itertools.permutations(range(3))))
@pytest.mark.parametrize(
    "flips", [(False, False, False), (True, False, False), (False, True, True)]
)
def test_channels_follow_the_axes_under_permutation_and_flip(perm, flips):
    """`lsd(T(labels)) == T'(lsd(labels))`, where T' moves the spatial axes as T does and also
    re-indexes the channels: offsets and variances permute with the axes, a flipped axis negates
    its offset (`v -> 1 - v`), a Pearson pair permutes with its two axes and negates when exactly
    one of them flips, and size is invariant. A wrong channel order fails this and nothing else.
    """
    labels = _blobby_labels((10, 10, 10), ids=4, block=3, seed=3)
    sigma = [1.6, 1.6, 1.6]
    rank = 3
    original, _ = lsd_from_labels(labels[None], sigma)
    transformed, _ = lsd_from_labels(_transform(labels, perm, flips)[None], sigma)

    moved = _transform(original[0], perm, flips)          # spatial axes moved, channels not yet
    foreground = _transform(labels, perm, flips) > 0
    groups = channel_groups(rank)
    expected = torch.zeros_like(moved)
    pairs = [(a, b) for a in range(rank) for b in range(a + 1, rank)]
    for new_axis in range(rank):
        old_axis = perm[new_axis]
        offset = moved[groups["offset"].start + old_axis]
        expected[groups["offset"].start + new_axis] = torch.where(
            foreground, 1.0 - offset if flips[new_axis] else offset, 0.0
        )
        expected[groups["variance"].start + new_axis] = moved[groups["variance"].start + old_axis]
    for index, (a, b) in enumerate(pairs):
        old_pair = tuple(sorted((perm[a], perm[b])))
        pearson = moved[groups["pearson"].start + pairs.index(old_pair)]
        expected[groups["pearson"].start + index] = torch.where(
            foreground, 1.0 - pearson if flips[a] != flips[b] else pearson, 0.0
        )
    expected[groups["size"]] = moved[groups["size"]]
    assert torch.allclose(transformed[0], expected, atol=1e-5)


# ------------------------------------------------------------------------------ masks


@pytest.mark.unit
def test_background_and_ignore_are_zero_and_masked_out_by_default():
    labels = _blobby_labels((8, 8, 8), ids=3, block=2, seed=4)
    labels[0, :2] = -1
    target, mask = lsd_from_labels(labels[None], [1.5] * 3)
    assert not target[0][:, labels <= 0].any()
    assert torch.equal(mask[0, 0], labels > 0)


@pytest.mark.unit
def test_background_zero_supervises_background_but_never_ignore():
    labels = _blobby_labels((8, 8, 8), ids=3, block=2, seed=4)
    labels[0, :2] = -1
    labels[7, :2] = -7  # negative but not the ignore id: background-like, as in the affinity target
    target, mask = lsd_from_labels(labels[None], [1.5] * 3, background="zero")
    assert torch.equal(mask[0, 0], labels != -1)
    assert not target[0][:, labels <= 0].any()


@pytest.mark.unit
def test_a_negative_id_is_not_an_object():
    """Ignore voxels next to an object must not pull its centre of mass or count as its size."""
    plain = torch.zeros(1, 12, 6, 6, dtype=torch.long)
    plain[:, 3:9] = 1
    with_ignore = plain.clone()
    with_ignore[:, 9:] = -1
    a, _ = lsd_from_labels(plain, [1.5] * 3)
    b, _ = lsd_from_labels(with_ignore, [1.5] * 3)
    assert torch.equal(a[:, :, 3:9], b[:, :, 3:9])


@pytest.mark.unit
def test_downsample_leaves_a_vanished_object_undefined():
    """An object the strided grid skips has no coarse statistics; the reference silently gives it
    garbage, this marks it out of the mask instead."""
    labels = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    labels[0, 1, 1, 1] = 1          # odd coordinates: absent from the ::2 grid
    labels[0, 4:8, 4:8, 4:8] = 2    # present on it
    target, mask = lsd_from_labels(labels, [2.0] * 3, downsample=2)
    assert not mask[0, 0, 1, 1, 1] and not target[0, :, 1, 1, 1].any()
    assert mask[0, 0, 4:8, 4:8, 4:8].all()


@pytest.mark.unit
def test_per_sample_sigma_rows_are_honoured():
    labels = torch.stack([_blobby_labels((8, 8, 8), 3, 2, seed=5)] * 2)
    both, _ = lsd_from_labels(labels, torch.tensor([[1.5, 1.5, 1.5], [2.5, 2.5, 2.5]]))
    first, _ = lsd_from_labels(labels[:1], [1.5, 1.5, 1.5])
    second, _ = lsd_from_labels(labels[1:], [2.5, 2.5, 2.5])
    assert torch.equal(both[0], first[0]) and torch.equal(both[1], second[0])
    assert not torch.equal(both[0], both[1])


@pytest.mark.unit
def test_chunking_does_not_change_the_result(monkeypatch: pytest.MonkeyPatch):
    labels = _blobby_labels((10, 9, 8), ids=7, block=2, seed=6)
    whole, _ = lsd_from_labels(labels[None], [1.5] * 3)
    monkeypatch.setattr(lsd_module, "CHUNK_ELEMENTS", 1)  # one label per chunk
    piecewise, _ = lsd_from_labels(labels[None], [1.5] * 3)
    # Not `equal`: a convolution over a batch of one picks a different kernel than one over seven,
    # and they round differently at the 1e-7 level. Anything larger is a chunking error.
    assert torch.allclose(whole, piecewise, atol=1e-6, rtol=0)


@pytest.mark.unit
def test_runs_in_float32_under_autocast():
    labels = _blobby_labels((8, 8, 8), ids=3, block=2, seed=7)
    plain, _ = lsd_from_labels(labels[None], [1.5] * 3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        under, _ = lsd_from_labels(labels[None], [1.5] * 3)
    assert under.dtype == torch.float32 and torch.equal(plain, under)


# ------------------------------------------------------------------------------ contracts


@pytest.mark.unit
def test_rejects_a_downsample_that_does_not_tile_the_crop():
    with pytest.raises(ValueError, match="does not divide"):
        lsd_from_labels(torch.ones(1, 9, 8, 8, dtype=torch.long), [2.0] * 3, downsample=2)


@pytest.mark.unit
def test_rejects_a_sigma_whose_kernel_is_one_voxel():
    with pytest.raises(ValueError, match="radius"):
        lsd_from_labels(torch.ones(1, 8, 8, 8, dtype=torch.long), [0.1, 2.0, 2.0])


@pytest.mark.unit
def test_rejects_bad_arguments():
    labels = torch.ones(2, 8, 8, 8, dtype=torch.long)
    with pytest.raises(ValueError, match="background"):
        lsd_from_labels(labels, [2.0] * 3, background="mask")
    with pytest.raises(ValueError, match="rows"):
        lsd_from_labels(labels, torch.ones(3, 3))
    with pytest.raises(ValueError, match="positive"):
        lsd_from_labels(labels, [2.0, -1.0, 2.0])
    with pytest.raises(ValueError, match="spatial"):
        lsd_from_labels(torch.ones(2, 8, dtype=torch.long), [2.0])
