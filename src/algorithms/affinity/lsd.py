"""Local shape descriptors (LSDs) as an auxiliary target beside affinities.

Sheridan et al., *Local shape descriptors for neuron segmentation*, Nature Methods 2023; reference
implementation `lsd.train.local_shape_descriptor` at github.com/funkelab/lsd. For every voxel `v`
with label `i`, the object is intersected with a Gaussian window centred on `v`, and the
intersection is described by its Gaussian-weighted size, the offset of its centre of mass from `v`,
and the covariance of its coordinates. Ten channels in 3D, six in 2D, in **the batch's spatial axis
order** -- x, y, z here, the same convention the affinity offsets follow:

    [0:rank]        mean offset per axis, `offset / sigma * 0.5 + 0.5`
    [rank:2*rank]   variance per axis, `variance / sigma**2`
    [2*rank:-1]     Pearson correlation per axis pair (xy, xz, yz), `pearson * 0.5 + 0.5`
    [-1]            size, the Gaussian-weighted fraction of the window inside the object

Every channel is clipped to [0, 1] and is 0 at background, exactly as the reference normalises
them (`gaussian` mode, kernel truncated at 3 sigma), so a network trained here predicts the same
quantities as one trained with funkelab's pipeline. `tests/unit/test_lsd_targets.py` pins that
against a port of the reference arithmetic and against a brute-force evaluation of the paper's
Eq. 3-7.

**Written against offset kernels, not coordinate grids.** The reference convolves the object mask
with a Gaussian, and separately convolves coordinate volumes `x`, `x*y`, ... masked by the object,
then subtracts `v` afterwards. A Gaussian window is shift-invariant, so the same statistics come
out of convolving the mask alone with `g(u)`, `u*g(u)` and `u*u*g(u)`, where `u` is the offset from
the centre. That is the paper's Eq. 7 evaluated directly; it needs no coordinate volumes; it is
translation invariant, so a crop's position in its volume cannot leak in; and it is numerically
centred, where the reference's `E[x^2] - E[x]^2` in absolute world coordinates cancels
catastrophically for small objects far from the origin.

**Every voxel's statistics are its own object's.** They cannot come out of one convolution over the
label volume, because a window mixes neighbouring objects. The reference loops over labels; this
one-hots the labels present in the crop a chunk at a time, runs the separable passes over the
chunk as a batch, and gathers each voxel's own channel. Cost is therefore proportional to *labels
times voxels*, which is why `downsample` exists: the reference computes on a strided grid and
nearest-upsamples, and so does this, with the gather addressed by the voxel's own fine-resolution
label so that a voxel sitting in a coarse cell of another object still receives its own object's
statistics -- the reference's `descriptor * mask` semantics.

**Each chunk runs over its objects' box, not the crop.** An object's statistics are needed only in
the cells its voxels fall in, and each of those reads at most `kernel_radius` cells around it, so
the passes need only the cells of its bounding box widened by that radius: the same numbers, since
the band matrix of a shorter axis is the leading block of the longer one and every cell kept sees
its whole window. Objects are taken in Morton order of their boxes' centres, so a chunk's members
are neighbours and their union box stays small, and a chunk grows while its objects times its box
stay within `CHUNK_ELEMENTS`. On crops of many small objects -- 2,265 in a 316^3 CREMI crop, 9 s
per crop over the whole volume -- that is most of the cost.

**Each 1-D pass is a banded matrix product, not a convolution.** Measured on a B300 over real NISB
crops (`/nrs/scicompsoft/orhane/mia-train-experiments/lsd_aux_v1/probes/lsd_target_cost`), `conv3d`
with a `(k, 1, 1)` kernel ran at about 3 G elements/s, some 200x under the memory bound, taking a
second per 256^3 sample at downsample 2 and ten at downsample 1: cuDNN has no good kernel for that
shape. Along an axis of length L the same pass is `x @ M` with `M[i, j] = kernel(i - j)`, an
ordinary GEMM that cuBLAS runs near peak; the extra arithmetic (L instead of k multiply-adds per
output) is cheap next to that. It is the same trade the dense heads make for their resampling.

**Where this runs.** On the training device, from the labels as they reach the step, never in a
dataloader worker. Under `defer_image_ops` the geometric augmentation runs on the device *after*
the workers, and it transforms only the image and the labels: a descriptor built in a worker would
arrive unrotated, its offset and covariance channels neither permuted nor sign-flipped, and nothing
would raise. Built here it is correct under either pipeline.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator, Sequence

import numpy as np
import torch

#: Where the truncated Gaussian stops, in standard deviations. scipy's `gaussian_filter` default,
#: which the reference relies on; the kernel radius is `int(TRUNCATE * sigma + 0.5)`.
TRUNCATE = 3.0
#: Variance floor before the Pearson coefficients divide by it, so a one-voxel-thin object yields
#: 0/floor = 0 (a Pearson of 0.5 after normalisation) rather than 0/0. The reference floors at 1e-3
#: in world units squared; here it is in voxels squared on the grid the statistics are computed
#: on, which only differs where a variance is already too small to mean anything.
VARIANCE_FLOOR = 1e-3
#: What the LSD loss does at background (label 0): `ignore` masks it out, the reference's default;
#: `zero` supervises it towards a descriptor of all zeros, which is what the target holds there.
BACKGROUND_MODES = ("ignore", "zero")
#: Elements per `(labels, coarse voxels)` working tensor, so the label chunk shrinks as the crop
#: grows: 2**23 float32 is 32 MiB, and the moments tree below holds up to about 20 of them, so
#: the target costs well under a gigabyte of the training device's memory at any crop size.
CHUNK_ELEMENTS = 1 << 23
#: Spatial ranks the descriptors are defined for.
RANKS = (2, 3)


def lsd_channels(rank: int) -> int:
    """Descriptor channels for `rank` spatial axes: offsets, variances, pairwise Pearsons, size."""
    return 2 * rank + rank * (rank - 1) // 2 + 1


def channel_groups(rank: int) -> dict[str, slice]:
    """Named slices of the channel axis, for a loss that wants to report them apart."""
    pairs = rank * (rank - 1) // 2
    return {
        "offset": slice(0, rank),
        "variance": slice(rank, 2 * rank),
        "pearson": slice(2 * rank, 2 * rank + pairs),
        "size": slice(2 * rank + pairs, 2 * rank + pairs + 1),
    }


def kernel_radius(sigma: float) -> int:
    """Half-width of the truncated Gaussian, in voxels of the grid `sigma` is expressed on."""
    return int(TRUNCATE * sigma + 0.5)


def moment_kernels(
    sigma: float, device: torch.device, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """1-D kernels `g(u)`, `u g(u)`, `u^2 g(u)` over `u in [-r, r]`, with `g` summing to one.

    `g` matches scipy's `gaussian_filter` kernel (normalised, truncated at `TRUNCATE` sigma), so
    the size channel is the same Gaussian-weighted fraction the reference computes. The other two
    weight the offset and its square, which is what turns a convolution of the object mask into
    first and second moments about the centre voxel.
    """
    radius = kernel_radius(sigma)
    if radius < 1:
        raise ValueError(
            f"sigma={sigma} voxels truncates to a kernel of radius {radius}: a window of one voxel "
            "makes every descriptor a constant. Use a larger sigma or a smaller downsample."
        )
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    gaussian = torch.exp(-0.5 * (offsets / sigma) ** 2)
    gaussian = gaussian / gaussian.sum()
    return gaussian, offsets * gaussian, offsets * offsets * gaussian


def _band_matrix(kernel: torch.Tensor, length: int) -> torch.Tensor:
    """`(length, length)` matrix `M` with `M[i, j] = kernel(i - j)`, zero beyond the kernel's reach.

    `x @ M` along an axis is then `out[j] = sum_i kernel(i - j) x[i]`: the cross-correlation that
    `conv*d` computes, so `u g(u)` weights each object voxel by its offset *from* the centre and
    the first moment points from the voxel towards its object's mass. Zero padding falls out of
    the matrix's finite extent.
    """
    radius = kernel.numel() // 2
    index = torch.arange(length, device=kernel.device)
    difference = index[:, None] - index[None, :]
    padded = torch.cat([kernel, kernel.new_zeros(1)])   # index -1 -> the trailing zero
    return padded[torch.where(difference.abs() <= radius, difference + radius, -1)]


@contextlib.contextmanager
def _exact_float32_matmul() -> Iterator[None]:
    """Hold the GEMMs to true float32 while the moments are accumulated.

    A run may allow TF32 (`torch.set_float32_matmul_precision("high")`), whose 10-bit mantissa
    would put ~1e-3 relative error on `E[u^2]` -- of order `r^2 * 1e-3`, a fifth of a voxel
    squared at radius 15 -- and the covariance is that minus `E[u]^2`, so the loss of precision
    lands on the small number that is left. The one-hot is exact in float32, and so are the sums.
    """
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _apply_along(x: torch.Tensor, matrix: torch.Tensor, axis: int) -> torch.Tensor:
    """`(N, *spatial) @ matrix` along spatial axis `axis`.

    Without moving the axis: the last one is a plain `x @ matrix`, any other `matrix.T @ x` over a
    view that puts it in the middle of three. Moving it last instead made every pass copy the whole
    tensor before multiplying, and back after -- twice the memory traffic of the product itself,
    which is what bounds these passes.
    """
    dim = axis + 1
    shape = x.shape
    if dim == x.ndim - 1:
        return (x.reshape(-1, shape[-1]) @ matrix).view(*shape[:-1], matrix.shape[1])
    middle = x.reshape(math.prod(shape[:dim]), shape[dim], math.prod(shape[dim + 1 :]))
    return torch.matmul(matrix.T, middle).view(*shape[:dim], matrix.shape[1], *shape[dim + 1 :])


def _moments(
    onehot: torch.Tensor, matrices: Sequence[Sequence[torch.Tensor]]
) -> dict[tuple[int, ...], torch.Tensor]:
    """Every Gaussian-weighted moment of total degree <= 2, keyed by its per-axis exponents.

    Separable, so it is a tree of 1-D passes: each axis multiplies the running product by one of
    its three banded matrices, and a branch stops growing once its exponents sum to two. Sharing
    the prefixes is what makes it 19 passes rather than the 30 that ten independent separable
    filters would take, and it is exact -- the product of the per-axis kernels *is* the separable
    kernel `u_x^a u_y^b u_z^c g(u)`. Parents are dropped as soon as their children exist, which
    bounds what is live at once.
    """
    rank = len(matrices)
    states: list[tuple[tuple[int, ...], torch.Tensor]] = [((), onehot)]
    with _exact_float32_matmul():
        for axis in range(rank):
            grown: list[tuple[tuple[int, ...], torch.Tensor]] = []
            while states:
                exponents, tensor = states.pop()
                for degree in range(2 - sum(exponents) + 1):
                    matrix = matrices[axis][degree]
                    grown.append(((*exponents, degree), _apply_along(tensor, matrix, axis)))
                del tensor
            states = grown
    return dict(states)


def _normalised_channels(
    moments: dict[tuple[int, ...], torch.Tensor], sigma: Sequence[float]
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Moments about the centre voxel -> the reference's normalised channels, plus the raw count.

    Returned in the channel order the module docstring gives, each in [0, 1]. The count comes
    back separately because a zero count marks a coarse cell the object never reaches, where the
    descriptor is undefined rather than zero.
    """
    rank = len(sigma)
    unit = lambda axis: tuple(1 if a == axis else 0 for a in range(rank))  # noqa: E731

    count = moments.pop((0,) * rank)
    reach = count > 0
    inverse = torch.where(reach, count.clamp_min(torch.finfo(count.dtype).tiny).reciprocal(), 0.0)

    offset = [moments.pop(unit(a)) * inverse for a in range(rank)]
    variance = []
    for a in range(rank):
        second = tuple(2 if b == a else 0 for b in range(rank))
        variance.append(
            (moments.pop(second) * inverse - offset[a] * offset[a]).clamp_min(VARIANCE_FLOOR)
        )
    pearson = []
    for a in range(rank):
        for b in range(a + 1, rank):
            mixed = tuple(1 if c in (a, b) else 0 for c in range(rank))
            covariance = moments.pop(mixed) * inverse - offset[a] * offset[b]
            pearson.append(covariance / torch.sqrt(variance[a] * variance[b]))

    channels = (
        [offset[a] / sigma[a] * 0.5 + 0.5 for a in range(rank)]
        + [variance[a] / sigma[a] ** 2 for a in range(rank)]
        + [p * 0.5 + 0.5 for p in pearson]
        + [count]
    )
    # Zero where the object never reaches the window, so an undefined descriptor reads as the
    # background one rather than as the neutral 0.5 the divisions above leave behind.
    return [c.clamp_(0.0, 1.0).mul_(reach) for c in channels], count


def _object_boxes(
    order: torch.Tensor, counts: torch.Tensor, shape: Sequence[int], downsample: int
) -> torch.Tensor:
    """`(n_ids, 2, rank)`: per dense id, the first and last coarse cell its voxels fall in.

    `order` lists the voxels sorted by id and `counts` how many each id has, so every id is one
    contiguous run and a segment reduction finds its box without atomics. A scatter of every voxel
    onto its id's slot instead serialises on crops of few large objects: 148 of 172 ms per 316^3
    crop of 23 objects on a B300.
    """
    bounds = []
    for axis, extent in enumerate(shape):
        stride = math.prod(shape[axis + 1 :])
        # float32 holds every cell index exactly, and segment_reduce reduces floating types.
        cells = (order // stride % extent // downsample).to(torch.float32)
        bounds.append(torch.stack([
            torch.segment_reduce(cells, "min", lengths=counts),
            torch.segment_reduce(cells, "max", lengths=counts),
        ]))
    return torch.stack(bounds, dim=-1).transpose(0, 1).to(torch.int64)


def _morton_order(points: np.ndarray) -> np.ndarray:
    """Indices that sort integer points `(n, rank)` along a Z-order curve, so neighbours stay near."""
    key = np.zeros(len(points), dtype=np.int64)
    rank = points.shape[1]
    for bit in range(int(points.max(initial=0)).bit_length()):
        for axis in range(rank):
            key |= ((points[:, axis] >> bit) & 1) << (bit * rank + axis)
    return np.argsort(key, kind="stable")


def _chunks(
    boxes: np.ndarray, radius: Sequence[int], shape: Sequence[int]
) -> list[tuple[int, int, tuple[int, ...], tuple[int, ...]]]:
    """Consecutive objects (`boxes` in processing order) grouped into chunks: `(first, stop, lo,
    hi)` per chunk, `[lo, hi)` the cells its passes run over -- the members' union box widened by
    `radius` and clipped to `shape` -- grown while members times cells stay within
    `CHUNK_ELEMENTS`, but never empty."""
    def span(lo, hi):
        lo = tuple(max(a - r, 0) for a, r in zip(lo, radius, strict=True))
        hi = tuple(min(b + 1 + r, n) for b, r, n in zip(hi, radius, shape, strict=True))
        return lo, hi, math.prod(b - a for a, b in zip(lo, hi, strict=True))

    chunks = []
    first, union = 0, None
    for index, (lo, hi) in enumerate(boxes.tolist()):
        grown = (lo, hi) if union is None else (
            [min(a, b) for a, b in zip(union[0], lo)], [max(a, b) for a, b in zip(union[1], hi)]
        )
        if union is not None and (index - first + 1) * span(*grown)[2] > CHUNK_ELEMENTS:
            chunks.append((first, index, *span(*union)[:2]))
            first, grown = index, (lo, hi)
        union = grown
    chunks.append((first, len(boxes), *span(*union)[:2]))
    return chunks


def _descriptors(
    labels: torch.Tensor, sigma: Sequence[float], downsample: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """One label volume -> `(descriptor (C, *spatial), defined (*spatial))`.

    `defined` is true at every foreground voxel whose object reaches its coarse cell's window;
    with `downsample == 1` that is every foreground voxel, since the window always holds the voxel
    itself. Background and negative ids are never descriptors and stay zero and undefined.
    """
    rank = labels.ndim
    device = labels.device
    if any(extent % downsample for extent in labels.shape):
        raise ValueError(
            f"downsample={downsample} does not divide the label crop {tuple(labels.shape)}; the "
            "descriptors are computed on a strided grid that has to tile the crop exactly"
        )
    coarse_sigma = [s / downsample for s in sigma]
    coarse_shape = tuple(extent // downsample for extent in labels.shape)
    radius = [kernel_radius(s) for s in coarse_sigma]
    # Whole-axis band matrices; a chunk's box uses their leading blocks, which are its own.
    matrices = [
        [_band_matrix(kernel, extent) for kernel in moment_kernels(s, device)]
        for s, extent in zip(coarse_sigma, coarse_shape, strict=True)
    ]

    # Dense ids: `unique` sorts, so negatives (ignore) come first, then 0, then the objects.
    ids, dense = torch.unique(labels, return_inverse=True)
    n_objects = int((ids > 0).sum())
    descriptor = torch.zeros(
        (lsd_channels(rank), *labels.shape), device=device, dtype=torch.float32
    )
    defined = torch.zeros(labels.shape, device=device, dtype=torch.bool)
    if n_objects == 0:
        return descriptor, defined
    first_object = ids.numel() - n_objects

    # Objects renumbered 0.. in Morton order of their boxes (everything else -1), so a chunk is a
    # contiguous range of them and its one-hot is a comparison against `arange`.
    flat_dense, order = torch.sort(dense.reshape(-1))
    counts = torch.bincount(flat_dense, minlength=ids.numel())
    boxes = _object_boxes(order, counts, labels.shape, downsample)[first_object:]
    host_boxes = boxes.cpu().numpy()
    order_of_objects = _morton_order(host_boxes.sum(axis=1) // 2)
    position = torch.full((ids.numel(),), -1, dtype=torch.int64, device=device)
    position[first_object + torch.as_tensor(order_of_objects, device=device)] = torch.arange(
        n_objects, device=device
    )
    ranked = position[dense]
    coarse = ranked[tuple(slice(None, None, downsample) for _ in range(rank))]

    # Voxels sorted by object, so that the voxels of one chunk are one contiguous run: the gather
    # below then touches every foreground voxel exactly once over the whole loop, and the loop
    # bounds come from one host round trip rather than one per chunk.
    flat_ranked, order = torch.sort(ranked.reshape(-1))
    bounds = (torch.bincount(flat_ranked + 1, minlength=n_objects + 1).cumsum(0)).tolist()

    for first, stop, lo, hi in _chunks(host_boxes[order_of_objects], radius, coarse_shape):
        box = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
        extent = [b - a for a, b in zip(lo, hi, strict=True)]
        channel_ids = torch.arange(first, stop, device=device).reshape(-1, *(1,) * rank)
        onehot = (coarse[box].unsqueeze(0) == channel_ids).to(torch.float32)
        blocks = [[m[:n, :n] for m in axis] for axis, n in zip(matrices, extent, strict=True)]
        channels, count = _normalised_channels(_moments(onehot, blocks), coarse_sigma)

        start, end = bounds[first], bounds[stop]   # bounds[k]: voxels ranked below k, -1 included
        voxels = order[start:end]
        channel = flat_ranked[start:end] - first
        box_strides = [math.prod(extent[a + 1 :]) for a in range(rank)]
        cell = torch.zeros_like(voxels)
        for axis, index in enumerate(torch.unravel_index(voxels, labels.shape)):
            cell += (index // downsample - lo[axis]) * box_strides[axis]

        gathered = torch.stack([c.reshape(stop - first, -1)[channel, cell] for c in channels])
        descriptor.view(descriptor.shape[0], -1)[:, voxels] = gathered
        defined.view(-1)[voxels] = count.reshape(stop - first, -1)[channel, cell] > 0
    return descriptor, defined


@torch.compiler.disable
def lsd_from_labels(
    labels: torch.Tensor,
    sigma: torch.Tensor | Sequence[float],
    downsample: int = 1,
    ignore_index: int = -1,
    background: str = "ignore",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Instance ids -> `(descriptor target, loss mask)`, `(B, C, *spatial)` float32 and
    `(B, 1, *spatial)` bool.

    `sigma` is per spatial axis **in voxels** of `labels`, either one row for the batch or one per
    sample `(B, rank)` -- a batch drawn from volumes of different voxel size needs the latter,
    since the descriptor is defined in physical units and the caller has divided by each sample's
    voxel size.

    The mask is where the target is knowable and wanted: foreground voxels whose object reaches
    their coarse cell, plus -- under `background="zero"` -- every voxel that is neither an object
    nor `ignore_index`, whose target is all zeros. Broadcast over the channel axis rather than
    expanded, since every channel shares it.

    Not compiled, deliberately. The number of objects in a crop decides the loop count and every
    intermediate shape, so inside a compiled step this would recompile or break the graph on every
    batch; and the work is a handful of GEMMs per label chunk, which inductor would not improve.
    Autocast is disabled inside: a bf16 one-hot loses the second moments' cancellation.
    """
    if labels.ndim < 3:
        raise ValueError(
            "labels must be (B, *spatial) with at least two spatial axes, got "
            f"{tuple(labels.shape)}"
        )
    if background not in BACKGROUND_MODES:
        raise ValueError(f"background must be one of {BACKGROUND_MODES}, got {background!r}")
    if downsample < 1:
        raise ValueError(f"downsample must be at least 1, got {downsample}")
    rank = labels.ndim - 1
    if rank not in RANKS:
        raise ValueError(f"descriptors are defined for 2 or 3 spatial axes, got {rank}")

    sigmas = torch.as_tensor(sigma, dtype=torch.float64).reshape(-1, rank)
    if sigmas.shape[0] not in (1, labels.shape[0]):
        raise ValueError(
            f"sigma has {sigmas.shape[0]} rows for a batch of {labels.shape[0]}; give one row, or "
            "one per sample"
        )
    if not bool((sigmas > 0).all()):
        raise ValueError(f"sigma must be positive on every axis, got {sigmas.tolist()}")
    sigmas = sigmas.expand(labels.shape[0], rank)

    with torch.no_grad(), torch.autocast(labels.device.type, enabled=False):
        per_sample = [
            _descriptors(labels[b], sigmas[b].tolist(), downsample) for b in range(labels.shape[0])
        ]
        target = torch.stack([d for d, _ in per_sample])
        defined = torch.stack([m for _, m in per_sample])
        if background == "zero":
            defined = defined | ((labels <= 0) & (labels != ignore_index))
    return target, defined.unsqueeze(1)
