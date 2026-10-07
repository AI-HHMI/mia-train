"""Dense prediction over a region too large for memory: in blocks, shared by many workers, to disk.

`predict_volume` holds the whole region's float32 sums in memory -- (4C + 4) bytes a voxel, plus
the float16 result, ~40 B/voxel at 6 channels -- which ends at a few tens of gigavoxels: LSD's
zebrafinch benchmark region (478 GVox) would need ~19 TB. Here the same lattice is cut into blocks
of output voxels, and each block runs every tile that meets it and keeps only its own voxels
(`dense.accumulate`). A block's values are therefore *bit-identical* to the same voxels of the
whole-region prediction: the same tiles, added in the same order, divided the same way. The price
is compute -- a tile straddling a block face runs once for every block it meets, about
prod(1 + stride / block) over the axes, 1.27x at 1536^3 blocks with a 256 patch.

**Many workers, no coordination.** Blocks are dealt round-robin -- worker k of n takes blocks k,
k + n, ... -- and each is written straight into its own chunks of one artifact: a block spans whole
chunks on every axis it does not end on, so no two workers ever write the same chunk. A marker,
`<name>.zarr.blocks/<index>.json`, then records the block done. A rerun, after a crash or a
TERM_RUNLIMIT and with any worker count, skips marked blocks; an unmarked block is recomputed
whole, so a worker killed mid-write costs only the time. What the workers share is made once,
whichever starts first: the plan is created and never replaced (`_write_json_once`), and the
empty artifact, attributes and all, is built under a name of the worker's own and renamed into
place, where exactly one rename succeeds (`_publish`).

**The artifact appears only when complete.** Workers write into `<name>.zarr.partial`, and
whichever finishes the last block renames it to `<name>.zarr` -- one atomic rename -- so a scorer
can never take a half-written prediction for a finished one (missing chunks read as zero
affinity: a boundary everywhere). A run that differs from the partial it finds -- another
checkpoint, data config, lattice or block shape -- is refused, by a fingerprint of all of them.
Across nodes, "the last one" can be nobody: an NFS client caches lookups for up to a minute, so
two workers finishing together may each miss the other's final marker. One more run after the
array has ended (any worker count; `deploy/lsf/README.md` submits it as a dependent job) finds
every block done, or does what is missing, and completes the artifact.

**One GPU type per artifact.** The same code gives slightly different predictions on different GPU
generations (H100 against B300: mean |diff| 1.3e-3), so blocks from two of them would meet at faces
that do not agree. Every marker records its device, and completion is refused if they differ.
"""

from __future__ import annotations

import errno
import hashlib
import itertools
import json
import os
import shutil
import socket
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

from .artifact import LEVEL, create_ome_artifact, default_chunks
from .dense import accumulate, blend
from .grid import VolumeGrid

Box = tuple[tuple[int, ...], tuple[int, ...]]


def block_boxes(shape: Sequence[int], block: Sequence[int], chunks: Sequence[int]) -> list[Box]:
    """[low, high) boxes of `block` output voxels tiling `shape`, in C order.

    The last block along an axis is cut to the shape. Every other block ends on a chunk boundary --
    `block` must be a multiple of the chunk wherever it is shorter than the axis -- or two blocks
    would write into one chunk, and concurrent writers would then overwrite each other's voxels.
    """
    if len(block) != len(shape):
        raise SystemExit(f"--block has {len(block)} axes but the lattice has {len(shape)}")
    for axis, (b, c, s) in enumerate(zip(block, chunks, shape, strict=True)):
        if b <= 0:
            raise SystemExit(f"--block {list(block)}: every axis must be positive")
        if b < s and b % c:
            raise SystemExit(
                f"--block {list(block)}: axis {axis} is {b} voxels, which is not a multiple of the "
                f"artifact's {c}-voxel chunks, so two blocks would write into one chunk"
            )
    ranges = [[(low, min(low + b, s)) for low in range(0, s, b)]
              for b, s in zip(block, shape, strict=True)]
    return [(tuple(r[0] for r in combo), tuple(r[1] for r in combo))
            for combo in itertools.product(*ranges)]


def fingerprint(attrs: dict[str, Any], geometry: dict[str, Any], shape: Sequence[int],
                chunks: Sequence[int], block: Sequence[int]) -> str:
    """A digest of everything that decides the artifact's values and layout."""
    record = {"attrs": attrs, "geometry": geometry, "shape": list(shape),
              "chunks": list(chunks), "block": list(block)}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def predict_blocks(
    algorithm: Any,
    grid: VolumeGrid,
    device: torch.device,
    *,
    path: Path,
    attrs: dict[str, Any],
    geometry: dict[str, Any],
    block: Sequence[int],
    worker: int = 0,
    workers: int = 1,
    chunks: tuple[int, ...] | None = None,
) -> Path | None:
    """Predict worker `worker` of `workers`'s share of `grid`'s blocks into the artifact at `path`.

    `attrs` and `geometry` are what `write_ome_artifact` would be given for the whole-region
    prediction. Returns `path` once the artifact is complete -- this worker finished the last
    block, or another already had -- and None while blocks remain.
    """
    if not 0 <= worker < workers:
        raise SystemExit(f"--worker {worker} is not one of --workers {workers} (0-based)")
    path = Path(path)
    if path.exists():
        print(f"{path} is complete already", flush=True)
        return path
    partial = path.with_name(path.name + ".partial")
    markers = path.with_name(path.name + ".blocks")

    shape = (int(algorithm.prediction_channels), *grid.output_shape)
    chunks = default_chunks(shape) if chunks is None else tuple(chunks)
    boxes = block_boxes(grid.output_shape, block, chunks[1:])
    plan = {"fingerprint": fingerprint(attrs, geometry, shape, chunks, block),
            "block": list(block), "blocks": len(boxes)}
    markers.mkdir(parents=True, exist_ok=True)
    if not (markers / "plan.json").exists():
        _write_json_once(markers / "plan.json", plan)
    found = json.loads((markers / "plan.json").read_text())
    if found != plan:
        raise SystemExit(
            f"{partial} is being written by a different prediction (its plan is {found}, this "
            f"run's is {plan}): another checkpoint, data config, lattice or block shape. Rerun "
            f"with the settings that made it, or delete {partial} and {markers} to start over."
        )
    # Every worker tries to publish, even when the partial is there already. Asking first
    # (`partial.exists()`) is what NFS remembers: a worker told "missing" that then loses the race
    # still cannot open the partial the rename found -- every worker on one node failed that way in
    # mia-evals' copy of this code under a stress test (2026-10-05). Seen before publishing, these
    # markers cannot be for a partial this worker makes.
    stale = sorted(p.name for p in markers.glob("[0-9]*.json"))
    made = _publish(partial, lambda where: create_ome_artifact(
        where, shape, np.float16, chunks=chunks, attrs=attrs, name=path.name, **geometry))
    if made and stale:
        # Markers without the partial they describe: completing now would leave those blocks'
        # chunks unwritten, i.e. zero, in an artifact that looks finished.
        shutil.rmtree(partial)
        if path.exists():                      # completed while this worker was starting up
            return path
        raise SystemExit(
            f"{markers} marks {len(stale)} block(s) done, but {partial} was missing and has "
            f"just been made empty. Delete {markers} too, then rerun."
        )
    level = zarr.open_array(str(partial / LEVEL), mode="r+")   # this worker's, another's, a rerun's

    todo = [i for i in range(worker, len(boxes), workers)
            if not (markers / f"{i}.json").exists()]
    print(f"{path.name}: {len(boxes)} blocks of {list(block)} over the {list(grid.output_shape)} "
          f"lattice; worker {worker} of {workers} has {len(todo)} to do", flush=True)
    handle = grid.image_handle()
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
    for index in todo:
        if path.exists():                      # other workers, rerun alongside, have finished it
            break
        low, high = boxes[index]
        started = time.perf_counter()
        total, weight, tiles = accumulate(
            algorithm, grid, device, low, tuple(h - lo for lo, h in zip(low, high, strict=True)),
            handle=handle, progress=False,
        )
        values = blend(total, weight)
        del total, weight
        level[(slice(None), *(slice(lo, h) for lo, h in zip(low, high, strict=True)))] = values
        del values
        seconds = time.perf_counter() - started
        _write_json(markers / f"{index}.json", {
            "index": index, "low": list(low), "high": list(high), "tiles": tiles,
            "seconds": round(seconds, 1), "device": device_name, "host": socket.gethostname(),
            "worker": worker, "workers": workers,
        })
        print(f"  block {index} [{list(low)}, {list(high)}): {tiles} tiles, {seconds:.0f} s",
              flush=True)
    try:
        return _complete(partial, path, markers, plan)
    except FileNotFoundError:       # another worker finished at the same moment and completed it
        if path.exists():
            return path
        raise


def _publish(final: Path, build: Callable[[Path], Any]) -> bool:
    """Build directory `final` under a name of this call's own and rename it into place, unless
    another worker has published it first. True if this call did.

    Workers started together all find no partial, and making it in place would race: zarr's
    open-or-create looks, then creates, so a worker between the two raises ContainsGroupError (one
    of nine did in mia-evals' copy of this pattern, 2026-10-05), and two that both looked first
    both create -- the slower overwriting the attributes the faster one had already written. A
    rename onto a directory that exists and is not empty fails, so exactly one worker's copy lands,
    whole, and every other worker discards its own and uses that one.
    """
    temporary = final.with_name(f".{final.name}.{socket.gethostname()}.{os.getpid()}."
                                f"{uuid.uuid4().hex[:8]}")
    try:
        build(temporary)
        try:
            os.rename(temporary, final)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            return False
        return True
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _complete(partial: Path, path: Path, markers: Path, plan: dict[str, Any]) -> Path | None:
    """Rename the partial into place if every block is done; otherwise say how far it has got."""
    if path.exists():
        return path
    count = int(plan["blocks"])
    missing = [i for i in range(count) if not (markers / f"{i}.json").exists()]
    if missing:
        print(f"{count - len(missing)} of {count} blocks done; {path.name} is completed by "
              "whichever worker finishes the last one, or by rerunning any worker", flush=True)
        return None
    records = [json.loads((markers / f"{i}.json").read_text()) for i in range(count)]
    devices = sorted({record["device"] for record in records})
    if len(devices) > 1:
        by_device = {d: [r["index"] for r in records if r["device"] == d] for d in devices}
        raise SystemExit(
            f"{path.name}'s blocks were predicted on {len(devices)} GPU types {by_device}. "
            "Predictions differ across GPU generations, so blocks from two would disagree at "
            f"their faces. Delete the markers of one type's blocks from {markers} and rerun on "
            "the other's queue."
        )
    summary = {
        "block": plan["block"], "blocks": count, "device": devices[0],
        "tiles": sum(r["tiles"] for r in records),
        "seconds": round(sum(r["seconds"] for r in records), 1),
    }
    group = zarr.open_group(str(partial), mode="r+")
    group.attrs.update(blockwise=summary)
    group[LEVEL].attrs.update(blockwise=summary)
    os.rename(partial, path)
    shutil.rmtree(markers)
    print(f"completed {path}: {count} blocks, {summary['tiles']} tiles, "
          f"{summary['seconds'] / 3600:.2f} GPU-hours on {devices[0]}", flush=True)
    return path


def _write_json(path: Path, record: dict[str, Any]) -> None:
    """Write-then-rename, so a reader sees the whole record or none of it."""
    temporary = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}")
    temporary.write_text(json.dumps(record, indent=1))
    os.replace(temporary, path)


def _write_json_once(path: Path, record: dict[str, Any]) -> None:
    """Create `path` holding `record` unless it exists; a file already there is never replaced.

    A hard link to a finished file either creates `path` or fails because it is there. Replacing
    it instead, as `_write_json` does, would swap the file under a worker on another node that is
    reading it, and NFS answers that read with ESTALE -- which workers started together, all
    writing the plan, did (2026-10-05).
    """
    temporary = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}."
                               f"{uuid.uuid4().hex[:8]}")
    temporary.write_text(json.dumps(record, indent=1))
    try:
        os.link(temporary, path)
    except FileExistsError:
        pass
    finally:
        temporary.unlink()
