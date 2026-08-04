from __future__ import annotations

import multiprocessing as mp
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch.distributed as dist


def _worker(
    rank: int,
    world_size: int,
    store_path: str,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result_queue: mp.Queue,
) -> None:
    # File rendezvous rather than a TCP port. Picking a "free" port means binding it, reading
    # the number, closing the socket and hoping nothing else takes it before the workers bind —
    # a race that surfaces as EADDRINUSE on a busy shared node. All ranks are on one host, so a
    # node-local file needs no port at all.
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        result_queue.put((rank, fn(rank, world_size, *args, **kwargs), None))
    except Exception as exc:  # forward worker failures to the parent process
        result_queue.put((rank, None, repr(exc)))
    finally:
        dist.destroy_process_group()


@pytest.fixture
def run_distributed() -> Callable[..., list[Any]]:
    """Run fn(rank, world_size, *args, **kwargs) across world_size Gloo CPU processes."""

    def _run(
        fn: Callable[..., Any],
        world_size: int = 2,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        timeout: float = 60.0,
    ) -> list[Any]:
        kwargs = kwargs or {}
        ctx = mp.get_context("spawn")
        result_queue: mp.Queue = ctx.Queue()
        # torch creates the rendezvous file itself, so hand it a fresh path in an empty dir.
        store_dir = tempfile.mkdtemp(prefix="mia_rendezvous_")
        store_path = str(Path(store_dir) / "store")
        try:
            procs = [
                ctx.Process(
                    target=_worker,
                    args=(rank, world_size, store_path, fn, args, kwargs, result_queue),
                )
                for rank in range(world_size)
            ]
            for p in procs:
                p.start()

            results: list[Any] = [None] * world_size
            errors: list[tuple[int, str]] = []
            for _ in range(world_size):
                rank, result, error = result_queue.get(timeout=timeout)
                if error is not None:
                    errors.append((rank, error))
                results[rank] = result

            for p in procs:
                p.join(timeout=timeout)
        finally:
            shutil.rmtree(store_dir, ignore_errors=True)

        if errors:
            raise RuntimeError(f"distributed worker(s) failed: {errors}")
        return results

    return _run


def _write_ome_ngff_zarr(
    root_path: Path,
    group_key: str,
    base_shape: tuple[int, ...],
    num_scales: int,
    dtype: str = "float32",
) -> None:
    """Write a minimal OME-NGFF v0.4 multiscale group, each level downsampled 2x.

    Adapted from miao's own test scaffolding: miao reads real zarr through tensorstore, so
    exercising the dataset wrapper needs an actual container rather than a mock.
    """
    import json

    import numpy as np
    import zarr
    from zarr.storage import LocalStore

    root = zarr.open_group(LocalStore(str(root_path)), mode="a", zarr_format=2)
    group = root
    for part in group_key.split("/"):
        group = group.create_group(part, overwrite=False) if part not in group else group[part]

    datasets = []
    for level in range(num_scales):
        factor = 2**level
        level_shape = tuple(size // factor for size in base_shape)
        array = group.create_array(
            str(level),
            shape=level_shape,
            chunks=tuple(min(32, size) for size in level_shape),
            dtype=dtype,
            overwrite=True,
        )
        array[:] = np.random.RandomState(42 + level).rand(*level_shape).astype(dtype)
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [{"type": "scale", "scale": [float(factor)] * 3}],
            }
        )

    attrs_path = root_path / group_key / ".zattrs"
    attrs = json.loads(attrs_path.read_text()) if attrs_path.exists() else {}
    attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [{"name": n, "type": "space", "unit": "micrometer"} for n in ("z", "y", "x")],
            "datasets": datasets,
        }
    ]
    attrs_path.write_text(json.dumps(attrs))


@pytest.fixture
def ome_zarr_volume(tmp_path: Path) -> Path:
    """A 64^3 three-level OME-NGFF volume under the group key "raw"."""
    path = tmp_path / "volume.zarr"
    _write_ome_ngff_zarr(path, group_key="raw", base_shape=(64, 64, 64), num_scales=3)
    return path
