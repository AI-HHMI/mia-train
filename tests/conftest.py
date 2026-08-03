from __future__ import annotations

import multiprocessing as mp
import os
import socket
from collections.abc import Callable
from typing import Any

import pytest
import torch.distributed as dist


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(
    rank: int,
    world_size: int,
    port: int,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result_queue: mp.Queue,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
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
        port = _free_tcp_port()
        ctx = mp.get_context("spawn")
        result_queue: mp.Queue = ctx.Queue()
        procs = [
            ctx.Process(
                target=_worker,
                args=(rank, world_size, port, fn, args, kwargs, result_queue),
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

        if errors:
            raise RuntimeError(f"distributed worker(s) failed: {errors}")
        return results

    return _run
