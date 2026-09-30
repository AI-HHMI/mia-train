#!/usr/bin/env python
"""How large an input volume can a model train on per B300 node, and at what throughput?

Copied 2026-09-29 from `experiments/b300_capability_run/capability_sweep.py` (the 7B DINOv3 run) so
this experiment can change it as it moves to other architectures. Changes so far: `--decode-chunks`
(an explicit count, or `auto` = the fewest slabs that keep each sub-pixel-head slab under cuDNN's
2^31-element limit, so one config serves every crop size), and the record carries the chunk count,
the readout width, the encoder's width and depth, and its attention layout (`attn_window`,
`attn_window_mode`, `attn_global_blocks`, `attn_global_kv_pool`). It also builds the plain `vit3d`
as well as `dinov3_vit3d`: a per-axis `img_size` / `patch_size`, no `device` argument, no prefix
tokens. And it builds `convnet3d`, whose patch size is the stride its stages add up to, so the
slab count is read off the built model rather than the config, and whose record carries its block,
widths and depths. With `decoder = "unet"`, `--decode-chunks auto` sizes the slabs by that head's
largest tensor instead of the sub-pixel head's.

One `torchrun` entry point that measures **one** crop size: it builds the model and algorithm named
in an ordinary mia-train config, parallelizes them exactly as `engine.trainer` does, and runs a
handful of real training steps -- forward, backward, grad clip, optimizer -- recording peak memory
and step time. `submit.sh` walks the sizes; the largest one that exits zero is the answer.

One size per process rather than a loop inside it, because a CUDA out-of-memory inside a collective
leaves the other ranks waiting rather than raising, so there is nothing reliable to catch. Letting
the process die and having the caller stop is both simpler and more honest about what happened.

Three things about the method, because each is a way to measure the wrong thing:

**`[data]` is not read.** The batch is synthetic and already on the device, so nothing here measures
the input pipeline. That cost is real but it is a separate, already-characterised problem, and
folding it in would make the memory frontier depend on how many dataloader workers happened to keep
up. Everything downstream of the batch *is* measured, including the affinity target construction,
which allocates several tensors per voxel and turns out to be where much of the memory goes.

**`split_disconnected` is forced off.** The connected-components pass runs in dataloader workers in
a real run (`AffinitySegmentation.sample_transform`), so it is off the step's critical path by
design; leaving it on would put a CPU pass in the middle of a GPU measurement.

**`img_size` comes from `--size`, not from the config.** That is the swept variable.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import components  # noqa: E402,F401  (populates the registries, as a real run does)
from algorithms.registry import AlgorithmRegistry  # noqa: E402
from distributed.grad_norm import clip_grad_norm_  # noqa: E402
from distributed.parallel_dims import ParallelDims  # noqa: E402
from distributed.parallelize import (  # noqa: E402
    apply_tensor_parallel,
    shard_algorithm,
)
from distributed.setup import destroy_distributed, device_type  # noqa: E402
from engine.activation_checkpoint import apply_activation_checkpointing  # noqa: E402
from engine.optimizer import build_optimizer, optimizer_step  # noqa: E402
from models.registry import ModelRegistry  # noqa: E402
from utils.config import load_run_config  # noqa: E402
from utils.hardware_flops import peak_flops  # noqa: E402

#: Side of a constant-instance-id block in the synthetic label field, in voxels. With
#: `split_disconnected` off nothing in the step is data-dependent, so the label's content changes
#: the reported accuracies and nothing else -- but blocks this size land the positive rate near
#: NISB's ~83%, which keeps the logged metrics recognisable rather than degenerate.
LABEL_BLOCK = 32

#: Collective timeout. NCCL defaults to 10 minutes and a step at the top of this sweep takes longer
#: than that -- attention is quadratic in the token count -- so the watchdog would abort a run that
#: is merely slow.
PG_TIMEOUT = timedelta(hours=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--size", required=True, type=int, help="cubic crop side, in voxels")
    parser.add_argument("--results", required=True, type=Path, help="JSON-lines output, appended")
    parser.add_argument(
        "--dims",
        default="",
        help="dp_replicate,dp_shard,tp overriding [parallelism]. For running an arm on fewer "
        "nodes than it is written for -- a single-node smoke test -- without a second config "
        "whose only difference is three numbers",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="trace one step and print the ops by self device time, on rank 0. For answering "
        "*where* a step goes rather than how long it takes -- step-time arithmetic across "
        "sizes and parallelisms can bound that but cannot identify it",
    )
    parser.add_argument(
        "--compile-blocks",
        action="store_true",
        help="compile each transformer block instead of the whole algorithm, which is how "
        "torchtitan composes torch.compile with tensor parallelism: every hook -- torch's "
        "SequenceParallel input_fn and this repo's sequence entry/exit -- then sits outside "
        "any traced region. Only meaningful with [trainer].compile set",
    )
    parser.add_argument(
        "--trace",
        default="",
        help="with --profile, also export rank 0's chrome trace here. The op table sums "
        "kernel durations and so cannot say whether collectives overlapped compute or "
        "serialised against it; only the stream timeline can",
    )
    parser.add_argument(
        "--decode-chunks",
        default="",
        help="override [algorithm].decode_chunks: an integer, or `auto` for the fewest slabs whose "
        "largest voxel-resolution tensor (slab plus its halo planes, times the readout width) stays "
        "under 0.9 x 2^31 elements -- past that the head's convolutions fall back to cuDNN's int64 "
        "kernels. The right count depends on the crop, so a fixed one is wrong at all but one size",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--measure-flops",
        action="store_true",
        help="count the step's real FLOPs, head included. Costs an extra forward/backward, which "
        "at the top of the sweep is minutes; the analytic encoder-only figure is always reported",
    )
    return parser.parse_args()


def synthetic_batch(size: int, batch_size: int, device: torch.device, seed: int) -> dict[str, Any]:
    """One data-parallel rank's batch, in the layout `output_axes = "lcxyz"` produces.

    Seeded on the *data-parallel* rank rather than the global one, so tensor-parallel peers hold
    identical tensors -- which is what `ScatterSequence` assumes when it takes a local slice without
    a collective, and what a real run gives them, since `ParallelDims.dp_rank` feeds the sampler.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    image = torch.randn(batch_size, 1, 1, size, size, size, device=device, generator=generator)
    blocks = size // LABEL_BLOCK
    ids = torch.randint(
        1, 64, (batch_size, 1, blocks, blocks, blocks),
        device=device, dtype=torch.int32, generator=generator,
    )
    label = (
        ids.view(batch_size, 1, blocks, 1, blocks, 1, blocks, 1)
        .expand(batch_size, 1, blocks, LABEL_BLOCK, blocks, LABEL_BLOCK, blocks, LABEL_BLOCK)
        .reshape(batch_size, 1, size, size, size)
    )
    return {"img": image, "label": label}


#: Headroom under cuDNN's 32-bit element limit for the head's slab tensors.
CUDNN_ELEMENT_BUDGET = int(0.9 * 2**31)


def auto_decode_chunks(size: int, patch: int, readout: int, reach: int = 2) -> int:
    """Fewest slabs along the first axis whose (slab + 2 x reach) x size^2 x readout fits the budget.

    A slab is whole patch planes; `AffinitySegmentation` hands the refinement convolutions the slab
    plus `refine_reach` voxels of each neighbour -- one per 3^3 convolution, `decoder_refine_depth`
    in all. Falls back to one plane per slab when even that is over the budget (above ~2590^3 at
    readout 16, where the head then runs on the slow kernels).
    """
    planes = size // patch
    for chunks in range(1, planes + 1):
        depth = -(-planes // chunks) * patch + 2 * reach
        if depth * size * size * readout <= CUDNN_ELEMENT_BUDGET:
            return chunks
    return planes


def first_axis_patch(model: torch.nn.Module) -> int:
    """The patch extent along the first spatial axis, which the head's slabs are cut along.

    Read off the built model: `dinov3_vit3d` holds one int for a cube, `vit3d` one per axis, and
    `convnet3d` takes no patch size at all, its patch being the stride its stages add up to.
    """
    patch = model.patch_size
    return patch if isinstance(patch, int) else int(patch[0])


def model_kwargs_at(
    name: str, model_kwargs: dict[str, Any], size: int, device: torch.device
) -> dict[str, Any]:
    """The config's model arguments with the crop set to `size`, in the form the model takes.

    `img_size` is an int for `dinov3_vit3d` and a per-axis list for `vit3d`, so the override keeps
    whichever form the config used. `device` is passed only to a model that takes it: building on
    the device is what keeps a 7B from materialising on the host, and a ViT-L built on the host
    and moved is only ~1 GB.
    """
    configured = model_kwargs.get("img_size")
    kwargs = {
        **model_kwargs,
        "img_size": [size] * len(configured) if isinstance(configured, list | tuple) else size,
    }
    if "device" in inspect.signature(ModelRegistry.get(name)).parameters:
        kwargs["device"] = device
    return kwargs


def prefix_tokens(model: torch.nn.Module) -> int:
    """Tokens in front of the patch grid: DINOv3's CLS and storage tokens, none for `vit3d`."""
    storage = getattr(model, "n_storage_tokens", None)
    return 0 if storage is None else 1 + storage


def main() -> int:
    args = parse_args()
    config = load_run_config(args.config)
    dims = (
        ParallelDims(*(int(part) for part in args.dims.split(",")))
        if args.dims
        else config.parallelism
    )
    trainer = config.trainer

    # Not `distributed.setup.init_distributed`, which is otherwise exactly this: it does not expose
    # the process-group timeout, and the default is shorter than one step at the top of this sweep.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=PG_TIMEOUT)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    primary = rank == 0

    try:
        mesh = dims.build_mesh(device_type())
        device = torch.device("cuda", local_rank)
        size = args.size

        if primary:
            print(
                f"[sweep] {config.experiment_name} {size}^3  dims={dims}  world={world_size}  "
                f"device={torch.cuda.get_device_name(device)}",
                flush=True,
            )

        torch.cuda.reset_peak_memory_stats(device)
        # `device=` rather than a `.to()` after the fact: 6.7B fp32 parameters are 27 GB, and
        # initialising them on the host would cost that per rank, eight ranks to a node, plus a
        # minute of `trunc_normal_` on one core. The blocks and the learned tokens take the device
        # argument; the norms and the patch convolution do not, so `.to()` still moves those.
        model = ModelRegistry.build(
            config.model.name,
            **model_kwargs_at(config.model.name, config.model.kwargs, size, device),
        )
        if args.decode_chunks:
            kwargs = config.algorithm.kwargs
            # The widest voxel-resolution tensor in a slab, and the planes past the slab it spans.
            # For the sub-pixel head, the readout over the reach of its refinement convolutions.
            # For the U-Net head, the full-resolution concatenation of the stream and the image
            # path (2 x its finest width) over the two planes its residual block reads each side.
            if kwargs.get("decoder") == "unet":
                width, reach = 2 * list(kwargs.get("decoder_widths", [16]))[0], 2
            else:
                width = kwargs.get("decoder_readout_dim", 16)
                reach = kwargs.get("decoder_refine_depth", 2)
            kwargs["decode_chunks"] = (
                auto_decode_chunks(args.size, first_axis_patch(model), width, reach)
                if args.decode_chunks == "auto"
                else int(args.decode_chunks)
            )
        # `split_disconnected` is overridden, not merely left to the config. Built without a
        # dataset there is no `sample_transform` to delegate the components pass to, so the
        # algorithm would run its *device* implementation inside the step -- an iterative
        # union-find whose termination test is a `torch.equal`, i.e. 30-odd host synchronizations
        # per sample. It is off the critical path in a real run and it has to be off here too, or
        # the step time measures cc3d rather than the model.
        algorithm = AlgorithmRegistry.build(
            config.algorithm.name,
            model,
            None,
            **{**config.algorithm.kwargs, "split_disconnected": False},
        ).to(device)
        encoder_parameters = model.num_parameters()
        # Summed rather than asked for: `num_parameters` is on `BaseModel`, not `BaseAlgorithm`.
        # `parameters()` deduplicates by identity, so the encoder -- which `affinity_seg` registers
        # under both `model` and `encoder` -- is counted once.
        head_parameters = (
            sum(p.numel() for p in algorithm.parameters()) - encoder_parameters
        )

        # Before activation checkpointing, which replaces each block with a `CheckpointWrapper`
        # and so trips `flops()`'s assertion that its blocks are `SelfAttentionBlock`s. Asking the
        # architecture what it costs before anything wraps it is the right order anyway: this is a
        # property of the model, not of how the engine chose to run it.
        encoder_flops = model.flops((1, size, size, size))

        # The trainer's order, and each boundary is load-bearing there; see its comment.
        apply_tensor_parallel(model, mesh, dims)
        if trainer.activation_checkpointing:
            apply_activation_checkpointing(algorithm)
        shard_algorithm(algorithm, mesh, dims)

        optimizer = build_optimizer(algorithm, trainer)
        if trainer.compile and args.compile_blocks:
            for index, block in enumerate(model.blocks):
                model.blocks[index] = torch.compile(block)
            step_fn = algorithm
        else:
            step_fn = torch.compile(algorithm) if trainer.compile else algorithm
        batch = synthetic_batch(size, trainer.batch_size, device, 1000 + dims.dp_rank(mesh))
        build_peak = torch.cuda.max_memory_allocated(device)

        def one_step() -> dict[str, torch.Tensor]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                metrics = step_fn(batch)
            metrics["loss"].backward()
            clip_grad_norm_(algorithm.parameters(), trainer.grad_clip_norm)
            # engine.optimizer.optimizer_step, as the trainer does: a tp-only run (dp_shard 1) mixes
            # DTensor and plain parameters, which foreach AdamW rejects without it.
            optimizer_step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            return metrics

        def run_steps(count: int, label: str) -> tuple[float, dict[str, torch.Tensor]]:
            """Run `count` steps, reporting each as it lands. Returns their total wall time.

            Printed per step, not per size, because at the top of this sweep a step is tens of
            minutes and a size that reports only at the end is indistinguishable from one that has
            hung -- which is exactly what a 1600-cube trial looked like for an hour before it was
            killed. The per-step synchronize this needs costs nothing at those durations and ~1%
            at the small end.
            """
            total = 0.0
            for index in range(count):
                started = time.perf_counter()
                step_metrics = one_step()
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - started
                total += seconds
                if primary:
                    print(
                        f"[step] {size}^3 {label} {index + 1}/{count}: {seconds:.2f} s, "
                        f"{torch.cuda.max_memory_reserved(device) / 1024**3:.1f} GiB reserved",
                        flush=True,
                    )
            return total, step_metrics

        _, metrics = run_steps(args.warmup, "warmup")
        dist.barrier()

        # After the warmup, so the figure is the steady state a long run sits at rather than the
        # allocator still growing and cuDNN still choosing algorithms.
        torch.cuda.reset_peak_memory_stats(device)
        timed, metrics = run_steps(args.steps, "timed")
        elapsed = torch.tensor([timed], device=device)
        step_peak = torch.cuda.max_memory_allocated(device)
        step_reserved = torch.cuda.max_memory_reserved(device)

        # The slowest rank sets the pace: every rank synchronizes with every other one each step,
        # so a mean over ranks would report a step nobody took.
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        step_seconds = float(elapsed.item()) / args.steps

        if args.profile:
            # The algorithm labels its encoder, decoder and affinity-target regions; the head's
            # own share of the decoder is labelled here, on this instance only, so the split
            # between the head and the targets/loss around it can be read off without touching
            # the algorithm.
            decode = algorithm._decode

            def labelled_decode(*decode_args: Any, **decode_kwargs: Any) -> torch.Tensor:
                with torch.profiler.record_function("head"):
                    return decode(*decode_args, **decode_kwargs)

            algorithm._decode = labelled_decode  # type: ignore[method-assign]
            # After the timed window, so it costs the measurement nothing. One step: at these
            # sizes a step is tens of seconds and the trace is already large.
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                one_step()
                torch.cuda.synchronize(device)
            if primary and args.trace:
                prof.export_chrome_trace(args.trace)
                print(f"[trace] wrote {args.trace}", flush=True)
            if primary:
                averages = prof.key_averages()
                # The labelled regions' device time. Forward and the checkpoint recompute only:
                # autograd runs the backward outside them, so backward kernels are in the op table.
                for region in ("encoder", "decoder", "head", "affinity_targets"):
                    total = sum(e.device_time_total for e in averages if e.key == region)
                    print(f"[region] {region:17s} {total / 1e6:9.3f} s device time", flush=True)
                print(averages.table(sort_by="self_device_time_total", row_limit=40), flush=True)
                print("\n===== by input shape =====", flush=True)
                print(
                    prof.key_averages(group_by_input_shape=True).table(
                        sort_by="self_device_time_total", row_limit=25
                    ),
                    flush=True,
                )
                print("\n===== by device memory =====", flush=True)
                print(
                    prof.key_averages().table(
                        sort_by="self_device_memory_usage", row_limit=25
                    ),
                    flush=True,
                )

        measured_flops = None
        if args.measure_flops:
            from engine.mfu import measure_step_flops

            measured_flops = measure_step_flops(
                algorithm, batch, lambda: torch.autocast("cuda", dtype=torch.bfloat16)
            )

        samples = trainer.batch_size * dims.dp_world_size
        # Model FLOPs in the PaLM sense (forward plus backward, no recompute) for the *encoder*
        # alone -- `BaseModel.flops` knows nothing about the affinity head, whose voxel-resolution
        # arithmetic is a large share at these crop sizes. `--measure-flops` gives the honest
        # whole-step number where it is affordable; the two are reported side by side rather than
        # reconciled.
        per_gpu_flops = 3 * encoder_flops * samples / world_size
        peak, _ = peak_flops(torch.cuda.get_device_name(device), "bf16")
        gib = 1024**3

        record = {
            "experiment": config.experiment_name,
            "size": size,
            "tokens": model.num_patches + prefix_tokens(model),
            "dp_replicate": dims.dp_replicate,
            "dp_shard": dims.dp_shard,
            "tp": dims.tp,
            "world_size": world_size,
            "decoder": config.algorithm.kwargs.get("decoder", "interpolate"),
            "decode_chunks": config.algorithm.kwargs.get("decode_chunks", 1),
            "decoder_readout_dim": config.algorithm.kwargs.get("decoder_readout_dim", 16),
            "decoder_widths": config.algorithm.kwargs.get("decoder_widths"),
            "model": config.model.name,
            "embed_dim": config.model.kwargs.get("embed_dim"),
            "depth": config.model.kwargs.get("depth"),
            "attn_window": config.model.kwargs.get("attn_window"),
            "attn_window_mode": config.model.kwargs.get("attn_window_mode", "block"),
            "attn_global_blocks": list(config.model.kwargs.get("attn_global_blocks", [])),
            "attn_global_kv_pool": config.model.kwargs.get("attn_global_kv_pool", 1),
            "block": config.model.kwargs.get("block"),
            "widths": config.model.kwargs.get("widths"),
            "depths": config.model.kwargs.get("depths"),
            "activation_checkpointing": trainer.activation_checkpointing,
            "compile": trainer.compile,
            "status": "ok",
            "step_seconds": step_seconds,
            "global_batch": samples,
            "samples_per_s": samples / step_seconds,
            "voxels_per_s": samples * size**3 / step_seconds,
            "peak_allocated_gib": step_peak / gib,
            "peak_reserved_gib": step_reserved / gib,
            "build_peak_gib": build_peak / gib,
            "encoder_tflops_per_gpu_s": per_gpu_flops / step_seconds / 1e12,
            "encoder_mfu": per_gpu_flops / step_seconds / peak if peak else None,
            "encoder_parameters": encoder_parameters,
            "head_parameters": head_parameters,
            "loss": float(metrics["loss"].detach()),
            "target_positive_rate": float(metrics["target_positive_rate"].detach()),
        }
        if measured_flops is not None:
            record["measured_step_tflops_per_gpu"] = measured_flops / 1e12
            record["measured_mfu"] = measured_flops / step_seconds / peak if peak else None

        if primary:
            with args.results.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"[sweep] {size}^3 ({record['tokens']} tokens): {step_seconds:.2f} s/step, "
                f"peak {record['peak_reserved_gib']:.1f} GiB reserved, "
                f"{record['voxels_per_s'] / 1e6:.1f} Mvoxel/s, "
                f"encoder MFU {record['encoder_mfu']:.3f}",
                flush=True,
            )
        dist.barrier()
    finally:
        destroy_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
