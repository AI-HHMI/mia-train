#!/usr/bin/env python
"""One 3D ConvNeXt block, forward + backward, on one GPU, at every stage of a 1024^3 crop.

Written 2026-09-29, before porting DINOv3's ConvNeXt to 3D. Its 7x7 depthwise convolution becomes
7x7x7: 343 multiply-adds per output instead of 49, on CUDA cores rather than tensor cores, through
PyTorch's much less used 3D depthwise kernels. This measures what that costs.

Measured 2026-09-29 on one B300 per case (torch 2.13, cuDNN 9.20; job 154473598, results in
mia-train-experiments/large_inputs/probes/convnext_block/summary.txt). Encoder blocks at 1024^3,
compiled, with activation checkpointing, in seconds per GPU-step for 7^3/5^3/3^3 kernels: ConvNeXt-T
10.3/4.3/1.1, -S 13.0/5.4/1.4, -L 26.0/10.8/2.8, against 1.82 for the windowed ViT-L encoder.
PyTorch's depthwise kernel (contiguous layout) runs at 1.5-2 TFLOP/s, ~2% of the CUDA-core peak,
with a backward ~6x its forward; cuDNN's (channels-last) backward takes 35-80 s per stage-1 block
at every kernel size. Unchunked past 2^31 elements the PyTorch kernel is 4x slower.

Cases are the block's input at each stage of ConvNeXt-T (t1-t4) and ConvNeXt-L (l1-l4) for a 1024^3
crop: strides 4/8/16/32 give 256^3/128^3/64^3/32^3 positions, at widths 96/192/384/768 (T) and
192/384/768/1536 (L). Each case runs kernels 7, 5 and 3 in two layouts,

  ncdhw     contiguous, which PyTorch sends to its own depthwise-3D kernel
  ndhwc     channels_last_3d, which it sends to cuDNN

and measures

  dwconv    the depthwise convolution alone, eager (ndhwc also with cudnn.benchmark on)
  eager     the whole block: DINOv3's, as written, with 3D kernels
  compiled  the whole block under torch.compile

under bf16 autocast with fp32 parameters and an fp32 residual stream, as training runs. A tensor of
2^31 elements or more (l1: 192 x 256^3) is past what cuDNN and PyTorch's kernels index with 32
bits, so the block runs its depthwise convolution in channel chunks under that, which is exact
because depthwise channels are independent. `--whole` times it unchunked, to show what the limit
costs.

  vit       the baseline: the whole vitl_dp_win16 encoder (ViT-L/16, 16^3-token block windows) at
            1024^3 with activation checkpointing, eager and compiled -- the encoder these blocks
            would replace, measured the same way.

Run one GPU's share:  python experiments/large_inputs/convnext_block_bench.py --cases t1 --out DIR
Summarize a run:      python experiments/large_inputs/convnext_block_bench.py --summarize DIR
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CASES = {  # block input at a 1024^3 crop: (width, positions per axis)
    "t1": (96, 256),
    "t2": (192, 128),
    "t3": (384, 64),
    "t4": (768, 32),
    "l1": (192, 256),
    "l2": (384, 128),
    "l3": (768, 64),
    "l4": (1536, 32),
}
MODELS = {  # (case, blocks) per stage; ConvNeXt-S has T's widths with L's depths
    "convnext_t": (("t1", 3), ("t2", 3), ("t3", 9), ("t4", 3)),
    "convnext_s": (("t1", 3), ("t2", 3), ("t3", 27), ("t4", 3)),
    "convnext_l": (("l1", 3), ("l2", 3), ("l3", 27), ("l4", 3)),
}
KERNELS = (7, 5, 3)
LAYOUTS = {"ncdhw": torch.contiguous_format, "ndhwc": torch.channels_last_3d}
INDEX_LIMIT = 2**31
VIT_CONFIG = REPO / "experiments/large_inputs/vitl_dp_win16.toml"


class DepthwiseConv3d(nn.Conv3d):
    """A depthwise convolution run in equal channel chunks of fewer than 2^31 elements each."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = math.ceil(x.numel() / (INDEX_LIMIT - 1))
        if chunks == 1:
            return super().forward(x)
        step = math.ceil(self.in_channels / chunks)
        return torch.cat(
            [
                F.conv3d(
                    x[:, i : i + step],
                    self.weight[i : i + step],
                    self.bias[i : i + step],
                    padding=self.padding,
                    groups=min(step, self.in_channels - i),
                )
                for i in range(0, self.in_channels, step)
            ],
            dim=1,
        )


class Block(nn.Module):
    """DINOv3's ConvNeXt block (dinov3/models/convnext.py, its permute form) with 3D kernels."""

    def __init__(self, dim: int, kernel: int) -> None:
        super().__init__()
        self.dwconv = DepthwiseConv3d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dwconv(x).permute(0, 2, 3, 4, 1)  # (N, C, D, H, W) -> (N, D, H, W, C)
        y = self.pwconv2(self.act(self.pwconv1(self.norm(y))))
        return x + (self.gamma * y).permute(0, 4, 1, 2, 3)


class Autocast(nn.Module):
    """`inner` under bf16 autocast, the way the trainer runs a step."""

    def __init__(self, inner: nn.Module, method: str = "forward") -> None:
        super().__init__()
        self.inner, self.method = inner, method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = getattr(self.inner, self.method)(x)
        return out[0] if isinstance(out, tuple) else out


def median_ms(fn: Callable[[], Any], steps: int, warmup: int, slow_ms: float = 20_000) -> float:
    """Median of `steps` timed calls after `warmup`; a call past `slow_ms` ends the timing early."""
    times: list[float] = []
    for i in range(warmup + steps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end)
        if i >= warmup or (i > 0 and ms > slow_ms):
            times.append(ms)
        if i > 0 and ms > slow_ms:
            break
    return statistics.median(times)


def measure(
    module: nn.Module, x: torch.Tensor, steps: int, warmup: int, input_grad: bool = True
) -> dict[str, float]:
    """Forward and forward+backward milliseconds, and the backward's peak memory above its inputs.

    The first call of each is a warmup call, so compilation and cuDNN's setup are not timed.
    """
    with torch.no_grad():
        fwd = median_ms(lambda: module(x), steps, warmup)
    x = x.detach().requires_grad_(input_grad)
    grad: list[torch.Tensor] = []

    def step() -> None:
        x.grad = None
        for p in module.parameters():
            p.grad = None
        out = module(x)
        if not grad:
            grad.append(torch.randn_like(out))
        out.backward(grad[0])

    step()
    x.grad = None
    for p in module.parameters():
        p.grad = None
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fwdbwd = median_ms(step, steps, max(warmup - 1, 0))
    peak = (torch.cuda.max_memory_allocated() - base) / 2**30
    return {"fwd_ms": fwd, "fwdbwd_ms": fwdbwd, "peak_gib": peak}


def conv_backend(conv: nn.Conv3d, layout: torch.memory_format) -> str:
    """Which of PyTorch's convolution backends takes this layout (probed on a small input)."""
    x = torch.empty(1, conv.in_channels, 8, 8, 8, device="cuda", dtype=torch.bfloat16)
    x = x.contiguous(memory_format=layout)
    weight = conv.weight.to(torch.bfloat16).contiguous(memory_format=layout)
    try:
        backend = torch._C._select_conv_backend(
            x, weight, None, conv.stride, conv.padding, conv.dilation, False, (0, 0, 0), conv.groups
        )
    except (AttributeError, RuntimeError, TypeError) as error:
        return f"unknown ({type(error).__name__})"
    return str(backend).rsplit(".", 1)[-1]


def run_case(
    name: str, steps: int, warmup: int, write: Callable[[dict[str, Any]], None], whole: bool
) -> None:
    """Every layout and kernel for one case; `whole` times the unchunked convolution instead."""
    width, grid = CASES[name]
    for layout_name, layout in LAYOUTS.items():
        x = torch.randn(1, width, grid, grid, grid, device="cuda").contiguous(memory_format=layout)
        for kernel in KERNELS:
            torch.manual_seed(0)
            block = Block(width, kernel).cuda().to(memory_format=layout)
            base = {"case": name, "width": width, "grid": grid, "layout": layout_name}
            base |= {"kernel": kernel, "backend": conv_backend(block.dwconv, layout)}
            # `dwconv` is the block's own convolution, so it chunks exactly when the block does.
            trials: list[tuple[str, nn.Module]] = [("dwconv", Autocast(block.dwconv))]
            if layout_name == "ndhwc":
                trials.append(("dwconv_cudnn_benchmark", Autocast(block.dwconv)))
            trials.append(("eager", Autocast(block)))
            trials.append(("compiled", torch.compile(Autocast(block), dynamic=False)))
            if whole:
                plain = nn.Conv3d(width, width, kernel, padding=kernel // 2, groups=width)
                plain.load_state_dict(block.dwconv.state_dict())
                trials = [("dwconv_whole", Autocast(plain.cuda().to(memory_format=layout)))]
            for what, module in trials:
                torch._dynamo.reset()
                torch.backends.cudnn.benchmark = what == "dwconv_cudnn_benchmark"
                record = {**base, "what": what}
                try:
                    record |= measure(module, x, steps, warmup)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                    record["error"] = f"{type(error).__name__}: {str(error)[:400]}"
                torch.backends.cudnn.benchmark = False
                torch.cuda.empty_cache()
                write(record)
        torch.cuda.empty_cache()


def run_vit(steps: int, warmup: int, write: Callable[[dict[str, Any]], None]) -> None:
    import components  # noqa: F401  (populates the registries, as a real run does)
    from engine.activation_checkpoint import apply_activation_checkpointing
    from models.registry import ModelRegistry
    from utils.config import load_run_config

    config = load_run_config(VIT_CONFIG)
    kwargs = {**config.model.kwargs, "img_size": 1024}
    if "device" in inspect.signature(ModelRegistry.get(config.model.name)).parameters:
        kwargs["device"] = torch.device("cuda")
    x = torch.randn(1, 1, 1024, 1024, 1024, device="cuda")
    for what in ("eager", "compiled"):
        torch._dynamo.reset()
        torch.manual_seed(0)
        model = ModelRegistry.build(config.model.name, **kwargs).cuda()
        apply_activation_checkpointing(model)
        module: nn.Module = Autocast(model, "patch_features")
        if what == "compiled":
            module = torch.compile(module, dynamic=False)
        record: dict[str, Any] = {"case": "vit", "config": VIT_CONFIG.name, "what": what}
        try:
            record |= measure(module, x, steps, warmup, input_grad=False)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            record["error"] = f"{type(error).__name__}: {str(error)[:400]}"
        write(record)
        del model, module
        torch.cuda.empty_cache()


def _fmt(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def summarize(out: Path) -> None:
    records = [json.loads(line) for path in sorted(out.glob("*.jsonl")) for line in path.open()]
    meta = next(r for r in records if "device" in r)
    print(f"{meta['device']}, torch {meta['torch']}, cuDNN {meta['cudnn']}, CUDA {meta['cuda']}")
    rows = {(r["case"], r.get("layout"), r.get("kernel"), r["what"]): r for r in records}

    def get(case: str, layout: str | None, kernel: int | None, what: str, key: str) -> float | None:
        record = rows.get((case, layout, kernel, what))
        return None if record is None or "error" in record else record[key]

    print("\nms per call, one block; dwconv TF/s counts 2*k^3*C*V FLOPs forward, 3x that for f+b")
    header = "case  C     V      layout k  backend          | dwconv fwd   f+b  TF/s"
    print(header + " | cudnn.bench f+b | eager fwd   f+b  GiB | compiled fwd   f+b  GiB")
    for case, (width, grid) in CASES.items():
        for layout in LAYOUTS:
            for kernel in KERNELS:
                dw = rows.get((case, layout, kernel, "dwconv"))
                if dw is None:
                    continue

                def cell(what: str, key: str, width_: int, case=case, layout=layout, kernel=kernel):
                    return f"{_fmt(get(case, layout, kernel, what, key)):>{width_}}"

                fb = get(case, layout, kernel, "dwconv", "fwdbwd_ms")
                tfs = None if fb is None else 3 * 2 * kernel**3 * width * grid**3 / fb / 1e9
                print(
                    f"{case:5} {width:<5} {grid:>3}^3  {layout:6} {kernel}  {dw['backend']:16} | "
                    f"{cell('dwconv', 'fwd_ms', 10)} {_fmt(fb):>6} {_fmt(tfs):>5} | "
                    f"{cell('dwconv_cudnn_benchmark', 'fwdbwd_ms', 15)} | "
                    f"{cell('eager', 'fwd_ms', 9)} {cell('eager', 'fwdbwd_ms', 6)} "
                    f"{cell('eager', 'peak_gib', 4)} | {cell('compiled', 'fwd_ms', 12)} "
                    f"{cell('compiled', 'fwdbwd_ms', 6)} {cell('compiled', 'peak_gib', 4)}"
                )
    for r in records:
        if r["what"] == "dwconv_whole" and "error" not in r:
            print(
                f"{r['case']} unchunked ({r['width']} x {r['grid']}^3 >= 2^31 elements), "
                f"{r['layout']} k{r['kernel']}: fwd {r['fwd_ms']:.1f} ms, "
                f"f+b {r['fwdbwd_ms']:.1f} ms"
            )
    for r in records:
        if "error" in r:
            where = f"{r['case']} {r.get('layout')} k{r.get('kernel')} {r['what']}"
            print(f"error {where}: {r['error']}")

    print("\nencoder blocks at a 1024^3 crop, s per GPU-step (stem and downsampling excluded);")
    print("'AC' adds the forward that activation checkpointing recomputes")
    print("model       layout k | eager  AC     | compiled  AC     | compiled activations GiB")
    for model, stages in MODELS.items():
        blocks = [n for _, n in stages]
        for layout in LAYOUTS:
            for kernel in KERNELS:
                cells: list[float | None] = []
                for what in ("eager", "compiled"):
                    fb = [get(c, layout, kernel, what, "fwdbwd_ms") for c, _ in stages]
                    fw = [get(c, layout, kernel, what, "fwd_ms") for c, _ in stages]
                    if None in fb or None in fw:
                        cells += [None, None]
                        continue
                    total = sum(n * t for n, t in zip(blocks, fb, strict=True)) / 1e3
                    ac = total + sum(n * t for n, t in zip(blocks, fw, strict=True)) / 1e3
                    cells += [total, ac]
                mem = [get(c, layout, kernel, "compiled", "peak_gib") for c, _ in stages]
                act = None if None in mem else sum(n * m for n, m in zip(blocks, mem, strict=True))
                print(
                    f"{model:11} {layout:6} {kernel} | {_fmt(cells[0], 2):>6} "
                    f"{_fmt(cells[1], 2):>6} | {_fmt(cells[2], 2):>8} {_fmt(cells[3], 2):>6} | "
                    f"{_fmt(act, 0):>6}"
                )
    for what in ("eager", "compiled"):
        record = rows.get(("vit", None, None, what))
        if record is None:
            continue
        if "error" in record:
            print(f"vit {what}: {record['error']}")
            continue
        print(
            f"vitl_dp_win16 encoder, {what}, activation checkpointing on: fwd "
            f"{record['fwd_ms'] / 1e3:.2f} s, fwd+bwd {record['fwdbwd_ms'] / 1e3:.2f} s, "
            f"peak {record['peak_gib']:.1f} GiB"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cases", nargs="+", choices=[*CASES, "vit"], default=[])
    parser.add_argument("--out", type=Path, help="directory for one <case>.jsonl per case")
    parser.add_argument("--device", type=int, default=0, help="which visible GPU to run on")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--summarize", type=Path, help="print the tables for a finished run")
    parser.add_argument(
        "--whole",
        action="store_true",
        help="time only the unchunked depthwise convolution. Run it in a process of its own: a "
        "kernel indexing past 32 bits can fault, and a fault takes the CUDA context with it",
    )
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return
    if not args.cases or args.out is None:
        parser.error("name --cases and --out, or pass --summarize DIR")

    torch.cuda.set_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "cuda": torch.version.cuda,
    }
    for case in args.cases:
        path = args.out / f"{case}{'_whole' if args.whole else ''}.jsonl"

        def write(record: dict[str, Any], path: Path = path) -> None:
            with path.open("a") as handle:
                handle.write(json.dumps({**record, **meta}) + "\n")
            shown = {k: round(v, 2) if isinstance(v, float) else v for k, v in record.items()}
            print(json.dumps(shown), flush=True)

        if case == "vit":
            run_vit(args.steps, args.warmup, write)
        else:
            run_case(case, args.steps, args.warmup, write, args.whole)


if __name__ == "__main__":
    main()
