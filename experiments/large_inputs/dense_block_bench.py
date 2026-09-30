#!/usr/bin/env python
"""Dense-convolution blocks in 3D, forward + backward, on one GPU, at every stage of a 1024^3 crop.

Written 2026-09-30, after `convnext_block_bench.py` found 3D depthwise convolution running at ~2%
of the CUDA cores' peak. The question here is whether blocks whose spatial mixing is a dense
convolution, which runs on the tensor cores, do better. Four residual blocks, each with its
standard normalization and activation:

  basic       ResNet's basic block: two dense 3^3 convolutions
  bottleneck  ResNet's bottleneck: a 1^3 down to width/4, a dense 3^3 there, a 1^3 back up
  fused       EfficientNetV2's Fused-MBConv: a dense 3^3 from width to 4x width, a 1^3 back
  pconv       FasterNet's block: a dense 3^3 over the first quarter of the channels, a 2x MLP

They run at the stage shapes of three encoder layouts at a 1024^3 crop, where strides 4/8/16/32
give 256^3/128^3/64^3/32^3 positions: ResNet-18/34 widths 64/128/256/512 (n1-n4), ConvNeXt-T
widths 96/192/384/768 (t1-t4, as in convnext_block_bench.py) and ResNet-50 widths
256/512/1024/2048 (b1-b4). Each runs in both layouts (ncdhw contiguous, ndhwc channels_last_3d)
and measures

  conv       the block's 3^3 convolution alone, eager (and again with cudnn.benchmark on)
  eager      the whole block
  compiled   the whole block under torch.compile

under bf16 autocast with fp32 parameters. The residual stream is bf16, as it is in a network of
these blocks, where nothing fp32 (such as ConvNeXt's layer scale) promotes it. 1^3 convolutions
are matmuls, because cuBLAS has no 2^31-element limit and needs no layout change. A 3^3
convolution whose output reaches 2^31 elements runs in output-channel chunks, which is exact.

The summary puts the blocks into whole encoders, beside two references from
convnext_block_bench.py's run: ConvNeXt-T with 3^3 depthwise kernels, and the windowed ViT-L.

Measured 2026-09-30 (job 154474411; mia-train-experiments/large_inputs/probes/dense_block/
summary.txt). Encoders at 1024^3, compiled, channels-last, with activation checkpointing, in
s/GPU-step: ResNet-34 layout 0.25, the same at ConvNeXt-T widths 0.66, ResNet-50 layout 0.35,
Fused-MBConv at ConvNeXt-T widths 1.60, FasterNet 0.35. For comparison, the 3^3 ConvNeXt-T took
1.09 and the windowed ViT-L 1.82. A dense 3^3 convolution alone runs at 38-71% of the tensor-core
peak from 64 channels up; channels-last beats contiguous everywhere; cudnn.benchmark changes
nothing.

Run one GPU's share:  python experiments/large_inputs/dense_block_bench.py --runs basic:n1 --out DIR
Summarize a run:      python experiments/large_inputs/dense_block_bench.py --summarize DIR
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from convnext_block_bench import INDEX_LIMIT, LAYOUTS, Autocast, Block, measure

CASES = {  # block input at a 1024^3 crop: (width, positions per axis)
    "n1": (64, 256),
    "n2": (128, 128),
    "n3": (256, 64),
    "n4": (512, 32),
    "t1": (96, 256),
    "t2": (192, 128),
    "t3": (384, 64),
    "t4": (768, 32),
    "b1": (256, 256),
    "b2": (512, 128),
    "b3": (1024, 64),
    "b4": (2048, 32),
}
ENCODERS = {  # (family, case, blocks) per stage; stem and downsampling excluded
    "resnet34_3d": (("basic", "n1", 3), ("basic", "n2", 4), ("basic", "n3", 6), ("basic", "n4", 3)),
    "resnet34_3d_wide": (
        ("basic", "t1", 3),
        ("basic", "t2", 4),
        ("basic", "t3", 6),
        ("basic", "t4", 3),
    ),
    "resnet50_3d": (
        ("bottleneck", "b1", 3),
        ("bottleneck", "b2", 4),
        ("bottleneck", "b3", 6),
        ("bottleneck", "b4", 3),
    ),
    "fused_t": (("fused", "t1", 3), ("fused", "t2", 3), ("fused", "t3", 9), ("fused", "t4", 3)),
    "fused_then_convnext3_t": (
        ("fused", "t1", 3),
        ("fused", "t2", 3),
        ("convnext3", "t3", 9),
        ("convnext3", "t4", 3),
    ),
    "pconv_t": (("pconv", "t1", 3), ("pconv", "t2", 3), ("pconv", "t3", 9), ("pconv", "t4", 3)),
    "convnext3_t": (
        ("convnext3", "t1", 3),
        ("convnext3", "t2", 3),
        ("convnext3", "t3", 9),
        ("convnext3", "t4", 3),
    ),
}
# Dense-conv stages at strides 4 and 8 in front of the windowed ViT-L, whose cost is added whole.
HYBRID = (("basic", "n1", 2), ("basic", "n2", 2))


class Pointwise(nn.Module):
    """A 1^3 convolution without bias, computed as a matmul over channels in the input's layout."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cout, cin))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_contiguous(memory_format=torch.channels_last_3d):
            return F.linear(x.permute(0, 2, 3, 4, 1), self.weight).permute(0, 4, 1, 2, 3)
        n, c = x.shape[:2]
        return torch.matmul(self.weight, x.reshape(n, c, -1)).view(n, -1, *x.shape[2:])


class Conv3x3(nn.Conv3d):
    """A dense 3^3 convolution, run in output-channel chunks at 2^31 output elements."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(cin, cout, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[0] * self.out_channels * math.prod(x.shape[2:])
        chunks = math.ceil(size / (INDEX_LIMIT - 1))
        if chunks == 1:
            return super().forward(x)
        step = math.ceil(self.out_channels / chunks)
        starts = range(0, self.out_channels, step)
        return torch.cat([F.conv3d(x, self.weight[i : i + step], padding=1) for i in starts], dim=1)


class Basic(nn.Module):
    """ResNet's basic block (He et al., 2016): two dense 3^3 convolutions."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.conv, self.bn1 = Conv3x3(width, width), nn.BatchNorm3d(width)
        self.conv2, self.bn2 = Conv3x3(width, width), nn.BatchNorm3d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv(x)))
        return F.relu(x + self.bn2(self.conv2(y)))


class Bottleneck(nn.Module):
    """ResNet's bottleneck (He et al., 2016): a 1^3 down to width/4, a dense 3^3, a 1^3 back up."""

    def __init__(self, width: int) -> None:
        super().__init__()
        inner = width // 4
        self.reduce, self.bn1 = Pointwise(width, inner), nn.BatchNorm3d(inner)
        self.conv, self.bn2 = Conv3x3(inner, inner), nn.BatchNorm3d(inner)
        self.expand, self.bn3 = Pointwise(inner, width), nn.BatchNorm3d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.reduce(x)))
        y = F.relu(self.bn2(self.conv(y)))
        return F.relu(x + self.bn3(self.expand(y)))


class FusedMBConv(nn.Module):
    """EfficientNetV2's Fused-MBConv (Tan & Le, 2021): a dense 3^3 to 4x width, a 1^3 back."""

    def __init__(self, width: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = expansion * width
        self.conv, self.bn1 = Conv3x3(width, hidden), nn.BatchNorm3d(hidden)
        self.project, self.bn2 = Pointwise(hidden, width), nn.BatchNorm3d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bn2(self.project(F.silu(self.bn1(self.conv(x)))))


class PartialConv(nn.Module):
    """FasterNet's block (Chen et al., 2023): a dense 3^3 over a channel quarter, 2x MLP."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.part = width // 4
        self.conv = Conv3x3(self.part, self.part)
        self.fc1, self.bn = Pointwise(width, 2 * width), nn.BatchNorm3d(2 * width)
        self.fc2 = Pointwise(2 * width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.conv(x[:, : self.part]), x[:, self.part :]], dim=1)
        return x + self.fc2(F.relu(self.bn(self.fc1(y))))


FAMILIES: dict[str, type[nn.Module]] = {
    "basic": Basic,
    "bottleneck": Bottleneck,
    "fused": FusedMBConv,
    "pconv": PartialConv,
}


def run(family: str, case: str, steps: int, warmup: int, write: Callable[[dict], None]) -> None:
    width, grid = CASES[case]
    for layout_name, layout in LAYOUTS.items():
        torch.manual_seed(0)
        block = FAMILIES[family](width).cuda().to(memory_format=layout)
        conv = block.conv
        shape = (1, width, grid, grid, grid)
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16).contiguous(memory_format=layout)
        xc = torch.randn(1, conv.in_channels, grid, grid, grid, device="cuda", dtype=torch.bfloat16)
        xc = xc.contiguous(memory_format=layout)
        base = {"family": family, "case": case, "width": width, "grid": grid, "layout": layout_name}
        base |= {"conv_in": conv.in_channels, "conv_out": conv.out_channels}
        trials = [
            ("conv", Autocast(conv), xc),
            ("conv_cudnn_benchmark", Autocast(conv), xc),
            ("eager", Autocast(block), x),
            ("compiled", torch.compile(Autocast(block), dynamic=False), x),
        ]
        for what, module, inp in trials:
            torch._dynamo.reset()
            torch.backends.cudnn.benchmark = what == "conv_cudnn_benchmark"
            record = {**base, "what": what}
            try:
                record |= measure(module, inp, steps, warmup)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                record["error"] = f"{type(error).__name__}: {str(error)[:400]}"
            torch.backends.cudnn.benchmark = False
            torch.cuda.empty_cache()
            write(record)
        del block, conv, x, xc, trials
        torch.cuda.empty_cache()


def block_parameters(family: str, width: int) -> int:
    with torch.device("meta"):
        block = Block(width, 3) if family == "convnext3" else FAMILIES[family](width)
    return sum(p.numel() for p in block.parameters())


def _load(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for path in sorted(directory.glob("*.jsonl")) for line in path.open()]


def _fmt(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def summarize(out: Path, reference: Path) -> None:
    records, previous = _load(out), _load(reference)
    meta = records[0]
    print(f"{meta['device']}, torch {meta['torch']}, cuDNN {meta['cudnn']}, CUDA {meta['cuda']}")
    rows = {(r["family"], r["case"], r["layout"], r["what"]): r for r in records}
    # ConvNeXt-T blocks at 3^3 from the depthwise run: contiguous, its only usable layout, so
    # they stand in for both layout columns below.
    for r in previous:
        if (
            r.get("kernel") == 3
            and r.get("layout") == "ncdhw"
            and r["what"] in ("eager", "compiled")
        ):
            for layout in LAYOUTS:
                rows[("convnext3", r["case"], layout, r["what"])] = r
    vit = {r["what"]: r for r in previous if r["case"] == "vit" and "error" not in r}

    def get(family: str, case: str, layout: str, what: str, key: str) -> float | None:
        record = rows.get((family, case, layout, what))
        return None if record is None or "error" in record else record[key]

    print("\nms per call, one block; conv TF/s counts 2*27*Cin*Cout*V FLOPs forward, 3x for f+b")
    print(
        "family      case C     V      layout | conv Cin->Cout   fwd    f+b  TF/s | "
        "cudnn.bench f+b | eager fwd    f+b   GiB | compiled fwd    f+b   GiB"
    )
    for family in FAMILIES:
        for case, (width, grid) in CASES.items():
            for layout in LAYOUTS:
                conv = rows.get((family, case, layout, "conv"))
                if conv is None:
                    continue

                def cell(what: str, key: str, pad: int, family=family, case=case, layout=layout):
                    return f"{_fmt(get(family, case, layout, what, key)):>{pad}}"

                fb = get(family, case, layout, "conv", "fwdbwd_ms")
                flops = 2 * 27 * conv["conv_in"] * conv["conv_out"] * grid**3
                tfs = None if fb is None else 3 * flops / fb / 1e9
                print(
                    f"{family:11} {case:4} {width:<5} {grid:>3}^3  {layout:6} | "
                    f"{conv['conv_in']:>5}->{conv['conv_out']:<5} {cell('conv', 'fwd_ms', 6)} "
                    f"{_fmt(fb):>6} {_fmt(tfs, 0):>5} | "
                    f"{cell('conv_cudnn_benchmark', 'fwdbwd_ms', 15)} | "
                    f"{cell('eager', 'fwd_ms', 9)} {cell('eager', 'fwdbwd_ms', 6)} "
                    f"{cell('eager', 'peak_gib', 5)} | {cell('compiled', 'fwd_ms', 12)} "
                    f"{cell('compiled', 'fwdbwd_ms', 6)} {cell('compiled', 'peak_gib', 5)}"
                )
    for r in records:
        if "error" in r:
            where = f"{r['family']} {r['case']} {r['layout']} {r['what']}"
            print(f"error {where}: {r['error']}")

    def encoder(stages: tuple[tuple[str, str, int], ...], layout: str, what: str) -> float | None:
        """Seconds per GPU-step for these stages, with activation checkpointing's extra forward."""
        total = 0.0
        for family, case, blocks in stages:
            fb, fw = (get(family, case, layout, what, key) for key in ("fwdbwd_ms", "fwd_ms"))
            if fb is None or fw is None:
                return None
            total += blocks * (fb + fw) / 1e3
        return total

    print("\nencoder blocks at a 1024^3 crop, s per GPU-step with activation checkpointing")
    print("(stem and downsampling excluded; the ConvNeXt stages are contiguous in both columns)")
    print("encoder                  params | compiled ncdhw  ndhwc | eager ncdhw  ndhwc")
    for name, stages in ENCODERS.items():
        params = sum(n * block_parameters(f, CASES[c][0]) for f, c, n in stages) / 1e6
        cells = [
            encoder(stages, layout, what) for what in ("compiled", "eager") for layout in LAYOUTS
        ]
        print(
            f"{name:24} {params:5.0f}M | {_fmt(cells[0], 2):>14} {_fmt(cells[1], 2):>6} | "
            f"{_fmt(cells[2], 2):>11} {_fmt(cells[3], 2):>6}"
        )
    if "compiled" in vit:
        cells = []
        for what in ("compiled", "eager"):
            for layout in LAYOUTS:
                stem = encoder(HYBRID, layout, what)
                cells.append(None if stem is None else stem + vit[what]["fwdbwd_ms"] / 1e3)
        print(
            f"{'resnet stem + vitl_win16':24} {'':>6} | {_fmt(cells[0], 2):>14} "
            f"{_fmt(cells[1], 2):>6} | {_fmt(cells[2], 2):>11} {_fmt(cells[3], 2):>6}"
        )
        print(
            f"{'vitl_dp_win16 (reference)':24} {'303M':>6} | "
            f"{vit['compiled']['fwdbwd_ms'] / 1e3:>14.2f} {'':>6} | "
            f"{vit['eager']['fwdbwd_ms'] / 1e3:>11.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", nargs="+", default=[], help="family:case pairs, e.g. basic:n1")
    parser.add_argument("--out", type=Path, help="directory for one <family>_<case>.jsonl each")
    parser.add_argument("--device", type=int, default=0, help="which visible GPU to run on")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--summarize", type=Path, help="print the tables for a finished run")
    parser.add_argument(
        "--reference",
        type=Path,
        help="convnext_block_bench.py's output directory (default: convnext_block beside the run)",
    )
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize, args.reference or args.summarize.parent / "convnext_block")
        return
    runs = [tuple(run.split(":")) for run in args.runs]
    for family, case in runs:
        if family not in FAMILIES or case not in CASES:
            parser.error(f"unknown run {family}:{case}")
    if not runs or args.out is None:
        parser.error("name --runs and --out, or pass --summarize DIR")

    torch.cuda.set_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "cuda": torch.version.cuda,
    }
    for family, case in runs:
        path = args.out / f"{family}_{case}.jsonl"

        def write(record: dict[str, Any], path: Path = path) -> None:
            with path.open("a") as handle:
                handle.write(json.dumps({**record, **meta}) + "\n")
            shown = {k: round(v, 2) if isinstance(v, float) else v for k, v in record.items()}
            print(json.dumps(shown), flush=True)

        run(family, case, args.steps, args.warmup, write)


if __name__ == "__main__":
    main()
