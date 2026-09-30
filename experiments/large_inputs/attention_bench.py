#!/usr/bin/env python
"""One layer's attention, forward + backward, on one GPU: global, block windows, sliding windows.

Written 2026-09-29 to find where sliding-window attention's time went. Windows centred on each
token (the first sliding implementation) measured, per layer on one B300 at a 64^3 grid (16 heads x
64, bf16): block windows 35 ms, exact per-token sliding 625 ms, the same with the predicate read
from per-token tables 277 ms, and the same key blocks all treated as full -- sliding tile by tile
-- 84 ms; cuDNN global 1071 ms, FlexAttention global 5007 ms. The per-score predicate in key blocks
only partly inside a window was the cost; FlexAttention's kernel on full blocks matches cuDNN per
pair. Sliding windows were then reimplemented to move tile by tile, which is what `sliding` runs.

  global        cuDNN SDPA over every token
  flex_global   FlexAttention with no mask: its kernel's speed against cuDNN's
  block         block windows: SDPA over the batch of windows
  sliding       windows sliding tile by tile, on FlexAttention

Run on one GPU:  python experiments/large_inputs/attention_bench.py --grid 32
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention.flex_attention import flex_attention  # noqa: E402

from layers.common.window_attention import (  # noqa: E402
    attention_pairs,
    sliding_window_attention,
    windowed_attention,
)


def sdpa(q, k, v):  # (B, N, H, D) layout, as the helpers take it
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=32, help="patches per axis (32 = 512^3 at p16)")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--prefix", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda")
    grid, window = (args.grid,) * 3, (args.window,) * 3
    n = args.prefix + math.prod(grid)
    flex = torch.compile(flex_attention, dynamic=False)
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(1, n, args.heads, args.head_dim, device=device, dtype=torch.bfloat16,
                    requires_grad=True)
        for _ in range(3)
    )
    scale = args.head_dim**-0.5

    def block(q, k, v):
        return windowed_attention(
            q, k, v, grid=grid, window=window, attend=sdpa, prefix=args.prefix
        )

    def sliding(q, k, v):
        return sliding_window_attention(
            q, k, v, grid=grid, window=window, attend=sdpa, scale=scale, prefix=args.prefix
        )

    def flex_global(q, k, v):
        return flex(*(t.transpose(1, 2) for t in (q, k, v)), scale=scale)

    variants = {"global": sdpa, "flex_global": flex_global, "block": block, "sliding": sliding}
    pairs = {
        "block": attention_pairs(grid, args.prefix, window),
        "sliding": attention_pairs(grid, args.prefix, window, window_mode="sliding"),
    }
    print(f"grid {grid} window {window}, {args.heads} x {args.head_dim} heads, {n} tokens")
    for name, fn in variants.items():
        try:
            for _ in range(3):
                out = fn(q, k, v)
                torch.autograd.grad(out, (q, k, v), torch.ones_like(out))
            torch.cuda.synchronize()
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.steps):
                out = fn(q, k, v)
                torch.autograd.grad(out, (q, k, v), torch.ones_like(out))
            stop.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(stop) / args.steps
            scored = pairs.get(name, n * n)
            # Forward is 4 FLOPs per pair per channel; backward ~2.5x that.
            tflops = 3.5 * 4 * scored * args.heads * args.head_dim / (ms / 1e3) / 1e12
            print(f"  {name:12s} {ms:9.2f} ms fwd+bwd   {tflops:7.1f} TFLOP/s", flush=True)
        except Exception as error:  # a variant failing should not hide the others
            print(f"  {name:12s} failed: {type(error).__name__}: {str(error)[:200]}", flush=True)


if __name__ == "__main__":
    main()
