# How large an input can a model train on per B300 node, and at what throughput?

The goal is models that take very large volumes in one pass: 2048³ voxels as a working target,
8192³ as the ambition. `b300_capability_run` answered this for the largest model in the repo, a
DINOv3 ViT-7B/16: 1600³ (1M tokens) on 64 B300s at ~460 s/step, with the wall set by the
voxel-resolution affinity head and a label dtype rather than by the encoder or the parallelism.

This experiment starts with the model the repo actually trains -- **DINOv3 ViT-L/16**, gary_comparison's
production arm -- and asks three things:

1. **Does the smaller encoder move the crop frontier?** A ViT-L's residual stream is 4x narrower and
   24 blocks deep instead of 40, so its activation-checkpointed memory per token is ~1/7 of the 7B's.
   But the head's memory is proportional to *voxels*, not to the encoder, and it is replicated across
   tensor-parallel ranks, so the frontier may barely move.
2. **What is the throughput** -- s/step, Mvoxel/s per node, encoder MFU -- at 512³, 1024³ and 2048³?
3. **Which parallelism is right for it** on one node: every GPU its own crop (pure data parallel), or
   one crop split across the node (tensor + sequence parallel)?

Later in this experiment: other architectures chosen to raise the input size and the throughput --
a non-attention model (e.g. ResNeXt-style 3D convolutions), subquadratic attention, and so on. The
harness here takes any registered model and algorithm, so those arrive as new configs. Phase 2
(below) is the first: local attention with cheap global mixing, in both ViTs.

## Arms

All on one B300 node (8 GPUs), batch 1 per data-parallel rank, bf16, activation checkpointing,
eager unless stated, ViT-L/16 (1024 wide x 24 blocks x 16 heads, axial RoPE), affinity_seg with the
production sub-pixel head (readout 16, which every arm keeps: the head's capacity is not a knob here).
Each config differs from `vitl_dp.toml` only in the lines named here (`diff` them).

| arm | parallelism | what it asks |
|---|---|---|
| `vitl_dp` | dp_shard 8 x tp 1 (8 crops per node) | the crop ONE GPU can hold, and its throughput |
| `vitl_tp8` | dp_shard 1 x tp 8 (1 crop per node) | does splitting the encoder 8 ways move the frontier past the replicated head? |
| `vitl_dp_compile` | dp 8, `compile = true` | does compile pay for a ViT-L, and to what crop? |

## Method

`capability_sweep.py` is a copy of b300_capability_run's driver (see its docstring for what changed):
it builds the model and algorithm from an ordinary config, parallelizes them exactly as the trainer
does, and times real training steps -- forward, backward, grad clip, optimizer -- on a synthetic batch
already on the device, so the input pipeline is excluded and everything downstream of the batch,
affinity targets included, is measured. `submit.sh` runs one `torchrun` per crop size, ascending, and
stops at the first size that records nothing; the last row in an arm's results file is its limit.
Sizes 512³ -> 3072³ (16³ patches: 32k to 7.1M tokens), 2 warmup + 4 timed steps.

**Chunking the head, per crop.** The sub-pixel head decodes in slabs along the first axis
(`decode_chunks`). Its convolutions fall off cuDNN's fast kernels when a tensor exceeds 2^31
elements, and the right slab count depends on the crop. Since 2026-09-29 the refinement convolutions
see each slab plus only the 2 voxels of its neighbours they actually reach (whole halo tokens are
expanded, then cropped *before* the convolutions); before that they ran on the slab plus a whole
16-voxel halo plane per side, 48 voxels at one plane per slab against 20 now, which capped readout 16
on the fast kernels at ~1664³. `--decode-chunks auto` picks the fewest slabs whose
(slab + 2 x 2 voxels) x crop² x 16 stays under 0.9 x 2^31:

| crop | slabs | convolution input per slab |
|---|---|---|
| 512³ | 2 | 260 x 512² x 16 |
| 1024³ | 11 | 100 x 1024² x 16 |
| 1536³ | 48 | 36 x 1536² x 16 |
| 2048³ | 128 (one plane each) | 20 x 2048² x 16 = 0.63 x 2^31 |
| 2560³ | 160 (one plane each) | 0.98 x 2^31: under the limit, past the 0.9 margin |
| 3072³ | 192 (one plane each) | 1.4 x 2^31: slow kernels |

**History.** The first sweep (2026-09-29, before the halo crop) also had readout-8 arms, removed
since: the head's capacity stays at production 16. Its records are in `jobs/pre_halo_fix/`: at
readout 16, `vitl_tp8` 512³ 1.09 s / 1024³ 10.2 s / 1536³ 111.6 s per step, `vitl_dp` 512³ 1.38 s /
1024³ 41.2 s, `vitl_dp_compile` 512³ 1.00 s / 1024³ 38.2 s (compile ran out of memory at 1536³).

## Results

2026-09-29, after the halo crop, readout 16, one B300 node per arm (`jobs/<arm>.jsonl`; regenerate
the full tables with `python experiments/large_inputs/table.py .../jobs/*.jsonl`). Mvoxel/s is per
node. Each arm stopped on a CUDA out-of-memory at the next size.

| crop | `vitl_dp` (8 crops/node) | `vitl_dp_compile` | `vitl_tp8` (1 crop/node) |
|---|---|---|---|
| 512³ | 1.37 s, 30 GiB, **785** Mvox/s | 0.99 s, 27 GiB, **1087** | 1.08 s, 26 GiB, 124 |
| 1024³ | 40.3 s, 83 GiB, **213** | 37.1 s, 157 GiB, **231** | 9.3 s, 59 GiB, 115 |
| 1536³ | 456.6 s, 180 GiB, **64** | OOM | 91.1 s, 99 GiB, 40 |
| 2048³ | OOM | - | 392.9 s, 193 GiB, **22** (encoder MFU 0.18) |
| 2560³ | - | - | OOM |

* **The frontier:** one GPU holds a 1536³ crop; one node holds 2048³ (tp 8). The 7B needed 8 nodes for 1600³.
* **Throughput per node vs the 7B** (`b300_capability_run`, fsdp_tp8_chunked: 48 / 28 / 9 Mvox/s at
  512³ / 1024³ / 1600³): ~23x at 512³ and ~8x at 1024³ (best arm each), ~7x at 1536³ vs its 1600³.
* **Data parallel beats tensor parallel wherever it fits** (1.6-6x); tp 8 is only for 2048³.
* **Compile** pays 1.4x at 512³ and 1.09x at 1024³, at up to 1.9x the memory; it cannot reach 1536³.
* **Voxels/s falls ~36x from 512³ to 2048³**: attention is quadratic in tokens, so its cost per
  voxel grows with the crop.
* **The halo crop** (vs `jobs/pre_halo_fix/`): `vitl_tp8` 1024³ 10.2 -> 9.3 s (8%), 1536³ 111.6 ->
  91.1 s (18%), and 2048³ now runs at readout 16 on the fast kernels.

## Phase 2: local attention with cheap global mixing (2026-09-29)

**Why.** A two-point fit of phase 1's `vitl_dp` rows (512³ and 1024³; one term per voxel, one per
token pair) splits a GPU-step into attention, ~4.9e-10 s × tokens², and a per-voxel part:
~6.3 ns/voxel eager, ~3.4 compiled. The per-voxel part is mostly the head, targets and loss, and
compile's whole gain is there: 2.9 ns/voxel at both sizes. At 1024³ attention is ~33 of the 40 s
(83%), and more than 90% at 2048³. The fit runs 13% fast at 1536³.

**What.** ViTDet-style attention for both ViTs:
- Most blocks attend within 3D windows of the patch grid.
- A few global blocks carry information between windows, either with full attention or over keys
  and values averaged per box (PVT).
- DINOv3's CLS and storage tokens attend globally in every block.

The code is `layers/common/window_attention.py`, shared by `dinov3_vit3d` and the plain `vit3d`,
with three `[model]` keys: `attn_window`, `attn_global_blocks` and `attn_global_kv_pool`.
- No parameters are added, so every checkpoint loads.
- A window at least as large as the crop is exactly global attention.
- RoPE is unchanged: tokens are rotated for their place on the whole grid before windowing.

Every arm uses 16³-token windows (256³ voxels, gary_comparison's crop), and the global arms use
blocks 5, 11, 17 and 23.

| arm | model, parallelism | attention |
|---|---|---|
| `vitl_dp_win16` | DINOv3 ViT-L, dp 8 | windows only: the cost floor, not a trainable model |
| `vitl_dp_win16_g4` | DINOv3 ViT-L, dp 8 | + 4 blocks of full global attention |
| `vitl_dp_win16_g4p2` | DINOv3 ViT-L, dp 8 | + 4 global blocks over keys/values pooled per 2³ |
| `vitl_dp_win16_g4p4` | DINOv3 ViT-L, dp 8 | + 4 global blocks over keys/values pooled per 4³ |
| `vitl_dp_win16_g4p2_compile` | DINOv3 ViT-L, dp 8, compiled | as `g4p2` |
| `vitl_tp8_win16_g4p4` | DINOv3 ViT-L, tp 8 | as `g4p4`, for the node frontier |
| `vit_l_dp` | plain ViT3D at ViT-L size, dp 8 | global (the baseline) |
| `vit_l_dp_win16_g4p2` | plain ViT3D at ViT-L size, dp 8 | as `g4p2` |

The plain ViT3D has no tensor-parallel plan, so it runs data parallel only.

**Sliding windows** (`attn_window_mode = "sliding"`, added the same day) remove the fixed seams
block windows leave. With block windows, an edge token sees context on one side only in 20 of the 24
blocks.
- **The windows move tile by tile.** Tokens are grouped into 4×4×8 tiles (one 128-token
  FlexAttention block each). Every tile attends to the box of tiles that the 16³ windows centred on
  its tokens cover: itself plus about 8 tokens either side, 20×20×24 tokens. So every token sees at
  least its own centred window, and none sits at the edge of its window.
- **Why tiles, not tokens.** The first implementation centred a window on each token. That leaves
  88% of key blocks only partly inside a window, and FlexAttention then evaluates a predicate per
  score. `attention_bench.py`, one layer forward + backward on one B300 at a 64³ grid, 16 heads × 64:

  | attention | ms |
  |---|---|
  | block windows (cuDNN) | 35 |
  | centred on each token | 625 |
  | centred on each token, predicate from lookup tables | 277 |
  | the same blocks, all treated as full: sliding by tile | 84 |
  | global, cuDNN | 1071 |
  | global, FlexAttention | 5007 |

  On full blocks FlexAttention matches cuDNN per pair, so sliding by tile costs only its 2.4× more
  pairs (9,600 keys against 4,096). The runs with per-token windows are archived in
  `jobs/exact_sliding/`: 22.9–25.9 s/step at 1024³ (data parallel), against 7.7–8.3 s with blocks.
- **Kernel details:** the block mask is built from the geometry, not by testing every pair. Its
  index lists have one column per key block, which is what BlockMask's backward transposition needs:
  about 4 GB at 2048³, built once per crop size and cached. `flops()` counts the box pairs.
- **Tests:** CPU FlexAttention ignores the block lists and the fused kernel trusts them, so the tests
  check the predicate and the lists separately on CPU. The fused kernel, its backward and
  `torch.compile` are checked on a B300 (`tests/unit/test_window_attention_gpu.py`).

Each sliding arm is its block-window arm plus one line:

| arm | block-window counterpart |
|---|---|
| `vitl_dp_slide16` | `vitl_dp_win16` |
| `vitl_dp_slide16_g4p2` | `vitl_dp_win16_g4p2` |
| `vitl_dp_slide16_g4p4` | `vitl_dp_win16_g4p4` |
| `vitl_dp_slide16_g4p2_compile` | `vitl_dp_win16_g4p2_compile` |
| `vitl_tp8_slide16_g4p4` | `vitl_tp8_win16_g4p4` |
| `vit_l_dp_slide16_g4p2` | `vit_l_dp_win16_g4p2` |

**Predicted before the runs,** from the fit above: attention seconds for one crop per GPU-step.

| arm | 1024³ | 2048³ |
|---|---|---|
| global (phase 1) | 33 | ~2,100 |
| windows only | 0.5 | 4 |
| + 4 global blocks | 6 | 360 |
| + 4 global, pooled 2³ | 1.1 | 48 |
| + 4 global, pooled 4³ | 0.5 | 9 |

The per-voxel part adds ~3.7 s at 1024³ compiled and ~6.8 s eager. Windows should cut time, not
memory: SDPA's memory already grows linearly with tokens, so one GPU should still stop near 1536³.

### Phase 2 results

2026-09-29, one B300 node per arm. Seconds per step, block windows → sliding windows (tile by
tile, FLASH backend); `jobs/<arm>.jsonl`. Every arm stopped on a CUDA out-of-memory at the next
size.

| arm | 512³ | 1024³ | 1536³ | 2048³ |
|---|---|---|---|---|
| DINOv3 dp, global (phase 1) | 1.37 | 40.28 | 456.57 | OOM |
| DINOv3 dp, windows only | 1.05 → 1.21 | 7.88 → 9.27 | 49.75 → 56.00 | OOM |
| DINOv3 dp, + 4 full global blocks | 1.10 → 1.24 | 13.27 → 14.37 | 118.56 → 122.47 | OOM |
| DINOv3 dp, + 4 global, pooled 2³ | 1.02 → 1.16 | 8.27 → 9.45 | 56.98 → 62.43 | OOM |
| DINOv3 dp, + 4 global, pooled 4³ | 1.01 → 1.15 | 7.70 → 8.85 | 49.77 → 54.94 | OOM |
| DINOv3 dp, pooled 2³, compiled | 0.58 → 0.91 | 4.86 → 7.43 | OOM | - |
| DINOv3 tp 8, global (phase 1) | 1.08 | 9.34 | 91.12 | 392.91 |
| DINOv3 tp 8, pooled 4³ | 1.12 → 1.15 | 5.69 → 5.84 | 42.32 → 42.89 | 101.51 → 103.40 |
| DINOv3 tp 8, + 4 full global blocks (sliding only) | 1.15 | 6.48 | 51.16 | 152.01 |
| plain ViT3D dp, global | 1.36 | 38.60 | 412.76 | OOM |
| plain ViT3D dp, pooled 2³ | 1.00 → 1.05 | 7.75 → 8.36 | 56.27 → 58.39 | OOM |

* **Windows cut the step 5-9x at 1024³-1536³,** as predicted from the phase-1 fit (the predictions
  above were within ~8% at 1024³). They cut time, not memory: the frontier is unchanged at 1536³
  per GPU and 2048³ per node.
* **Sliding costs +10-18% over block windows** on the DINOv3 data-parallel arms, +4-8% on the plain
  ViT3D and +1-3% under tp 8 (where each GPU runs 2 of the 16 heads). Compiled it costs +53%,
  because under `torch.compile` the block mask is rebuilt in every layer instead of cached; not yet
  fixed.
* **Full global blocks are quadratic:** +5.5 s/step at 1024³ and +67 s at 1536³ over 4³-pooled ones,
  whose cost is close to nothing. They still fit at 2048³ on one node (152 s).
* **The per-voxel path is now the floor.** Compiled, ~3.7 of the 4.86 s at 1024³ is the head,
  targets and loss, and its cost per voxel doubles from 1024³ to 1536³ in every arm. 1M steps at
  1024³ is still ~56 days on one node; the next lever is the output side, not attention.

### Intermediate sizes (after the channels-last head)

The sub-pixel head was then switched to channels-last (`SubPixelHead._expand_tokens`). cuDNN had
been transposing every refinement convolution and falling back to Ampere kernels for their
backward. The switch took 1024³ from 7.70 to 6.88 s eager (block windows, 4³-pooled) and from 4.86
to 4.37 s compiled (2³-pooled), and saved 23 GiB compiled (`probes/head_channels_last/`).

Two other changes gave no speed:
- computing the prefix queries' attention as matmuls (reverted);
- replicating the 303M-parameter model instead of FSDP-sharding it (`--dims 8,1,1`). The
  collectives were already hidden, but it halves compiled memory (135 -> 69 GiB at 1024³).

Seconds per step, uncompiled, data parallel, all with the new head (`probes/intermediate_sizes/`;
~10% faster than the rows above). Block windows need the 16-token window to tile the grid, so they
cannot run 640³ or 896³ (40 and 56 tokens per axis); the other dashes were not measured. The last
row is phase 1's model (`vitl_dp`, global attention in every block) re-measured with the new head:

| design | 512³ | 640³ | 768³ | 896³ | 1024³ |
|---|---|---|---|---|---|
| block windows | 0.95 | - | 3.01 | - | 7.07 |
| sliding windows | 1.11 | 2.09 | 3.58 | 5.70 | 8.46 |
| block + 4 full global blocks | 1.00 | - | 3.86 | - | 12.48 |
| sliding + 4 full global blocks | 1.14 | - | 4.33 | - | 13.59 |
| full global attention in all 24 blocks | 1.28 | - | 8.26 | - | 39.74 |

## Phase 3: a 3D ConvNeXt encoder? (2026-09-29)

A convolutional encoder is linear in voxels and needs no windows, and DINOv3 released ConvNeXt
T/S/B/L. In 3D its 7x7 depthwise convolution becomes 7x7x7: 343 multiply-adds per output, on CUDA
cores rather than tensor cores. So before porting the model, one block was timed at the shape each
stage sees in a 1024³ crop (`convnext_block_bench.py`; `probes/convnext_block/`). Each case ran on
one B300 under bf16 autocast, beside the windowed ViT-L encoder timed the same way.

Encoder blocks at a 1024³ crop, forward + backward with activation checkpointing, compiled, seconds
per GPU-step. Stem and downsampling are excluded, and the convolution uses PyTorch's own depthwise
kernel (contiguous layout):

| encoder | 7³ kernels | 5³ | 3³ |
|---|---|---|---|
| ConvNeXt-T | 10.33 | 4.30 | 1.09 |
| ConvNeXt-S | 12.97 | 5.40 | 1.38 |
| ConvNeXt-L | 25.96 | 10.80 | 2.78 |
| ViT-L/16 with 16³-token block windows (`vitl_dp_win16`) | 1.82 | | |

* **The depthwise convolution is the cost.** PyTorch's 3D depthwise kernel runs at 1.5-2 TFLOP/s,
  ~2% of the B300's 80 TFLOP/s CUDA-core peak. Its backward takes ~6x its forward. At stage 1 it
  is 73% (3³) to 97% (7³) of a compiled block's time.
* **cuDNN is worse.** Channels-last sends the convolution to cuDNN, whose backward takes 35-80 s
  per stage-1 block at every kernel size (ConvNeXt-T at 3³: 133 s per step). `cudnn.benchmark`
  does not help, so a port would keep the contiguous layout.
* **Tensors of 2^31 or more elements need the convolution split into channel chunks.** That covers
  ConvNeXt-L's stage 1 (192 x 256³), and the split is exact. Unchunked, the kernel takes 4x longer
  (16.9 vs 4.0 s at 7³).
* **Activation checkpointing is required at 1024³.** Without it the blocks keep ~200 GiB
  (ConvNeXt-T) to ~510 GiB (L) of activations.
* **Only 3³ kernels beat the ViT, and only in ConvNeXt-T and -S.** Their encoder costs 1.09-1.38 s
  against 1.82 s, which would take the 4.37 s compiled step to about 3.6-3.9 s. At DINOv3's 7³
  kernels, the ConvNeXt is 6-14x slower than the windowed ViT-L.

### Dense-convolution blocks instead (2026-09-30)

The same benchmark was repeated with blocks whose spatial mixing is a dense 3³ convolution, which
runs on the tensor cores (`dense_block_bench.py`; `probes/dense_block/`). The table covers encoders
at a 1024³ crop, compiled, with activation checkpointing. Layouts are channels-last, except the
ConvNeXt, which is contiguous. Stem and downsampling are excluded. MFU is 3x the forward FLOPs per
step over the time, against 2,250 TFLOP/s:

| encoder | block (forward FLOPs per position, width C) | params | FLOPs/voxel | s/GPU-step | MFU |
|---|---|---|---|---|---|
| ViT-L/16, 16³ block windows | attention + MLP (24d² + 4Nd per token) | 303M | 246k | 1.82 | 19% |
| Fused-MBConv, ConvNeXt-T widths and depths | 3³ to 4C, 1³ back (224C²) | 362M | 230k | 1.60 | 21% |
| ResNet-34 layout, ConvNeXt-T widths | two 3³ (108C²) | 153M | 107k | 0.66 | 23% |
| ResNet-34 layout | two 3³ (108C²) | 68M | 48k | 0.25 | 27% |
| ResNet-50 layout | 1³ to C/4, 3³, 1³ back (4.4C²) | 44M | 31k | 0.35 | 13% |
| FasterNet, ConvNeXt-T widths and depths | 3³ on C/4, 2x MLP (11.4C²) | 18M | 12k | 0.35 | 5% |
| ConvNeXt-T, 3³ depthwise | depthwise + 4x MLP (16C²) | 26M | 17k | 1.09 | 2% |

* **A dense 3³ convolution alone runs at 38-71% of peak from 64 channels up**, channels-last. The
  3³ depthwise one ran at ~2 TFLOP/s.
* **Most of the loss is at the high-resolution stage.** At 256³ positions a dense convolution runs
  at 18-39% of peak, and Fused-MBConv spends 66% of its time in stage 1 for 42% of its FLOPs. At
  strides 8-32 the compiled basic and Fused-MBConv blocks run at 37-63%.
* **The FLOP-lean designs leave the tensor cores idle:** depthwise, FasterNet and the bottleneck
  run at 2-13% MFU.

### End to end: `convnet3d` against the ViT (2026-09-30)

Both dense-conv blocks were then built into a whole encoder, `src/models/convnet3d.py`. It is
sized to the windowed ViT-L's arithmetic: ~250k forward FLOPs per voxel against the ViT's 246k,
and ~400M parameters against its 303M. Three quarters of the FLOPs sit at stride 16, the ViT's
token grid, and most of the rest at stride 8. The runs used the same sub-pixel head and the same
sweep, on one B300 node, data parallel (`jobs/cnn_*.jsonl`). Every arm stopped on a CUDA
out-of-memory at the next size. Seconds per step:

| arm | 512³ | 768³ | 1024³ | 1536³ |
|---|---|---|---|---|
| convnet3d, basic blocks (`cnn_basic_dp`) | 0.64 | 2.00 | 4.71 | 18.03 |
| convnet3d, Fused-MBConv (`cnn_fused_dp`) | 0.65 | 2.02 | 4.78 | 18.34 |
| ViT-L, block windows (table above) | 0.95 | 3.01 | 7.07 | - |
| convnet3d, basic blocks, compiled | 0.42 | 1.31 | 3.02 | OOM |
| convnet3d, Fused-MBConv, compiled | 0.42 | 1.30 | 3.03 | OOM |
| ViT-L, block windows + 4 global (2³-pooled), compiled | - | - | 4.37 | OOM |

* **At the ViT-L's FLOP budget, the convolutional encoder takes a third off the step at 1024³:**
  7.07 → 4.71 s eager and 4.37 → 3.02 s compiled. The two block types are within 2% of each other.
* **The memory frontier is unchanged:** 1536³ eager and 1024³ compiled per GPU. Memory is set by
  the head and targets, not the encoder: 78 GiB at 1024³ eager, against the ViT's 73.
* **Per-voxel cost stays nearly flat from 1024³ to 1536³** (4.4 → 5.0 ns/voxel eager). Every ViT
  arm roughly doubled there, but its 1536³ rows predate the head change, so this does not show
  which side that slowdown came from.
* **1M steps at 1024³ compiled** would take ~35 days on one node, against ~51 for the ViT.
* The sweep ran the model before its parameters were renamed for layerwise learning-rate decay.
  The arithmetic is the same, except that the renamed layout also recomputes the two downsampling
  layers under activation checkpointing.

### A U-Net decoder on `convnet3d` (2026-09-30)

`UNetHead` (`layers/common/dense_heads.py`, `decoder = "unet"`) is UNETR's decoder for an encoder
with a real pyramid. Its skips are `convnet3d`'s own stages at strides 8 and 4, plus a residual
block on the raw image at full resolution. Its widths are 16/32/64/128 at strides 1/2/4/8, UNETR's
defaults. It decodes in exact slabs, each stride cut to what the next finer one reads, and
`--decode-chunks auto` sizes the slabs to its full-resolution concatenation. The arm is
`cnn_basic_dp_unet`, which is `cnn_basic_dp` with the head swapped. Seconds per step, with the
peak allocated memory in brackets:

| head | 512³ | 768³ | 1024³ | 1536³ |
|---|---|---|---|---|
| sub-pixel, compiled | 0.42 (19 GiB) | 1.31 (44) | 3.02 (124) | OOM |
| U-Net, compiled | 0.79 (43) | 2.59 (215) | OOM | - |
| sub-pixel, eager | 0.64 (23) | 2.00 (48) | 4.71 (72) | 18.03 (177) |
| U-Net, eager | 3.72 (63) | 12.80 (92) | 32.20 (115) | 135.24 (221) |

* **Compiled, the U-Net head doubles the step** (+0.37 s at 512³, +1.28 s at 768³, i.e. about
  +2.8 ns/voxel). It does ~2.5x the sub-pixel head's FLOPs per voxel, ~89k against ~36k. Most of
  them sit in 16- and 32-channel convolutions at full and half resolution, where the tensor cores
  run at ~5% of peak.
* **It runs out of memory at 1024³ compiled, and the cause is compile, not the head.** Compiled
  slab decoding holds 215 GiB at 768³ against 92 GiB eager: the slabs stop bounding memory once
  the loop sits inside one compiled graph. The sub-pixel head shows the same effect, less
  severely (124 against 72 GiB at 1024³). Not investigated further.
* **Eager it is 5-7x the sub-pixel head.** Autocast leaves the norm and activation outputs of every
  full-resolution block in fp32, and only compilation fuses them away. As on the ViT's UNETR,
  compilation is mandatory for this head.

## Where things are

`/nrs/scicompsoft/orhane/mia-train-experiments/large_inputs/`: `jobs/<arm>.jsonl` (one record per crop
that fitted), `jobs/<arm>_<jobid>.{log,err}`, the generated `jobs/<arm>.sh`; smoke runs under `jobs/smoke/`.
