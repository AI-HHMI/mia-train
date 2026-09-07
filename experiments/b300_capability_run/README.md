# How large a volume can a 7B DINOv3 fine-tune on, on 8 B300 nodes?

Sixteen B300 nodes were added to the cluster, eight GPUs apiece, and up to ten are routinely free.
This asks what that buys for the largest model this repo can run: **DINOv3 ViT-7B/16 with axial
rotary embeddings, supervised on the affinity-segmentation task**, at batch 1 per data-parallel
rank so the crop is the only variable. Two parallelisms, one question each:

* **HSDP** — shard within a node, replicate across the eight. What the repo could already do.
* **HSDP + tensor parallelism** — does splitting the model across GPUs raise the ceiling, and by
  how much?

And the number that prompted it: **1600³ voxels is 1,000,000 tokens at this model's 16³ patch**,
so "can we train a 1M-context vision transformer" and "can we train a 1600-cube" are the same
question.

**Status: run and complete. The short answer is 1600³ — the full 1,000,005-token target — and
getting there needed two changes to the *algorithm*, not to the parallelism.**

## Tensor parallelism did not exist, so it was written

`ParallelDims` accepted a `tp` dimension and `parallelize.py` looked for a
`tensor_parallel_plan()`, but no model in the repo defined one — `tp > 1` was an error message.
Four things had to land before this experiment could ask its question. All are in `src/`, not here,
because none of them is experiment-specific.

**A fused q/k/v projection cannot be column-parallel the obvious way.**
`SelfAttention.qkv` is one `Linear(dim -> 3 * dim)` whose output is `[q | k | v]`, and a plain
`Shard(0)` cuts that concatenation into contiguous thirds of the *fused* extent rather than into
whole heads: at `tp = 4` and `dim = 8`, rank 2 receives global rows 12–17, which is the tail of `k`
and the head of `v`. Attention cannot be computed from that — and nothing about the shapes says so,
because the reshape to `(B, N, 3, heads, head_dim)` still succeeds and still produces numbers.
`distributed/tensor_parallel.py` uses `_StridedShard(0, split_factor=3)` instead, the placement
that means "this extent was already cut into 3 pieces before you sharded it". It hands rank *r*
exactly `[q_r | k_r | v_r]`, and `full_tensor()` still reassembles the ordinary `[q | k | v]`
global weight — so **a checkpoint written under tensor parallelism is laid out identically to one
written without it**. It is also the placement FSDP2 itself composes with a TP shard, so 2D
`(dp, tp)` works rather than being a special case.

**Sequence parallelism, because plain TP would not have moved this experiment's number.** Plain
Megatron TP shards the projections and leaves the residual stream replicated — and the residual
stream is precisely what activation checkpointing stores, 40 blocks of `(B, N, embed_dim)`, which
is the term that decides whether a crop fits. Sharding the token axis divides it by `tp` at
*identical* communication volume, since the all-reduce a plain TP block does is already an
all-gather plus a reduce-scatter. Torch's `PrepareModuleInput`/`PrepareModuleOutput` would do the
entry and exit but index `inputs[0]` expecting a tensor, and `SelfAttentionBlock.forward` takes a
*list* of per-crop token tensors; `ScatterSequence`/`GatherSequence` do the same job through that
signature, applied to the first and last block so the stream between them is never gathered.

**Gradient clipping across two device meshes.** Under TP the projections become DTensors on the 2D
`(dp, tp)` mesh while everything the plan deliberately leaves replicated — the patch embedding, the
learned tokens, the output norms, and every parameter the *algorithm* owns rather than the model —
stays on the 1D data-parallel one. `torch.nn.utils.clip_grad_norm_` takes one norm over all of them
by `torch.stack`, and rejects the mix:

```
All operands in aten.stack.default must have the same mesh, but got
DeviceMesh((dp_shard=2, tp=2)) and DeviceMesh((dp_shard=2))
```

That is to say `tp > 1` and `grad_clip_norm` were mutually exclusive. `distributed/grad_norm.py`
takes the norm per mesh and combines the plain scalars — `‖g‖_p` over a partition is the p-norm of
the parts' p-norms, so nothing is approximated. A run whose gradients are all on one mesh is handed
straight to torch, so the common case keeps the exact code path it has always had.

**Per-block FSDP units.** FSDP2 all-gathers a unit's parameters for the whole of that unit's
forward. With the model as the only unit — which is what `parallelize.py` did — a sharded 7B still
materializes 27 GB of fp32 weights on every rank and 27 GB of gradients beside them: the optimizer
state is sharded and nothing else is. `BaseModel.fsdp_units()` names the transformer blocks, so a
block's weights are gathered when it runs and resharded when it finishes.

### One ordering constraint, now enforced

**Tensor parallelism must be applied before activation checkpointing.** `checkpoint_wrapper`
re-parents a block under `_checkpoint_wrapped_module`, and `parallelize_module` resolves a plan's
paths through `named_children()` — so a plan written against `blocks.0.attn.qkv` matches nothing
once the block is wrapped, and *matching nothing is not an error there*. The run would train, fully
replicated, at `tp` times the memory the setting was chosen to avoid. `engine.trainer` now
sequences TP → checkpointing → sharding, and `apply_tensor_parallel` refuses a model that is
already wrapped rather than silently doing nothing.

### What was verified, and against what

`tests/distributed/test_tensor_parallel.py`, 10 cases on 2 and 4 Gloo ranks. Every check is against
an unparallelized copy of the same weights rather than against a shape or a finite-value assertion,
because the failure this guards is silent — a wrongly sharded fused qkv attends with half of `k`
where `v` belongs and nothing in a loss curve says so.

| what | result |
|---|---|
| forward vs one process, `tp = 2` and `tp = 4` | max abs diff < 1e-5 |
| forward with an *indivisible* sequence (11 tokens over 4 ranks) | < 1e-5 |
| every parameter's gradient vs one process | < 1e-5 |
| fused qkv local rows are this rank's heads' q, k, v | exact |
| `full_tensor()` reassembles the unsharded global weight | exact |
| `LinearKMaskedBias.bias_mask` cut where the bias is cut | exact |
| grad-norm clipping across the 2D and 1D meshes | matches one process to 1e-4 |
| TP after checkpointing | refused, with the reason |
| stochastic depth under TP | refused (its subset path bypasses the hooks) |

## Method

`capability_sweep.py` measures **one** crop size: it builds the model and algorithm named in an
ordinary mia-train config, parallelizes them exactly as `engine.trainer` does, and runs a warm-up
plus a handful of real training steps — forward, backward, grad clip, optimizer. `submit.sh` walks
the sizes and stops at the first that does not fit, so the last size in the results file is the
answer.

One process per size rather than a loop inside one, because a CUDA out-of-memory inside a
collective leaves the other ranks waiting rather than raising: there is nothing reliable to catch,
so the process is allowed to die and the shell loop stops.

Three things the numbers do and do not include:

* **The input pipeline is excluded.** The batch is synthetic and already on the device. That cost
  is real but it is a separate, already-characterised problem, and folding it in would make the
  memory frontier depend on how many dataloader workers happened to keep up. Everything downstream
  of the batch is measured, **including the affinity target construction**, which allocates several
  tensors per voxel and turns out to matter a great deal.
* **`split_disconnected` is off**, as it effectively is in a real run: the connected-components
  pass runs in dataloader workers (`AffinitySegmentation.sample_transform`), off the step's
  critical path. Leaving it on would have put a CPU pass with ~30 host synchronizations per sample
  inside a GPU measurement.
* **The head is `subpixel`.** It is the cheaper of the two — roughly a sixth of the multiply-adds
  and a quarter of the activation memory of the interpolating head at 256³ — so this is the
  frontier at its most favourable.

The architecture is read off the released checkpoint
(`dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth`): `embed_dim` 4096, depth 40, 32 heads, SwiGLU
with `w1`/`w2` 8192 wide, LayerScale on both branches, 4 storage tokens, no qkv bias. Two
deliberate departures, both about what is being measured: `in_chans = 1` (EM is single-channel,
which *grows* the model — the 3D stem is 16.8M parameters against the 2D stem's 3.1M), and
`untie_global_and_local_cls_norm = false` (that norm is reached only by DINOv3's multi-crop SSL
path, so a supervised fine-tune would carry 8K parameters that never receive a gradient).
**6,729,666,560 parameters** in the encoder, 4.47M in the head.

`pos_embed_rope_type = "vanilla"` is the axial rotary embedding asked for: depth gets its own third
of the rotary channels, rather than the superposition variant's zero-gated overlay on the 2D
layout.

### One hardware fact worth recording

A B300 reports **268.5 GB usable**, not the 288 GB the spec sheet quotes. That is the budget every
number below is against.

## Results

### The answer

**1M context is reachable. 1600³ = 1,000,005 tokens trains on 8 B300 nodes**, at 459.79 s/step and
241.3 GiB of 267.7 -- 27 GiB of headroom, not a squeak past.

| crop | HSDP `8x8x1` | HSDP+TP `8x2x4` | + chunked head | + chunked, `tp=8`, int32 labels |
|---|---|---|---|---|
| 512³ | 23.67 s / 85.3 GiB | 20.41 s / 60.9 GiB | 5.85 s / 58.6 GiB | -- |
| 1024³ | **OOM** | **OOM** | 374 s / 264.2 GiB | 60.14 s / 113.0 GiB |
| 1600³ | -- | -- | -- | **459.79 s / 241.3 GiB** |

The frontier moved 512³ -> 1024³ -> 1600³ over four findings, and **only one of them was a
parallelism setting**:

| what moved it | why |
|---|---|
| `decode_chunks` | the head's voxel-resolution tensors, not the 7B encoder, were the wall |
| sizing the chunk count against 2^31 | 8 chunks at 1024³ landed back on the cuDNN cliff; 100 did not |
| `tp = 8`, `dp_shard = 8` | shards the encoder's sequence and the optimizer state |
| **the label dtype** | a redundant int64 copy, ~24 GiB -- more than every parallelism change combined |

**Neither parallelism reaches the frontier on its own.** Both stock arms die at 1024³ on the *same*
32 GiB allocation, and it is not in the encoder:

```
HSDP     1024³: OOM, tried to allocate 32.00 GiB, 241.3 GiB in use of 267.7
HSDP+TP  1024³: OOM, tried to allocate 32.00 GiB, 247.2 GiB in use of 267.7
```

`1024³ = 1,073,741,824 voxels x decoder_readout_dim 16 x 2 bytes = 32.0 GiB` -- `SubPixelHead`'s
readout at voxel resolution. The affinity head is *replicated* across tensor-parallel ranks, so
every rank builds that tensor however hard the encoder is sharded. And shrinking it does not help
either: halving `decoder_readout_dim` halves the failing allocation and 1024³ still needs ~273 GiB.
The whole voxel-resolution stage is the wall, which is what pointed at decomposing it rather than
shrinking it.

**The last 24 GiB came from a dtype, not a mesh.** `_prepare_labels` ended in `.long()`, which on an
int32 label volume allocates a second copy at twice the size while the caller's batch still holds
the first -- 45.8 GiB of a ~265 GiB step at 1600³. Nothing downstream reads the width: the affinity
target only ever evaluates `labels > 0`, `labels != ignore_index` and `a == b`. Preserving a signed
integer dtype took 1600³ from a crash at 264.8 GiB to a clean run at 241.3, *and* made the step 12%
faster by taking the allocator off the ceiling.

### Chunk count has to be sized, not picked

`decode_chunks = 8` at 1024³ puts each slab at `1.074e9 / 8 x 16 = 2.147e9` elements -- **exactly
2^31 again**, the very limit chunking exists to stay under. That run came back at MFU 0.173 against
the 512-cube's 0.261 for precisely that reason. Sized properly (100 chunks, 0.38 x 2^31) together
with `tp = 4`:

| 1024³, 64 GPUs | s/step | peak GiB | Mvoxel/s | encoder MFU |
|---|---|---|---|---|
| `tp=1`, 8 chunks | 374.44 | 264.2 | 183.5 | 0.173 |
| **`tp=4`, 100 chunks** | **60.14** | **113.0** | **285.7** | **0.269** |

6.2x on the step and 2.3x on memory. The step time flatters it -- TP quarters the global batch --
so the honest figure is **1.56x more voxels per second**, which is exactly the MFU ratio.

### FlashAttention-4 is the wrong choice here, and forward-only benchmarks say otherwise

| | SDPA | FA4 |
|---|---|---|
| kernel, forward only, 32,773 tokens | 1504 TFLOP/s | 1611 -- **FA4 +7%** |
| kernel, **forward+backward**, 4,101 tokens | 1035 | 856 -- **SDPA +21%** |
| kernel, forward+backward, 32,773 tokens | 1382 | 1411 -- tie |
| **end to end**, 512³ chunked + compile | **4.70 s** | 5.33 s -- **SDPA +13%** |
| **memory**, same | **53.0 GiB** | 59.9 GiB |

FA4's forward is consistently 5-7% faster and its backward is slower; backward is ~3.5x the work,
so training sees the opposite of what a forward-only benchmark reports. The end-to-end gap is wider
than the kernel gap because FA4 is a CuTeDSL kernel Inductor cannot trace: it breaks the graph at
every attention call and forfeits the rotary-embedding fusion `compile` exists for, which is also
where its +6.9 GiB comes from. This *confirms* the repo's existing "FA4 on H200, off on B300" note
and supplies the mechanism it lacked -- the deficit is in the backward, and the crossover sits
between 4k and 33k tokens, so a result measured at 256³ should not be generalised.

Torch's default dispatch picks cuDNN at these shapes. Forcing `FLASH_ATTENTION` is 3.6x slower and
`EFFICIENT_ATTENTION` 8.7x slower, so the default is already the right one.

### `torch.compile`: a 2x win at 256³ that is worthless above it

It did not work with `tp > 1` at all until this experiment, and the bug is worth recording because
it was silent. Under compile the run died with

```
RuntimeError: aten.native_layer_norm.default got mixed torch.Tensor and DTensor
```

at `self.norm` -- the *final* norm, whose weights are deliberately replicated -- with a `Shard(1)`
DTensor arriving as its input. The cause: **Dynamo does not run a forward hook that replaces a
module's output**, and the sequence-parallel exit was such a hook. Forward *pre*-hooks are honoured,
which is why the entry worked and the failure surfaced forty blocks downstream, where nothing
connected it to a skipped hook. The exit now happens in the models' own `forward_features_list`,
where Dynamo traces reliably, and `GatherSequence` is gone.

With that fixed, the measured trade (one node, `dp_shard 2 x tp 4`, chunked, one config key apart):

| crop | attention share | eager | compile | speedup | eager mem | compile mem | mem ratio |
|---|---|---|---|---|---|---|---|
| 256³ | 17% | 1.33 s | **0.66 s** | **2.02x** | 29.0 GiB | 29.0 GiB | **1.00x** |
| 512³ | 62% | 3.30 s | 2.47 s | 1.34x | 30.0 GiB | 58.0 GiB | 1.93x |
| 1024³ | 93% | 60.20 s | 53.76 s | 1.12x | 105.0 GiB | 210.5 GiB | 2.00x |
| 1600³ | 98% | 459.79 s | **OOM** | -- | 241.3 GiB | >267.7 GiB | -- |

**Use it at 256³ and nowhere above.** Both trends are monotone and have the same cause: Inductor
fuses elementwise work, and elementwise work falls from 83% of the encoder to 2% as attention takes
over, while its fused buffers grow with the tensor. At 256³ it is 2x for free; at 512³ 1.34x for
+28 GiB; at 1024³ 1.12x for +105 GiB; at the frontier it does not fit.

A widely-quoted "compile saves ~17 GiB" figure from earlier in this experiment was measured at
`tp = 1` with 8 chunks and does **not** generalise -- in the configuration above compile *costs*
memory at every size past 256³. Compilation itself costs ~135 s per input shape, which is nothing
in a 100k-step run and most of the wall clock in a sweep.

**One caveat, unresolved.** At 64 ranks across 8 nodes (`dp_shard 8 x tp 8`) a compiled step
**hangs** in a collective -- CPU time frozen, GPUs at 100% utilisation with 0% memory activity and
idle power, the signature of NCCL spin-waiting. The same configuration at 8 ranks on one node runs
fine. So compile and tensor parallelism now compose *functionally*, but multi-node compiled TP is
not usable and has no diagnosis here.

### Levers, all measured at 512³

Four knobs touch this workload. They are orthogonal, which is why each was isolated rather than
bundled. The single-GPU-equivalent configuration is held fixed (`dp_shard 8`, batch 1/rank), so the
step times are directly comparable.

| configuration | s/step | peak GiB | encoder MFU |
|---|---|---|---|
| **interpolating head** (`decoder = "interpolate"`, the repo default) | 120.77 | 169.1 | 0.013 |
| sub-pixel head, `readout 16` | 23.67 | 85.3 | 0.064 |
| … + `torch.compile` | 22.73 | 67.2 | 0.067 |
| … `readout 8` | 5.41 | 74.1 | 0.282 |
| … `readout 8` + `torch.compile` | 4.53 | 57.6 | 0.337 |
| **… `readout 16` + `decode_chunks = 8`** | **5.85** | **58.6** | **0.261** |

**20.6x on time and 2.9x on memory** separate the repo's current default head from chunked
sub-pixel, at the same crop on the same hardware.

**The interpolating head is not a viable choice at these sizes**, and it fails for the same reason
by a wider margin. It upsamples to `decoder_hidden_dim = 64` channels at voxel resolution against
the sub-pixel head's `readout = 16`, so at 512³ it holds 8.59e9 elements -- *four times* past 2^31
rather than exactly on it. At 256³, where both are under the limit, it is only 1.29x slower (0.76 s
against 0.59 s), which is what its 4x arithmetic alone predicts. At 512³ it is **5.0x slower and
takes 2.0x the memory**. The cliff, again, and steeper.

**`torch.compile` is a memory lever here, not a time lever.** It saves ~17 GiB whichever head width
it is applied to (-18.1 GiB at readout 16, -16.5 at readout 8) and buys only 1.06-1.19x on time.
Both halves follow from the profile: it fuses the rotary embedding's cast/rotate/concat/multiply
chain, which churns 349 GB of `aten::mul` and 130 GB of `aten::cat` per step across 40 blocks x
(forward + recompute + backward) -- so the intermediates stop being materialised -- while leaving
`convolution_backward` to cuDNN, which is 72% of the time.

**`decode_chunks` beats halving `readout`, and does not cost head capacity.** The two reach almost
the same step time (5.85 s against 5.41 s), but chunking gives back more memory (58.6 GiB against
74.1) and is exact -- the head keeps all 16 readout channels. Halving `readout` is a change to what
the model *is*; chunking is a change to how it is evaluated. The ~1.5x extra arithmetic the halo
costs is repaid about six-fold by the kernels a smaller tensor unlocks.

### What this means for the repo

The head, not the 7B encoder, is what limits this workload -- in memory and in time, and by a wide
margin in each. In order of payoff:

1. **`decode_chunks` is the lever that moves the frontier**, and its value has to be *sized against
   cuDNN's 2^31 element limit* rather than picked round. `chunk_voxels x readout` must stay under
   2^31 including the halo, or the setting silently buys memory and forfeits the speed. There is a
   case for computing it rather than configuring it.
2. **`SubPixelHead.refine` and `.out` deserve the treatment `.expand` already got.**
   `_expand_tokens` exists because cuDNN evaluated the transposed convolution at 0.3% of peak, and
   it was replaced with an equivalent matmul. The refine convolutions are the remaining instance of
   the same pathology; chunking works around it rather than fixing it.
3. **`decoder = "interpolate"` is the wrong default at any large crop.** Its 64 voxel-resolution
   channels are 4x the sub-pixel head's readout, so it is four times *past* 2^31 at 512³ rather
   than on it: 5.0x slower and 2.0x the memory there, and 0.013 encoder MFU.
4. **`decoder_readout_dim` is a throughput knob, not a capacity knob.** 4.44x on the step at 512³,
   and halving it does not make 1024³ fit. Reach for it for speed, never for headroom -- and prefer
   chunking, which reaches the same speed without giving up head capacity.
5. **Tensor parallelism cannot help a replicated head.** It is worth having -- it is what takes
   1024³ from 374 s to 60 s -- but only once the head is chunked. Alone it moves the frontier not
   at all.
6. **`torch.compile` and `tp > 1` are currently mutually exclusive** (see above). Worth fixing:
   compile is ~17 GiB, which is twice the shortfall at 1600³.

The one thing none of this touches is that **attention is O(N²)**, so cost per voxel rises with
crop volume -- 1.0x at 256³, 2.2x at 512³, 11.5x at 1024³, 41.5x at 1600³ -- and measured
throughput follows: 1807, 1469, 286 Mvoxel/s. Bigger crops do less work per second, not more. They
buy *context*, and the price is quadratic. If ~1M-voxel context is genuinely wanted, the lever is
sub-quadratic attention -- windowed over the 3D grid, or a hierarchical encoder like the `muvit3d`
already in this repo -- not more hardware and not more parallelism.

## Reproducing


```bash
# both arms, 8 nodes each
bash experiments/b300_capability_run/submit.sh

# single-node smoke test of the whole path, ~4 minutes
NODES=1 SIZES="128 256" DIMS=2,4,1 WALL=1:00 WARMUP=2 STEPS=4 \
  bash experiments/b300_capability_run/submit.sh hsdp

# regenerate the tables above
python experiments/b300_capability_run/table.py \
  /nrs/scicompsoft/orhane/mia-train-scratch/b300_capability/*.jsonl
```

`FLOPS=1` adds `--measure-flops`, which counts the step's real arithmetic — affinity head and
activation-checkpoint recompute included — instead of only the analytic encoder-only figure. It is
opt-in because it costs an extra forward/backward *after* the timed window, and at the top of the
sweep a size that trains fine could fail there and be recorded as not fitting.

`hsdp.toml` and `hsdp_tp.toml` are ordinary mia-train configs, not sweep-only files: the sweep
overrides `[model].img_size` with the size it is measuring and ignores `[data]`, but set
`[data].patch_size` to the size that fitted and either one trains through `src/train.py` unchanged.
Add an `[init]` section as `configs/dinov3_nisb_finetune.toml` does to start from the released
weights; the sweep measures a from-scratch model because initialisation costs no memory or time.
