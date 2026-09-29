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
harness here takes any registered model and algorithm, so those arrive as new configs.

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

(pending: `python experiments/large_inputs/table.py /nrs/scicompsoft/orhane/mia-train-experiments/large_inputs/jobs/*.jsonl`)

## Where things are

`/nrs/scicompsoft/orhane/mia-train-experiments/large_inputs/`: `jobs/<arm>.jsonl` (one record per crop
that fitted), `jobs/<arm>_<jobid>.{log,err}`, the generated `jobs/<arm>.sh`; smoke runs under `jobs/smoke/`.
