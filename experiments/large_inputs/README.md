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
sub-pixel head. Each config differs from `vitl_dp.toml` only in the lines named here (`diff` them).

| arm | parallelism | head readout | what it asks |
|---|---|---|---|
| `vitl_dp` | dp_shard 8 x tp 1 (8 crops per node) | 16 (production) | the crop ONE GPU can hold, and its throughput |
| `vitl_tp8` | dp_shard 1 x tp 8 (1 crop per node) | 16 | does splitting the encoder 8 ways move the frontier past the replicated head? |
| `vitl_dp_ro8` | dp 8 | 8 | the readout-halved head: throughput through cuDNN's 2^31 limit |
| `vitl_tp8_ro8` | tp 8 | 8 | the largest-crop arm |
| `vitl_dp_compile` | dp 8, `compile = true` | 16 | does compile pay for a ViT-L, and to what crop? |

## Method

`capability_sweep.py` is a copy of b300_capability_run's driver (see its docstring for what changed):
it builds the model and algorithm from an ordinary config, parallelizes them exactly as the trainer
does, and times real training steps -- forward, backward, grad clip, optimizer -- on a synthetic batch
already on the device, so the input pipeline is excluded and everything downstream of the batch,
affinity targets included, is measured. `submit.sh` runs one `torchrun` per crop size, ascending, and
stops at the first size that records nothing; the last row in an arm's results file is its limit.
Sizes 512³ -> 3072³ (16³ patches: 32k to 7.1M tokens), 2 warmup + 4 timed steps.

**Chunking the head, per crop.** The sub-pixel head decodes in slabs along the first axis
(`decode_chunks`), each slab plus one halo patch-plane per side. Its convolutions fall off cuDNN's
fast kernels when a tensor exceeds 2^31 elements, and the right slab count depends on the crop: too
few slabs cross the limit, too many triple the halo arithmetic. `--decode-chunks auto` picks the
fewest slabs whose (slab + 2 halo planes) x crop² x readout stays under 0.9 x 2^31:

| crop | readout 16 | readout 8 |
|---|---|---|
| 512³ | 2 slabs | 1 (undivided) |
| 1024³ | 13 | 6 |
| 1536³ | 96 (one plane each) | 32 |
| 2048³ | over the limit even at one plane (128) | 128 |

So at 2048³ only a readout-8 head stays on the fast kernels; readout 16 runs there on the slow ones.

## Results

(pending: `python experiments/large_inputs/table.py /nrs/scicompsoft/orhane/mia-train-experiments/large_inputs/jobs/*.jsonl`)

## Where things are

`/nrs/scicompsoft/orhane/mia-train-experiments/large_inputs/`: `jobs/<arm>.jsonl` (one record per crop
that fitted), `jobs/<arm>_<jobid>.{log,err}`, the generated `jobs/<arm>.sh`; smoke runs under `jobs/smoke/`.
