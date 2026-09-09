# Promptable ("segment this") instance segmentation, v1

The first runs of `algorithms/promptable_seg.py` — a Segment Anything-style model for EM/ExM/LM
volumes. A click, a box or a previous mask goes in; one object comes out, with a predicted IoU so
several candidates can be ranked.

This directory holds two things: the run that answers *does it learn at all on real data*, and five
timing arms that answer *which device to run it on*. Neither is a result about segmentation
quality — that needs whole-volume mask generation and `mia-evals`, which do not exist yet.

## Why this task rather than `affinity_seg`

Instance identity in these volumes is unbounded and arbitrary, so there is no vocabulary to predict
into. `affinity_seg` gets around that by predicting *relationships* between neighbouring voxels and
recovering instances in post-processing. A promptable model gets around it differently: it never
enumerates objects, it answers a question about one. That is the interface proofreading, sparse
annotation and whole-volume mask generation all want, and it is the only one of the two that can be
asked for a *particular* object.

## The corpus

`configs/data/lmd_instances.yaml`, generated and verified by
`/nrs/scicompsoft/orhane/mia-train-scratch/sam3d/make_instances_config.py`. Six training volumes
plus one held out, from the 138 `cell`/`neurite` label groups in `lmd-v0.0.1`.

Two measurements shaped it (`sam3d/instance_census.py`):

* **A 256-cube holds 8 to 5,533 objects, median ~70.** Far too many to decode per step, which is
  why objects are sampled — 8 per crop here, against the reference's up to 64 per GPU.
* **Four of the corpus's instance-labelled volumes have ZERO objects at their volume centre.**
  Their annotation is an offset sub-box. Every volume therefore carries a `bounding_box`; without
  one the model would be trained mostly on the claim that there is nothing to segment.

## Arm 1 — `escape.toml`: does it learn on real data?

The smoke run (`configs/promptable_smoke.toml`, 50 steps) only ever showed the collapse to
"predict nothing" that every mask model starts from, so it could not tell a strategy that works
slowly from one that does not work. 4000 steps, 2xH100, from-scratch ViT3D at a 128-cube.

**Result: yes.** Job 154205156.

| step | loss | val IoU (round 0) | val IoU (final round) | val oracle |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3.45 | 0.00 | 0.00 | 0.00 |
| 50 | 1.11 | 0.00 | 0.00 | — |
| 500 | 0.91 | 0.150 | 0.201 | 0.166 |
| 1000 | 0.80 | 0.195 | 0.266 | 0.204 |
| 2000 | 0.83 | 0.212 | 0.303 | 0.227 |
| 3000 | 0.78 | 0.221 | 0.320 | 0.262 |
| **4000** | **0.78** | **0.227** | **0.331** | **0.280** |

Monotone on every column, and **the interactive rounds are doing real work**: the final round is
0.10 IoU above round 0 throughout, so the corrections are corrections rather than noise. The
oracle-versus-picked gap (0.280 against 0.227 at round 0) is the ambiguity headroom the IoU head
has not yet learned to rank — about a fifth of the achievable score is being left on the table by
choosing the wrong one of the three candidates.

The 50-step plateau at loss ~1.1 is the trivial predictor: an empty mask scores dice ~1.0 and
almost no focal loss, so that is where the optimiser sits until the dice gradient pulls it off.
Reading the smoke run as a failure would have been wrong, and reading it as a success would have
been wrong too — it measured neither.

Two things worth carrying forward:

* **`valid_fraction` oscillates between 0.5 and 1.0** at batch size 2, i.e. one of the two crops
  routinely contains no eligible object. Those crops are correctly excluded from the loss rather
  than counted as empty, but half the batch is doing no work a good fraction of the time. Tightening
  the per-volume boxes, or weighting sampling by annotation density, is the cheapest available
  throughput win — bigger than anything in the timing arms below.
* **`final_oracle_iou` is identical to `final_iou` by construction**, since only round 0 is
  ambiguous and later rounds emit a single mask. It is not a bug and not informative; read
  `first_oracle_iou` for the ambiguity headroom.

## Arms 2-6 — `time_*.toml`: is a Blackwell device worth using here?

Prior measurement in this repo (2026-09-04, recorded in `b300-torch-compile-2x`) found that on
**simmim / affinity_seg / dinov3**, an eager B300 was *slower* than an eager H200 despite doing the
GPU work 1.33x faster — 57% GPU-busy against 99%, i.e. launch-bound — and that
`torch.compile(algorithm)` recovered 1.57x-2.30x, with SDPA rather than FlashAttention-4 on B300.

That is evidence, not a fact about *this* algorithm: it was a different device pair (H200, not
H100) and different step shapes. This decoder fires more small kernels than any of those, so the
same story is plausible and untested. Five arms, identical apart from the three variables:

| arm | queue | attention | compile |
| :--- | :--- | :--- | :--- |
| `time_h100_fa4_eager` | `gpu_h100` | FlashAttention-4 | no |
| `time_h100_fa4_compile` | `gpu_h100` | FlashAttention-4 | yes |
| `time_h100_sdpa_eager` | `gpu_h100` | torch SDPA | no |
| `time_h100_sdpa_compile` | `gpu_h100` | torch SDPA | yes |
| `time_b300_fa4_eager` | `gpu_b300` | FlashAttention-4 | no |
| `time_b300_sdpa_eager` | `gpu_b300` | torch SDPA | no |
| `time_b300_sdpa_compile` | `gpu_b300` | torch SDPA | yes |

Three rounds were needed. The first used an 11M-parameter encoder at a 128-cube and reached the
opposite conclusion, because at that scale the step is almost entirely the decoder's small kernels
-- the part of the workload that scales least with a real model. `time2_*`/`time3_*` repeat it at
DINOv3 ViT-L/16 (306M) and a 256-cube; `time4_*` isolate FSDP2 by dropping to one GPU; `time5_*`
turn on the repo's profiler. Wall clock on these nodes drifts up to 16% between identical runs, so
a difference under ~20% is unresolved.

**Results: `RESULTS.md`.** Summary: **this step is host-bound -- the GPU is idle ~75% of every
step on both devices** (measured with `python -m engine.profiler`), so the accelerator is not the
lever. Compiled, H100 and B300 are tied to within 4%. The B300 hardware is not throttled: it
completes the step's device work 1.65x faster and wins every microbenchmark. Always set
`compile = true` with `attention_backend = "sdpa"` -- worth 1.30x on H100 and 1.66x on B300, and
FA4 cannot be compiled at all (inductor emits an undefined symbol inside flash-attn's wrapper).

### What was changed to make compilation viable

`promptable/targets.pooled_masks` originally read three tensor values on the host per step
(`labels.max()`, `object_ids.max()`, `keep.sum()`) and looped over the batch in Python. Each is a
device-to-host synchronization on the critical path, and under `torch.compile` they are graph
breaks and shape recompilations. It now resolves each voxel's object slot with a batched
`searchsorted` over the drawn ids and scatters unclaimed voxels into a discard row, so every shape
is static and nothing is read back. `tests/unit/test_promptable_targets.py` pins it against
`F.avg_pool3d` either way.

## Submitting

```bash
JOBS=/nrs/scicompsoft/orhane/mia-train-jobs
bsub -P miaai -q gpu_h100 -gpu "num=2" -n 24 -W 2:00 -J psv1_escape \
  -cwd $JOBS -o $JOBS/psv1_escape_%J.log -e $JOBS/psv1_escape_%J.err \
  $JOBS/sam3d_smoke.sh experiments/promptable_seg_v1/escape.toml
```

`sam3d_smoke.sh` takes the config path as its one argument and is otherwise the standard
`torchrun --standalone --nproc_per_node=2 src/train.py` recipe from `deploy/lsf/README.md`.
