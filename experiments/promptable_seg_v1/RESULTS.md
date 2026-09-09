# Results

## Arms 2-7 — device, attention kernel and `torch.compile`

**Headline: the device is not the lever. This step is host-bound — the GPU is idle roughly
three-quarters of every step on both an H100 and a B300 — so which accelerator it runs on barely
matters, and the thing worth attacking is the 75% idle.**

That conclusion took three rounds of measurement, and the first two were wrong for reasons worth
recording.

### Round 1 (v1, `time_*.toml`) — wrong scale, wrong conclusion

A from-scratch 11M-parameter ViT3D at a 128-cube, 300 steps, 2 GPUs. Median of the last six
`samples_per_s` windows:

| arm | samples/s | vs baseline |
| :--- | ---: | ---: |
| `h100_fa4_eager` | 19.82 | 1.00x |
| `h100_sdpa_eager` | 20.29 | 1.02x |
| `h100_sdpa_compile` | 28.82 | 1.45x |
| `b300_fa4_eager` | 9.33 | 0.47x |
| `b300_sdpa_eager` | 9.44 | 0.48x |
| `b300_sdpa_compile` | 11.15 | 0.56x |

Read naively this says a B300 is half an H100. **It does not generalise, because the encoder was
too small to matter.** With an 11M encoder the step is almost entirely the decoder's many tiny
kernels, so the arm measured the part of the workload that scales *least* with a real model.

### Round 2 (v2/v3, `time2_*.toml`, `time3_*.toml`) — realistic scale

DINOv3 ViT-L/16 (306M) at a 256-cube, 4096 patch tokens, 2 GPUs, 20 dataloader workers so that
`data_wait_frac` is ~0.001 in every arm and nothing is input-bound. (At 8 workers two H100 arms
went input-bound at `dwait` 0.35-0.63 — a *faster* step outruns the loader, so those numbers were
loader ceilings, not compute.)

| arm | samples/s | vs H100 eager |
| :--- | ---: | ---: |
| `h100_fa4_eager` | 4.499 | 1.02x |
| `h100_sdpa_eager` | 4.420 | 1.00x |
| **`h100_sdpa_compile`** | **5.740** | **1.30x** |
| `b300_fa4_eager` | 3.203 | 0.72x |
| `b300_sdpa_eager` | 3.331 | 0.75x |
| **`b300_sdpa_compile`** | **5.527** | **1.25x** |

The gap closes from 0.47x to 0.75x eager, and **compiled the two devices are within 4% — tied,
well inside the ~16% run-to-run drift these nodes show.** So the scale of the encoder decided the
answer, and the v1 conclusion was an artefact.

Single GPU (`time4_*.toml`, `dp_shard = 1`) shows the same shape, so FSDP2 is not the explanation:
H100 3.015 / B300 1.997 eager, H100 5.332 / B300 4.797 compiled.

### Round 3 (v5, `time5_*.toml`) — the repo's profiler, on the real step

`[trainer].profile` plus `python -m engine.profiler <run dir>`. This is the measurement that
actually answers the question, because it separates *device* time from *step* time:

| | H100 eager | H100 eager (repeat) | B300 eager | B300 compiled |
| :--- | ---: | ---: | ---: | ---: |
| mean step | 1319.5 ms | 548.7 ms | 762.1 ms | 565.1 ms |
| GPU busy | 301.2 ms | 261.7 ms | 182.5 ms | 152.1 ms |
| **GPU busy %** | 22.8% | **47.7%** | 24.0% | 26.9% |

**Read the busy percentage with care, and never from one trace.** Two runs of the identical H100
config produced 22.8% and 47.7%. Worse, profiling is itself expensive here -- throughput inside
the window drops to 1.25 samples/s against ~4 outside -- so the busy *fraction* is systematically
understated: the device work is roughly right, the step it is divided by is not. Against the
unprofiled step time from the same runs (~490 ms H100, ~610 ms B300), busy is nearer **60% on
H100 and 30% on B300**.

Three things follow, and they replace everything above:

1. **A large fraction of every step the device does nothing** -- roughly 40% on H100 and 70% on
   B300 once profiler overhead is discounted. The workload is substantially host-bound on both, and
   more so on B300. A faster GPU cannot fix that; it only idles more.
2. **The B300 hardware is behaving exactly as advertised.** It completes the step's device work in
   182.5 ms against the H100's 301.2 ms — **1.65x faster** — and separately wins the microbenchmarks
   (`device_probe.py`: 8192-cube bf16 GEMM 1700 vs 767 TFLOP/s, conv3d 1.55x, a RoPE application
   2.37x, kernel launch overhead marginally *lower*). Nothing about the device is throttled.
   The end-to-end deficit is host-side, and since it moves with node and mode it is not a property
   of the accelerator.
3. **The device split is uneven and instructive.** B300 is much faster on the compute-bound work
   (encoder 97.1 vs 158.7 ms of device time) and *slower* on the bandwidth-bound elementwise work
   (AdamW 101.7 vs 75.2 ms, grad clip 44.6 vs 16.4 ms). A step that is 25% optimizer will not show
   a device's arithmetic advantage.

~~Incidental but large: H100's FSDP2 all-gather costs 46.8 ms/step, 11.7x B300's 4.0 ms.~~
**RETRACTED -- see "The all-gather number was noise" below. A repeat of the identical config
measured 7.0 ms.**

### FlashAttention-4 and `torch.compile` cannot be combined

`h100_fa4_compile` dies at the first step with a NameError inside inductor-generated code --
`buf0 = empty_strided_cuda((s56, s8, s80, s28), ...)` with `s28` undefined -- raised from
`flash_attn/cute/interface.py` under a dynamo resume. A torch.compile x flash-attn-4 interaction,
not this algorithm's code.

It is a live landmine because `attention_backend` defaults to `"auto"`, which selects FA4 on
Hopper and Blackwell: **`[trainer].compile = true` requires SDPA.** That costs nothing -- FA4 is
1.02x SDPA on H100 and 0.96x on B300 at these shapes, i.e. a wash either way. Suspected trigger,
untested: the decoder's *cross*-attention has unequal query and key lengths (20 tokens against
4096 patches), a shape no other algorithm here presents.

### The all-gather number was noise, and so are the busy percentages from a single trace

A first trace put H100's `ncclDevKernel_AllGather_RING_LL` at 46.8 ms/step, the biggest kernel in
it. Repeating the *identical* config measured **7.0 ms** -- 6.7x less -- and the same repeat moved
the step from 1319.5 to 548.7 ms and GPU-busy from 22.8% to 47.7%. One trace is not a measurement
of anything here.

The physics says the same. 0.57 GiB moves between two GPUs per all-gather; `nvidia-smi topo -m`
reports **NV18 on both queues**, i.e. 18 NVLink connections, ~450 GB/s. That is ~1.3 ms of
transfer. A kernel reading 46.8 ms is not moving data for 46.8 ms -- NCCL device kernels spin-wait
on the peer, so a collective's duration is where **rank skew** accumulates, and on a host-bound
step the ranks drift. The collective is the accounting sink, not the cost.

### Does `fsdp_unit_budget_gb` help? No.

The feature (`[parallelism].fsdp_unit_budget_gb`, default `"auto"` = 5% of device memory) packs a
model's declared FSDP units into groups under a byte budget. Two things make it a non-lever here:

**At the default it is already at the coarsest setting.** ViT-L is 1.14 GiB in fp32 against a
4 GiB budget on an 80 GB H100, so all 24 blocks pack into one group -- and `_fsdp_unit_groups`
drops a single group, leaving the root as the only unit. There is nothing to coarsen. Tuning it
therefore means going *finer*, opposite to the direction the feature was built for.

Finer does not pay. H100, 2 GPUs, 20 workers, `dwait` ~0.0008 in every arm except where noted:

| `fsdp_unit_budget_gb` | units | samples/s |
| :--- | ---: | ---: |
| `auto` (default) | 1 (root only) | 4.647 |
| 0.5 | 3 | 4.739 |
| 0.25 | 5 | 4.940 |
| 0.05 | 24 (one per block) | 4.207 |

The spread across the first three is +6%, inside the ~16% drift these nodes show. Only the
per-block extreme is clearly worse (-9.5%), which reproduces the 7.7% loss the feature's own
docstring already measured on SimMIM and is the reason the budget exists.

The profiled pair says the same from the other side: at 3 units the all-gather measured 19.8 ms
against the root-only run's 7.0 ms, and the step was slower (659.3 vs 548.7 ms). If the collective
were exposed transfer on the critical path, splitting into units it could overlap with would have
helped; it does not, because there is little transfer to hide.

**Where the budget *would* matter is memory, not speed, and at a scale this model does not reach.**
Its docstring records the case it was built for: the same per-block split saves ~28 GB at 7B and is
what makes that run fit.

### Does `miao`'s `defer_image_ops` help? No -- and not for the reason expected.

The timing arms above all ran with `defer_image_ops` unset, i.e. miao's default `False`, so the
workers did the image resample/normalize/cast on CPU. `MiaoVolumeDataset.finish_batch` puts that at
235 ms of one core per 256-cube sample, 47% of it in the trilinear interpolation, and moving it to
the device is measured elsewhere in this repo at 2.97x on the loader alone. So it was worth asking.

Two mechanisms could have made it matter, and `data_wait_frac` only rules out the first:

1. **Loader throughput.** `dwait` ~0.0008 in every trusted arm says the loop never waits for a
   batch, so a faster loader has nothing to give.
2. **Host core contention.** `dwait` cannot see this. Workers can keep up while still taking the
   cores the training process needs to *launch kernels*, which on a host-bound step costs time
   with nobody ever waiting -- the failure `data/base.py` documents, where moving work into the
   workers left GPU busy time unchanged (235.5 vs 237.2 ms) and raised GPU *idle* by 16.5 ms/step.
   With 20 workers x 2 ranks = 40 processes on a 24-slot allocation, this looked live.

Measured, ViT-L @ 256^3, sdpa, eager, `dwait` ~0.0008 throughout:

| device | `defer_image_ops` | workers/rank | samples/s | avg cores used |
| :--- | :--- | ---: | ---: | ---: |
| H100 | false | 20 | 4.734 | 13.5 |
| H100 | false | 8 | 4.880 | 8.6 |
| H100 | **true** | 20 | **5.011** | 10.1 |
| H100 | true | 8 | 4.837 | 8.2 |
| B300 | false | 20 | 3.289 | 8.3 |
| B300 | true | 8 | 3.311 | 7.9 |

**No throughput effect.** The H100 spread is +5.8% across four arms, inside the ~16% drift these
nodes show; on B300 it is +0.7%. Both mechanisms are dead, and the second one died for a reason
worth recording: **the allocation was never saturated.** At `defer=false` with 20 workers the whole
job averaged **13.5 of its 24 slots**. The workers are blocked on zarr reads over network storage,
not burning CPU, so there was no contention to relieve. The "40 processes on 24 slots" worry was
wrong, and only the cores measurement could have shown that.

`defer_image_ops` does do what it claims -- at the same 20 workers it cuts average CPU from 13.5 to
10.1 cores, a quarter less host work for identical throughput. There is simply no bottleneck there
to collect.

**This also closes out what the host is actually doing.** The step is host-bound (GPU ~60% busy on
H100, ~30% on B300) and it is now clear that the input pipeline is not why: not its throughput, not
its CPU. What remains is single-threaded Python and ATen dispatch inside the training loop itself --
24 encoder blocks plus `masks_per_sample x rounds` = 48 decoder passes per step, each a long tail of
small ops. `defer_image_ops` cannot touch that. `torch.compile` can, which is why it is the one
setting in this whole exercise that moved anything (1.30x on H100, 1.66x on B300).

## What to actually do

* **Run on whichever queue is free.** Compiled, H100 and B300 are tied. Use H100 if you want the
  attention kernel choice to be irrelevant; use B300 if the FSDP2 collective cost matters to you.
* **Always set `compile = true` and `attention_backend = "sdpa"`.** 1.30x on H100, 1.66x on B300,
  and the two settings are coupled.
* **Set `defer_image_ops = true` anyway.** It is throughput-neutral here but costs a quarter less
  host CPU for it, which is free insurance the day a node is contended or the worker count rises --
  and it is what lets a *smaller* worker count keep the loader ahead.
* **The real headroom is the idle time, not the hardware.** Compilation recovers part of it
  (76.0% -> 73.1% idle on B300 while cutting the step 762 -> 565 ms). What remains is host work
  the profiler names directly: the optimizer step is 25.1% of the compiled step's host time, and
  `masks_per_sample x rounds` = 48 small decoder passes per step is a lot of dispatch. CUDA graphs,
  a fused optimizer, and reconsidering `rounds = 3` are the next things to try -- all of them
  larger levers than the choice of GPU.

## Method notes

* Wall clock on these nodes drifts up to 16% between identical runs; differences under ~20% are
  unresolved.
* `data_wait_frac` must be checked before any throughput number is believed. Twice in this
  exercise an arm was silently loader-bound, and a loader ceiling looks exactly like a compute
  result.
* A single Python-loop timing on each node suggested the B300 host was 2.1x slower single-core.
  That contradicts the CPUs' relative specification (the B300 node's Xeon 6747P is newer than the
  H100 node's Platinum 8468) and was one sample each with unknown co-tenancy, so it is recorded
  here as unreliable and was not used to draw any conclusion.

### What was changed to make compilation viable

`promptable/targets.pooled_masks` originally read three tensor values on the host per step
(`labels.max()`, `object_ids.max()`, `keep.sum()`) and looped over the batch in Python. Each is a
device-to-host synchronization on the critical path, and under `torch.compile` they are graph
breaks and shape recompilations. It now resolves each voxel's object slot with a batched
`searchsorted` over the drawn ids and scatters unclaimed voxels into a discard row, so every shape
is static and nothing is read back. `tests/unit/test_promptable_targets.py` pins it against
`F.avg_pool3d` either way.
