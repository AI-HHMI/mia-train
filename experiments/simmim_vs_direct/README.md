# Does SSL pretraining on NISB help downstream instance segmentation?

One controlled comparison. Three arms differ **only** in where the encoder's weights come from; a
fourth revisits the winner at twice the batch.

```
                                        ┌─ arm A ─ 1_simmim_pretrain ─┐
Meta DINOv3 ViT-L/16 (2D, LVD-1689M) ───┤                             ├─ affinity_seg on NISB
                                        └─ arm B ─────────────────────┤
random initialisation ───── arm C ───────────────────────────────────-┘
```

| | arm A | arm B | arm C | arm D |
|---|---|---|---|---|
| stage 1 | SimMIM on the NISB training cubes | — | — | — |
| stage 2 | `affinity_seg`, init from stage 1 | `affinity_seg`, init from the Meta checkpoint | `affinity_seg`, no init at all | arm B at global batch 16 |
| GPUs | 8 (1 node) | 8 (1 node) | 8 (1 node) | **16 (2 nodes)** |

Everything else is identical across arms — same seed, steps, schedule, crop size, data, and
architecture. The only difference is the `[init]` section, which is the point.

**Arm C is what makes the other two readable.** A and B alone answer "does *more* pretraining
help", and a small gap between them could mean either that SSL adds little or that the whole
pretraining question is unimportant here. Arm C sets the scale: it is the floor that says how much
of the final number is attributable to pretraining at all.

## Running it

```bash
bash experiments/simmim_vs_direct/submit.sh          # every arm
bash experiments/simmim_vs_direct/submit.sh D        # just one, once the others have run
```

`bsub -w` makes arm A's fine-tune wait on its pretraining; the rest start immediately.

To watch them together — including arms that started later:

```bash
bash experiments/simmim_vs_direct/tensorboard.sh     # then open localhost:6006
```

## Reading the result

`affinity_seg` reports `loss` and `affinity_accuracy` on the held-out **val** cube (seed100). That
is a proxy, not the benchmark number: the real NISB metrics need affinities turned into instances
and scored against the skeletons, which happens in `~/projects/banis` and needs dependencies this
repo deliberately does not carry. Use the val curves to compare the arms during training, then run
the benchmark's own tooling on the final predictions for a number worth quoting.

## Scale, and what it costs

Crop **512³** at global batch **8** — 512x the voxels per step of the first version of this
experiment (128³ at batch 2). At patch 16 that is a **32³ = 32768-token** sequence, and a
transformer's activation memory is linear in sequence length while the dense head's is linear in
voxels. Measured on one H200, ViT-L, batch 1:

| | 128³ | 256³ | 512³ | 512³ + activation checkpointing |
|---|---|---|---|---|
| `simmim` | 5.9 GiB | 10.4 GiB | 54.7 GiB | **11.1 GiB**, 1.5 s/step |
| `affinity_seg` | 6.5 GiB | 22.3 GiB | OOM (>140 GiB) | **112.7 GiB**, 6.1 s/step |

So `[trainer].activation_checkpointing = true` is what makes this run exist, and `gpu_h200` is
required — 112.7 GiB does not fit an 80 GiB H100.

Where `affinity_seg`'s memory goes, and why the head is shaped the way it is:

- The **encoder** costs only 4.8 GiB once checkpointed. It is not the problem.
- The **dense head** is. Its tensors are `decoder_hidden_dim x crop³`: 16 GiB each at 512³.
  Checkpointing `decoder_out` alone left 121.6 GiB, because a checkpointed region stores its own
  *inputs* and the upsampled tensor was one. Folding the interpolation into the module (`VoxelHead`)
  moved that boundary back to the patch grid and dropped retained memory from 50 GiB to 18 GiB.
- Autocast runs `F.interpolate` in **fp32**, which doubles every tensor in the head. Forcing it to
  bf16 would save ~40 GiB and let this fit on an H100. It was measured and **rejected**: bf16
  accumulation over eight neighbours is 1.4x less accurate than accumulating in fp32 and rounding
  once, with worst-case deviations of several percent of the feature scale. A bigger GPU is the
  cheaper thing to spend.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set in `submit.sh`) is also load-bearing: the
head allocates and frees 16 GiB tensors every step, and the default caching allocator strands
enough between segments to OOM a run whose live set fits.

## Choices worth knowing

- **ViT-L/16**, 303 M parameters. `layerscale_init = 1.0e-05` and `mask_k_bias = true` are *not*
  optional: the released checkpoints were trained with both, and a model without them silently
  computes a different function. The loader rejects that now rather than letting it through.
- **`pos_embed_rope_type = "superposition"`**, so the inflated 3D model starts out behaving
  exactly like its 2D self (`depth_scale` initialises at 0) and learns how much depth to mix in.
  Arm C keeps it too — not because random weights need it, but because varying the architecture
  and the initialisation together would make the comparison unreadable.
- **Local batch size stays 1**; the global batch of 8 comes from 8 GPUs. One 512-cube per GPU is
  what fits, so the batch grows with devices rather than with samples per device.
- **The token grid is still 32³ against a 512³ output**, so the head upsamples 16x. A larger crop
  buys the transformer more *context*, not finer output granularity — the patch size is fixed by
  the checkpoint. All arms share the limitation, so the comparison is unaffected.
- **`max_steps` is unchanged from the 128³ version** (8000 SSL / 6000 fine-tune) even though each
  step now sees 512x the voxels — roughly 105 passes over the training cubes rather than 0.4.
  Kept deliberately so the arms remain comparable to each other; worth revisiting once the first
  curves are in.
- **Only the NISB training cubes are ever read**, in every arm and both stages. The benchmark's
  rules forbid training on val or test, and that includes self-supervised pretraining.

## Result at 512³, global batch 8

| arm | val `affinity_accuracy` | val `loss` |
|---|---|---|
| **B** — Meta DINOv3 → fine-tune | **0.8713** | **0.2840** |
| A — SimMIM on NISB → fine-tune | 0.8648 | 0.2959 |
| C — from scratch | 0.7584 | 0.4832 |

Arm C is what makes this readable. A trivial predictor that answers "same object" everywhere scores
**~0.72** (the classes are that unbalanced — see `target_positive_rate`, logged beside the
accuracy). From-scratch reaches 0.758, barely above that floor. Pretraining is worth **+11 points**;
SimMIM on top of it cost a little rather than adding.

Two readings, and this experiment cannot separate them: NISB's five training cubes may be too
little data for SSL to add anything Meta's LVD-1689M pretraining has not already given, or 8000
SimMIM steps is too short. Arm D tests neither — it asks a different question.

## Arm D: is arm B batch-limited?

Arm B was the strongest, so arm D runs it again at global batch 16 (16 GPUs across 2 nodes, still
one 512-cube per GPU). It is also this repo's **first multi-node run**.

Read it with two caveats, both deliberate and both in the config:

- **The learning rate is not scaled.** Doubling the batch halves the gradient noise and the usual
  recipes raise the LR to compensate. Left alone so exactly one thing differs from arm B. If D
  loses, an unscaled LR is the first thing to try before concluding the batch size did not help.
- **`max_steps` is still 6000**, so D sees twice the samples of B, not the same samples in half
  the steps. "Same optimizer steps" was chosen to match every other arm.

Parallelism is **HSDP** (`dp_replicate = 2, dp_shard = 8`): sharding stays inside each node's
NVLink and only the gradient all-reduce crosses InfiniBand. Measured 8.02 s/step, against 8.05 for
flat FSDP over all 16 ranks and 7.1 for arm B on one node — so twice the samples for 1.13x the wall
clock, and the two multi-node shapes are indistinguishable at 306M parameters because the step is
compute-bound. NCCL uses InfiniBand with GDRDMA between nodes.

Getting here needed one real fix: DCP scatters its save plan as a pickled *object*, which over NCCL
crosses IB and faulted the queue pair, killing the first 16-rank run at its first checkpoint after
every training step had succeeded. `CheckpointManager` now routes plan coordination over a Gloo
group. See `deploy/lsf/README.md`.

## Previous result (128³, global batch 2)

The first version of this experiment finished with the two arms indistinguishable:

| | arm A (SimMIM → fine-tune) | arm B (direct) |
|---|---|---|
| val `affinity_accuracy` | 0.8318 | 0.8296 |
| val `loss` | 0.3585 | 0.3605 |

A 0.002 gap on one seed is noise. That null result is part of why this version runs at a scale
where the encoder has enough context to be worth pretraining, and adds arm C so the axis has a
zero point.

## Scaling up: the other NISB variants

A metadata-only sidecar, written once by a throwaway script since removed, unlocks the NISB
variants downloaded in August, which
`miao` could not read. It turned out **not** to be a bad download: NISB publishes each cube as a
plain zarr v2 group -- `data.zarr` holding flat `img`/`seg` arrays, exactly what the benchmark's
own BANIS code reads -- while miao needs OME-NGFF multiscale metadata. `base` and `liconn` are the
exceptions, converted to OME-NGFF by a colleague back in June.

The pixel data was always fine, so the script writes only the missing metadata and symlinks the
level `s0` at the published arrays: **~600 KB of JSON for 2.7 TB of data**, nothing copied,
nothing under the source tree modified.

| variant | cubes | now readable at |
|---|---|---|
| `train_100` | 100 train + val + test | `train_100-ngff/` |
| `multichannel` | 5 + val + test (**8 channels**) | `multichannel-ngff/` |
| `neg_guidance`, `pos_guidance`, `no_touch_thick`, `slice_perturbed`, `touching_thin` | 5 + val + test each | `<variant>-ngff/` |

`configs/data/nisb_train_100.yaml` is ready to use in place of `nisb_base.yaml` — **20x the
training data**, same crop size and conventions. Verified before use:

- `train_100/seed0` is byte-identical to `base/seed0`, which pins both the axis order the arrays
  are stored in (`x,y,z,c` for images, channel *trailing*, unlike `base`) and the 9x9x20 nm voxel
  size the published cubes never state.
- miao's `lcxyz` output at a given coordinate is bit-identical to a raw x,y,z read, so nothing is
  transposed — the failure mode that would otherwise train perfectly and score as nonsense.
- `train_100`'s val cube is byte-identical to `base`'s, so numbers stay directly comparable, and
  its train split (seeds 0-99) does not overlap val (100) or test (101).

Caveat: the published cubes carry no resolution pyramid, so only `resolutions: [[9, 9, 20]]`
works; a coarser request fails rather than silently downsampling. `multichannel` needs
`in_chans = 8`.
