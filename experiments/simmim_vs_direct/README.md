# Does SSL pretraining on NISB help downstream instance segmentation?

One controlled comparison, two arms, differing **only** in where the encoder's weights come from.

```
                                        ┌─ arm A ─ 1_simmim_pretrain ─┐
Meta DINOv3 ViT-L/16 (2D, LVD-1689M) ───┤                             ├─ affinity_seg on NISB
                                        └─ arm B ─────────────────────┘
```

| | arm A | arm B |
|---|---|---|
| stage 1 | SimMIM on the NISB training cubes | — |
| stage 2 | `affinity_seg`, init from stage 1 | `affinity_seg`, init from the Meta checkpoint |

Everything else in stage 2 is identical between arms — same seed, steps, schedule, crop size, data.
The only difference is the `[init]` section, which is the point.

## Running it

```bash
# arm A stage 1 (SSL), then both fine-tunes
bash experiments/simmim_vs_direct/submit.sh
```

`submit.sh` makes arm A's fine-tune depend on its pretraining with `bsub -w`, so the three jobs can
be submitted at once.

## Reading the result

`affinity_seg` reports `loss` and `affinity_accuracy` on the held-out **val** cube (seed100). That
is a proxy, not the benchmark number: the real NISB metrics need affinities turned into instances
and scored against the skeletons, which happens in `~/projects/banis` and needs dependencies this
repo deliberately does not carry. Use the val curves to compare the arms during training, then run
the benchmark's own tooling on the final predictions for a number worth quoting.

## Choices worth knowing

- **ViT-L/16**, 303 M parameters. `layerscale_init = 1.0e-05` and `mask_k_bias = true` are *not*
  optional: the released checkpoints were trained with both, and a model without them silently
  computes a different function. The loader rejects that now rather than letting it through.
- **`pos_embed_rope_type = "superposition"`**, so the inflated 3D model starts out behaving
  exactly like its 2D self (`depth_scale` initialises at 0) and learns how much depth to mix in.
- **Crop 128³ at patch 16** gives an 8×8×8 token grid. Coarse for dense prediction — the affinity
  head upsamples 16× — but the patch size is fixed by the checkpoint, and both arms share the
  limitation, so the *comparison* is unaffected even though the absolute numbers suffer.
- **Only the NISB training cubes are ever read**, in both stages. The benchmark's rules forbid
  training on val or test, and that includes self-supervised pretraining.
