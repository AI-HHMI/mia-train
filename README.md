# mia-train

PyTorch-native distributed training for volumetric microscopy. There is a single entrypoint 
([`train.py`](src/train.py)) for all runs, one config file per run, and everything a config 
can name lives in a registry.

Core dependencies are `torch`, `tensorboard` and `miao-io`. Optional extras add
capabilities without touching the core (see [Installing](#installing)).

## Installing

```bash
pip install -e .                    # core
pip install -e '.[affinity]'        # + cc3d, for the affinity instance-segmentation task
pip install -e '.[cellmap]'         # + HuggingFace datasets, for the CellMap tasks
pip install -e '.[dev]'             # + pytest, ruff, mypy
```

`affinity` is worth installing before any serious `affinity_seg` run. Without it the algorithm
still trains, and to the same targets, but it splits disconnected label components on the training
device instead of in the dataloader's workers — measured at 107 ms of every 377 ms step at 256³.
A run that falls back says so on startup.

Machine-local paths (dataset roots, checkpoint directory, venv, scheduler project) live in
`configs/cluster/active.toml` (untracked), which can be created by copying 
`configs/cluster/template.toml` and filling it in.

## Running a training job

Always run jobs through `torchrun`, never `python src/train.py` directly:

```bash
torchrun --standalone --nproc_per_node=<gpus> src/train.py --config configs/<run>.toml
```

Useful flags:

- `--resume` continues the newest run of this `experiment_name`, or starts fresh if there is none,
  so one submission script serves the first launch and every resubmission after a wall-time limit.
- `--resume <dir>` continues that exact run directory.
- `--output-root <dir>` overrides where artifacts go (default: `[environment].checkpoint_dir`).

Each run writes `<checkpoint_dir>/<experiment_name>_<timestamp>/` containing checkpoints,
TensorBoard logs, `resolved_config.json` (every setting expanded), the full git commit hash, 
and a copy of any referenced data config.

To submit training jobs on a cluster, see [`deploy/lsf/README.md`](deploy/lsf/README.md)
for single-node, multi-node and resume recipes.

## Configuration

A run is one TOML file with these sections:

```toml
experiment_name = "my_run"

[model]        # name = a registered model, plus its constructor kwargs
[init]         # optional: start from weights trained elsewhere (path, prefix, inflate_2d_to_3d, ...)
[algorithm]    # name = a registered algorithm, plus its kwargs
[data]         # name = a registered dataset; or config_path = a shared data config
[val_data]     # optional: same shape as [data]
[trainer]      # steps, batch size, lr, schedule, precision, checkpointing cadence
[augment]      # optional: training-data augmentation (never applied to [val_data])
[parallelism]  # dp_replicate, dp_shard, tp; must multiply to the torchrun world size
```

`[trainer].batch_size` is **per rank**, so the global batch is `batch_size × dp_replicate × dp_shard`.

See `configs/*.toml` for working examples.

## What's available

**Models** (`[model].name`)

| name | |
|---|---|
| `dinov3_vit` | DINOv3 vision transformer, 2D. Can load Meta's released checkpoints. |
| `dinov3_vit3d` | The same architecture for volumes, with 2D/3D RoPE variants. A released 2D checkpoint can be inflated into it via `[init].inflate_2d_to_3d`. |
| `vit3d` | Plain 3D ViT with split `embed`/`encode`, so an algorithm can drop tokens between them. |
| `muvit3d` | Multi-scale ViT consuming several resolution levels at once. |

**Algorithms** (`[algorithm].name`)

| name | |
|---|---|
| `mae` | Masked autoencoding; drops masked tokens from the encoder. |
| `muvit_mae` | Masked autoencoding across scale levels, one decoder per level. |
| `simmim` | Masked image modelling by mask-token substitution. Works with DINOv3 encoders, which MAE cannot use because they cannot consume a scattered token subset. |
| `dinov3` | The DINOv3 self-supervised objective: teacher/student EMA, DINO + iBOT losses, Sinkhorn centring, KoLeo. Rank-agnostic. |
| `affinity_seg` | Supervised instance segmentation by affinity prediction (e.g. the NISB task). |
| `semantic_seg` | Supervised per-voxel classification (serves both 2D and 3D). |

**Datasets** (`[data].name`)

| name | |
|---|---|
| `miao_volumes` | Multi-scale OME-NGFF volumes via `miao`, for data far larger than memory. Configure inline or point `config_path` at a shared YAML (`configs/data/`). |
| `hf_semantic_seg` | Segmentation datasets from the HuggingFace Hub, read as memory-mapped Arrow. `preset = "cellmap_2d"` or `"cellmap_3d"`; other Hub datasets need only an entry in `PRESETS` or the fields spelled out inline. Needs the `cellmap` extra. |

**Evaluations**

| name | |
|---|---|
| `semantic_seg` | Whole-volume scoring with overlap-blended tiled inference, plus `mode = "orthoplane"` to apply a 2D model to a 3D volume by averaging the x, y and z passes. Reports IoU and Dice from one confusion matrix over the whole set. |

Adding a component means writing it and adding one line to `src/components.py`; the engine and the
registries never change.

## Other capabilities

- **Parallelism:** FSDP2 sharding, DDP replication, HSDP (shard within a node, replicate across),
  and tensor parallelism for models that declare a plan. Multi-node is validated; see
  `deploy/lsf/launch_multinode.sh`.
- **Activation checkpointing:** `[trainer].activation_checkpointing = true`. Models and
  algorithms declare which regions are worth recomputing.
- **Throughput and MFU:** every run logs `mfu`, `tflops_per_s` and `samples_per_s` alongside the
  loss. A step's FLOPs are measured once at startup with `torch.utils.flop_counter`, not derived
  from `model.flops()`. Peak throughput comes from a per-GPU table (`src/utils/hardware_flops.py`); 
  override it with `[trainer].peak_tflops`, or set `[trainer].measure_mfu = false` to skip the probe. 
  On an untabulated GPU the utilization is omitted and the throughput rates are still reported.
- **Profiling:** `[trainer].profile = true` traces a few steps and writes them to
  `<run>/profile/`. See [Profiling a run](#profiling-a-run) below.
- **Work in the dataloader's workers:** an algorithm may declare per-sample preprocessing via
  `BaseAlgorithm.sample_transform()`, and the engine attaches it to the training *and* validation
  datasets — unlike `[augment]`, since this builds targets rather than perturbing inputs. It is
  for work that depends only on one sample and would otherwise sit between the batch arriving and
  the loss. `affinity_seg` uses it for the connected-components pass over its label crop, which
  needs the `affinity` extra; the profiler section below is how that was found.
- **Augmentation:** `[augment]` adds volumetric EM augmentations: dropped and shifted sections,
  intensity jitter, noise, to the training data. Applied to the training dataset only; the engine
  never wraps `[val_data]`, so no setting can silently change what a validation number means.
- **Pretrained weights:** `[init]` loads a released or earlier-run checkpoint, reports what was
  copied, inflated, skipped or unused, and refuses silently-wrong loads.
- **Optimizer:** AdamW with layerwise LR decay, a separate patch-embedding multiplier, weight-decay
  scheduling, and rank-based weight-decay exemption. Linear or cosine LR decay after warmup.
- **Checkpointing:** Saves PyTorch distributed checkpoint (DCP), so a run can resume under a 
  different parallelism layout than it was saved with.

## Profiling a run

**1. Turn it on:** Add to `[trainer]`, or copy [`configs/profile_example.toml`](configs/profile_example.toml),
which is a small runnable run with every knob commented:

```toml
[trainer]
profile = true              # everything below is optional
profile_start_step = 50     # counted from this process's first step, so resumed runs still profile
profile_steps = 6
profile_all_ranks = false   # true to catch load imbalance between ranks
profile_memory = false      # allocator activity, for activation-checkpointing decisions
```

Run the job exactly as usual. The profiler trace lands in `<run>/profile/`.

**2. View the trace:**

```bash
python -m engine.profiler <run directory>
```

That prints where the step went:

```
  profiled steps        6
  mean step                446.0 ms
  GPU busy                 359.9 ms/step    81.5%
  GPU idle                  81.6 ms/step    18.5%   <- host could not keep the device fed

  --- annotated regions, device time ---
  region                              ms/step   % step
  forward                               199.8    44.8%
  relabel_connected                     106.8    23.9%
  encoder                                48.7    10.9%
  ...
  host blocked in cudaStreamSynchronize & co: 124.1 ms/step over 41 calls/step

  --- slowest kernels ---
  void at::native::...::upsample_trilinear3d_backward...             90.19    20.2%
  Memcpy HtoD (Pageable -> Device)                                   21.70     4.9%
```

It accepts a run directory, its `profile/` subdirectory, or a trace file, and picks the newest
trace it finds.

**3. For the full timeline:** Open `<run>/profile/*.pt.trace.json` at
**[ui.perfetto.dev](https://ui.perfetto.dev)** and drag the file in (it
reads the gzipped form too). The command in step 2 prints the exact path to drag.

## Experiments

[`experiments/`](experiments/) holds complete, runnable examples: configs, submission scripts and
a README each explaining what the experiment tested and what it found, e.g.:

- [`simmim_vs_direct/`](experiments/simmim_vs_direct/): does SSL pretraining help downstream
  instance segmentation? Four arms differing only in initialisation, plus one at twice the batch
  across two nodes.
- [`banis_parity/`](experiments/banis_parity/): closing the gap to the published NISB baselines,
  and the measurements that identified which knobs mattered.
- [`subpixel_decoder/`](experiments/subpixel_decoder/): a learned per-token readout in place of
  trilinear upsampling, which is both sharper and cheaper. Includes the scoring pipeline the NISB
  experiments share.
- [`init_comparison/`](experiments/init_comparison/): does pretraining help, and do EM
  augmentations? Four arms differing only in initialisation.

These may be a good starting point for a new run: simply copy what you need from here and edit it.

## Tests

```bash
pytest -m unit                      # fast, single-process, safe to run interactively
pytest tests/                       # everything, including multi-process tiers
```

`tests/distributed/` spawns several processes and must be submitted with enough slots rather than
run in a one-slot interactive session. `ruff check .` and `mypy` are expected to be clean.

Repo conventions and cluster rules are in [`CLAUDE.md`](CLAUDE.md); the architecture rationale is
in [`DESIGN.md`](DESIGN.md).
