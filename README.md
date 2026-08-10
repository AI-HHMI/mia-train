# mia-train

PyTorch-native distributed training for volumetric microscopy. There is a single entrypoint 
([`train.py`](src/train.py)) for all runs, one config file per run, and everything a config 
can name lives in a registry.

Core dependencies are `torch`, `tensorboard` and `miao-io`. Optional extras add
capabilities without touching the core (see [Installing](#installing)).

## Installing

```bash
pip install -e .                    # core
pip install -e '.[cellmap]'         # + HuggingFace datasets, for the CellMap tasks
pip install -e '.[dev]'             # + pytest, ruff, mypy
```

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
- **Pretrained weights:** `[init]` loads a released or earlier-run checkpoint, reports what was
  copied, inflated, skipped or unused, and refuses silently-wrong loads.
- **Optimizer:** AdamW with layerwise LR decay, a separate patch-embedding multiplier, weight-decay
  scheduling, and rank-based weight-decay exemption. Linear or cosine LR decay after warmup.
- **Checkpointing:** Saves PyTorch distributed checkpoint (DCP), so a run can resume under a 
  different parallelism layout than it was saved with.

## Experiments

[`experiments/`](experiments/) holds complete, runnable examples: configs, submission scripts and
a README each explaining what the experiment tested and what it found, e.g.:

- [`simmim_vs_direct/`](experiments/simmim_vs_direct/): does SSL pretraining help downstream
  instance segmentation? Four arms differing only in initialisation, plus one at twice the batch
  across two nodes.
- [`banis_parity/`](experiments/banis_parity/): closing the gap to the published NISB baselines,
  and the measurements that identified which knobs mattered.

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
