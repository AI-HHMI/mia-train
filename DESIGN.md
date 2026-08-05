# Design Document for `mia-train`

## 1. Configuration & Experiment Management

* **TOML-Based Configuration:**
  * All hyperparameter choices, model architectures, dataset parameters, distributed strategies, and algorithm selection must be explicitly defined in `.toml` config files inside `configs/`.
* **Experiment Tracking & Artifact Isolation:**
  * Every run automatically creates a unique output directory: `<checkpoint_dir>/<experiment_name>_<timestamp>/`, where `checkpoint_dir` comes from `[environment]` in the git-ignored `configs/cluster/active.toml`. That path is required to be **absolute**, so artifacts never scatter according to the directory a job happened to be launched from; `--output-root` overrides it for a one-off. Rank 0 alone creates the directory, and a barrier holds the other ranks until it exists.
  * **Saved Artifacts:** the source `.toml` copied verbatim (`config.toml`), the fully resolved settings (`resolved_config.json`), PyTorch Distributed Checkpoints (DCP), stdout/stderr logs, TensorBoard metrics, and visualization artifacts.
    * *Resolved* means every value the run actually used, including defaults the source file left implicit — so a run remains reproducible even after those defaults later change in the code.
    * The resolved settings are JSON rather than `.toml` because the standard library can read TOML but not write it, and core code must stay free of non-PyTorch dependencies.
  * **Reproducibility:** The full Git commit hash and dirty-diff patch are logged alongside the config at execution start, as `git_commit.txt` (suffixed `(dirty)` when the tree has uncommitted changes) and `dirty.patch`. A repository with no commits cannot resolve `HEAD`, so provenance is reported as unavailable rather than failing the run.

---

## 2. Directory Layout & Core Modules

The codebase enforces strict separation of concerns under `src/`:

```text
mia-train/
├── configs/               # Config files (.toml) and cluster templates
├── deploy/                # Scheduler scripts (LSF, Slurm, local)
├── tests/                 # Unit, multi-process distributed, and sanity tests
├── src/
│   ├── algorithms/        # Training strategy logic (MAE, SimMIM, Supervised, etc.)
│   ├── models/            # Runnable architectures: exactly what `[model].name` can select
│   ├── layers/            # Reusable building blocks (attention, blocks, position encodings)
│   ├── data/              # Datasets and loaders (integrating `miao`)
│   ├── engine/            # Training loop, step orchestration, FSDP/DDP execution
│   ├── distributed/       # Process groups, 3D parallelism, NCCL/Gloo wrappers
│   ├── evals/             # Downstream evaluation tasks and benchmarks
│   ├── utils/             # Logging, memory tracking, metrics, NCCL watchdogs
│   └── train.py           # Single entrypoint for every job (launch via torchrun)
├── CLAUDE.md              # AI agent operational guidelines & CLI commands
├── DESIGN.md              # Master system blueprint & module invariants
└── pyproject.toml         # Package definition and pytest configuration
```

---

## 3. Architecture & Pluggable Registry Interfaces

To allow collaborators and AI agents to add features without modifying the core codebase, `mia-train` relies on **Registry Design Patterns** across four key interfaces:

| Registry | Directory | Abstract Base Class | Function |
| :--- | :--- | :--- | :--- |
| **Algorithm** | `src/algorithms/` | `BaseAlgorithm` | Defines forward pass, custom masking/loss functions, and logged metrics. |
| **Model** | `src/models/` | `BaseModel` | Pure `nn.Module` definition; exposes parameter counts and FLOP calculators. |

`src/models/` is reserved for architectures a user can actually run: every module in it registers a
model, so `ls src/models/` is the list of valid `[model].name` values. Reusable pieces that are never
named in a config — `SelfAttention`, `TransformerBlock`, `AxialRotaryEmbedding` — live in
`src/layers/`, which never imports from `models/`, `algorithms/`, or `engine/`. Both halves of that
rule are enforced by `tests/unit/test_package_layout.py` rather than left to convention.

A decoder is not automatically a layer: a pretraining decoder exists only to serve its objective and
is discarded afterwards, so it belongs to the algorithm that owns it (see `MAE` and `MuViTMAE`). A
decoder that ships as part of a runnable model belongs in `models/` with it.
| **Data** | `src/data/` | `BaseDataset` | Wraps data sources (using `miao`) into distributed-aware dataloaders. |
| **Evaluation** | `src/evals/` | `BaseEvalTask` | Defines downstream zero-shot or fine-tuning evaluation loops. |

---

## 4. Single Entrypoint & The Algorithm Strategy Pattern

* **Unified Execution Driver (`src/train.py`):**
  * **Anti-Pattern Prevention:** There are **no separate top-level scripts** like `train_mae.py` or `train_supervised.py`. All jobs launch through `src/train.py`. Always use `torchrun` for launching distributed jobs. 
  * The central engine handles distributed process group initialization, FSDP/DDP wrapping, learning rate scheduling, mixed-precision scaling, and PyTorch DCP checkpointing **once**.
* **`BaseAlgorithm` Abstraction:**
  * Algorithms (e.g., MAE, SimMIM, Supervised, DINO) encapsulate custom forward-pass, masking, and loss logic into `training_step()` and `validation_step()` methods.
  * The core engine executes training loops blindly by calling `algorithm.training_step(batch)`.

---

## 5. Dataset Infrastructure & `miao` Integration

* **Dataset Engine (`src/data/`):**
  * All dataset utilities and raw data abstractions must build upon classes provided by the **`miao`** library.
  * Data pipelines must use distributed-aware samplers (`DistributedSampler` or stateful stream sharding) to ensure rank-wise data partitioning across multi-node clusters.

---

## 6. HPC Cluster Portability & Deployment

* **Environment Decoupling:**
  * Core Python code in `src/` must remain completely agnostic of job schedulers (IBM Spectrum LSF, Slurm, local, etc.) and cluster-specific paths.
* **Deployment Templates (`deploy/`):**
  * Cluster submission scripts reside in `deploy/lsf/`, `deploy/slurm/`, or `deploy/local/`.
  * Local cluster configurations (e.g., scratch directory paths, CUDA/NCCL module versions) are stored in git-ignored files (e.g., `configs/cluster/active.toml`).

---

## 7. Multi-Tier Testing & Verification Harness

To allow rapid local testing (and autonomous verification by coding agents without needing GPU allocation), the test suite uses a multi-tier structure:

* **Unit Tests (`tests/unit/`):** Fast (< 5s), single-process CPU tests for registries, configurations, and single layers.
* **Distributed Multi-Process Tests (`tests/distributed/`):** Multi-process CPU tests utilizing PyTorch's **Gloo backend** to test collectives, pipeline stage handshakes, and tensor parallel layers without requiring GPUs.
* **Convergence Sanity Checks (`tests/sanity/`):** Short 20-step runs verifying that mini-models can overfit synthetic batches.

---

## 8. Agentic Memory & Project Governance

To maintain codebase health when using coding agents:

* **`CLAUDE.md` (Operational Memory):** Loaded automatically every session. Contains strict developer rules, CLI commands for tests/linting, and architectural non-negotiables.
* **`DESIGN.md` (System Memory):** Document describing module boundaries, data flow diagrams, and step lifecycles. Read by agents on demand during planning phases.
* **`.claude/rules/` (Modular Rules):** Specialized Markdown files (e.g., `.claude/rules/testing.md`, `.claude/rules/algorithms.md`) that guide specific feature implementations.