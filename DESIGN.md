# Design Document for `mia-train`

## 1. Configuration & Experiment Management

* **TOML-Based Configuration:**
  * All hyperparameter choices, model architectures, dataset parameters, distributed strategies, and algorithm selection must be explicitly defined in `.toml` config files inside `configs/`.
* **Experiment Tracking & Artifact Isolation:**
  * Every run automatically creates a unique output directory: `outputs/<experiment_name>_<timestamp>/`.
  * **Saved Artifacts:** Fully resolved `.toml` configuration, PyTorch Distributed Checkpoints (DCP), stdout/stderr logs, TensorBoard metrics, and visualization artifacts.
  * **Reproducibility:** The full Git commit hash and dirty-diff patch are logged alongside the config at execution start.

---

## 2. Directory Layout & Core Modules

The codebase enforces strict separation of concerns under `src/`:

```text
mia-train/
├── configs/               # Config files (.toml) and cluster templates
├── deploy/                # Scheduler scripts (LSF, Slurm, local)
├── tests/                 # Unit, multi-process distributed, and sanity tests
├── src/
│   ├── algorithms/        # Training strategy logic (MAE, SimMIM, Supervised, etc.)
│   ├── models/            # Pure neural network architectures
│   ├── data/              # Datasets and loaders (integrating `miao`)
│   ├── engine/            # Training loop, step orchestration, FSDP/DDP execution
│   ├── distributed/       # Process groups, 3D parallelism, NCCL/Gloo wrappers
│   ├── evals/             # Downstream evaluation tasks and benchmarks
│   └── utils/             # Logging, memory tracking, metrics, NCCL watchdogs
├── CLAUDE.md              # AI agent operational guidelines & CLI commands
├── DESIGN.md              # Master system blueprint & module invariants
└── pyproject.toml         # Package definition and pytest configuration
```

---

## 3. Architecture & Pluggable Registry Interfaces

To allow collaborators and AI agents to add features without modifying the core codebase, `mia-train` relies on **Registry Design Patterns** across four key interfaces:

| Registry | Directory | Abstract Base Class | Function |
| :--- | :--- | :--- | :--- |
| **Algorithm** | `src/algorithms/` | `BaseAlgorithm` | Defines forward pass, custom masking/loss functions, and logged metrics. |
| **Model** | `src/models/` | `BaseModel` | Pure `nn.Module` definition; exposes parameter counts and FLOP calculators. |
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