# Submitting `mia-train` jobs on LSF (Janelia)

Cluster-wide policy, queue tables, and the slot model live in the admin-authored HPC hint sheet,
kept locally at `.claude/rules/cluster.md`. That file is **not committed** — it contains internal
infrastructure detail — so ask the HPC team for a copy. Read it first; this file only covers
**`mia-train`-specific** recipes.

## Substitute your own values

Every recipe below uses these four variables. Set them once per shell; the real values come from
your `configs/cluster/active.toml`, which is machine-local and not version controlled.

```bash
export MIA_TRAIN=/path/to/mia-train              # this repo
export VENV=/path/to/venv                        # [environment].python_venv
export JOBS=/nrs/<lab>/<user>/mia-train-jobs     # your own job scratch area (see below)
export PROJECT=<lsf-project>                     # [scheduler].project
```

`$JOBS` is a scratch area **you** own for job scripts and LSF logs. Create it under your own NRS
path — never write into someone else's. NRS is the right tier: not backed up, which is correct for
regenerable job artifacts.

## Non-negotiables for this repo

| Rule | Why |
| :--- | :--- |
| Always pass `-P $PROJECT` | Members of `scicompsoft` must set the billing project explicitly; the default bills the wrong group. Run `lsfgroup $USER` to see your default. |
| Never stage scripts/logs under `/tmp` | `/tmp` is node-local. A job submitted from one host runs on another, the path doesn't exist, and you get a bare non-zero exit with a near-zero CPU time. Use `/nrs/...` (scratch-tier) or `/groups/...`. |
| Always set `-W` | GPU queues default to a **2-hour** limit, not the 14-day CPU default. |
| Always set thread env vars | Unset, every torch process threads to the node's full physical core count (48–128), not your slot count. |
| Always `-o`/`-e` to a shared path | Otherwise LSF emails you per job. |

Run artifacts (checkpoints, TensorBoard logs, resolved config, git provenance) go to
`[environment].checkpoint_dir` from `configs/cluster/active.toml`. Each run creates
`<checkpoint_dir>/<experiment_name>_<timestamp>/`. That path is required to be absolute, so
artifacts do not scatter based on the job's working directory. Pass `--output-root` to override for
a one-off.

## Running the test suite

`tests/unit/` is single-process and safe to run directly in an interactive session. **The
`tests/distributed/` tier is not** — it spawns up to 4 Gloo processes, and a typical interactive
VSCode allocation is 1 slot (`echo $LSB_DJOB_NUMPROC` to confirm). Oversubscribing risks LSF's
automatic memory kill, which is unrecoverable.

```bash
# Fast tier — fine inline (single process, no allocation concerns)
$VENV/bin/python -m pytest -m unit

# Multi-process tiers — submit with enough slots
bsub -P $PROJECT -n 4 -W 0:20 -J mia_tests \
  -cwd $JOBS \
  -o $JOBS/tests_%J.log \
  -e $JOBS/tests_%J.err \
  "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; \
   cd $MIA_TRAIN && $VENV/bin/python -m pytest -v"
```

`-W 0:20` auto-routes to the `short` queue (best turnaround). Add `-K` to block until the job
finishes instead of polling `bjobs`. Measured footprint of the full suite as of the entrypoint
work: ~64s wall, ~1.7GB peak RSS on 4 slots.

## Single-node training (multi-GPU)

One `torchrun` process group per node, one process per GPU. Slot count should follow the queue's
slots-per-GPU ratio (12 for A100/H100/H200, 8 for L4) to avoid stranding GPUs for other users.

```bash
# 4 H100s on one node: 4 GPUs x 12 slots/GPU = 48 slots
bsub -P $PROJECT -q gpu_h100 -gpu "num=4" -n 48 -W 8:00 -J mia_train \
  -cwd $JOBS \
  -o $JOBS/train_%J.log \
  -e $JOBS/train_%J.err \
  "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; \
   cd $MIA_TRAIN && $VENV/bin/torchrun \
     --standalone --nproc_per_node=4 src/train.py --config configs/<your>.toml"
```

Notes:

- **Do not set `CUDA_VISIBLE_DEVICES`.** LSF sets it from the GPU allocation; `torchrun` +
  `LOCAL_RANK` handles per-process device selection on top of that.
- `OMP_NUM_THREADS` above is slots ÷ processes (48 ÷ 4 = 12 would also be defensible; 4 is
  conservative and avoids layered-threading blowup through the data pipeline).
- Pick the queue by VRAM need: L4 24GB → A100 80GB → H100 80GB → H200 141GB.
- Artifacts do **not** depend on the `cd` above — they go to the configured `checkpoint_dir`.
  The `cd` is only so `src/train.py` and the relative `--config` path resolve.

## Multi-node training

> **Untested in this repo so far.** The recipe below follows cluster policy and the standard
> `torchrun` rendezvous pattern, but it has not yet been validated end-to-end here. Validate on a
> 2-node job before relying on it.

Multi-node GPU work goes to the `*_parallel` queues, which allocate **whole nodes** — never submit
a partial-node request there (it disables CPU fencing and wastes the rest of the node). LSF gives
you the host list in `$LSB_MCPU_HOSTS`; one `torchrun` must be launched per node (via `blaunch`),
each with a distinct `--node_rank` and a shared `--master_addr`/`--master_port`.

Interconnect constraint that matters here: **H100 and H200 sit on separate InfiniBand fabrics with
no IB path between them.** A multi-node job must stay within one GPU generation — i.e. one queue,
either `gpu_h100_parallel` or `gpu_h200_parallel`, never a mix. A100/L4/T4 nodes have no IB at all
(Ethernet only), so multi-node NCCL there will be markedly slower and is not recommended for
sharded strategies (FSDP/HSDP) that are communication-bound.

```bash
# 2 nodes x 8 H100s = 16 ranks; 96 slots/node -> -n 192
bsub -P $PROJECT -q gpu_h100_parallel -app parallel-96 -gpu "num=8:mode=shared" \
  -n 192 -W 24:00 -J mia_train_mn \
  -cwd $JOBS \
  -o $JOBS/train_mn_%J.log \
  -e $JOBS/train_mn_%J.err \
  $JOBS/launch_multinode.sh
```

Where `launch_multinode.sh` derives the rendezvous endpoint from LSF's host list and uses
`blaunch` to start one `torchrun` per node. Keep this logic in `deploy/lsf/` — `src/` must stay
scheduler-agnostic (DESIGN.md §6).

## Mapping `ParallelDims` to an allocation

`ParallelDims(dp_replicate, dp_shard, tp)` must multiply to the `torchrun` world size (total GPUs).
Rules of thumb:

| Situation | Setting |
| :--- | :--- |
| Model fits comfortably on one GPU | `dp_replicate=N, dp_shard=1, tp=1` (DDP) |
| Model/optimizer state too large for one GPU | `dp_shard=N, tp=1` (FSDP) |
| Multi-node, want to confine sharding within a node | `dp_replicate=<nodes>, dp_shard=<gpus_per_node>` (HSDP) |
| A single layer's weights won't fit, or activation memory dominates | add `tp>1`, keeping `tp` **within a node** (NVLink) — the model must implement `tensor_parallel_plan()` |

Because H100/H200 IB is per-generation and TP is the most communication-intensive dimension, keep
`tp` inside a node and use `dp_replicate` as the cross-node dimension.

## Monitoring

Follow the hint sheet's backoff policy — never poll `bjobs` more than once a minute, batch job IDs
into one call, and prefer the dashboards:

- GPU utilization: https://gpustats.int.janelia.org
- Job metrics (right-size future jobs from actual usage): https://lsf-rtm.int.janelia.org
