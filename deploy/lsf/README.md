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

## Attention kernel

Transformer self-attention goes through `src/models/attention.py`, not `nn.MultiheadAttention`,
so the kernel is swappable. `[model].attention_backend` selects it:

| value | behaviour |
| :--- | :--- |
| `auto` (default) | FlashAttention-4 when the device and dtype allow, else torch SDPA |
| `flash4` | demand FA4; construction fails with the reason if it is unusable |
| `sdpa` | always torch SDPA |

FA4 needs **Hopper or Blackwell** (`gpu_h100`, `gpu_h200`; not `gpu_a100`, `gpu_l4` or `gpu_t4`)
and half precision — it is inactive under `precision = "fp32"`, where `auto` silently falls back
and `flash4` raises. Use `flash4` when timing the kernel, so a run that could not use it fails
instead of quietly reporting SDPA's numbers.

`flash-attn-4` is an optional dependency: absent, everything runs on SDPA.

### When FA4 actually pays

Measured on `gpu_h100` and `gpu_h200` (bf16, non-causal, 64-wide heads), FA4 wins only at long
context, and the crossover is high enough to matter for the configs in this repo:

| tokens per sample | attention kernel alone | whole MAE step |
| :--- | :--- | :--- |
| 128 | **0.5x** — FA4 loses at every batch size tried (1–32) | 0.92x |
| 512–1024 | **0.5x** at batch ≤ 2, reaching ~1.1x by batch 32 | 0.88x |
| 4096 | 1.2x | 1.00x |
| 8192+ | 1.2–1.26x | 1.04x |

Below ~2k tokens FA4's fixed per-call cost is roughly twice SDPA's and dominates. Batch size
amortises it the same way sequence length does, so the crossover moves with total work rather
than sequence length alone — but at 128 tokens there is too little work to amortise at any batch
size worth using. The whole-step column is compressed because attention is a small share of a
short-sequence step: the MLP and patch-embedding GEMMs set the pace, so a 2x loss on attention
costs only ~8% of the step.

`mae_pretrain.toml`'s 128³ volume at 16³ patches is 512 patches, and MAE masks 75% of them, so the
encoder attends over 128 tokens — the regime where `sdpa` is the faster choice. FA4 starts paying
for itself around 256³–320³ volumes (4k–8k patches).

Two things worth knowing before benchmarking this yourself:
- **The SDPA baseline is cuDNN, not flash.** `F.scaled_dot_product_attention` dispatches, and on
  Hopper it picks its cuDNN kernel, which is good. Forcing its `FLASH_ATTENTION` backend instead
  is ~0.65x, so comparing against *that* would overstate FA4's win as ~1.9x.
- **H100 and H200 are indistinguishable here** (within 2% at every size). Attention at these
  shapes is tensor-core bound, not bandwidth bound, so the H200's faster HBM buys nothing; its
  advantage is the 141GB of VRAM. At $0.80 vs $0.50 per GPU-hour, prefer `gpu_h100` unless you
  need the capacity.

## Resuming: write the submission script once, resubmit unchanged

Long runs get interrupted — wall-time limits, node failures, preemption. Pass `--resume` and the
same script serves both the first launch and every resubmission: it continues the newest run of
this `experiment_name`, or starts fresh if there isn't one. No timestamp to look up, nothing to
edit between attempts.

```bash
#!/bin/bash    # submit_pretrain.sh — safe to resubmit as-is
cd "$MIA_TRAIN"
"$VENV/bin/torchrun" --standalone --nproc_per_node=4 \
    src/train.py --config configs/mae_pretrain.toml --resume
```

```bash
# -r makes LSF requeue on node failure; because the script is idempotent, the requeue resumes
bsub -P $PROJECT -q gpu_h100 -gpu "num=4" -n 48 -W 24:00 -r -J mia_pretrain \
  -cwd $JOBS -o $JOBS/pretrain_%J.log -e $JOBS/pretrain_%J.err \
  $JOBS/submit_pretrain.sh
```

To walk a run through several wall-time windows unattended, chain submissions with a dependency
so each one starts when the last ends: `bsub -w "ended(<jobid>)" ... $JOBS/submit_pretrain.sh`.

Notes:

- `--resume <dir>` continues one exact directory, for when "newest" is not what you want.
- Changing settings between attempts is allowed and reported: a new `lr` or a higher `max_steps`
  logs a `[resume] continuing ... with changed settings` line. Changing the **architecture** is
  refused, because the checkpoint holds the old parameters.
- Each attempt appends to `attempts.log` in the run directory, since `git_commit.txt` and
  `resolved_config.json` describe only the attempt that wrote them.

## Multi-node training

> **Validated** on 2 x 8 H200 (16 ranks): rendezvous, FSDP training, validation, and DCP
> checkpoint save/load. NCCL selects `NET/IB` with GDRDMA, so the InfiniBand fabric is used rather
> than falling back to Ethernet.

Multi-node GPU work goes to the `*_parallel` queues, which allocate **whole nodes** — never submit
a partial-node request there (it disables CPU fencing and wastes the rest of the node). LSF gives
you the host list in `$LSB_MCPU_HOSTS`; one `torchrun` must be launched per node, via `blaunch`.

[`launch_multinode.sh`](launch_multinode.sh) does this. It uses **c10d rendezvous**, so every node
runs the identical command and the backend assigns ranks — there is no `--node_rank` to compute
per host. That matters because getting it wrong is silent: two nodes both claiming rank 0 simply
hang in `init_process_group` until the wall clock kills the job.

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
  "MIA_TRAIN=$MIA_TRAIN VENV=$VENV \
   $MIA_TRAIN/deploy/lsf/launch_multinode.sh 8 experiments/<your>.toml"
```

Set `NCCL_DEBUG=INFO` in the environment to confirm the transport; look for `NET/IB` (good) rather
than `NET/Socket` (Ethernet fallback — check you are on one generation's queue).

Keep launch logic in `deploy/lsf/` — `src/` must stay scheduler-agnostic (DESIGN.md §6); it learns
the topology only from the environment `torchrun` sets.

### Checkpointing across nodes needs a Gloo group

Found the hard way, and already fixed in `engine/checkpoint.py` — recorded here because the
symptom points nowhere near the cause. DCP agrees on *what* each rank will write by scattering a
pickled save plan. That is an **object** collective; over NCCL it is staged through a GPU buffer,
and once a job spans hosts that buffer crosses InfiniBand, where a ~400 KB plan faulted outright:

```
NET/IB : mlx5_5:1 async fatal event on QP: local access violation work queue error
ncclRemoteError: A call failed possibly due to a network error ...
```

The run had trained and validated correctly for every step before it and died at its first
`checkpoint_every`. `CheckpointManager` now builds a Gloo group and passes it to `dcp.save`/
`dcp.load`. Only plan metadata goes through it — each rank still writes its own shards straight to
storage — so there is no measurable cost. If you add another code path that saves state across
ranks, route its coordination the same way.

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
