#!/usr/bin/env bash
# Submit the B300 capability sweep: how large a crop a 7B DINOv3 affinity fine-tune can train on.
#
#   bash experiments/b300_capability_run/submit.sh                 # both arms, 8 nodes
#   bash experiments/b300_capability_run/submit.sh hsdp             # one arm
#   NODES=1 SIZES="128 256" DIMS=2,4,1 bash .../submit.sh hsdp      # single-node smoke test
#
# Each arm walks its sizes in ascending order, one `torchrun` per size, stopping at the first that
# does not fit -- so the last size in the results file is the answer. One process per size because
# a CUDA out-of-memory inside a collective leaves the other ranks waiting rather than raising:
# there is nothing to catch, so the process is allowed to die and the next size is not attempted.
#
# "Did it fit?" is decided by whether a record appeared, not by the exit code. Measured 2026-09-06:
# the 512-cube trial wrote its result, printed its summary, and *then* exited non-zero during
# process-group teardown -- no traceback, no child-failure report, and LSF called the job Done. An
# exit-code gate read that as the memory limit and stopped the sweep three sizes early. The
# results file cannot lie the same way: rank 0 appends one line per size that completed.
#
# Wall clock: attention is quadratic in the token count, so the top of the sweep is minutes per
# step, not the ~0.3 s the 256-cube runs in other experiments do. 8 h leaves room for a build,
# a warm-up and 8 timed steps at every size plus the size that fails.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
PROJECT=miaai
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
STAGE=/nrs/scicompsoft/orhane/mia-train-scratch/b300_capability   # NOT /tmp: that is node-local
LOGS="$RUNS/jobs"
mkdir -p "$LOGS" "$STAGE"

NODES=${NODES:-8}
GPUS=8
SLOTS=96
WALL=${WALL:-8:00}
# 16^3 patches, so these are 16..100 patches a side, i.e. 4k to 1M tokens. 1600^3 is the 1M-context
# target; the sweep stops well before it if the memory does.
SIZES=${SIZES:-"256 512 768 1024 1280 1600"}
STEPS=${STEPS:-8}
WARMUP=${WARMUP:-3}
# Overrides each arm's [parallelism] so a smoke test can run an 8-node arm on one node without
# a second config whose only difference is three numbers.
DIMS=${DIMS:-}
# Count the step's real FLOPs, affinity head included, rather than only the analytic
# encoder-only figure. Costs one extra forward/backward per size -- ~9% of a warmup-plus-steps
# window -- so it is opt-in rather than always on.
FLOPS=${FLOPS:-}
# Trace one step and print the op breakdown. For answering where a step goes.
PROFILE=${PROFILE:-}
# With PROFILE, also export rank 0's chrome trace here, for overlap analysis.
TRACE=${TRACE:-}
# Compile each transformer block instead of the whole algorithm; see the driver's --compile-blocks.
COMPILE_BLOCKS=${COMPILE_BLOCKS:-}

ARMS=("$@"); [[ ${#ARMS[@]} -eq 0 ]] && ARMS=(hsdp hsdp_tp)

for arm in "${ARMS[@]}"; do
  cfg="$HERE/${arm}.toml"
  [[ -f "$cfg" ]] || { echo "no config $cfg" >&2; exit 2; }
  # `RESULTS` overrides where an arm appends, so a one-off -- a profiled step, a smoke test
  # -- does not drop a row into the sweep's own file that differs from the rest of the
  # table in a way only the parallelism fields record.
  results="${RESULTS:-$STAGE/${arm}.jsonl}"

  # The arm's work as its own script, so nothing has to survive several layers of shell quoting --
  # the same reason lmd_ssl_v1 and new_ssl_recipe write theirs out.
  cmd="$STAGE/${arm}.sh"
  { echo "#!/usr/bin/env bash"
    echo "set -uo pipefail"
    echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
    echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    echo "cd $REPO"
    echo "for size in $SIZES; do"
    echo "  echo \"=== $arm \${size}^3 ===\""
    echo "  before=\$(wc -l < $results 2>/dev/null || echo 0)"
    if [[ $NODES -eq 1 ]]; then
      echo "  $VENV/bin/torchrun --standalone --nproc_per_node=$GPUS \\"
      echo "    experiments/b300_capability_run/capability_sweep.py \\"
      echo "    --config $cfg --size \$size --results $results \\"
      echo "    --warmup $WARMUP --steps $STEPS ${DIMS:+--dims $DIMS} ${FLOPS:+--measure-flops} ${PROFILE:+--profile} ${TRACE:+--trace $TRACE} ${COMPILE_BLOCKS:+--compile-blocks}"
    else
      # launch_multinode.sh reads LSF's host list and starts one torchrun per node under a c10d
      # rendezvous; ENTRYPOINT points it at the sweep instead of src/train.py.
      echo "  ENTRYPOINT=experiments/b300_capability_run/capability_sweep.py \\"
      echo "  MIA_TRAIN=$REPO VENV=$VENV \\"
      echo "    $REPO/deploy/lsf/launch_multinode.sh $GPUS $cfg \\"
      echo "    --size \$size --results $results --warmup $WARMUP --steps $STEPS \\"
      echo "    ${DIMS:+--dims $DIMS} ${FLOPS:+--measure-flops} ${PROFILE:+--profile} ${TRACE:+--trace $TRACE} ${COMPILE_BLOCKS:+--compile-blocks}"
    fi
    echo "  after=\$(wc -l < $results 2>/dev/null || echo 0)"
    echo "  if [ \"\$after\" -le \"\$before\" ]; then"
    echo "    echo \"$arm: \${size}^3 recorded nothing -- this is the limit\"; break"
    echo "  fi"
    echo "done"
    echo "echo \"$arm finished; results in $results\""
  } > "$cmd"

  if [[ $NODES -eq 1 ]]; then
    args=(-q gpu_b300 -gpu "num=$GPUS" -n "$SLOTS")
  else
    # `*_parallel` allocates whole nodes; `mode=shared` so one torchrun per node sees all 8 GPUs.
    args=(-q gpu_b300_parallel -app "parallel-$SLOTS" -gpu "num=$GPUS:mode=shared"
          -n "$((SLOTS * NODES))")
  fi

  bsub -P "$PROJECT" "${args[@]}" -W "$WALL" -J "b300cap_$arm" -cwd "$REPO" \
    -o "$LOGS/b300cap_${arm}_%J.log" -e "$LOGS/b300cap_${arm}_%J.err" \
    "bash '$cmd'"
  echo "  arm $arm: $NODES node(s), sizes $SIZES -> $results"
done
