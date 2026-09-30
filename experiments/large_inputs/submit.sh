#!/usr/bin/env bash
# Submit large_inputs capability sweeps: the largest crop, and the throughput at each crop, that a
# model trains on per B300 node. One arm = one config = one job on one node.
#
#   bash experiments/large_inputs/submit.sh vitl_tp8_ro8 vitl_dp       # arms, default sizes
#   SIZES="512 1024" bash experiments/large_inputs/submit.sh vitl_dp_compile
#   SMOKE=1 SIZES=256 STEPS=1 WARMUP=1 bash experiments/large_inputs/submit.sh vitl_dp vitl_tp8
#   PROBE=profile_1024 PROFILE=1 SIZES=1024 WARMUP=1 STEPS=2 bash experiments/large_inputs/submit.sh vitl_dp_win16_g4p4
#
# Adapted 2026-09-29 from experiments/b300_capability_run/submit.sh (the 7B run): single node by
# default (a ViT-L needs no more than one node to hold a crop, and one node per arm lets several arms
# run at once), `--decode-chunks auto` by default, results under this experiment's /nrs home, and
# SMOKE=1 putting everything a smoke run writes under jobs/smoke/.
#
# Each arm walks its sizes in ascending order, one `torchrun` per size, stopping at the first that
# records nothing -- so the last size in the results file is the arm's limit. One process per size
# because a CUDA out-of-memory inside a collective leaves the other ranks waiting rather than
# raising. "Did it fit?" is decided by whether rank 0 appended a record, not by the exit code: the 7B
# run saw a size write its record and then exit non-zero during process-group teardown.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
PROJECT=miaai
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/large_inputs   # runs/ jobs/ probes/ (nrs layout of 2026-09-16)
STAGE=$EXP/jobs${SMOKE:+/smoke}   # job scripts, LSF logs and results; NOT /tmp, which is node-local
[[ -n "${PROBE:-}" ]] && STAGE=$EXP/probes/$PROBE   # a one-off measurement, kept out of jobs/
mkdir -p "$STAGE"

GPUS=8
SLOTS=96                  # 12 per GPU on B300 (user directive 2026-09-21)
WALL=${WALL:-12:00}
SIZES=${SIZES:-"512 1024 1536 2048 2560 3072"}
STEPS=${STEPS:-4}
WARMUP=${WARMUP:-2}
CHUNKS=${CHUNKS:-auto}    # --decode-chunks: `auto`, an integer, or empty for the config's own
DIMS=${DIMS:-}            # dp_replicate,dp_shard,tp overriding [parallelism]
PROFILE=${PROFILE:-}      # trace one step and print the op breakdown
FLOPS=${FLOPS:-}          # also count the whole step's FLOPs, head included (one extra fwd/bwd)

ARMS=("$@"); [[ ${#ARMS[@]} -eq 0 ]] && { echo "name at least one arm (a config in $HERE)" >&2; exit 2; }

for arm in "${ARMS[@]}"; do
  cfg="$HERE/${arm}.toml"
  [[ -f "$cfg" ]] || { echo "no config $cfg" >&2; exit 2; }
  results="${RESULTS:-$STAGE/${arm}.jsonl}"
  cmd="$STAGE/${arm}.sh"
  { echo "#!/usr/bin/env bash"
    echo "set -uo pipefail"
    echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
    echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    echo "cd $REPO"
    echo "for size in $SIZES; do"
    echo "  echo \"=== $arm \${size}^3 ===\""
    echo "  before=\$(wc -l < $results 2>/dev/null || echo 0)"
    echo "  $VENV/bin/torchrun --standalone --nproc_per_node=$GPUS \\"
    echo "    experiments/large_inputs/capability_sweep.py \\"
    echo "    --config $cfg --size \$size --results $results \\"
    echo "    --warmup $WARMUP --steps $STEPS ${CHUNKS:+--decode-chunks $CHUNKS} ${DIMS:+--dims $DIMS} ${PROFILE:+--profile} ${FLOPS:+--measure-flops}"
    echo "  after=\$(wc -l < $results 2>/dev/null || echo 0)"
    echo "  if [ \"\$after\" -le \"\$before\" ]; then"
    echo "    echo \"$arm: \${size}^3 recorded nothing -- this is the limit\"; break"
    echo "  fi"
    echo "done"
    echo "echo \"$arm finished; results in $results\""
  } > "$cmd"

  bsub -P "$PROJECT" -q gpu_b300 -gpu "num=$GPUS" -n "$SLOTS" -W "$WALL" \
    -J "largein_${SMOKE:+smoke_}$arm" -cwd "$REPO" \
    -o "$STAGE/${arm}_%J.log" -e "$STAGE/${arm}_%J.err" \
    "bash '$cmd'"
  echo "  arm $arm: sizes $SIZES -> $results"
done
