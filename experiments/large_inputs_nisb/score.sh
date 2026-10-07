#!/usr/bin/env bash
# Score one large_inputs_nisb arm at one checkpoint on mia-evals' nisb_base_neurite_tracing task: predict NISB's
# base val (fit) and test cubes, then a blockwise mutex watershed route, each step its own LSF job chained on the last.
#
#   bash experiments/large_inputs_nisb/score.sh w256_1cube 100000
#   bash experiments/large_inputs_nisb/score.sh w512_1cube 100000 --dry-run
#   bash experiments/large_inputs_nisb/score.sh w256_1cube 100000 --after <job id>   # scoring also waits for it
#   bash experiments/large_inputs_nisb/score.sh w256_1cube 100000 --route mws_blockwise_1024 --no-predict
#
# Predictions cover each cube whole (predict.py --cover-box), so every row scores the same region, at the arm's own
# window: each split's data config is the task's own with patch_size set to the window, written next to the
# predictions, and predict.py checks it against the model's img_size. Not --patch, which keeps the config's
# 256-voxel read and resamples it up to the model's window (a 512 arm on a 2x lattice). They run on gpu_b300 like the training:
# another GPU generation predicts slightly differently, and blockwise completion refuses mixed devices. A cube is
# 12.2 G voxels and prediction holds ~40 B a voxel, so it goes in blocks of 1536 output voxels per axis -- 2 x 2 x 1
# per cube, ~127 GB of RAM each -- one GPU per block, then a finisher that completes the artifact
# (deploy/lsf/README.md). --no-predict skips this for affinities that exist, or that the jobs named by --after
# (repeatable) are still making.
#
# Routes (--route, default mws_blockwise), the task's scoring configs of that name:
#   mws_blockwise        512 x 512 x 256 blocks: one CPU job with 32 slots segments them 8 at a time and scores.
#   mws_blockwise_1024   1024 x 1024 x 1350 blocks, ~425 GB each: a 9-element array (36 slots each; every element
#                        segments one block of each cube) first, then the scorer stitches, relabels and scores.
#   cc_threshold         the earlier (BANIS) pipeline: the short-range channels thresholded over logits 3-11, no size
#                        filter, fitted on val; one CPU job with 32 slots labels each cube in memory.
#
# Where things land (layout of 2026-09-16):
#   predictions   $EXP/eval/<arm>/step<N>/{fit,test}/<volume>.zarr, their data configs in .../data/{fit,test}.yaml
#   scored labellings and scratch   /nrs/scicompsoft/orhane/mia-evals/nisb_base_neurite_tracing/{scored,scorescratch}/<record>
#   LSF logs      $EXP/jobs/
# The record is named <run>.step<N>.<route> by mia-evals; nothing here chooses a label.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
EVALS=/groups/scicompsoft/home/orhane/projects/mia-evals
VENV=/groups/scicompsoft/home/orhane/myvenv
MIA_EVALS=$VENV/bin/mia-evals
NUMA_LOCAL=$REPO/experiments/gary_comparison/numa_local.py
PROJECT=miaai
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/large_inputs_nisb
RUNS=$EXP/runs
LOGS=$EXP/jobs
TASK=nisb_base_neurite_tracing
TASK_DIR=/nrs/scicompsoft/orhane/mia-evals/$TASK
BLOCK=1536
BLOCKS=4   # ceil(3000 / 1536)^2 x ceil(1350 / 1536)

DRY=0 PREDICT=1 ROUTE=mws_blockwise
AFTER=()
ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --no-predict) PREDICT=0 ;;
    --after) AFTER+=("$2"); shift ;;
    --route) ROUTE=$2; shift ;;
    -*) echo "unknown flag $1" >&2; exit 2 ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
USAGE="usage: score.sh <arm> <step> [--dry-run] [--no-predict] [--after <job id>]... [--route <route>]"
ARM=${ARGS[0]:?$USAGE}
STEP=${ARGS[1]:?$USAGE}
case $ROUTE in
  mws_blockwise) SEGMENTERS=0 ;;
  mws_blockwise_1024) SEGMENTERS=9 ;;   # 3 x 3 x 1 blocks per cube, one per element
  cc_threshold) SEGMENTERS=0 ;;
  *) echo "unknown route $ROUTE (mws_blockwise | mws_blockwise_1024 | cc_threshold)" >&2; exit 2 ;;
esac
WINDOW=${ARM%%_*}; WINDOW=${WINDOW#w}
[[ $WINDOW =~ ^[0-9]+$ ]] || { echo "cannot read the window from arm $ARM (expected w<size>_<n>cube)" >&2; exit 2; }

run=$(ls -dt "$RUNS/large_inputs_nisb__${ARM}_"*/ 2>/dev/null | head -1 || true); run=${run%/}
[[ -n "$run" ]] || { echo "no run directory matching large_inputs_nisb__${ARM}_* under $RUNS" >&2; exit 1; }
[[ -d "$run/checkpoints/step_$STEP" ]] || { echo "no checkpoint step_$STEP in $run" >&2; exit 1; }
RUN_NAME=$(basename "$run")
ART=$EXP/eval/$ARM/step$STEP
mkdir -p "$LOGS" "$ART" "$TASK_DIR/scored" "$TASK_DIR/scorescratch"

jobid () { grep -o 'Job <[0-9]*>' | head -1 | tr -dc 0-9; }
submit () {      # submit <name> <log name> <bsub args...> -- <command>   -> job id or DRYRUN
  local name=$1 log=$2; shift 2; local args=(); while [[ $1 != -- ]]; do args+=("$1"); shift; done; shift
  if [[ $DRY -eq 1 ]]; then printf 'bsub -P %s %s -J %s ...\n    %s\n' "$PROJECT" "${args[*]}" "$name" "$1" >&2; echo DRYRUN
  else bsub -P "$PROJECT" "${args[@]}" -J "$name" -cwd "$REPO" -o "$LOGS/$log.log" -e "$LOGS/$log.err" "$1" | jobid; fi
}
waits () { local c=() j; for j in "$@"; do c+=("done($j)"); done; local IFS=' '; echo "${c[*]}" | sed 's/) done(/) \&\& done(/g'; }

# 1. predictions: per cube, one GPU per block, then the finisher once every block has ended.
deps=("${AFTER[@]}")
if (( PREDICT )); then
  mkdir -p "$ART/data"
  for split in fit test; do
    data=$ART/data/$split.yaml
    { echo "# Written by mia-train experiments/large_inputs_nisb/score.sh: mia-evals configs/$TASK/data/$split.yaml"
      echo "# with patch_size set to arm $ARM's window."
      sed "s/^patch_size: .*/patch_size: [$WINDOW, $WINDOW, $WINDOW]/" "$EVALS/configs/$TASK/data/$split.yaml"; } > "$data"
    cmd="export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$REPO/src; \
cd '$REPO' && '$VENV/bin/python' src/predict.py '$run' --step $STEP \
--data-config '$data' --out '$ART/$split' --cover-box --block $BLOCK"
    name="linisb_p_${ARM}_${STEP}_${split}"
    id=$(submit "$name[1-$BLOCKS]" "${name}_%J_%I" -q gpu_b300 -gpu "num=1" -n 12 -W 6:00 -- \
         "$cmd --worker \$((LSB_JOBINDEX - 1)) --workers $BLOCKS")
    args=(-q gpu_b300 -gpu "num=1" -n 12 -W 6:00)
    [[ $id != DRYRUN ]] && args+=(-w "ended($id)")
    fin=$(submit "${name}_finish" "${name}_finish_%J" "${args[@]}" -- "$cmd")
    echo "predict $split: jobs $id[1-$BLOCKS] + finisher $fin -> $ART/$split"
    deps+=("$fin")
  done
fi

# 2. the route: for large blocks, the per-block watershed on an array first; then the scorer, a CPU job chained on
#    everything before it. Scratch and scored labellings are filed under the record's name.
rec="${RUN_NAME}.step${STEP}.${ROUTE}"
tag="${ARM}_${STEP}"; [[ $ROUTE != mws_blockwise ]] && tag+="_${ROUTE#mws_blockwise_}"   # _1024, _cc_threshold
scratch=$TASK_DIR/scorescratch/$rec
if (( SEGMENTERS )); then
  # Imported, then main() called -- not `python -m postprocess.mws_blockwise` as mia-evals documents: run as
  # __main__ the module registers "mws_blockwise", and main()'s `import components` imports it again under its own
  # name and registers it a second time, which the registry refuses (every element of 154532421 failed so).
  cmd="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; cd '$EVALS' && '$VENV/bin/python' '$NUMA_LOCAL' '$VENV/bin/python' -c 'from postprocess.mws_blockwise import main; main()' configs/$TASK/$ROUTE.toml \
--test '$ART/test' --val '$ART/fit' --scratch '$scratch' --processes 1 --worker \$((LSB_JOBINDEX - 1)) --workers $SEGMENTERS"
  args=(-q local -n 36 -W 8:00)
  [[ ${#deps[@]} -gt 0 && ${deps[0]} != DRYRUN ]] && args+=(-w "$(waits "${deps[@]}")")
  id=$(submit "linisb_ws_${tag}[1-$SEGMENTERS]" "linisb_ws_${tag}_%J_%I" "${args[@]}" -- "$cmd")
  echo "segment $ROUTE: jobs $id[1-$SEGMENTERS] -> $scratch"
  deps=("$id")
fi
cmd="export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8; cd '$EVALS' && '$VENV/bin/python' '$NUMA_LOCAL' '$MIA_EVALS' score configs/$TASK/$ROUTE.toml \
--test '$ART/test' --val '$ART/fit' --run-dir '$run' \
--scored-out '$TASK_DIR/scored/$rec' --scratch '$scratch'"
args=(-q local -n 32 -W 24:00)
[[ ${#deps[@]} -gt 0 && ${deps[0]} != DRYRUN ]] && args+=(-w "$(waits "${deps[@]}")")
id=$(submit "linisb_sc_${tag}" "linisb_sc_${tag}_%J" "${args[@]}" -- "$cmd")
echo "score $ROUTE: job $id -> record $rec"
