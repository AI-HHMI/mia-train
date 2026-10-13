#!/usr/bin/env bash
# Score one lm_zebrafish_cell_instance arm at one checkpoint on mia-evals' lm_zebrafish_cell_instance task: predict the
# fit frames (1 and 25) and the test frame (10), then the task's scoring route, each step its own LSF job chained on
# the last.
#
#   bash experiments/lm_zebrafish_cell_instance/score.sh w288_em 100000
#   bash experiments/lm_zebrafish_cell_instance/score.sh w288_em 100000 --dry-run
#   bash experiments/lm_zebrafish_cell_instance/score.sh w288_em 100000 --no-predict    # the affinities exist
#   bash experiments/lm_zebrafish_cell_instance/score.sh w288_em 100000 --after <job id>  # scoring also waits for it
#   bash experiments/lm_zebrafish_cell_instance/score.sh w288_em 50000 --leaderboard <dir>  # record into another board
#
# large_inputs_nisb/score.sh without the blocks: a frame is 113 M voxels, so one GPU predicts a split whole. Each
# split's data config is the task's own (z, y, x order) with patch_size set to the run's window, written next to the
# predictions, and predict.py checks it against the model's img_size. Not --patch, which resamples instead. The
# predictions cover each frame whole (--cover-box), the region the task's first record fixed. They run on gpu_b300
# like the training, because another GPU generation predicts slightly differently.
#
# Both steps import miao from the pinned 0.4.2 export that the training used (submit.sh): predict.py and the scorer
# read one frame of the store through miao's fixed_axes, which myvenv's miao 0.4.1 does not have. Each job prints
# the miao it imported and stops if it is not the export.
#
# Routes (--route, default mws_size_filter) are the task's scoring configs of that name.
#
# --leaderboard puts the record into another leaderboard root instead of mia-evals' own, to look at a score before it
# joins the table: the scored labellings still go to the task's directory below, so moving the record into
# mia-evals' leaderboard/<task>/records/ and running `mia-evals leaderboard --refresh-views` lists it as if it had
# been scored there. A root without fileglancer_shares.json gets no views page.
#
# Where things land (layout of 2026-09-16):
#   predictions   $EXP/eval/<arm>/step<N>/{fit,test}/<volume>.zarr, their data configs in .../data/{fit,test}.yaml
#   scored labellings and scratch   /nrs/scicompsoft/orhane/mia-evals/lm_zebrafish_cell_instance/{scored,scorescratch}/<record>
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
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/lm_zebrafish_cell_instance
RUNS=$EXP/runs
LOGS=$EXP/jobs
MIAO_SRC=$EXP/jobs/miao-6b0533a/src   # submit.sh says how it was made
TASK=lm_zebrafish_cell_instance
TASK_DIR=/nrs/scicompsoft/orhane/mia-evals/$TASK

DRY=0 PREDICT=1 ROUTE=mws_size_filter BOARD=""
AFTER=()
ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --no-predict) PREDICT=0 ;;
    --after) AFTER+=("$2"); shift ;;
    --route) ROUTE=$2; shift ;;
    --leaderboard) BOARD=$2; shift ;;
    -*) echo "unknown flag $1" >&2; exit 2 ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
USAGE="usage: score.sh <arm> <step> [--dry-run] [--no-predict] [--after <job id>]... [--route <route>] [--leaderboard <dir>]"
ARM=${ARGS[0]:?$USAGE}
STEP=${ARGS[1]:?$USAGE}
[[ -f $EVALS/configs/$TASK/$ROUTE.toml ]] || { echo "no scoring config $EVALS/configs/$TASK/$ROUTE.toml" >&2; exit 2; }
[[ -f $MIAO_SRC/miao/__init__.py ]] || { echo "no miao export at $MIAO_SRC" >&2; exit 2; }

run=$(ls -dt "$RUNS/lm_zebrafish_cell_instance__${ARM}_"*/ 2>/dev/null | head -1 || true); run=${run%/}
[[ -n "$run" ]] || { echo "no run directory matching lm_zebrafish_cell_instance__${ARM}_* under $RUNS" >&2; exit 1; }
[[ -d "$run/checkpoints/step_$STEP" ]] || { echo "no checkpoint step_$STEP in $run" >&2; exit 1; }
RUN_NAME=$(basename "$run")
# The window from the run's own config: img_size is x, y, z there, and the task's data configs are z, y, x.
read -r X Y Z < <(sed -n 's/^img_size = \[\([0-9]*\), *\([0-9]*\), *\([0-9]*\)\].*/\1 \2 \3/p' "$run/config.toml")
[[ -n "${Z:-}" ]] || { echo "cannot read a per-axis img_size from $run/config.toml" >&2; exit 1; }
ART=$EXP/eval/$ARM/step$STEP
mkdir -p "$LOGS" "$ART" "$TASK_DIR/scored" "$TASK_DIR/scorescratch"

jobid () { grep -o 'Job <[0-9]*>' | head -1 | tr -dc 0-9; }
submit () {      # submit <name> <log name> <bsub args...> -- <command>   -> job id or DRYRUN
  local name=$1 log=$2; shift 2; local args=(); while [[ $1 != -- ]]; do args+=("$1"); shift; done; shift
  if [[ $DRY -eq 1 ]]; then printf 'bsub -P %s %s -J %s ...\n    %s\n' "$PROJECT" "${args[*]}" "$name" "$1" >&2; echo DRYRUN
  else bsub -P "$PROJECT" "${args[@]}" -J "$name" -cwd "$REPO" -o "$LOGS/$log.log" -e "$LOGS/$log.err" "$1" | jobid; fi
}
waits () { local c=() j; for j in "$@"; do c+=("done($j)"); done; local IFS=' '; echo "${c[*]}" | sed 's/) done(/) \&\& done(/g'; }
CHECK_MIAO="'$VENV/bin/python' -c \"import miao; print('[miao]', miao.__file__, flush=True); assert miao.__file__.startswith('$MIAO_SRC/'), 'not the pinned miao'\""

# 1. predictions: one GPU per split, the whole split in one job.
deps=("${AFTER[@]}")
if (( PREDICT )); then
  mkdir -p "$ART/data"
  for split in fit test; do
    data=$ART/data/$split.yaml
    { echo "# Written by mia-train experiments/lm_zebrafish_cell_instance/score.sh: mia-evals configs/$TASK/data/$split.yaml"
      echo "# with patch_size set to $RUN_NAME's window (img_size [$X, $Y, $Z], x, y, z)."
      sed "s/^patch_size: .*/patch_size: [$Z, $Y, $X]   # z, y, x: the run's window/" "$EVALS/configs/$TASK/data/$split.yaml"; } > "$data"
    cmd="export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$MIAO_SRC:$REPO/src; \
cd '$REPO' && $CHECK_MIAO && '$VENV/bin/python' src/predict.py '$run' --step $STEP \
--data-config '$data' --out '$ART/$split' --cover-box"
    name="lmzf_p_${ARM}_${STEP}_${split}"
    id=$(submit "$name" "${name}_%J" -q gpu_b300 -gpu "num=1" -n 12 -W 2:00 -- "$cmd")
    echo "predict $split: job $id -> $ART/$split"
    deps+=("$id")
  done
fi

# 2. the route: one CPU job chained on everything before it. Scratch and scored labellings are filed under the
#    record's name.
rec="${RUN_NAME}.step${STEP}.${ROUTE}"
cmd="export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=$MIAO_SRC; \
cd '$EVALS' && $CHECK_MIAO && '$VENV/bin/python' '$NUMA_LOCAL' '$MIA_EVALS' score configs/$TASK/$ROUTE.toml \
--test '$ART/test' --val '$ART/fit' --run-dir '$run' \
--scored-out '$TASK_DIR/scored/$rec' --scratch '$TASK_DIR/scorescratch/$rec'${BOARD:+ --leaderboard '$BOARD'}"
args=(-q local -n 32 -W 8:00)
[[ ${#deps[@]} -gt 0 && ${deps[0]} != DRYRUN ]] && args+=(-w "$(waits "${deps[@]}")")
id=$(submit "lmzf_sc_${ARM}_${STEP}" "lmzf_sc_${ARM}_${STEP}_%J" "${args[@]}" -- "$cmd")
echo "score $ROUTE: job $id -> record $rec"
