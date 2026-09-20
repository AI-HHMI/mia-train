#!/usr/bin/env bash
# Score one gary_comparison arm at one checkpoint in mia-evals: predict the fit and test blocks,
# then run every scoring route over them, each as its own LSF job chained on the predictions.
#
#   bash experiments/gary_comparison/score.sh 1a_dinov3_axial_subpixel 500000
#   bash experiments/gary_comparison/score.sh 1a_dinov3_axial_subpixel 500000 --dry-run
#   bash experiments/gary_comparison/score.sh 1a_dinov3_axial_subpixel 500000 --routes mws
#
# Data configs and scoring configs live in mia-evals under the task's own directory
# (configs/gary_comparison_neuron_instance/{mws,cc_threshold}.toml and .../data/{fit,test}.yaml), so
# prediction and scoring read the same volume definitions. Predictions run on gpu_b300 like the training: the same code gives different
# instance counts on a different GPU generation, so every arm is predicted on one architecture.
#
# Where things land (layout of 2026-09-16):
#   predictions   $EXP/eval/<arm>/step<N>/{fit,test}/<volume>.zarr (+ .gt.zarr)
#   scored labellings and scratch   /nrs/scicompsoft/orhane/mia-evals/gary_comparison_neuron_instance/{scored,scorescratch}/<record>
#   LSF logs      $EXP/jobs/
# The record is named <run>.step<N>.<route> by mia-evals; nothing here chooses a label.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
EVALS=/groups/scicompsoft/home/orhane/projects/mia-evals
VENV=/groups/scicompsoft/home/orhane/myvenv               # CUDA torch, for predict.py
MIA_EVALS=/groups/scicompsoft/home/orhane/banisvenv/bin/mia-evals   # the scorer's own environment
PROJECT=miaai
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/gary_comparison
RUNS=$EXP/runs
LOGS=$EXP/jobs
TASK=gary_comparison_neuron_instance
TASK_DIR=/nrs/scicompsoft/orhane/mia-evals/$TASK
QUEUE=${QUEUE:-gpu_b300}

DRY=0 ROUTES="mws cc"
ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --routes) ROUTES=${2//,/ }; shift ;;
    -*) echo "unknown flag $1" >&2; exit 2 ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
ARM=${ARGS[0]:?usage: score.sh <arm> <step> [--dry-run] [--routes mws,cc]}
STEP=${ARGS[1]:?usage: score.sh <arm> <step> [--dry-run] [--routes mws,cc]}

run=$(ls -dt "$RUNS/gary__${ARM}_"*/ 2>/dev/null | head -1 || true); run=${run%/}
[[ -n "$run" ]] || { echo "no run directory matching gary__${ARM}_* under $RUNS" >&2; exit 1; }
[[ -d "$run/checkpoints/step_$STEP" ]] || { echo "no checkpoint step_$STEP in $run" >&2; exit 1; }
RUN_NAME=$(basename "$run")
ART=$EXP/eval/$ARM/step$STEP
mkdir -p "$LOGS" "$ART" "$TASK_DIR/scored" "$TASK_DIR/scorescratch"

jobid () { grep -o 'Job <[0-9]*>' | head -1 | tr -dc 0-9; }
submit () {      # submit <name> <bsub args...> -- <command>   -> job id or DRYRUN
  local name=$1; shift; local args=(); while [[ $1 != -- ]]; do args+=("$1"); shift; done; shift
  if [[ $DRY -eq 1 ]]; then printf 'bsub -P %s %s -J %s ...\n    %s\n' "$PROJECT" "${args[*]}" "$name" "$1" >&2; echo DRYRUN
  else bsub -P "$PROJECT" "${args[@]}" -J "$name" -cwd "$REPO" -o "$LOGS/${name}_%J.log" -e "$LOGS/${name}_%J.err" "$1" | jobid; fi
}

# 1. predictions: one GPU per block. 896^3 x 6 channels of affinities plus the co-registered truth.
deps=()
for split in fit test; do
  cmd="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$REPO/src; \
cd '$REPO' && '$VENV/bin/python' src/predict.py '$run' --step $STEP \
--data-config '$EVALS/configs/$TASK/data/$split.yaml' --out '$ART/$split'"
  id=$(submit "gary_p_${ARM}_${STEP}_${split}" -q "$QUEUE" -gpu "num=1" -n 12 -W 4:00 -- "$cmd")
  echo "predict $split: job $id -> $ART/$split"
  deps+=("$id")
done
dep="done(${deps[0]}) && done(${deps[1]})"

# 2. scoring routes, each its own CPU job chained on both predictions. The record identifier is
#    <run>.step<N>.<route>; the scored labellings and scratch are filed under it.
for route in $ROUTES; do
  case $route in
    mws) cfg=configs/$TASK/mws.toml;          slots=32; wall=24:00 ;;   # ~4 G edges in memory per volume, ~2 h each
    cc)  cfg=configs/$TASK/cc_threshold.toml; slots=8;  wall=6:00;  route=cc_threshold ;;
    *) echo "unknown route $route (mws|cc)" >&2; exit 2 ;;
  esac
  rec="${RUN_NAME}.step${STEP}.${route}"
  cmd="export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8; cd '$EVALS' && '$MIA_EVALS' score '$cfg' \
--test '$ART/test' --val '$ART/fit' --run-dir '$run' \
--scored-out '$TASK_DIR/scored/$rec' --scratch '$TASK_DIR/scorescratch/$rec'"
  args=(-q local -n "$slots" -W "$wall")
  [[ ${deps[0]} != DRYRUN ]] && args+=(-w "$dep")
  id=$(submit "gary_sc_${ARM}_${STEP}_${route}" "${args[@]}" -- "$cmd")
  echo "score   $route: job $id -> record $rec"
done
