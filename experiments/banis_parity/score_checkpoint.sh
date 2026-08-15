#!/usr/bin/env bash
# Predict, score and visualise one checkpoint of this experiment, in three chained LSF jobs.
#
#   bash experiments/banis_parity/score_checkpoint.sh 160000
#   bash experiments/banis_parity/score_checkpoint.sh 160000 --reuse        # re-score, no GPU
#   bash experiments/banis_parity/score_checkpoint.sh 200000 --logits "5 6 7"
#
# The three stages run in different environments on purpose, and that split is the reason this is
# a submission script rather than one program:
#
#   predict    myvenv     1x H100    rebuilds the model from the run's own resolved_config.json
#   score      banisvenv  CPU        imports BANIS' segmentation + metric functions
#   visualise  myvenv     CPU        numpy/zarr/PIL only
#
# mia-train never acquires funlib.evaluate or numba, and BANIS is never imported by anything that
# touches a GPU. Scoring and visualising both depend only on the affinity zarr, so they are
# submitted in parallel behind prediction rather than in series.
#
# Mutex watershed over the long-range channels is deliberately *not* wired in here: measured
# head-to-head on one 512^3 block of step 160000 it scored nERL 0.014 against thresholded
# connected components' 0.419, so it is a diagnostic (`banis/mia_score_mws.py`), not a step in the
# routine path.
set -euo pipefail

STEP=${1:-}
if [[ -z "$STEP" || "$STEP" == -* ]]; then
  echo "usage: $(basename "$0") <step> [--run DIR] [--cube DIR] [--tag NAME] [--logits \"4 5 6\"]" >&2
  echo "                          [--patch N] [--stride N] [--origin \"X Y Z\"] [--size N]" >&2
  echo "                          [--reuse] [--no-viz] [--no-score] [--dry-run]" >&2
  exit 2
fi
shift

BANIS=/groups/scicompsoft/home/orhane/projects/banis
MYVENV=/groups/scicompsoft/home/orhane/myvenv
BANISVENV=/groups/scicompsoft/home/orhane/banisvenv
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
SCRATCH=/nrs/scicompsoft/orhane/mia-train-scratch
EVAL=$SCRATCH/eval
PROJECT=miaai

RUN=""
CUBE=/groups/miaai/miaai/lmd-v0.0.1/dev/nisb/train_100/val/seed100.zarr
TAG=bp
# The sweep the benchmark rules allow on the *val* cube. Narrowed from mia_score.py's -1..11
# default because every scored checkpoint so far has peaked at +5 or +6 and each extra threshold
# costs ~10 min of the scoring job; widen it if a checkpoint's best lands on an endpoint.
LOGITS="4 5 6 7 8"
PATCH=256
STRIDE=128
ORIGIN="1024 1024 512"
SIZE=256
REUSE=0
VIZ=1
SCORING=1
DRY=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --run)     RUN=$2;    shift 2 ;;
    --cube)    CUBE=$2;   shift 2 ;;
    --tag)     TAG=$2;    shift 2 ;;
    --logits)  LOGITS=$2; shift 2 ;;
    --patch)   PATCH=$2;  shift 2 ;;
    --stride)  STRIDE=$2; shift 2 ;;
    --origin)  ORIGIN=$2; shift 2 ;;
    --size)    SIZE=$2;   shift 2 ;;
    --reuse)   REUSE=1;   shift ;;
    --no-viz)  VIZ=0;     shift ;;
    --no-score) SCORING=0; shift ;;
    --dry-run) DRY=1;     shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Newest run of this experiment unless one was named. `ls -dt` matches how tensorboard.sh resolves
# arms, so both scripts follow a resubmitted run to the same directory.
if [[ -z "$RUN" ]]; then
  RUN=$(ls -dt "$RUNS"/banis_parity__finetune_256_long_*/ 2>/dev/null | head -1 || true)
  RUN=${RUN%/}
  [[ -n "$RUN" ]] || { echo "no banis_parity run found under $RUNS" >&2; exit 1; }
fi

AFF=$EVAL/${TAG}_step${STEP}_aff.zarr
SCORES=$EVAL/${TAG}_step${STEP}_scores.json
mkdir -p "$EVAL"

# Fail here rather than 30 min into a GPU job: a step that was never written, or was already
# rotated away, is the likeliest thing to be wrong about a command typed from memory.
if [[ $REUSE -eq 0 && ! -d "$RUN/checkpoints/step_$STEP" ]]; then
  echo "no checkpoint at step $STEP in $RUN/checkpoints" >&2
  echo "available: $(ls "$RUN/checkpoints" 2>/dev/null | tr '\n' ' ')" >&2
  exit 1
fi
if [[ $REUSE -eq 1 && ! -d "$AFF" ]]; then
  echo "--reuse given but $AFF does not exist" >&2
  exit 1
fi
if [[ $REUSE -eq 1 && $SCORING -eq 0 && $VIZ -eq 0 ]]; then
  echo "--reuse with both --no-score and --no-viz leaves nothing to do" >&2
  exit 2
fi

echo "run    $RUN"
echo "step   $STEP"
echo "cube   $CUBE"
echo "aff    $AFF"
echo

THREADS="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4;"

# Submit and return just the job id, so the next stage can depend on it. `-P` is skipped only in
# the sense that it is always passed: scicompsoft members must name the project or the wrong group
# is billed. The anchored sed ignores the "This job will be billed to ..." line bsub prints first.
submit () {
  if [[ $DRY -eq 1 ]]; then
    { printf 'bsub'; printf ' %q' -P "$PROJECT" "$@"; printf '\n\n'; } >&2
    echo "DRYRUN"
    return
  fi
  bsub -P "$PROJECT" "$@" | sed -n 's/^Job <\([0-9]*\)>.*/\1/p'
}

# ---- stage 1: affinities over the cube (GPU) --------------------------------------------------
# 12 slots on gpu_h100 is the queue's slots-per-GPU ratio, and its 480 GB covers the float32
# accumulator + weight volume the blend needs over a 3000x3000x1350 cube (~73 GB stored float16).
if [[ $REUSE -eq 1 ]]; then
  echo "stage 1  skipped, reusing $AFF"
  PRED=""
else
  PRED=$(submit -q gpu_h100 -gpu "num=1" -n 12 -W 3:00 \
    -J "pred_${TAG}${STEP}" -cwd "$BANIS" \
    -o "$SCRATCH/pred_${TAG}${STEP}_%J.log" -e "$SCRATCH/pred_${TAG}${STEP}_%J.err" \
    "$THREADS export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
     $MYVENV/bin/python $BANIS/mia_predict.py '$RUN' --step $STEP \
       --cube '$CUBE' --out '$AFF' --patch $PATCH --stride $STRIDE")
  echo "stage 1  predict    job $PRED  (~35 min)"
fi

# Both remaining stages read only the affinity zarr, so they queue behind prediction together.
# `done(...)`, not `ended(...)`: a failed prediction must not launch jobs that would read a
# half-written zarr and report numbers for it.
WAIT=()
[[ -n "$PRED" ]] && WAIT=(-w "done($PRED)")
# `${WAIT[@]+...}` rather than a bare "${WAIT[@]}": under `set -u`, bash before 4.4 treats an empty
# array expansion as an unbound variable, which is exactly the --reuse path.

# ---- stage 2: instances + nERL/VOI (CPU, banisvenv) ------------------------------------------
# 20 slots on `local` is 300 GB: mia_score.py holds the whole cube's affinities plus the labelled
# segmentation at once. NUMBA_NUM_THREADS as well as the usual three -- funlib.evaluate's inner
# loops are numba, which ignores OMP_NUM_THREADS.
SCORE=""
if [[ $SCORING -eq 1 ]]; then
  SCORE=$(submit -q local -n 20 -W 8:00 ${WAIT[@]+"${WAIT[@]}"} \
    -J "score_${TAG}${STEP}" -cwd "$BANIS" \
    -o "$SCRATCH/score_${TAG}${STEP}_%J.log" -e "$SCRATCH/score_${TAG}${STEP}_%J.err" \
    "$THREADS export NUMBA_NUM_THREADS=4; \
     $BANISVENV/bin/python $BANIS/mia_score.py '$AFF' \
       --skeleton '$CUBE/skeleton.pkl' --out '$SCORES' --logits $LOGITS")
  echo "stage 2  score      job $SCORE  (~10 min per threshold, $(set -- $LOGITS; echo $#) here)"
else
  echo "stage 2  skipped"
fi

# ---- stage 3: predicted vs ground-truth affinities (CPU, myvenv) ------------------------------
VIZJOB=""
if [[ $VIZ -eq 1 ]]; then
  VIZJOB=$(submit -q local -n 4 -W 0:30 ${WAIT[@]+"${WAIT[@]}"} \
    -J "viz_${TAG}${STEP}" -cwd "$BANIS" \
    -o "$SCRATCH/viz_${TAG}${STEP}_%J.log" -e "$SCRATCH/viz_${TAG}${STEP}_%J.err" \
    "$THREADS $MYVENV/bin/python '$BANIS/visualize_affinities.py' \
       --affinities '$AFF' --cube '$CUBE' --origin $ORIGIN --size $SIZE")
  echo "stage 3  visualise  job $VIZJOB  (figures land beside the zarr, in $EVAL)"
fi

echo
echo "watch:   bjobs $PRED $SCORE $VIZJOB"
[[ $SCORING -eq 1 ]] && cat <<EOF
scores:  grep -E 'logit|best' $SCRATCH/score_${TAG}${STEP}_*.log
         $SCORES
EOF
exit 0
