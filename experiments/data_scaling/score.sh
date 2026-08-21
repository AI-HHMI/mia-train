#!/usr/bin/env bash
# Score the arms on nERL over the WHOLE seed100 validation cube.
#
#   bash experiments/data_scaling/score.sh                 # final stage-B checkpoint of each arm
#   bash experiments/data_scaling/score.sh --stage a        # stage A's checkpoints instead
#   bash experiments/data_scaling/score.sh --step 50000 1 5 # a specific step, specific arms
#
# Whole cube, not a block: every arm is then measured over the same extent, which is all the
# comparison needs. (nERL is *not* comparable across extents -- the same model scored 0.3045 whole
# cube and 0.4192 on a 512^3 block -- so these numbers compare to each other and to the whole-cube
# `init_comparison` figures, and NOT to the 0.5844 block figure from `subpixel_decoder`.)
#
# `--patch 256` is not a tuning knob. RoPE normalises coordinates by the *runtime* grid extent, so
# a patch size other than the one trained on silently changes every position the encoder sees.
#
# Two jobs per arm, chained: predict on a GPU, then score on CPU, then delete the affinities --
# each whole-cube prediction is ~51 GB on disk and there is no reason to keep it once the JSON
# exists. Deletion is in the scoring job, so it only happens after a successful score.
set -euo pipefail

REPO=/groups/scicompsoft/home/orhane/projects/mia-train
BANIS=/groups/scicompsoft/home/orhane/projects/banis
TRAINVENV=/groups/scicompsoft/home/orhane/myvenv     # torch + mia-train, for prediction
SCOREVENV=/groups/scicompsoft/home/orhane/banisvenv  # numba + the BANIS metrics, for scoring
PROJECT=miaai
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
OUT=/nrs/scicompsoft/orhane/mia-train-scratch/data_scaling/eval
LOGS="$RUNS/jobs"
CUBE=/groups/miaai/miaai/lmd-v0.0.1/nisb/train_100/val/seed100
SKEL="$CUBE/skeleton.pkl"
mkdir -p "$OUT" "$LOGS"

WHICH=b STEP=""
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --stage) WHICH=$2; shift ;;
    --step)  STEP=$2;  shift ;;
    *) echo "unknown flag $1" >&2; exit 2 ;;
  esac
  shift
done
ARMS=("$@"); [[ ${#ARMS[@]} -eq 0 ]] && ARMS=(1 3 5)
[[ $WHICH == a ]] && EXP=interp || EXP=subpixel

jobid () { sed -n 's/^Job <\([0-9]*\)>.*/\1/p'; }

for arm in "${ARMS[@]}"; do
  name="${arm}cube_$EXP"
  run=$(ls -dt "$RUNS/ds__${name}_"*/ 2>/dev/null | head -1) || true
  [[ -n "${run:-}" ]] || { echo "arm $arm: no run directory for ds__$name yet, skipping" >&2; continue; }
  run=${run%/}
  step=${STEP:-$(ls -d "$run"/checkpoints/step_* | sed 's|.*step_||' | sort -n | tail -1)}
  tag="ds_${name}_step${step}"
  aff="$OUT/${tag}_aff.zarr"

  p=$(bsub -P "$PROJECT" -q gpu_h100 -gpu "num=1" -n 12 -W 12:00 -J "sc_${tag}_p" -cwd "$REPO" \
        -o "$LOGS/sc_${tag}_p_%J.log" -e "$LOGS/sc_${tag}_p_%J.err" \
        "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
         $TRAINVENV/bin/python $BANIS/mia_predict.py '$run' --step $step \
           --cube $CUBE --out '$aff' --patch 256 --stride 128" | jobid)

  # 24 slots x 15 GB = 360 GB. The whole-cube segmentation alone is ~49 GB as uint32, and the
  # metrics hold more than one array of that size at once.
  s=$(bsub -P "$PROJECT" -q local -n 24 -W 8:00 -w "done($p)" -J "sc_${tag}_s" -cwd "$REPO" \
        -o "$LOGS/sc_${tag}_s_%J.log" -e "$LOGS/sc_${tag}_s_%J.err" \
        "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8; \
         $SCOREVENV/bin/python $BANIS/mia_score.py '$aff' --skeleton $SKEL \
           --out '$OUT/${tag}_scores.json' --logits 3 4 5 6 7 && rm -rf '$aff'" | jobid)

  printf "arm %s  %-22s step %-7s predict=%s -> score=%s\n" "$arm" "$name" "$step" "$p" "$s"
done

echo
echo "results land in $OUT/*_scores.json ; read best_by_nerl"
