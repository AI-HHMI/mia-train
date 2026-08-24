#!/usr/bin/env bash
# Score the arms on nERL over a WHOLE NISB validation cube.
#
#   bash experiments/data_scaling/score.sh                              # base, final stage-B ckpt
#   bash experiments/data_scaling/score.sh --subset slice_perturbed     # the robustness probe
#   bash experiments/data_scaling/score.sh --stage a                    # stage A instead
#   bash experiments/data_scaling/score.sh --step 50000 1 5             # specific step / arms
#
# Whole cube, not a block: every arm is then measured over the same extent, which is all the
# comparison needs. (nERL is *not* comparable across extents -- the same model scored 0.3045 whole
# cube and 0.4192 on a 512^3 block -- so these compare to each other and to the whole-cube
# `init_comparison` figures, and NOT to `subpixel_decoder`'s 0.5844, which is a block figure.)
#
# `--patch 256` is not a tuning knob. RoPE normalises coordinates by the *runtime* grid extent, so
# a patch size other than the one trained on silently changes every position the encoder sees.
#
# Subsets. `base` is seed100, the cube every run in this line of work has validated on and the one
# these arms' own `[val_data]` used. `slice_perturbed` is seed107, an out-of-distribution cube from
# a different NISB generator setting that NO arm trained on -- so it asks a different question:
# whether more training cubes buy robustness even when they do not buy in-distribution accuracy.
# The two cubes have near-equal extents (both 3000x3000x1350) and skeleton sizes (784,783 vs
# 795,813 nodes), but they are still different skeletons: read the *drop* from base to perturbed
# within an arm, and compare those drops across arms. Do not read one cube's absolute nERL against
# the other's.
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
NISB=/groups/miaai/miaai/lmd-v0.0.1/dev/nisb
mkdir -p "$OUT" "$LOGS"

WHICH=b STEP="" SUBSET=base
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --stage)  WHICH=$2;  shift ;;
    --step)   STEP=$2;   shift ;;
    --subset) SUBSET=$2; shift ;;
    *) echo "unknown flag $1" >&2; exit 2 ;;
  esac
  shift
done
ARMS=("$@"); [[ ${#ARMS[@]} -eq 0 ]] && ARMS=(1 3 5)
[[ $WHICH == a ]] && EXP=interp || EXP=subpixel

# The val cube of each subset. The skeleton lives *inside* the zarr, not beside it -- the flat
# `.../nisb/<subset>/val/<seed>/skeleton.pkl` layout the 2026-08 eval scripts used no longer exists.
case "$SUBSET" in
  base)            CUBE="$NISB/base/val/seed100.zarr" ;;
  slice_perturbed) CUBE="$NISB/slice_perturbed/val/seed107.zarr" ;;
  *) CUBE="$NISB/$SUBSET/val/$(ls "$NISB/$SUBSET/val" | head -1)" ;;
esac
SKEL="$CUBE/skeleton.pkl"
[[ -d "$CUBE" && -f "$SKEL" ]] || { echo "no cube/skeleton for subset '$SUBSET' at $CUBE" >&2; exit 2; }
echo "subset $SUBSET -> $CUBE"

jobid () { sed -n 's/^Job <\([0-9]*\)>.*/\1/p'; }

for arm in "${ARMS[@]}"; do
  name="${arm}cube_$EXP"
  run=$(ls -dt "$RUNS/ds__${name}_"*/ 2>/dev/null | head -1) || true
  [[ -n "${run:-}" ]] || { echo "arm $arm: no run directory for ds__$name yet, skipping" >&2; continue; }
  run=${run%/}
  step=${STEP:-$(ls -d "$run"/checkpoints/step_* | sed 's|.*step_||' | sort -n | tail -1)}
  tag="ds_${name}_step${step}_$SUBSET"
  aff="$OUT/${tag}_aff.zarr"

  p=$(bsub -P "$PROJECT" -q gpu_h100 -gpu "num=1" -n 12 -W 12:00 -J "sc_${tag}_p" -cwd "$REPO" \
        -o "$LOGS/sc_${tag}_p_%J.log" -e "$LOGS/sc_${tag}_p_%J.err" \
        "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
         $TRAINVENV/bin/python $BANIS/mia_predict.py '$run' --step $step \
           --cube '$CUBE' --out '$aff' --patch 256 --stride 128" | jobid)

  # 24 slots x 15 GB = 360 GB. The whole-cube segmentation alone is ~49 GB as uint32, and the
  # metrics hold more than one array of that size at once.
  s=$(bsub -P "$PROJECT" -q local -n 24 -W 8:00 -w "done($p)" -J "sc_${tag}_s" -cwd "$REPO" \
        -o "$LOGS/sc_${tag}_s_%J.log" -e "$LOGS/sc_${tag}_s_%J.err" \
        "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8; \
         $SCOREVENV/bin/python $BANIS/mia_score.py '$aff' --skeleton '$SKEL' \
           --out '$OUT/${tag}_scores.json' --logits 3 4 5 6 7 && rm -rf '$aff'" | jobid)

  printf "arm %s  %-20s step %-7s %-16s predict=%s -> score=%s\n" \
    "$arm" "$name" "$step" "$SUBSET" "$p" "$s"
done

echo; echo "results land in $OUT/*_scores.json ; read best_by_nerl"
