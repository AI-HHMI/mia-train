#!/usr/bin/env bash
# Predict this experiment's arms over the instance-segmentation eval set, for scoring in mia-evals.
#
#   bash experiments/lmd_ssl_v1/predict_eval.sh 2c                    # both halves, arm 2 stage C
#   bash experiments/lmd_ssl_v1/predict_eval.sh 2c --half test         # reported half only
#   bash experiments/lmd_ssl_v1/predict_eval.sh 2c --volume kasthuri15_ac4
#   bash experiments/lmd_ssl_v1/predict_eval.sh 2c --dry-run
#
# One job per volume, because the volumes differ in cost by two orders of magnitude: 16 tiles for
# Kasthuri against 2744 for the zebrafish doublecube, and 4 GB of blend accumulator against 264 GB.
# A single job sized for the largest would hold a 320 GB reservation for hours to do the small ones.
#
# **Both halves are needed.** This experiment finetunes on four of the eight eval volumes and
# reports on the other four. The post-processing threshold is fitted on the finetune half and
# applied to the held-out half, so scoring the reported half alone leaves nothing to fit on -- and
# fitting on the half being reported would select on the number being published. mia-evals refuses
# that; see configs/tasks/lmd_ssl_v1_neuron_instance_test.toml over there.
#
# **L4, not H100.** Nothing here needs a Hopper GPU: peak GPU memory is 5.1 GiB against the L4's
# 24 GB, and arm 2's checkpoint runs attention through SDPA -- its saved config carries no `use_fa4`,
# so FlashAttention-4 (which would require Hopper) is not in the path. The H100/H200 queues are also
# where the training jobs live, and prediction has no business competing with them for those.
#
# Slot sizing is memory-driven, and on the L4 queues one slot is **15 GB** (not the 40 GB of the
# A100/H100/H200 queues). The binding cost is the blend accumulator: 6 channels of float32 over the
# output grid, plus a weight map, plus the float16 result. Measured per volume by `VolumeGrid`.
#
# Queue follows the memory, because slots-per-GPU differs:
#   gpu_l4        8 slots/GPU ->  120 GB   small volumes
#   gpu_l4_16    16 slots/GPU ->  240 GB   the middle ones
#   gpu_l4_large 64 slots/GPU ->  960 GB, 1 GPU/node -- for the zebrafish doublecube's 264 GB,
#                                which exceeds what a single gpu_l4_16 GPU's slot budget allows.
set -euo pipefail

REPO=/groups/scicompsoft/home/orhane/projects/mia-train
VENV=/groups/scicompsoft/home/orhane/myvenv
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
OUT_ROOT=/nrs/scicompsoft/orhane/mia-train-scratch/lmd1_arm2_eval
LOGS="$RUNS/jobs"
PROJECT=miaai
STEP=50000

ARM="${1:-}"; shift || true
[[ -n "$ARM" ]] || { sed -n '2,20p' "$0"; exit 2; }

HALVES=(test finetune)
ONLY_VOLUME=""
DRY=0
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --half)   HALVES=("$2"); shift ;;
    --volume) ONLY_VOLUME="$2"; shift ;;
    --step)   STEP="$2"; shift ;;
    --dry-run) DRY=1 ;;
    *) echo "unknown flag $1" >&2; exit 2 ;;
  esac
  shift
done

run=$(ls -dt "$RUNS"/lmd1__${ARM}_*/ 2>/dev/null | head -1) || true
[[ -n "${run:-}" ]] || { echo "no run directory matching lmd1__${ARM}_* under $RUNS" >&2; exit 1; }
run=${run%/}
[[ -d "$run/checkpoints/step_$STEP" ]] || {
  echo "no checkpoint step_$STEP in $run; it has:" >&2
  ls "$run/checkpoints" >&2; exit 1; }
echo "run  $run"
echo "step $STEP"

# volume -> slots, sized from the predict-side accumulator and kept at or under each queue's
# slots-per-GPU ratio so no job strands a GPU it cannot use.
slots_for () {                               # 15 GB per slot on the L4 queues
  case "$1" in
    zebrafish_fish2_doublecube1) echo 20 ;;  # 264 GB -> 300 GB
    zebrafish_fish2_quadcube1)   echo 12 ;;  # 158 GB -> 180 GB
    hemibrain_ellipsoid_body)    echo 4 ;;   #  40 GB ->  60 GB
    liconn_expid82)              echo 4 ;;   #  38 GB ->  60 GB
    *)                           echo 1 ;;   #  <6 GB
  esac
}
queue_for () {
  case "$1" in
    zebrafish_fish2_doublecube1) echo gpu_l4_large ;;   # 20 slots exceeds gpu_l4_16's 16/GPU
    zebrafish_fish2_quadcube1)   echo gpu_l4_16 ;;
    hemibrain_ellipsoid_body)    echo gpu_l4_16 ;;
    liconn_expid82)              echo gpu_l4_16 ;;
    *)                           echo gpu_l4 ;;
  esac
}
# Wall time follows the tile count, with headroom for the L4 being several times slower than an
# H100 at bf16: ~2700 tiles at the top end, and the read and the blend are not free either. Generous
# rather than tight, because a job killed at the wall loses the whole accumulator.
wall_for () {
  case "$1" in
    zebrafish_fish2_doublecube1) echo 20:00 ;;
    zebrafish_fish2_quadcube1)   echo 12:00 ;;
    hemibrain_ellipsoid_body)    echo 4:00 ;;
    liconn_expid82)              echo 4:00 ;;
    *)                           echo 1:00 ;;
  esac
}

jobid () { sed -n 's/^Job <\([0-9]*\)>.*/\1/p'; }

for half in "${HALVES[@]}"; do
  # The two task halves in mia-evals read these same YAMLs, so the volume lists cannot diverge.
  case "$half" in
    test)     config="$REPO/experiments/lmd_ssl_v1/lmd_val_singlescale.yaml" ;;
    finetune) config="$REPO/experiments/lmd_ssl_v1/lmd_finetune_singlescale.yaml" ;;
    *) echo "--half must be test or finetune, got $half" >&2; exit 2 ;;
  esac
  out="$OUT_ROOT/$half"
  mkdir -p "$out" "$LOGS"

  volumes=$("$VENV/bin/python" - "$config" <<'PY'
import sys
from miao.config import load_config
print(" ".join(v.name for v in load_config(sys.argv[1]).volumes))
PY
)
  for volume in $volumes; do
    [[ -n "$ONLY_VOLUME" && "$volume" != "$ONLY_VOLUME" ]] && continue
    n=$(slots_for "$volume"); w=$(wall_for "$volume"); q=$(queue_for "$volume")
    tag="${ARM}_${half}_${volume}"
    cmd="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$REPO/src; \
$VENV/bin/python $REPO/src/predict.py '$run' --data-config '$config' \
--volume '$volume' --step $STEP --out '$out'"
    if (( DRY )); then
      printf 'bsub -P %s -q %s -gpu "num=1" -n %s -W %s -J p_%s\n' \
        "$PROJECT" "$q" "$n" "$w" "$tag"
      continue
    fi
    id=$(bsub -P "$PROJECT" -q "$q" -gpu "num=1" -n "$n" -W "$w" -J "p_$tag" -cwd "$REPO" \
      -o "$LOGS/p_${tag}_%J.log" -e "$LOGS/p_${tag}_%J.err" "$cmd" | jobid)
    printf '  %-28s %-9s %-13s slots=%-2s wall=%-6s job=%s\n' \
      "$volume" "$half" "$q" "$n" "$w" "$id"
  done
done

echo
echo "artifacts land in $OUT_ROOT/{test,finetune}/<volume>{,.gt}.zarr"
echo "then, in mia-evals:"
echo "  python src/evaluate.py score configs/tasks/lmd_ssl_v1_neuron_instance_test.toml \\"
echo "      --test $OUT_ROOT/test --val $OUT_ROOT/finetune \\"
echo "      --val-config configs/tasks/lmd_ssl_v1_neuron_instance_fit.toml \\"
echo "      --run-dir $run --label ${ARM}_step${STEP}"
