#!/usr/bin/env bash
# Segment-everything one SAM arm's round over the eight instance-segmentation eval volumes.
#
#   bash experiments/sam_lmd_v1/predict_eval.sh base 2                    # both halves, arm `base` round 2
#   bash experiments/sam_lmd_v1/predict_eval.sh stride2 0 --half test     # the reported half only
#   bash experiments/sam_lmd_v1/predict_eval.sh base 2 --volume kasthuri15_ac4
#   bash experiments/sam_lmd_v1/predict_eval.sh base 2 --dry-run
#
# Writes `instances` artifacts plus the co-registered ground truth beside each, on the same lattice
# the affinity arms were predicted on (same YAMLs, same `predict.py`, same patch), so the SAM rows
# score on the very region the leaderboard's MWS rows do. Scoring is `score.sh`.
#
# **Both halves are needed.** The size filter is fitted on the finetune half and applied to the
# reported half, exactly as for the MWS entries; mia-evals refuses to fit and report on one set.
#
# **Mask-generator settings are fixed here, not fitted.** They are the values
# promptable_seg_v1/RESULTS.md found best on a held-out block (14^3 clicks per tile, IoU and
# stability gates at 0.5, propagate-with-coverage tile reconciliation), applied identically to
# every arm and round, and recorded in each artifact's attrs. The only parameter fitted per arm is
# the size filter, which is also the only one fitted for the MWS rows. `points_per_batch` is a
# throughput knob with no effect on the result and is lowered for the finer-stride arms, whose
# per-prompt logits are 8x and 64x larger.
#
# **B300, one architecture for every SAM arm.** The same code on a different GPU generation gives
# a different instance count (530 vs 520 on one block, pq 0.0486 vs 0.0503 -- measured), so all
# SAM arms are predicted on the queue they trained on. The affinity rows were predicted on H100 /
# L4; that cross-family difference is noted in the README rather than avoidable.
#
# Slots follow host memory at 40 GB each: the labelling is int64 over the output lattice, the
# ground truth beside it is another, and `predict.py` holds both while writing.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
STAGE=/nrs/scicompsoft/orhane/mia-train-scratch/sam_lmd_v1
LOGS="$RUNS/jobs"
PROJECT=miaai
QUEUE=${QUEUE:-gpu_b300}

ARM="${1:-}"; ROUND="${2:-}"
[[ -n "$ARM" && -n "$ROUND" ]] || { sed -n '2,8p' "$0"; exit 2; }
shift 2

HALVES=(test finetune); ONLY_VOLUME=""; STEP=""; DRY=0
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --half)    HALVES=("$2"); shift ;;
    --volume)  ONLY_VOLUME="$2"; shift ;;
    --step)    STEP="$2"; shift ;;
    --dry-run) DRY=1 ;;
    *) echo "unknown flag $1" >&2; exit 2 ;;
  esac
  shift
done

run=$(ls -dt "$RUNS"/sam1__${ARM}_r${ROUND}_*/ 2>/dev/null | head -1) || true
[[ -n "${run:-}" ]] || { echo "no run matching sam1__${ARM}_r${ROUND}_* under $RUNS" >&2; exit 1; }
run=${run%/}
[[ -n "$STEP" ]] || STEP=$(ls -d "$run"/checkpoints/step_* | sed 's|.*step_||' | sort -n | tail -1)
[[ -d "$run/checkpoints/step_$STEP" ]] || { echo "no step_$STEP in $run/checkpoints" >&2; exit 1; }

# The arm's mask stride decides the throughput knob and the memory below.
upscale=$("$VENV/bin/python" -c "import json,sys; print(json.load(open(sys.argv[1]))['algorithm']['kwargs'].get('mask_upscale', 4))" "$run/resolved_config.json")
case "$upscale" in 16) batch=8; mem_scale=3 ;; 8) batch=32; mem_scale=2 ;; *) batch=64; mem_scale=1 ;; esac
# One string, because it is spliced into a command LSF hands to a shell: the TOML quotes around
# a string value must survive that second parse, so they are escaped here (`\"propagate\"`).
# A run was once lost to the shell stripping them and predict.py refusing the bare word.
OVERRIDES='--override algorithm.points_per_side=14 --override algorithm.pred_iou_thresh=0.5 \
--override algorithm.stability_thresh=0.5 --override algorithm.nms_iou=0.7 \
--override algorithm.tile_merge=\"propagate\" --override algorithm.propagate_min_coverage=0.8'
OVERRIDES+=" --override algorithm.points_per_batch=$batch"
echo "run   $run"
echo "step  $STEP"
echo "mask_upscale $upscale -> points_per_batch $batch"

slots_for () {                          # host slots at 40 GB; two int64 volumes plus write buffers
  local base
  case "$1" in
    zebrafish_fish2_doublecube1) base=5 ;;   # 7.1 G voxels
    zebrafish_fish2_quadcube1)   base=4 ;;   # 4.7 G
    hemibrain_ellipsoid_body)    base=2 ;;
    liconn_expid82)              base=2 ;;
    *)                           base=1 ;;
  esac
  echo $(( base + mem_scale - 1 ))
}
wall_for () {                           # tiles x ~6 s per tile at the base head, with headroom
  local h
  case "$1" in
    zebrafish_fish2_doublecube1) h=24 ;;     # 2744 tiles
    zebrafish_fish2_quadcube1)   h=16 ;;
    *)                           h=6 ;;
  esac
  printf '%d:00' $(( h * mem_scale ))
}

jobid () { sed -n 's/^Job <\([0-9]*\)>.*/\1/p'; }

for half in "${HALVES[@]}"; do
  case "$half" in                        # the same YAMLs mia-evals' two task halves read
    test)     config="$REPO/experiments/lmd_ssl_v1/lmd_val_singlescale.yaml" ;;
    finetune) config="$REPO/experiments/lmd_ssl_v1/lmd_finetune_singlescale.yaml" ;;
    *) echo "--half must be test or finetune, got $half" >&2; exit 2 ;;
  esac
  out="$STAGE/eval/${ARM}_r${ROUND}/$half"
  mkdir -p "$out" "$LOGS"
  volumes=$(grep '^- name: ' "$config" | sed 's/^- name: //')
  for volume in $volumes; do
    [[ -n "$ONLY_VOLUME" && "$volume" != "$ONLY_VOLUME" ]] && continue
    n=$(slots_for "$volume"); w=$(wall_for "$volume")
    tag="${ARM}_r${ROUND}_${half}_${volume}"
    cmd="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$REPO/src; \
$VENV/bin/python $REPO/src/predict.py '$run' --data-config '$config' \
--volume '$volume' --step $STEP --out '$out' $OVERRIDES"
    if (( DRY )); then
      printf 'bsub -P %s -q %s -gpu "num=1" -n %s -W %s -J p_%s\n   %s\n' "$PROJECT" "$QUEUE" "$n" "$w" "$tag" "$cmd"
      continue
    fi
    id=$(bsub -P "$PROJECT" -q "$QUEUE" -gpu "num=1" -n "$n" -W "$w" -J "p_$tag" -cwd "$REPO" \
      -o "$LOGS/p_${tag}_%J.log" -e "$LOGS/p_${tag}_%J.err" "$cmd" | jobid)
    printf '  %-28s %-9s slots=%-2s wall=%-6s job=%s\n' "$volume" "$half" "$n" "$w" "$id"
  done
done

echo
echo "artifacts land in $STAGE/eval/${ARM}_r${ROUND}/{test,finetune}/<volume>{,.gt}.zarr"
echo "then:  bash experiments/sam_lmd_v1/score.sh $ARM $ROUND"
