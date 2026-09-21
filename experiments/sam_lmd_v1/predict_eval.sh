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
# **Mask-generator settings are the labeller's, fixed, not fitted.** Every generator knob is read
# from `pseudolabel.LABEL_AMG` (14^3 clicks per window, IoU gate 0.7, stability gate 0.8, NMS 0.7,
# windows reconciled by agreement), so the pass the leaderboard scores is exactly the pass that
# writes the pseudo-labels and that every diagnostic (probe, block tables, pictures) describes.
# Until 2026-09-15 the gates here were 0.5 / 0.5, the values promptable_seg_v1/RESULTS.md found best
# on one block of liconn_expid82 with an early model (pq 0.142 against 0.139 for 0.5 / 0.8 -- within
# noise); no artifact was ever scored with them. Applied identically to every arm and round, and
# recorded in each artifact's attrs. The only parameter fitted per arm is the size filter, which is
# also the only one fitted for the MWS rows. `points_per_batch` is a throughput knob with no effect
# on the result and is lowered for the finer-stride arms, whose per-prompt logits are 8x and 64x
# larger.
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
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/sam_lmd_v1   # this experiment's home on /nrs: runs/ jobs/ eval/ probes/ (layout of 2026-09-16)
RUNS=$EXP/runs
STAGE=$EXP
LOGS="$EXP/jobs"
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

# The arm's mask stride (patch / mask_upscale, in voxels) decides the throughput knob and the
# memory below: a stride-2 arm has 8x the logits per prompt of a stride-4 one, whether it got there
# through mask_upscale 8 at patch 16 or through mask_upscale 4 at patch 8 (arm3_p8).
stride=$("$VENV/bin/python" -c "import json,sys; r=json.load(open(sys.argv[1])); print(int(r['model']['kwargs'].get('patch_size', 16)) // int(r['algorithm']['kwargs'].get('mask_upscale', 4)))" "$run/resolved_config.json")
case "$stride" in 1) batch=8; mem_scale=3 ;; 2) batch=32; mem_scale=2 ;; *) batch=64; mem_scale=1 ;; esac
# One string, because it is spliced into a command LSF hands to a shell: the TOML quotes around
# a string value must survive that second parse, so they are escaped (`\"consensus\"`). A run was
# once lost to the shell stripping them and predict.py refusing the bare word. The knobs come from
# the labeller's LABEL_AMG so the two passes cannot drift apart; points_per_batch is overridden last.
OVERRIDES=$("$VENV/bin/python" - "$REPO/experiments/sam_lmd_v1" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from pseudolabel import LABEL_AMG
def toml(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return '\\"%s\\"' % v if isinstance(v, str) else str(v)
print(" ".join(f"--override algorithm.{k}={toml(v)}" for k, v in LABEL_AMG.items() if k != "points_per_batch"))
PY
)
OVERRIDES+=" --override algorithm.points_per_batch=$batch"
echo "run   $run"
echo "step  $STEP"
echo "mask stride $stride voxels -> points_per_batch $batch, memory x$mem_scale"
echo "generator $OVERRIDES"

SLOTS=12                                # one GPU's share of an H100/H200/B300 node (12 slots/GPU, 40 GB each); never size by RAM
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
    n=$SLOTS; w=$(wall_for "$volume")
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
