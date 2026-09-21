#!/usr/bin/env bash
# Submit one arm of lsd_aux_v1.
#
#   bash experiments/lsd_aux_v1/submit.sh mtlsd_s80                 # the run: 8 GPUs, resumable
#   bash experiments/lsd_aux_v1/submit.sh mtlsd_s80_bgzero
#   bash experiments/lsd_aux_v1/submit.sh mtlsd_s80 --smoke         # 20 steps, 1 GPU, 30 min
#   QUEUE=gpu_b300 bash experiments/lsd_aux_v1/submit.sh mtlsd_s80  # another queue (see README)
#
# Mirrors experiments/subpixel_decoder/submit.sh, whose run is this experiment's control, so the
# two are launched the same way: same queue by default, same slots per GPU, `--resume` so a
# resubmission after the wall-time limit continues the newest run of the arm.
set -euo pipefail

ARM=${1:-}
if [[ -z "$ARM" || "$ARM" == -* ]]; then
  echo "usage: $(basename "$0") <arm> [--smoke]     arms: $(cd "$(dirname "$0")" && ls *.toml | sed 's/\.toml//' | tr '\n' ' ')" >&2
  exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CONFIG="$HERE/$ARM.toml"
[[ -f "$CONFIG" ]] || { echo "no such arm: $CONFIG" >&2; exit 2; }

VENV=/groups/scicompsoft/home/orhane/myvenv
PROJECT=miaai
QUEUE=${QUEUE:-gpu_h100}      # the control trained on gpu_h100; see the README before changing
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/lsd_aux_v1   # runs/ jobs/ eval/ probes/ (layout of 2026-09-16)
RUNS=$EXP/runs
LOGS=$EXP/jobs
mkdir -p "$LOGS"

THREADS="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4;"
THREADS="$THREADS export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;"

if [[ "${2:-}" == "--smoke" ]]; then
  # Derived from the arm's config rather than kept as a second file, so the smoke test exercises
  # what actually runs. Smoke artifacts go under jobs/smoke and runs/smoke, never beside the runs.
  mkdir -p "$LOGS/smoke" "$RUNS/smoke"
  sed -e "s/^experiment_name = .*/experiment_name = \"lsd_aux_v1_smoke__$ARM\"/" \
      -e 's/^max_steps = .*/max_steps = 20/' \
      -e 's/^warmup_steps = .*/warmup_steps = 2/' \
      -e 's/^val_every = .*/val_every = 10/' \
      -e 's/^checkpoint_every = .*/checkpoint_every = 20/' \
      -e 's/^samples_per_epoch = .*/samples_per_epoch = 20/' \
      -e 's/^dp_shard = .*/dp_shard = 1/' \
      "$CONFIG" > "$LOGS/smoke/$ARM.toml"
  set -x
  bsub -P "$PROJECT" -q "$QUEUE" -gpu "num=1" -n 12 -W 0:30 \
       -J "lsd_smoke_$ARM" -cwd "$REPO" \
       -o "$LOGS/smoke/${ARM}_%J.log" -e "$LOGS/smoke/${ARM}_%J.err" \
       "$THREADS \
        $VENV/bin/torchrun --standalone --nproc_per_node=1 src/train.py \
          --config '$LOGS/smoke/$ARM.toml' --output-root '$RUNS/smoke'"
  exit 0
fi

set -x
bsub -P "$PROJECT" -q "$QUEUE" -gpu "num=8" -n 96 -W 48:00 -r \
     -J "lsd_$ARM" -cwd "$REPO" \
     -o "$LOGS/${ARM}_%J.log" -e "$LOGS/${ARM}_%J.err" \
     "$THREADS \
      $VENV/bin/torchrun --standalone --nproc_per_node=8 src/train.py \
        --config '$CONFIG' --output-root '$RUNS' --resume"
