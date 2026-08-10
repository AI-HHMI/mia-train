#!/usr/bin/env bash
# Submit the recipe-parity run. One arm; add more here as the question narrows.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
PROJECT=miaai
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
LOGS="$RUNS/jobs"
mkdir -p "$LOGS"

QUEUE=gpu_h100
GPUS=8
SLOTS=96

# `--resume` makes this idempotent: the first launch starts the run, a resubmission after the
# wall-time limit continues the newest run of this experiment. 300k steps will not fit one window.
bsub -P "$PROJECT" -q "$QUEUE" -gpu "num=$GPUS" -n "$SLOTS" -W 48:00 -r \
     -J banis_parity -cwd "$REPO" \
     -o "$LOGS/banis_parity_%J.log" -e "$LOGS/banis_parity_%J.err" \
     "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; \
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
      $VENV/bin/torchrun --standalone --nproc_per_node=$GPUS src/train.py \
        --config '$HERE/finetune_256_long.toml' --resume"
