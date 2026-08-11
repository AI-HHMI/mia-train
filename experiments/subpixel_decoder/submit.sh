#!/usr/bin/env bash
# Submit the sub-pixel decoder arm.
#
#   bash experiments/subpixel_decoder/submit.sh              # the run
#   bash experiments/subpixel_decoder/submit.sh --smoke      # 20 steps, 1 GPU, 30 min
#
# `--smoke` exists because the two things most likely to be wrong here fail at startup, not at
# convergence: the warm start matching no parameter, and a crop the sub-pixel head cannot cover.
# Both are cheaper to find on one GPU than on eight.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
PROJECT=miaai
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
LOGS="$RUNS/jobs"
mkdir -p "$LOGS"

THREADS="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4;"
THREADS="$THREADS export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;"

if [[ "${1:-}" == "--smoke" ]]; then
  # Derived from the real config rather than kept as a second file: a hand-maintained copy drifts,
  # and then the smoke test stops exercising what actually runs. `train.py` takes no override
  # flags, so the edit happens here. It must land on shared storage -- the job runs on another
  # host, where /tmp is a different disk.
  SMOKE=/nrs/scicompsoft/orhane/mia-train-scratch/smoke
  mkdir -p "$SMOKE"
  sed -e 's/^experiment_name = .*/experiment_name = "subpixel_smoke"/' \
      -e 's/^max_steps = .*/max_steps = 20/' \
      -e 's/^warmup_steps = .*/warmup_steps = 2/' \
      -e 's/^val_every = .*/val_every = 10/' \
      -e 's/^checkpoint_every = .*/checkpoint_every = 20/' \
      -e 's/^samples_per_epoch = .*/samples_per_epoch = 20/' \
      -e 's/^dp_shard = .*/dp_shard = 1/' \
      "$HERE/subpixel_256.toml" > "$SMOKE/subpixel_smoke.toml"

  bsub -P "$PROJECT" -q gpu_h100 -gpu "num=1" -n 12 -W 0:30 \
       -J subpixel_smoke -cwd "$REPO" \
       -o "$LOGS/subpixel_smoke_%J.log" -e "$LOGS/subpixel_smoke_%J.err" \
       "$THREADS \
        $VENV/bin/torchrun --standalone --nproc_per_node=1 src/train.py \
          --config '$SMOKE/subpixel_smoke.toml' --output-root '$SMOKE'"
  exit 0
fi

# `--resume` makes this idempotent: first launch starts the run, a resubmission after the
# wall-time limit continues the newest run of this experiment.
bsub -P "$PROJECT" -q gpu_h100 -gpu "num=8" -n 96 -W 48:00 -r \
     -J subpixel_decoder -cwd "$REPO" \
     -o "$LOGS/subpixel_decoder_%J.log" -e "$LOGS/subpixel_decoder_%J.err" \
     "$THREADS \
      $VENV/bin/torchrun --standalone --nproc_per_node=8 src/train.py \
        --config '$HERE/subpixel_256.toml' --resume"
