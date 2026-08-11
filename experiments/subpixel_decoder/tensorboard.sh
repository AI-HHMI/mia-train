#!/usr/bin/env bash
# Serve this arm beside the control it is an ablation of, so the two curves share an axis.
#
#   bash experiments/subpixel_decoder/tensorboard.sh [port]
#
# `boundary_accuracy` is the panel to read. Pooled `affinity_accuracy` is close to useless on this
# task -- the target is ~83% positive, so a model predicting "same object" everywhere scores 0.83
# and cuts nothing -- and the first 20 steps of this arm demonstrated exactly that, reporting 0.72
# pooled while getting precisely zero boundaries right.
set -euo pipefail

PORT=${1:-6006}
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
VIEW=$RUNS/tb_subpixel_decoder
VENV=/groups/scicompsoft/home/orhane/myvenv

rm -rf "$VIEW"
mkdir -p "$VIEW"

link () {           # link <short name> <run directory glob>
  local name=$1 dir
  dir=$(ls -dt $2 2>/dev/null | head -1 || true)
  if [[ -n "$dir" && -d "${dir%/}/tensorboard" ]]; then
    ln -sfn "${dir%/}/tensorboard" "$VIEW/$name"
    echo "  $name -> ${dir%/}"
  else
    echo "  $name -- not started yet"
  fi
}

echo "arms found:"
link subpixel  "$RUNS/subpixel_decoder__subpixel_256_*/"
link control   "$RUNS/banis_parity__finetune_256_long_*/"

echo
echo "http://localhost:$PORT"
exec "$VENV/bin/tensorboard" --logdir "$VIEW" --port "$PORT"
