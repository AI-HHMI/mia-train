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

# TensorBoard fails outright on a taken port, which is the normal case here: the arms of a
# comparison are usually watched at the same time, and every one of these scripts defaults to 6006.
# Step to the next free port instead and print it, so two of these can run side by side.
port_busy () { ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1$"; }
while port_busy "$PORT"; do
  echo "port $PORT is in use, trying $((PORT + 1))"
  PORT=$((PORT + 1))
done

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
