#!/usr/bin/env bash
# Serve every arm of this experiment in one TensorBoard, under short readable names.
#
#   bash experiments/banis_parity/tensorboard.sh [port]
#
# Rebuilds the symlink tree each time it runs, so arms that started later -- A2 waits on A1 --
# appear as soon as they exist. Re-run it once A2 has started to pick that arm up.

set -euo pipefail

PORT=${1:-6006}
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
VIEW=$RUNS/tb_banis_parity
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
link recipe_256   "$RUNS/banis_parity__finetune_256_long_*/"

echo
echo "http://localhost:$PORT"
exec "$VENV/bin/tensorboard" --logdir "$VIEW" --port "$PORT"
