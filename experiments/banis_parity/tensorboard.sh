#!/usr/bin/env bash
# Serve every arm of this experiment in one TensorBoard, under short readable names.
#
#   bash experiments/banis_parity/tensorboard.sh [port]
#
# Rebuilds the symlink tree each time it runs, so arms that started later -- A2 waits on A1 --
# appear as soon as they exist. Re-run it once A2 has started to pick that arm up.
#
# Why a symlink tree and not `--logdir_spec name=path,name=path`: that flag is broken in
# TensorBoard 2.21 (the runs endpoint comes back empty, and the UI reports "No dashboards are
# active" with the whole spec string shown as the log directory). A plain `--logdir` over a
# directory of symlinked runs works, and gives the same per-arm names in the legend.
set -euo pipefail

PORT=${1:-6006}
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
VIEW=$RUNS/tb_banis_parity
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
link recipe_256   "$RUNS/banis_parity__finetune_256_long_*/"

echo
echo "http://localhost:$PORT"
exec "$VENV/bin/tensorboard" --logdir "$VIEW" --port "$PORT"
