#!/usr/bin/env bash
# Serve every arm of this experiment in one TensorBoard.
#
#   bash experiments/init_comparison_comparison/tensorboard.sh [port]
#
# Rebuilds the symlink tree each run, so arms that start later -- 3b waits on 3a -- appear as soon
# as they exist. `boundary_accuracy` is the panel to read: pooled `affinity_accuracy` sits near the
# target's ~83% positive rate whatever the model does.
set -euo pipefail

PORT=${1:-6006}
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
VIEW=$RUNS/tb_init_comparison
VENV=/groups/scicompsoft/home/orhane/myvenv

# TensorBoard refuses a taken port rather than falling back, and watching two experiments at once
# is the normal case here.
port_busy () { ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1$"; }
while port_busy "$PORT"; do
  echo "port $PORT is in use, trying $((PORT + 1))"
  PORT=$((PORT + 1))
done

rm -rf "$VIEW"; mkdir -p "$VIEW"

link () {
  local name=$1 dir
  dir=$(ls -dt $2 2>/dev/null | head -1 || true)
  if [[ -n "$dir" && -d "${dir%/}/tensorboard" ]]; then
    ln -sfn "${dir%/}/tensorboard" "$VIEW/$name"; echo "  $name -> ${dir%/}"
  else
    echo "  $name -- not started yet"
  fi
}

echo "stages found:"
link 1a_scratch_interp      "$RUNS/init__1a_scratch_interp_*/"
link 1b_scratch_subpixel    "$RUNS/init__1b_scratch_subpixel_*/"
link 2a_dinov3_interp       "$RUNS/init__2a_dinov3_interp_*/"
link 2b_dinov3_subpixel     "$RUNS/init__2b_dinov3_subpixel_*/"
link 3a_simmim_pretrain     "$RUNS/init__3a_simmim_pretrain_*/"
link 3b_simmim_interp       "$RUNS/init__3b_simmim_interp_*/"
link 3c_simmim_subpixel     "$RUNS/init__3c_simmim_subpixel_*/"
link 4a_dinov3_aug_interp   "$RUNS/init__4a_dinov3_aug_interp_*/"
link 4b_dinov3_aug_subpixel "$RUNS/init__4b_dinov3_aug_subpixel_*/"

echo; echo "http://localhost:$PORT"
exec "$VENV/bin/tensorboard" --logdir "$VIEW" --port "$PORT"
