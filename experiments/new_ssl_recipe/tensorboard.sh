#!/usr/bin/env bash
# Serve every arm of this experiment in one TensorBoard.
#
#   bash experiments/new_ssl_recipe/tensorboard.sh [port]
#   bash experiments/new_ssl_recipe/tensorboard.sh 6007 --smoke   # the 20-step check instead
#
# Rebuilds the symlink tree each run, so stages that start later -- 2b waits on 2a, 2c on 2b --
# appear as soon as they exist.
#
# What to read, by stage type:
#
#   SSL (2a, 3a)          `loss` is a masked reconstruction error and only ever compares against
#                         the other SSL arm, never against a finetune stage. `val` here is a
#                         progress readout on the *training* cubes (and augmented ones at that), so
#                         treat a gap between train and val as noise rather than generalisation.
#   finetune (everything  `boundary_accuracy` is the panel. Pooled `affinity_accuracy` sits near
#   else)                 the target's ~83% positive rate whatever the model does, so it looks
#                         healthy for a model that has learned nothing.
#
#   all stages            `mfu`, `tflops_per_s`, `samples_per_s` say whether the run is limited by
#                         arithmetic or by everything else. Measured beforehand on one H100 at this
#                         geometry: ~13% MFU for simmim and 4-8% for the affinity heads at local
#                         batch 1. If `samples_per_s` sits well below what those imply, the input
#                         pipeline is the limit, not the GPU.
#
# The arm-2 panel worth watching early is the freeze boundary: `2a_ssl_twophase` holds the backbone
# fixed for its first 10k steps, and the learning rate ramps over the 1k steps after it. A visible
# kink in `loss` at 10k is the phase change, not an instability.
set -euo pipefail

PORT=${1:-6006}
MODE=${2:-}
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
SCRATCH=/nrs/scicompsoft/orhane/mia-train-scratch/new_ssl_recipe
VIEW=$RUNS/tb_new_ssl_recipe
VENV=/groups/scicompsoft/home/orhane/myvenv

# TensorBoard refuses a taken port rather than falling back, and watching this alongside
# init_comparison is the normal case.
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

if [[ "$MODE" == "--smoke" ]]; then
  # `submit.sh --smoke` writes under its own output root, so the run dirs are somewhere else
  # entirely and carry a `smoke_` prefix.
  echo "smoke stages found:"
  for stage in 1a_dinov3_interp 1b_dinov3_subpixel \
               2a_ssl_twophase 2b_ssl_twophase_interp 2c_ssl_twophase_subpixel \
               3a_ssl_joint 3b_ssl_joint_interp 3c_ssl_joint_subpixel; do
    link "$stage" "$SCRATCH/$stage/smoke/smoke_${stage}_*/"
  done
else
  # Ordered so the graph legend reads arm by arm: control first, then the two SSL arms whose
  # difference is the experiment's question.
  echo "stages found:"
  link 1a_dinov3_interp         "$RUNS/sslrec__1a_dinov3_interp_*/"
  link 1b_dinov3_subpixel       "$RUNS/sslrec__1b_dinov3_subpixel_*/"
  link 2a_ssl_twophase          "$RUNS/sslrec__2a_ssl_twophase_*/"
  link 2b_ssl_twophase_interp   "$RUNS/sslrec__2b_ssl_twophase_interp_*/"
  link 2c_ssl_twophase_subpixel "$RUNS/sslrec__2c_ssl_twophase_subpixel_*/"
  link 3a_ssl_joint             "$RUNS/sslrec__3a_ssl_joint_*/"
  link 3b_ssl_joint_interp      "$RUNS/sslrec__3b_ssl_joint_interp_*/"
  link 3c_ssl_joint_subpixel    "$RUNS/sslrec__3c_ssl_joint_subpixel_*/"
  link 4a_ssl_joint_b32         "$RUNS/sslrec__4a_ssl_joint_b32_*/"
  link 4b_ssl_b32_interp        "$RUNS/sslrec__4b_ssl_b32_interp_*/"
  link 4c_ssl_b32_subpixel      "$RUNS/sslrec__4c_ssl_b32_subpixel_*/"
fi

echo; echo "http://localhost:$PORT"
exec "$VENV/bin/tensorboard" --logdir "$VIEW" --port "$PORT"
