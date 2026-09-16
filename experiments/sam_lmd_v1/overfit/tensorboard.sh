#!/usr/bin/env bash
# Serve the single-crop overfit tests side by side, with the version 3 round-0 run as the reference.
#
#   bash experiments/sam_lmd_v1/overfit/tensorboard.sh [port]
#   bash experiments/sam_lmd_v1/overfit/tensorboard.sh 6007 --list    # rebuild the tree, print, do not serve
#
# Runs are named by their test so the legend reads `8nm`, `4nm`, `8nm_stride2` and
# `ref_feat64_r0_200k`. Rebuilt
# on every call, so a test appears as soon as its run directory exists under the overfit scratch
# root; the newest run of each test wins if a test was rerun.
#
# WHAT TO READ
#
#   train/first_iou       The question these tests ask. One fixed crop, no augmentation, the same
#                         objects prompted every step: a pipeline that can represent the masks
#                         climbs past 0.9 well inside 2000 steps. `8nm` is the crop at the
#                         experiment's resolution (token 128 nm); `4nm` is the central 1 um of it
#                         read at 4 nm (token 64 nm), so the two curves separate the model's
#                         capacity from the token-versus-neurite size mismatch. `8nm_stride2` is
#                         the 8 nm crop with the mask drawn on 2-voxel cells (16 nm) instead of
#                         4-voxel cells (32 nm), token unchanged: it separates the mask grid from
#                         the token. The reference run plateaued at 0.53-0.55 on its training
#                         crops after 200k steps; 8nm reached 0.49, 4nm 0.80.
#   train/first_voxel_iou The same at voxel resolution: capped near 0.75 at stride 4 by
#                         quantisation alone (0.9 at stride 2), so read it against that ceiling.
#   val/*                 The same crop, unaugmented, so `val` tracks `train` here; a gap between
#                         them means the 1-voxel jitter miao requires, nothing else.
#   train/first_oracle_iou - train/first_iou
#                         How much the three candidates offer beyond the min-loss pick.
#   loss_round_0          Should fall to well under 0.2 if the crop is being memorised.
#   lr                    100-step warmup to 3e-4, linear to 3e-7 at step 2000.
set -euo pipefail

PORT=${1:-6007}
MODE=${2:-}
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/sam_lmd_v1   # this experiment's home on /nrs: runs/ jobs/ eval/ probes/ (layout of 2026-09-16)
RUNS=$EXP/runs
OVERFIT=/nrs/scicompsoft/orhane/mia-train-experiments/sam_lmd_v1/overfit
VIEW=$EXP/overfit/tensorboard
VENV=/groups/scicompsoft/home/orhane/myvenv
TESTS=(8nm 4nm 8nm_stride2)

port_busy () { ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1$"; }
if [[ "$MODE" != "--list" ]]; then
  while port_busy "$PORT"; do echo "port $PORT is in use, trying $((PORT + 1))"; PORT=$((PORT + 1)); done
fi

# Replace the links in place rather than removing the directory: a TensorBoard already serving
# it holds files open, and on NFS that leaves the directory undeletable.
mkdir -p "$VIEW"; find "$VIEW" -maxdepth 1 -type l -delete
link () {
  local name=$1 dir
  dir=$(ls -dt $2 2>/dev/null | head -1 || true)
  if [[ -n "$dir" && -d "${dir%/}/tensorboard" ]]; then
    ln -sfn "${dir%/}/tensorboard" "$VIEW/$name"; printf '  %-22s -> %s\n' "$name" "$(basename "${dir%/}")"
  else
    printf '  %-22s -- not started yet\n' "$name"
  fi
}

echo "runs found:"
for test in "${TESTS[@]}"; do
  link "$test" "$OVERFIT/sam1_overfit_${test}_[0-9]*/"   # [0-9]: the timestamp, so 8nm does not match 8nm_stride2
done
link "ref_feat64_r0_200k" "$RUNS/sam1__feat64_r0_*/"

echo
echo "8nm = one fixed 256^3 hemibrain crop at 8 nm (token 128 nm, mask cell 32 nm); 4nm = its central 1 um"
echo "at 4 nm (token 64 nm, cell 16 nm); 8nm_stride2 = the 8 nm crop with 16 nm mask cells, token unchanged."
echo "All: feat64 head, LVD start, no augmentation, crop is train and val, 2000 steps, one B300 GPU."
[[ "$MODE" == "--list" ]] && exit 0
echo; echo "http://localhost:$PORT"
exec "$VENV/bin/tensorboard" --logdir "$VIEW" --port "$PORT"
