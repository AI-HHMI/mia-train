#!/usr/bin/env bash
# Serve the five version-4 arms of sam_lmd_v1 in one TensorBoard.
#
#   bash experiments/sam_lmd_v1/tensorboard.sh [port]
#   bash experiments/sam_lmd_v1/tensorboard.sh 6008 --smoke     # the 20-step checks instead
#   bash experiments/sam_lmd_v1/tensorboard.sh 6006 --list      # rebuild the tree, print, do not serve
#
# Rebuilds the symlink tree each time, so an arm appears as soon as its run directory exists (the
# newest run of an arm wins if it was rerun). Only the version-4 arms are served; the version-3
# `feat64` run and the version 1/2 arms are superseded (README, "Status" and "Version 4").
#
# THE FIVE ARMS (all: feat64 head, LVD start, 3D axial RoPE, v1 prompting recipe, loss ignoring
# unlabelled voxels, 200k steps linear 3e-4 -> 3e-7, round 0 only)
#
#   arm1_4nm        the four volumes read at 4 nm (2x upsampled): token 64 nm, mask cell 16 nm,
#                   window 1 um, 4096 tokens, one crop per rank (global batch 8)
#   arm2_4nm_gb16   arm 1 with two crops per rank (global batch 16), same LR
#   arm3_p8         patch 8 at 8 nm: token 64 nm, mask cell 16 nm, window 2 um, 32768 tokens,
#                   global batch 8; ~3x the step cost of arm 1
#   arm4_8nm_gb16   version 3's geometry (8 nm, patch 16: token 128 nm, cell 32 nm, window 2 um)
#                   at two crops per rank, global batch 16 -- the batch-size question alone
#   arm5_8nm_gb32   the same at four crops per rank, global batch 32
#   arm6_8nm_gb16_musam
#                   arm 4 with Archit et al. 2025's training changes: 32 objects per crop (was 16),
#                   a foreground AND a background click per correction round, the previous mask fed
#                   back with probability 0.5 (was always). Compare with arm 4 at equal steps; its
#                   `final_iou` is measured half the time without the mask prompt, so read
#                   `first_iou` for the model and `final_iou` only against itself.
#   arm7_8nm_gb16_musam64
#                   arm 6 with 64 objects per crop instead of 32, otherwise identical: arm 7 vs
#                   arm 6 vs arm 4 is 64 vs 32 vs 16 objects (arm 4 also lacks the click pairs).
#
# WHAT TO READ
#
#   train/first_iou       One click, the min-loss candidate, on the mask grid. Arms 1-3 have 16 nm
#                         cells and arms 4-5 the version-3 32 nm cells, so compare within those two
#                         groups on this number (`first_voxel_iou` bridges them). The version-3 model
#                         (128 nm token, 32 nm cells) plateaued at 0.53-0.55 on its training crops
#                         and could not fit even one crop (0.49); the same crop read at 4 nm fit to
#                         0.80. The arms exist to find out whether that holds on the whole corpus.
#   val/first_iou         The same on 32 held-out crops. Version 3 ended at 0.52 (last eval), 0.46
#                         (last 25k-window mean), on 8 nm crops with 32 nm cells -- not the same
#                         target, so compare the arms with each other, not with that number.
#   train - val gap       Version 3 opened a 0.07 gap only at the very end; with 4 nm crops each
#                         training sample covers 1/8 of the tissue, so watch whether arms 1/2
#                         start to fit their (smaller) windows sooner.
#   arm1 vs arm2, arm4 vs arm5 (vs version 3's 0.53 train / 0.46 val at 200k)
#                         The batch-size question, at equal steps AND at equal samples (a batch-16
#                         arm sees twice the crops per step, so compare it at step N with the
#                         batch-8 arm at 2N too). Arms 4/5 answer it at version 3's own geometry,
#                         with only the RoPE differing from version 3.
#   arm1 vs arm3          Same token and cell; the difference is field of view (1 vs 2 um) and
#                         native vs interpolated voxels.
#   first_voxel_iou       Voxel-resolution IoU; at 4 nm the voxels are finer than at 8 nm, so the
#                         quantisation ceiling differs slightly between arms 1/2 and arm 3.
#   first_iou_error       |predicted - achieved| of the IoU head, round 0; sat at 0.12-0.13 in every
#                         earlier run.
#   samples_per_s, mfu    Arm 1 should run ~0.42 s/step like version 3; arm 2 somewhat more per
#                         step; arm 3 is the one to read off and check against its 288 h wall.
#   data_wait_frac        Near 0.001; spikes at the epoch period (100,000 / global batch steps)
#                         are worker respawns; a rank stuck near 1 is the stall that hit version 2.
#
# WHAT NOT TO READ ACROSS: any of these against version 1-3 curves (different RoPE, cells and
# voxels); `loss` between arm 3 and arms 1/2 (different voxel counts per object in the targets).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=${1:-6006}
MODE=${2:-}
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/sam_lmd_v1   # this experiment's home on /nrs: runs/ jobs/ eval/ probes/ (layout of 2026-09-16)
RUNS=$EXP/runs
SMOKE=$EXP/smoke
VIEW=$EXP/tensorboard
VENV=/groups/scicompsoft/home/orhane/myvenv
ARMS=(arm1_4nm arm2_4nm_gb16 arm3_p8 arm4_8nm_gb16 arm5_8nm_gb32 arm6_8nm_gb16_musam arm7_8nm_gb16_musam64)

port_busy () { ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1$"; }
if [[ "$MODE" != "--list" ]]; then
  while port_busy "$PORT"; do echo "port $PORT is in use, trying $((PORT + 1))"; PORT=$((PORT + 1)); done
fi

# Replace the links in place rather than removing the directory: a TensorBoard already serving
# it holds files open, and on NFS that leaves the directory undeletable.
mkdir -p "$VIEW"; find "$VIEW" -maxdepth 1 -type l -delete
missing=()
link () {
  local name=$1 dir
  dir=$(ls -dt $2 2>/dev/null | head -1 || true)
  if [[ -n "$dir" && -d "${dir%/}/tensorboard" ]]; then
    ln -sfn "${dir%/}/tensorboard" "$VIEW/$name"; printf '  %-16s -> %s\n' "$name" "$(basename "${dir%/}")"
  else
    missing+=("$name")
  fi
}

echo "arms found:"
for arm in "${ARMS[@]}"; do
  if [[ "$MODE" == "--smoke" ]]; then
    link "$arm" "$SMOKE/smoke_sam1__${arm}_r0_*/"
  else
    link "$arm" "$RUNS/sam1__${arm}_r0_*/"
  fi
done
(( ${#missing[@]} )) && echo "  not started: ${missing[*]}"

echo
echo "arm1_4nm = 4 nm read, 64 nm token, 1 um window, gb 8 | arm2_4nm_gb16 = arm 1 at gb 16 |"
echo "arm3_p8 = patch 8 at 8 nm, 64 nm token, 2 um window, 32k tokens, gb 8 |"
echo "arm4_8nm_gb16 / arm5_8nm_gb32 = version 3's geometry (8 nm, patch 16, 32 nm cells) at gb 16 / 32."
echo "arm6_8nm_gb16_musam = arm 4 + 32 objects/crop, click pairs, mask fed back at p=0.5 (Archit et al. 2025)."
echo "arm7_8nm_gb16_musam64 = arm 6 with 64 objects/crop."
echo "All: feat64 head, 3D axial RoPE, v1 recipe, 200k steps, LR 3e-4."
[[ "$MODE" == "--list" ]] && exit 0
echo; echo "http://localhost:$PORT"
exec "$VENV/bin/tensorboard" --logdir "$VIEW" --port "$PORT"
