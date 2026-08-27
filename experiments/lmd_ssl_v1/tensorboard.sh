#!/usr/bin/env bash
# Serve all eight stages of lmd_ssl_v1 in one TensorBoard.
#
#   bash experiments/lmd_ssl_v1/tensorboard.sh [port]
#   bash experiments/lmd_ssl_v1/tensorboard.sh 6008 --smoke   # the 20-step check instead
#
# Rebuilds the symlink tree each run, so a stage appears as soon as its run directory exists.
# Stages are held one behind the next, so B shows up only after A finishes and C after B.
#
# Runs are named `<stage>_arm<N>_<what>` so the legend sorts by stage first: all three arms' stage
# B sit together, which is the comparison this experiment is for. Arm 2 has no stage A.
#
# WHAT TO READ
#
#   ssl/loss            SimMIM's masked-voxel L1 (norm_pix_loss=false), so it is in the same
#                       units as the data: [0, 1] intensities. **The number to compare it against
#                       is 0.1004** -- the measured L1 of predicting the crop's visible mean under
#                       this exact 60% / 32-voxel masking. Above that line the model has learned
#                       nothing a single scalar did not already know, which is precisely where the
#                       first (batch 8) attempt sat for its whole life.
#
#                       At batch 84 the floor is crossed at ~step 900 and the margin keeps widening
#                       (+0.0137 by step 4000). At batch 256 expect the crossing sooner in steps,
#                       though not in samples. Judge the TREND, not a single crossing: a margin
#                       that stops widening is the stall this experiment was rebuilt to escape.
#
#                       Arm 3 (MuViT-MAE) is NOT running -- it collapsed to a positional constant
#                       and is parked pending the same batch fix. Its killed run IS still plotted,
#                       pinned at 1.0000: that is its own trivial floor (targets are standardised
#                       per patch, so predicting zero scores exactly 1.0). Do not read arm 3's
#                       numbers against arm 1's -- different objective, different normalisation,
#                       different floor.
#
#   boundary_accuracy   The finetune panel. Pooled `affinity_accuracy` sits near the target's
#                       positive rate whatever the model does, so it looks healthy for a model that
#                       has learned nothing; `boundary_accuracy` is the one that moves.
#
#   val/*               32 crops from the FOUR held-out volumes (kasthuri15_ac4,
#                       zebrafish_doublecube1, liconn_mouse_hippocampus, liconn_expid82). Identical
#                       for all three arms -- this is the number the arms are compared on, and
#                       nothing about how an encoder was pretrained can touch it.
#
#                       NOT a model selector. val boundary_accuracy is per-voxel and blind to
#                       instance-level fragmentation: scoring pseudo-labelling checkpoints once
#                       showed nERL swinging 0.5844 -> 0.3889 -> 0.5518 while val boundary accuracy
#                       sat flat at 0.925-0.945. Rank arms on nERL through the banis scoring path.
#                       And with only 4 validation volumes this set doubles as selection and score,
#                       so a checkpoint picked on it is selected-on.
#
#   masked_fraction     Finetune stages only, and it should be near-identical across the three arms
#                       and flat: these are real labels, so the mask is border slabs, with no
#                       abstention. Divergence between arms means they are not seeing the same data,
#                       which would be a bug in the split rather than a result.
#
#   grad_norm           Comparable between arms 1 and 2 (both 306.6M parameters) but NOT against
#                       arm 3 (314.9M, and a 3x longer sequence).
#
#   lr                  Each stage is its own linear ramp, so a sawtooth at every stage boundary is
#                       expected. The B->C step down is deliberate and is the design: stage C peaks
#                       at 1e-4 against stage B's 3e-4, because it continues an encoder that has
#                       already converged under a decaying schedule.
#
#   samples_per_s       Worth a look on arm 3 specifically: its cost was never measured before
#                       launch (12288 tokens against 4096, attention quadratic), and its wall-clock
#                       limits in submit.sh were set high rather than extrapolated. Read the real
#                       number here and tighten them.
#
# WHAT NOT TO READ: arm 1 vs arm 3 differs in architecture, objective AND scale ladder at once, so
# a gap between them says "this recipe beat that recipe", not "multi-scale helps". Model size is
# not one of the confounds -- they are matched to within 3%.
#
# And do not rank checkpoints on `val/*` here: it is 32 crops from 4 heterogeneous volumes, and
# across arm 2's 50k steps it swung 0.62-0.84 between adjacent evaluations with no trend after
# ~12k. That spread is far larger than any real change it could resolve. Raise
# `val_data.samples_per_epoch` to ~256 before trusting it to choose anything.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=${1:-6006}
MODE=${2:-}
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
SCRATCH=/nrs/scicompsoft/orhane/mia-train-scratch/lmd_ssl_v1
VIEW=$RUNS/tb_lmd_ssl_v1
VENV=/groups/scicompsoft/home/orhane/myvenv

# TensorBoard refuses a taken port rather than falling back, and watching this alongside another
# experiment is the normal case.
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
    ln -sfn "${dir%/}/tensorboard" "$VIEW/$name"; echo "  $name -> $(basename "${dir%/}")"
  else
    echo "  $name -- not started yet"
  fi
}

# arm number -> the tag used in its experiment_name
declare -A TAG=( [1]=dinov3_simmim [2]=dinov3_lvd [3]=muvit_mae )

echo "stages found:"
if [[ "$MODE" == "--smoke" ]]; then
  for n in 1 3; do link "A_ssl_arm${n}" "$SCRATCH/smoke/smoke_${n}a_${TAG[$n]}_pretrain_*/"; done
  for n in 1 2 3; do
    link "B_interp_arm${n}"   "$SCRATCH/smoke/smoke_${n}b_${TAG[$n]}_finetune_interpolate_*/"
    link "C_subpixel_arm${n}" "$SCRATCH/smoke/smoke_${n}c_${TAG[$n]}_finetune_subpixel_*/"
  done
else
  # Grouped by stage so the legend puts the arms side by side within each stage.
  for n in 1 3;     do link "A_ssl_arm${n}_${TAG[$n]}"      "$RUNS/lmd1__${n}a_${TAG[$n]}_pretrain_*/"; done
  for n in 1 2 3;   do link "B_interp_arm${n}_${TAG[$n]}"   "$RUNS/lmd1__${n}b_${TAG[$n]}_ft_interpolate_*/"; done
  for n in 1 2 3;   do link "C_subpixel_arm${n}_${TAG[$n]}" "$RUNS/lmd1__${n}c_${TAG[$n]}_ft_subpixel_*/"; done
fi

echo
echo "arms:"
echo "  1  DINOv3 ViT-L, random init -> SimMIM 100k   global batch 128 (16/rank x 8 ranks, 1 H200 node)"
echo "  2  DINOv3 ViT-L <- LVD-1689M checkpoint, no SSL                    [baseline, batch 8]"
echo "  3  MuViT multiscale, random init -> MuViT-MAE 100k        global batch 128 (1 H200 node)"
echo
echo "  SimMIM trivial floor (predict the crop visible mean): L1 0.1004"
echo
echo "finetune split (identical for every arm):"
for split in finetune val; do
  printf "  %-8s %s\n" "$split" \
    "$("$VENV/bin/python" -c "
import yaml,sys
c=yaml.safe_load(open('$HERE/lmd_${split}_singlescale.yaml'))
print(', '.join(v['name'] for v in c['volumes']))" 2>/dev/null || echo '?')"
done

echo; echo "http://localhost:$PORT"
exec "$VENV/bin/tensorboard" --logdir "$VIEW" --port "$PORT"
