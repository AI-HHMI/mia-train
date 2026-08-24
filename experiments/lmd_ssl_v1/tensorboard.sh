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
#   ssl/loss            Arm 1 is SimMIM (masked voxel regression, norm_pix_loss=false); arm 3 is
#                       MuViT-MAE (norm_pix_loss=true, and it reconstructs THREE levels). These are
#                       different objectives on differently-normalised targets over different token
#                       counts. **The two numbers are not comparable to each other** -- read each
#                       only for its own shape: falling and then flattening is what a healthy
#                       pretrain looks like. The comparison happens downstream, in stage C.
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
echo "  1  DINOv3 ViT-L, random init -> SimMIM 100k          (vanilla 3D RoPE, 8 nm)"
echo "  2  DINOv3 ViT-L <- LVD-1689M checkpoint, no SSL      (superposition RoPE, 8 nm)  [baseline]"
echo "  3  MuViT, random init -> MuViT-MAE 100k              (world-coord rotary, 8/16/32 nm)"
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
