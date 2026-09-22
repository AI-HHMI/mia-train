#!/usr/bin/env bash
# Serve the gary_comparison arms in one TensorBoard.
#
#   bash experiments/gary_comparison/tensorboard.sh [port]
#   bash experiments/gary_comparison/tensorboard.sh 6008 --smoke     # the 20-step smoke runs instead
#   bash experiments/gary_comparison/tensorboard.sh 6006 --list      # rebuild the tree, print, do not serve
#
# Rebuilds the symlink tree each time, so an arm appears as soon as its run directory exists (the
# newest run of an arm wins if it was rerun). The tree is `tensorboard/` in this experiment's /nrs
# directory, the one place the layout provides for it; smoke runs are linked under
# `tensorboard/smoke/` and served from there, so they never sit beside the production curves.
#
# THE ARMS (README.md, "Arms")
#
#   1a_dinov3_axial_subpixel   released DINOv3 ViT-L/16 inflated to 3D, axial RoPE, SDPA + compile,
#                              affinity_seg sub-pixel head from step 0, ONE stage of 500k steps,
#                              lr 3e-4 (5k warmup, linear to 3e-7), global batch 8, 256^3 at 8 nm
#   1b_scratch_axial_subpixel  arm 1a from random initialisation (no [init]); 1a vs 1b is the
#                              value of the pretrained encoder. Expect a longer trivial-predictor
#                              plateau at the start (boundary_accuracy 0); 1a's ended by step 3k.
#   1c_dinov3_axial_subpixel_1m
#                              arm 1a with max_steps 1M and checkpoint_every 100k; 1a vs 1c is the
#                              value of a longer schedule. Its lr curve decays half as fast: compare
#                              with 1a at equal steps AND at equal remaining-schedule fraction.
#   2a_simmim_lmd_mask60       SSL ONLY: SimMIM pretraining of the same ViT-L/16 from random init on the
#                              census-20260920 lmd corpus (226 volumes), 500k steps at global batch 128,
#                              mask ratio 0.6. Its curves are reconstruction losses, not affinity metrics.
#   2b_simmim_lmd_mask85       arm 2a with mask_ratio 0.85, nothing else. A harder mask has a HIGHER
#                              masked L1 by construction: never rank 2a against 2b on these curves.
#   3a_simmim60_axial_subpixel arm 1a's recipe (affinity_seg sub-pixel head, 500k steps) with the encoder
#                              initialised from arm 2a's final SSL checkpoint instead of the LVD weights.
#                              Read it against 1a (natural-image pretraining) and 1b (none).
#   3b_simmim85_axial_subpixel the same from arm 2b's checkpoint; 3a vs 3b is the SSL mask ratio.
#                              Both start a COLD sub-pixel head: the collapse check below applies.
#
# WHAT TO READ
#
#   train/affinity_accuracy vs train/target_positive_rate
#                         The collapse check. A head that predicts "everything connected" scores
#                         exactly the positive rate; the repo's three earlier cold single-stage
#                         sub-pixel runs sat on it to three decimals with a dip near step 400 and no
#                         recovery. Alive means the two curves separate and stay apart.
#   train/boundary_accuracy
#                         Zero for the trivial predictor. The smoke reached 0.29 within 15 steps; a
#                         flat zero after step 1k is the collapse.
#   train/loss, val/loss  val is 32 crops from the 3000 <= z < 4000 slab every 10k steps and is the
#                         model-selection signal; select checkpoints on val/loss, not on
#                         val/boundary_accuracy (a thresholded proxy that ranks noisily).
#   samples_per_s, mfu    Expected 36-53 samples/s (0.15-0.22 s/step at global batch 8) once
#                         compilation is done; the 96 h wall assumes no worse than ~0.7 s/step.
#   data_wait_frac        Near 0 with 8 workers and deferral on. Spikes every 12,500 steps are the
#                         worker respawn at the 100k-sample epoch boundary, not a loader problem.
#   grad_norm, lr         Clip is 1.0; lr reaches 3e-4 at step 5k and decays linearly to 3e-7.
#
# FOR THE SSL ARMS (2a, 2b) READ INSTEAD
#
#   train/loss            Masked L1 on the augmented crop. Every previous SimMIM run's smoothed masked loss
#                         plateaued by step 10-25k (probes/simmim_saturation on /nrs), so a flat curve from
#                         ~20k on is expected, not a fault. Augmentation is in-plane rotation only, so
#                         train and val sit on one scale -- unlike lmd_ssl_v1 arm 1, whose additive-noise
#                         augmentation entered the target (train 0.15 vs val 0.09).
#   train/visible_l1      Reconstruction error on the patches the encoder could see. Should sit well above
#                         the masked loss early and fall for a long time (0.66 -> 0.28 over 100k in lmd 1a);
#                         if it converges onto train/loss the model has stopped using the mask token.
#   val/loss              128 un-augmented crops from the hemibrain val slab every 10k steps: the clean
#                         progress readout. Smooth over >= 5 points before reading a trend (+-0.01 noise).
#                         Not a selection signal; the fine-tuned score is the only readout that ranks arms.
#   train/masked_fraction 0.5996 (2a) / 0.8496 (2b): the ratio rounded to whole 32-voxel blocks.
#   samples_per_s         >= 170 expected on a compiled B300 node (128 samples per step); below ~53 the
#                         500k steps do not fit the 14-day wall and the run needs a --resume resubmission.
#   data_wait_frac        Loader stalls; with 21.3 MB of stored voxels per sample the node reads 3.6-4.9
#                         GB/s at the expected rate. Spikes every 7,812 steps are the epoch respawn.
#
# WHAT NOT TO READ ACROSS: lmd_ssl_v1's 2b/2c curves. Those trained on four volumes with a 1024^3
# hemibrain box (inside this experiment's train AND val boxes), used superposition RoPE and FA4,
# and ran the sub-pixel head only as a warm-started second stage.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/gary_comparison   # this experiment's home on /nrs: runs/ jobs/ eval/ probes/ (layout of 2026-09-16)
RUNS=$EXP/runs
VIEW=$EXP/tensorboard
VENV=/groups/scicompsoft/home/orhane/myvenv
ARMS=(1a_dinov3_axial_subpixel 1b_scratch_axial_subpixel 1c_dinov3_axial_subpixel_1m
      2a_simmim_lmd_mask60 2b_simmim_lmd_mask85
      3a_simmim60_axial_subpixel 3b_simmim85_axial_subpixel)

PORT=6006 SMOKE=0 LIST=0
for arg in "$@"; do
  case $arg in
    --smoke) SMOKE=1 ;;
    --list) LIST=1 ;;
    [0-9]*) PORT=$arg ;;
    *) echo "unknown argument $arg" >&2; exit 2 ;;
  esac
done

if [[ $SMOKE -eq 1 ]]; then RUNS=$RUNS/smoke; VIEW=$VIEW/smoke; fi   # smoke artifacts live one level down

port_busy () { ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1$"; }
if [[ $LIST -eq 0 ]]; then
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
    ln -sfn "${dir%/}/tensorboard" "$VIEW/$name"; printf '  %-28s -> %s\n' "$name" "$(basename "${dir%/}")"
  else
    missing+=("$name")
  fi
}

echo "arms found:"
for arm in "${ARMS[@]}"; do
  if [[ $SMOKE -eq 1 ]]; then
    link "$arm" "$RUNS/smoke_gary__${arm}_*/"
  else
    link "$arm" "$RUNS/gary__${arm}_*/"
  fi
done
(( ${#missing[@]} )) && echo "  not started: ${missing[*]}"
[[ $LIST -eq 1 ]] && exit 0
echo; echo "http://localhost:$PORT"
exec "$VENV/bin/tensorboard" --logdir "$VIEW" --port "$PORT"
