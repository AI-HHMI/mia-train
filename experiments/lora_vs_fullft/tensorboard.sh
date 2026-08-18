#!/usr/bin/env bash
# Serve both arms of this experiment in one TensorBoard.
#
#   bash experiments/lora_vs_fullft/tensorboard.sh [port]
#   bash experiments/lora_vs_fullft/tensorboard.sh 6008 --smoke   # the 20-step check instead
#
# Rebuilds the symlink tree each run, so the second stage of each arm -- 1b waits on 1a, 2b on 2a --
# appears as soon as it exists.
#
# What to read:
#
#   boundary_accuracy   The panel. Pooled `affinity_accuracy` sits near the target's ~83% positive
#                       rate whatever the model does, so it looks healthy for a model that has
#                       learned nothing; `boundary_accuracy` is the one that moves.
#
#   val/*               A progress readout on 32 crops of the held-out cube, not a model selector.
#                       In subpixel_decoder the lowest val loss did not pick the best checkpoint
#                       (85k won on loss, 100k on nERL). The arms are compared by scoring
#                       checkpoints with banis -- see score_checkpoint.sh.
#
#   lr                  Worth one look at the stage boundary. Stage b restarts at 1.545e-4, which is
#                       where a single 100k schedule would have been at step 50000, and anneals to
#                       the same 3e-6 endpoint. A visible step down from 3e-6 to 1.545e-4 between
#                       stages is the intended handoff, not a bug -- see the note in 1b's [trainer].
#
#   mfu, samples_per_s  Both arms should sit within a couple of percent of each other. LoRA saves
#                       optimizer state and weight gradients, *not* compute -- backward still
#                       traverses the whole stack to reach the adapters -- so a large gap here means
#                       something other than the method differs, most likely the input pipeline.
#                       `data_wait_frac_max` separates the two.
#
# What NOT to read across the arms: `grad_norm`. Arm 1 clips over 10.6M trainable parameters and arm
# 2 over 306.6M, so the global norm is a different quantity in each and `grad_clip_norm = 1.0` fires
# at very different frequencies. Compare each arm's against itself over time, never one to the other.
set -euo pipefail

PORT=${1:-6006}
MODE=${2:-}
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
SCRATCH=/nrs/scicompsoft/orhane/mia-train-scratch/lora_vs_fullft
VIEW=$RUNS/tb_lora_vs_fullft
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
    ln -sfn "${dir%/}/tensorboard" "$VIEW/$name"; echo "  $name -> ${dir%/}"
  else
    echo "  $name -- not started yet"
  fi
}

STAGES=(1a_lora_interp 1b_lora_subpixel 2a_full_interp 2b_full_subpixel)

if [[ "$MODE" == "--smoke" ]]; then
  # `submit.sh --smoke` writes under its own output root, so the run dirs are elsewhere and carry a
  # `smoke_` prefix.
  echo "smoke stages found:"
  for s in "${STAGES[@]}"; do link "$s" "$SCRATCH/smoke/smoke_${s}_*/"; done
else
  # Ordered so the legend pairs the arms stage by stage: 1a beside 2a, 1b beside 2b.
  echo "stages found:"
  link 1a_lora_interp   "$RUNS/lvf__1a_lora_interp_*/"
  link 2a_full_interp   "$RUNS/lvf__2a_full_interp_*/"
  link 1b_lora_subpixel "$RUNS/lvf__1b_lora_subpixel_*/"
  link 2b_full_subpixel "$RUNS/lvf__2b_full_subpixel_*/"
fi

echo; echo "http://localhost:$PORT"
exec "$VENV/bin/tensorboard" --logdir "$VIEW" --port "$PORT"
