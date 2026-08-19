#!/usr/bin/env bash
# Serve both arms of the pseudo-labelling bootstrap in one TensorBoard.
#
#   bash experiments/pseudo_labeling/tensorboard.sh [port]
#   bash experiments/pseudo_labeling/tensorboard.sh 6008 --smoke   # the 20-step check instead
#
# Rebuilds the symlink tree each run, so round 2 of each arm appears as soon as it exists -- both
# round-2 stages wait on a *labelling* job, not just a checkpoint, so they show up later than the
# usual second stage would.
#
# What to read:
#
#   boundary_accuracy   The panel. Pooled `affinity_accuracy` sits near the target's positive rate
#                       whatever the model does, so it looks healthy for a model that has learned
#                       nothing; `boundary_accuracy` is the one that moves.
#
#   masked_fraction     **New here, and the number this experiment turns on.** It is the share of
#                       voxel/offset pairs excluded from the loss, and on pseudo-labelled data that
#                       is mostly abstention rather than border. Round 1's labels abstain on ~29.5%
#                       of voxels, so expect this well above what a `base`-only run shows. If it
#                       *climbs* between rounds, the teacher is getting less certain, not more --
#                       which is the failure mode iterating invites. Cross-check against the round's
#                       oracle JSON before reading a rising val number as progress.
#
#   target_positive_rate  Round 1's pseudo-labels are more fragmented than ground truth (a median
#                       of 62 true objects rendered as 305 pseudo-objects), so this sits *below* a
#                       GT-trained run's ~83%. Watch it across rounds: falling further means
#                       fragmentation is compounding, which is the specific risk of bootstrapping
#                       from a teacher whose bias is to split when unsure.
#
#   val/*               Measured on 32 crops of the held-out val cube with REAL labels, identical
#                       for every arm and round -- the one number here that pseudo-labelling cannot
#                       contaminate. Still a progress readout, not a model selector: in
#                       subpixel_decoder the lowest val loss did not pick the best checkpoint
#                       (85k won on loss, 100k on nERL). Arms are compared by scoring checkpoints
#                       with banis.
#
#   lr                  Each round is its own 50k cosine from 3e-4, not a continuation, so a
#                       sawtooth between rounds is expected in both arms.
#
# What NOT to read across the arms: nothing is parameter-count-dependent here (unlike
# lora_vs_fullft, both arms train all 306.6M), so `grad_norm` *is* comparable. What differs is the
# starting point -- arm 1 resets to the interpolation-trained encoder each round, arm 2 warm-starts
# from its own previous student -- so arm 2 should start each round at a lower loss and that is the
# arm's definition, not evidence it is better. Compare the arms at the *end* of a round.
set -euo pipefail

PORT=${1:-6006}
MODE=${2:-}
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
SCRATCH=/nrs/scicompsoft/orhane/mia-train-scratch/pseudo_labeling
VIEW=$RUNS/tb_pseudo_labeling
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

STAGES=(1a_reset_r1 1b_reset_r2 2a_warm_r1 2b_warm_r2)

if [[ "$MODE" == "--smoke" ]]; then
  # `submit.sh --smoke` writes under its own output root, so the run dirs are elsewhere and carry
  # a `smoke_` prefix.
  echo "smoke stages found:"
  for s in "${STAGES[@]}"; do link "$s" "$SCRATCH/smoke/smoke_${s}_*/"; done
else
  # Ordered so the legend pairs the arms round by round: 1a beside 2a, 1b beside 2b.
  echo "stages found:"
  link 1a_reset_r1 "$RUNS/pl__1a_reset_r1_*/"
  link 2a_warm_r1  "$RUNS/pl__2a_warm_r1_*/"
  link 1b_reset_r2 "$RUNS/pl__1b_reset_r2_*/"
  link 2b_warm_r2  "$RUNS/pl__2b_warm_r2_*/"
fi

# The labels each round trained on, so a curve can be read against the data that produced it
# rather than in isolation. Read-only oracle numbers -- they select nothing.
echo
for tag in r1 r2_reset r2_warm; do
  f="$SCRATCH/configs/oracle_$tag.json"
  [[ -f "$f" ]] || continue
  "$VENV/bin/python" -c "
import json; d = json.load(open('$f'))
print('  pseudo_%-9s precision %.4f  instance %.3f  abstain %.3f  merges %d  splits %d'
      % ('$tag', d['pair_precision'], d['frac_instance'], d['frac_ignore'],
         d['merged_pseudo'], d['split_gt']))"
done

echo; echo "http://localhost:$PORT"
exec "$VENV/bin/tensorboard" --logdir "$VIEW" --port "$PORT"
