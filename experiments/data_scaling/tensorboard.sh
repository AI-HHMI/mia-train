#!/usr/bin/env bash
# Serve all six stages of the data-scaling experiment in one TensorBoard.
#
#   bash experiments/data_scaling/tensorboard.sh [port]
#   bash experiments/data_scaling/tensorboard.sh 6008 --smoke   # the 20-step check instead
#
# Rebuilds the symlink tree each run, so a stage appears as soon as its run directory exists. Stage
# B of each arm shows up ~23 h after stage A starts, because it is held on `done(A)`.
#
# The run names are ordered so the legend groups the arms *within* a stage: 1cube, 3cube, 5cube
# interp together, then the three subpixel. That is the comparison this experiment is for -- three
# curves that should be read against each other at the same step, not a single run's trajectory.
#
# What to read:
#
#   boundary_accuracy   The panel. Pooled `affinity_accuracy` sits near the target's positive rate
#                       whatever the model does, so it looks healthy for a model that has learned
#                       nothing; `boundary_accuracy` is the one that moves.
#
#   train vs val gap    **The number this experiment turns on.** All three arms see the same
#                       held-out cube, and the training sets differ only in size, so the arms'
#                       *train* curves separating while their *val* curves do not is exactly what
#                       one cube overfitting looks like. Watch the gap, not either curve alone. If
#                       1cube's train accuracy runs above 5cube's while its val sits at or below,
#                       the extra cubes are buying generalisation; if all six curves lie on top of
#                       each other, NISB saturated at one cube and that is the result.
#
#   masked_fraction     Should be near-identical across the arms and flat: these are REAL labels, so
#                       the mask is border slabs only, with no abstention. Divergence between arms
#                       means the cubes differ in label density, which would be a confound worth
#                       knowing about before reading anything else. (Contrast pseudo_labeling, where
#                       this number carried the abstention rate and was the point.)
#
#   target_positive_rate  Ditto -- ~83% on GT labels. Arms differing here means the cubes differ,
#                       not the models.
#
#   grad_norm           Comparable across arms: all six stages train the same 306.6M parameters.
#                       A 1-cube run spiking here late is the memorisation signature.
#
#   lr                  One linear ramp 3e-4 -> 3e-7 per stage, so the sawtooth at the A->B boundary
#                       is expected: stage B is its own schedule, not a continuation.
#
# What NOT to read: do not rank the arms on `val/boundary_accuracy`. It is per-voxel and this task
# fails at the instance level -- scoring pseudo_labeling's checkpoints showed nERL swinging
# 0.5844 -> 0.3889 -> 0.5518 while val boundary accuracy sat flat at 0.925-0.945 the whole way. Use
# it to see that a run is training and to watch the train/val gap; rank with `score.sh`.
#
# Also: stage A and stage B are different architectures (interpolating vs sub-pixel head), so a
# jump at the boundary is the head changing, not the arm improving.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=${1:-6006}
MODE=${2:-}
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
SCRATCH=/nrs/scicompsoft/orhane/mia-train-scratch/data_scaling
VIEW=$RUNS/tb_data_scaling
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

echo "stages found:"
if [[ "$MODE" == "--smoke" ]]; then
  # `submit.sh --smoke` writes under its own output root, with a `smoke_ds_` prefix.
  for n in 1 3 5; do
    link "${n}cube_A_interp"   "$SCRATCH/smoke/smoke_ds_${n}a_interp_*/"
    link "${n}cube_B_subpixel" "$SCRATCH/smoke/smoke_ds_${n}b_subpixel_*/"
  done
else
  # Grouped by stage, ascending in cubes, so the legend reads as the data-scaling curve.
  for n in 1 3 5; do link "A_interp_${n}cube"   "$RUNS/ds__${n}cube_interp_*/";   done
  for n in 1 3 5; do link "B_subpixel_${n}cube" "$RUNS/ds__${n}cube_subpixel_*/"; done
fi

# Which cubes each arm is actually training on, so a curve is read against its data rather than
# against its name.
echo; echo "training sets:"
for n in 1 3 5; do
  printf "  %scube: %s\n" "$n" \
    "$(grep -oE 'seed[0-9]+' "$HERE/nisb_base_${n}cube_256.yaml" | sort -u | tr '\n' ' ')"
done
echo "  val (all arms): seed100, real labels, 32 crops"

# Anything already scored. Read-only: these select nothing, `score.sh` writes them.
if compgen -G "$SCRATCH/eval/*_scores.json" > /dev/null; then
  echo; echo "nERL scored so far (whole seed100 cube):"
  for f in "$SCRATCH"/eval/*_scores.json; do
    "$VENV/bin/python" -c "
import json,sys; d=json.load(open('$f')); b=d.get('best_by_nerl') or {}
print('  %-30s step %-7s nerl %s' % ('$(basename "$f" _scores.json)', d.get('step'), b.get('nerl')))" 2>/dev/null || true
  done
fi

echo; echo "http://localhost:$PORT"
exec "$VENV/bin/tensorboard" --logdir "$VIEW" --port "$PORT"
