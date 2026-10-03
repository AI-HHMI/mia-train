#!/usr/bin/env bash
# Serve the six large_inputs_nisb arms in one TensorBoard.
#
#   bash experiments/large_inputs_nisb/tensorboard.sh [port]
#   bash experiments/large_inputs_nisb/tensorboard.sh 6006 --list    # rebuild the tree, print, do not serve
#
# gary_comparison/tensorboard.sh with this experiment's arms. Rebuilds the symlink tree each time, so
# an arm appears as soon as its run directory exists (the newest run of an arm wins if it was rerun).
# The tree is `tensorboard/` in this experiment's /nrs directory and holds the six arms' links and
# nothing else: TensorBoard reads subdirectories too, so anything left there would be served
# alongside them. The smoke runs are never linked.
#
# THE ARMS (README.md): gary_comparison 5a (DINOv3 ViT-L/16, global attention, UNETR head), 500k
# steps, global batch 8, on 1, 3 or 5 NISB base cubes (seed0; seed0-2; seed0-4) at two windows.
#
#   w256_{1,3,5}cube   256^3 crops, ~0.21 s/step (500k steps in ~29 h)
#   w512_{1,3,5}cube   512^3 crops, ~1.24 s/step (500k steps in ~7.2 days)
#
# WHAT TO READ
#
#   train/affinity_accuracy vs train/target_positive_rate
#                         The collapse check: a head predicting "everything connected" scores exactly
#                         the positive rate. Alive means the two curves separate and stay apart.
#   train/boundary_accuracy, val/boundary_accuracy
#                         Zero for the trivial predictor; the one pooled metric that moves. Within a
#                         window the three arms see the same val cube, so their train/val gap is the
#                         overfitting readout: 1cube's train curve running above 5cube's while its val
#                         curve does not is one cube being memorised.
#   train/loss, val/loss  val is 32 crops of seed100 every 10k steps, at the arm's own window.
#   samples_per_s         ~38 (256^3) and ~6.5 (512^3) once compiled; data_wait_frac near 0, with
#                         spikes every 12,500 steps at the epoch boundary (worker respawn).
#   lr                    3e-4 after the 5k-step warmup, linear to 3e-7 at 500k.
#
# WHAT NOT TO READ ACROSS
#
#   The two windows' val numbers. They are measured on different crops: 512^3 val crops hold more
#   interior per border voxel (masked_fraction 0.989 against 0.979), so a 512^3 arm's per-voxel
#   metrics differ from a 256^3 arm's before any model difference. Compare arms within a window.
#   Compare the windows at equal steps, not equal wall time. No val curve ranks the arms: that is
#   nERL on the test cube (README.md, "Reading the result"), scored only on request.
set -euo pipefail

EXP=/nrs/scicompsoft/orhane/mia-train-experiments/large_inputs_nisb   # this experiment's home on /nrs
RUNS=$EXP/runs
VIEW=$EXP/tensorboard
VENV=/groups/scicompsoft/home/orhane/myvenv
ARMS=(w256_1cube w256_3cube w256_5cube w512_1cube w512_3cube w512_5cube)

PORT=6006 LIST=0
for arg in "$@"; do
  case $arg in
    --list) LIST=1 ;;
    [0-9]*) PORT=$arg ;;
    *) echo "unknown argument $arg" >&2; exit 2 ;;
  esac
done

port_busy () { ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1$"; }
if [[ $LIST -eq 0 ]]; then
  while port_busy "$PORT"; do echo "port $PORT is in use, trying $((PORT + 1))"; PORT=$((PORT + 1)); done
fi

# Replace the links in place rather than removing the directory: a TensorBoard already serving
# it holds files open, and on NFS that leaves the directory undeletable. Every link at any depth
# goes, then any subdirectory left empty, so nothing but the arms below is ever served.
mkdir -p "$VIEW"
find "$VIEW" -mindepth 1 -type l -delete
find "$VIEW" -mindepth 1 -type d -empty -delete
missing=()
link () {
  local name=$1 dir
  dir=$(ls -dt $2 2>/dev/null | head -1 || true)
  if [[ -n "$dir" && -d "${dir%/}/tensorboard" ]]; then
    ln -sfn "${dir%/}/tensorboard" "$VIEW/$name"; printf '  %-14s -> %s\n' "$name" "$(basename "${dir%/}")"
  else
    missing+=("$name")
  fi
}

echo "arms found:"
for arm in "${ARMS[@]}"; do
  link "$arm" "$RUNS/large_inputs_nisb__${arm}_*/"
done
(( ${#missing[@]} )) && echo "  not started: ${missing[*]}"
[[ $LIST -eq 1 ]] && exit 0
echo; echo "http://localhost:$PORT"
exec "$VENV/bin/tensorboard" --logdir "$VIEW" --port "$PORT"
