#!/usr/bin/env bash
# Serve the four lm_zebrafish_cell_instance arms in one TensorBoard.
#
#   bash experiments/lm_zebrafish_cell_instance/tensorboard.sh [port]
#   bash experiments/lm_zebrafish_cell_instance/tensorboard.sh 6006 --list    # rebuild the tree, print, do not serve
#
# large_inputs_nisb/tensorboard.sh with this experiment's arms. Rebuilds the symlink tree each time, so
# an arm appears as soon as its run directory exists (the newest run of an arm wins if it was rerun).
# The tree is `tensorboard/` in this experiment's /nrs directory and holds the four arms' links and
# nothing else: TensorBoard reads subdirectories too, so anything left there would be served
# alongside them. The smoke runs are never linked.
#
# THE ARMS (README.md): large_inputs_nisb's recipe (DINOv3 ViT-L/16 from the released weights, global
# attention, UNETR head), 100k steps, global batch 8, trained on frames 1 and 25 of the zebrafish store.
#
#   w496_em, w496_noem   496 x 496 x 144 crops (x, y, z voxels), with / without the EM section augmentations
#   w288_em, w288_noem   288 x 288 x 144 crops, likewise
#
# WHAT TO READ
#
#   train/affinity_accuracy vs train/target_positive_rate
#                         The collapse check: a head predicting "everything connected" scores exactly
#                         the positive rate. Alive means the two curves separate and stay apart.
#   train/boundary_accuracy
#                         Zero for the trivial predictor; the one pooled metric that moves.
#   train/loss
#   samples_per_s, data_wait_frac
#                         data_wait_frac near 0, with spikes every 12,500 steps at the epoch boundary
#                         (worker respawn).
#   lr                    3e-4 after the 5k-step warmup, linear to 3e-7 at 100k.
#
# WHAT NOT TO READ
#
#   Any curve as a ranking. There is no validation (no labelled frame is held out), so every curve is
#   measured on augmented training crops of the two frames the model is fitting. The ranking is pq on
#   frame 10 (README.md, "Scoring"), scored only on request.
#   Training loss across the EM switch: the _em arms' crops carry dropped and shifted sections, so
#   their loss is measured on harder inputs by construction.
#   Per-voxel metrics across windows: they differ with the share of voxels near a crop face before any
#   model difference. Compare the windows at equal steps, not equal wall time.
set -euo pipefail

EXP=/nrs/scicompsoft/orhane/mia-train-experiments/lm_zebrafish_cell_instance   # this experiment's home on /nrs
RUNS=$EXP/runs
VIEW=$EXP/tensorboard
VENV=/groups/scicompsoft/home/orhane/myvenv
ARMS=(w496_em w496_noem w288_em w288_noem)

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
    ln -sfn "${dir%/}/tensorboard" "$VIEW/$name"; printf '  %-10s -> %s\n' "$name" "$(basename "${dir%/}")"
  else
    missing+=("$name")
  fi
}

echo "arms found:"
for arm in "${ARMS[@]}"; do
  link "$arm" "$RUNS/lm_zebrafish_cell_instance__${arm}_*/"
done
(( ${#missing[@]} )) && echo "  not started: ${missing[*]}"
[[ $LIST -eq 1 ]] && exit 0
echo; echo "http://localhost:$PORT"
exec "$VENV/bin/tensorboard" --logdir "$VIEW" --port "$PORT"
