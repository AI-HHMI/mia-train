#!/usr/bin/env bash
# Serve every round of every sam_lmd_v1 arm in one TensorBoard, beside the 150k-step reference run.
#
#   bash experiments/sam_lmd_v1/tensorboard.sh [port]
#   bash experiments/sam_lmd_v1/tensorboard.sh 6008 --smoke     # the 20-step checks instead
#
# Rebuilds the symlink tree each time, so a stage appears as soon as its run directory exists.
# Runs are named `r<round>_<arm>` so the legend sorts by ROUND first: every arm's round 0 sits
# together (the architecture sweep on ground truth alone), then every round 1 (the first data-engine
# round), then round 2. `ref_promptable_lmd_v2` is the finished 150k-step run of the base head on
# five native-resolution volumes at LR 1e-4 -- a different corpus and schedule, so a reference
# curve for the metric names and plausible values, not a comparison.
#
# WHAT TO READ
#
#   first_voxel_iou     THE number for this experiment. Segment-everything can only issue a single
#                       click per prompt, so whole-volume quality follows round 0's mask; the
#                       interactive rounds' gains (`final_*`) never reach the eval. Scored at voxel
#                       resolution, so it is comparable ACROSS the stride arms -- `first_iou` is
#                       measured on each arm's own mask grid and is not (a finer grid is a harder
#                       target). The reference run ends near 0.50 (val, LR 1e-4, five volumes).
#
#   final_voxel_iou     After two correction clicks. Its gap over `first_voxel_iou` (~0.06-0.10) is
#                       what interaction buys; a gap that vanishes means the corrections stopped
#                       teaching anything.
#
#   first_iou_error     |predicted IoU - achieved IoU| of the IoU head, round 0. The mask generator
#                       gates candidates on this head's prediction (0.5 at eval, 0.7 when
#                       labelling), so its calibration decides what a pseudo-label round trains on.
#                       Watch it in rounds 1-2 especially: a head that grows over-confident on its
#                       own pseudo-labels is the confirmation-bias failure of self-training.
#
#   valid_fraction      Share of crops with at least one eligible object. ~1.0 on ground truth.
#                       In rounds 1-2 it also measures the pseudo-labels: a block whose teacher kept
#                       few masks yields crops with nothing to prompt for, which are excluded from
#                       the loss rather than counted -- so a lower value is wasted steps, not a
#                       worse loss. Compare it with the `mean claimed fraction` in the round's data
#                       config header.
#
#   loss, loss_round_*  Focal+dice+IoU per interactive round; round 2 < round 1 < round 0 is the
#                       healthy shape. Round 1-2 values are NOT comparable to round 0's: a
#                       pseudo-label is a softer, sparser target than ground truth.
#
#   val/*               32 crops from the four held-out volumes, identical for every arm and round,
#                       and the same set arms 1/2 of lmd_ssl_v1 validated on. NOT a model selector:
#                       every round is scored at its final step, as arms 1/2 were.
#
#   lr                  A fresh linear ramp per round, so a sawtooth at every round boundary: 3e-4
#                       peak for round 0, 1e-4 for rounds 1-2 (lmd_ssl_v1's stage-B and stage-C
#                       peaks, for the same reason -- a warm start continues a converged model).
#
#   samples_per_s, mfu  The stride arms are the ones to look at: `stride2` multiplies the decoder's
#                       output tensors by 8 and `stride1` by 64, and their step cost was never
#                       measured before launch -- submit.sh's walls were scaled 1.5x / 2x by guess.
#                       Read the real number here and tighten them.
#
#   data_wait_frac      Should sit near 0. Periodic spikes at the epoch period
#                       (samples_per_epoch / global batch = 1250 steps) are worker respawns.
#
# WHAT NOT TO READ ACROSS: `first_iou` between stride arms (different grids); round-1/2 losses
# against round 0 (different targets); anything against `ref_promptable_lmd_v2` as a result.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=${1:-6006}
MODE=${2:-}
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
SMOKE=/nrs/scicompsoft/orhane/mia-train-scratch/sam_lmd_v1/smoke
VIEW=$RUNS/tb_sam_lmd_v1
VENV=/groups/scicompsoft/home/orhane/myvenv
ARMS=(base stride2 stride1 stride2_small feat64 refine4 deep4 wide512)
REF=$RUNS/promptable_lmd_v2_20260910_104156

port_busy () { ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1$"; }
while port_busy "$PORT"; do echo "port $PORT is in use, trying $((PORT + 1))"; PORT=$((PORT + 1)); done

rm -rf "$VIEW"; mkdir -p "$VIEW"
link () {
  local name=$1 dir
  dir=$(ls -dt $2 2>/dev/null | head -1 || true)
  if [[ -n "$dir" && -d "${dir%/}/tensorboard" ]]; then
    ln -sfn "${dir%/}/tensorboard" "$VIEW/$name"; printf '  %-24s -> %s\n' "$name" "$(basename "${dir%/}")"
  else
    printf '  %-24s -- not started yet\n' "$name"
  fi
}

echo "stages found:"
for round in 0 1 2; do
  for arm in "${ARMS[@]}"; do
    if [[ "$MODE" == "--smoke" ]]; then
      link "r${round}_${arm}" "$SMOKE/smoke_sam1__${arm}_r${round}_*/"
    else
      link "r${round}_${arm}" "$RUNS/sam1__${arm}_r${round}_*/"
    fi
  done
done
[[ "$MODE" == "--smoke" ]] || link "ref_promptable_lmd_v2" "$REF/"

echo
echo "arms (one knob each, otherwise identical; see make_configs.py::ARMS):"
echo "  base           stride 4 (32 nm masks), 32 features, 2 decoder layers, dim 256   <- S2 at r0, S3 at r1-2"
echo "  stride2        masks at stride 2 (16 nm)        stride1        masks at voxel resolution"
echo "  stride2_small  stride 2 + objects >= 64 voxels  feat64         64 mask features"
echo "  refine4        4 refinement convs at mask res   deep4          4-layer two-way decoder"
echo "  wide512        512-wide neck and decoder"
echo
echo "rounds: r0 = GT only from the LVD checkpoint (100k)   r1 = +1 block/volume (50k)   r2 = +4 blocks/volume (50k)"
echo; echo "http://localhost:$PORT"
exec "$VENV/bin/tensorboard" --logdir "$VIEW" --port "$PORT"
