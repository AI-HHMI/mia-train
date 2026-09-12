#!/usr/bin/env bash
# Serve every sam_lmd_v1 stage that exists in one TensorBoard, beside the 150k-step reference run.
#
#   bash experiments/sam_lmd_v1/tensorboard.sh [port]
#   bash experiments/sam_lmd_v1/tensorboard.sh 6008 --smoke     # the 20-step checks instead
#   bash experiments/sam_lmd_v1/tensorboard.sh 6006 --list      # rebuild the tree, print, do not serve
#
# Rebuilds the symlink tree each time, so a stage appears as soon as its run directory exists.
# Runs are named `r<round>_<arm>` so the legend sorts by ROUND first. Version 3 (2026-09-12) runs
# ONE stage, `r0_feat64`, for 200k steps; the other 23 stages are listed as absent until the model
# earns them (README, "Status"). `ref_promptable_lmd_v2` is the finished 150k-step run of the base
# head on five native-resolution volumes at LR 1e-4 -- a different corpus and schedule, so a
# reference curve for the metric names and plausible values, not a comparison.
#
# WHAT TO READ IN VERSION 3
#
#   first_voxel_iou     THE curve. One interior click, no correction, scored at voxel resolution on
#                       the 32 held-out crops. Versions 1 and 2 ended at 0.44-0.45 (v1 recipe) and
#                       0.43 (v2 recipe, harder prompt mix) after 100k steps with the curve still
#                       rising as the linear schedule hit its floor; 200k steps exist to find out
#                       how much further it goes. Comparable across arms and versions ONLY for the
#                       v1 recipe (v2 mixed in boundary clicks). The reference run ends near 0.50.
#
#   train/first_voxel_iou vs val/first_voxel_iou
#                       Identical in versions 1 and 2 (0.43 both): the model had not fit even the
#                       four training volumes. A gap opening in version 3 is the first sign of
#                       fitting them, and the moment the data engine's extra volumes start to matter.
#
#   WHAT TENSORBOARD CANNOT SHOW  The number that decides the restart is single-window RECALL: how
#                       many of the objects in a 256^3 window get a mask at IoU >= 0.5 (0.13 at 100k
#                       against a perfect model's 0.95). It comes from the probe and the sweep, not
#                       from validation IoU, which averages over prompts that were placed on objects
#                       for it. At each 12.5k checkpoint of interest (50k, 100k, 150k, 200k):
#                           bash experiments/sam_lmd_v1/calibration_probe.sh feat64
#                           SETTINGS_ONLY="single_tile consensus" bash experiments/sam_lmd_v1/assembly_sweep.sh feat64
#                       (both take the arm's newest checkpoint; pass --step to pseudolabel.py for
#                       an older one).
#
#   final_voxel_iou     After two correction clicks. Its gap over `first_voxel_iou` (~0.10-0.13 so
#                       far) is what interaction buys; the labeller never uses it (one click per
#                       prompt), so it is a health signal for the training loop, not a result.
#
#   first_iou_error     |predicted IoU - achieved IoU| of the IoU head, round 0. Flat at 0.12-0.13
#                       through versions 1 and 2, i.e. it did not improve with training. The
#                       probe showed the head IS well ranked on grid prompts (precision 0.93 in the
#                       0.7-0.8 bin), so read this as a calibration width, not a defect. Off-object
#                       metrics (`offobject_*`) are absent in version 3: the v2 recipe is off.
#
#   valid_fraction      Share of prompt slots holding an object. ~1.0 on ground truth with the v1
#                       recipe (v2's off-object slots pulled it to 0.67-0.77). In a data-engine
#                       round it also measures the pseudo-labels: crops with nothing to prompt for
#                       are excluded from the loss, so a lower value is wasted steps, not a worse
#                       loss.
#
#   loss, loss_round_*  Focal + dice + IoU per interactive round; round 2 < round 1 < round 0 is
#                       the healthy shape. Not comparable between recipes (v2 averaged over harder
#                       prompts) or between round 0 and a pseudo-label round (softer, sparser
#                       targets, and since version 3 unlabelled voxels weigh nothing).
#
#   lr                  One linear ramp: 3k warmup to 3e-4, then down to 3e-7 at 200k.
#
#   samples_per_s, mfu  feat64 ran at 0.40 s/step (20 samples/s, 8 x B300) in version 1; the same
#                       is expected here. The v2 recipe's extra worker CPU cost ~0.13 s/step.
#
#   data_wait_frac      Should sit near 0.001. Spikes at the epoch period -- samples_per_epoch /
#                       global batch = 100,000 / 8 = 12,500 steps -- are worker respawns (~4 s).
#                       A rank stuck near 1 is the failure that stalled version 2's `base` run at
#                       step 70k (8 s/step); if it recurs, kill and `--resume`.
#
# WHAT NOT TO READ ACROSS: `first_iou` between stride arms (different grids); v2 curves against v1
# or v3 (different prompt mix); anything against `ref_promptable_lmd_v2` as a result.
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
if [[ "$MODE" != "--list" ]]; then
  while port_busy "$PORT"; do echo "port $PORT is in use, trying $((PORT + 1))"; PORT=$((PORT + 1)); done
fi

rm -rf "$VIEW"; mkdir -p "$VIEW"
missing=()
link () {
  local name=$1 dir
  dir=$(ls -dt $2 2>/dev/null | head -1 || true)
  if [[ -n "$dir" && -d "${dir%/}/tensorboard" ]]; then
    ln -sfn "${dir%/}/tensorboard" "$VIEW/$name"; printf '  %-24s -> %s\n' "$name" "$(basename "${dir%/}")"
  else
    missing+=("$name")
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
(( ${#missing[@]} )) && echo "  not run (${#missing[@]}): ${missing[*]}"

echo
echo "version 3 = r0_feat64 only: 200k steps, v1 prompting recipe, loss ignores unlabelled voxels,"
echo "checkpoints every 12.5k; the other arms and the data-engine rounds wait on its single-window recall."
echo "arms (one knob each, otherwise identical; see make_configs.py::ARMS):"
echo "  base           stride 4 (32 nm masks), 32 features, 2 decoder layers, dim 256"
echo "  stride2        masks at stride 2 (16 nm)        stride1        masks at voxel resolution"
echo "  stride2_small  stride 2 + objects >= 64 voxels  feat64         64 mask features   <- VERSION 3"
echo "  refine4        4 refinement convs at mask res   deep4          4-layer two-way decoder"
echo "  wide512        512-wide neck and decoder"
echo
echo "rounds: r0 = GT only from the LVD checkpoint (200k)   r1 = +1 block/volume (50k)   r2 = +4 blocks/volume (50k)"
[[ "$MODE" == "--list" ]] && exit 0
echo; echo "http://localhost:$PORT"
exec "$VENV/bin/tensorboard" --logdir "$VIEW" --port "$PORT"
