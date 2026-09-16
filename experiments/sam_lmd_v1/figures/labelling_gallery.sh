#!/usr/bin/env bash
# Pictures of one teacher's pseudo-labels beside the ground truth, on the four scored blocks.
#
#   bash experiments/sam_lmd_v1/figures/labelling_gallery.sh arm1_4nm            # newest checkpoint
#   STEP=200000 NM=4 bash experiments/sam_lmd_v1/figures/labelling_gallery.sh arm1_4nm
#   NM=8 bash experiments/sam_lmd_v1/figures/labelling_gallery.sh feat64        # an 8 nm arm
#   PRED_IOU=0.9 NM=8 bash experiments/sam_lmd_v1/figures/labelling_gallery.sh arm4_8nm_gb16
#                                                  # a stricter head gate; output dir gets _iou0.9
#   AMG="prefer=part" TAG=part VOLUMES=hemibrain_ellipsoid_body NM=8 bash .../labelling_gallery.sh arm4_8nm_gb16
#                                                  # any generator setting, one volume; dir gets _part
#
# One gpu_b300 array element per GT volume runs `labelling_gallery.py label` (the same block,
# partition and generator settings as `assembly_sweep.sh`'s `consensus` row, so the saved
# labelling IS the one the table scored), then a CPU job draws `gallery_overview.png`.
# Output: $STAGE/viz/<arm>_step<N>/{gallery_*.png, <volume>_{base,assembled}.npz, <volume>.json}.
# The npz files are ~0.3 GB per volume and regenerable by this script; delete them once looked at.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/sam_lmd_v1   # this experiment's home on /nrs: runs/ jobs/ eval/ probes/ (layout of 2026-09-16)
RUNS=$EXP/runs
STAGE=$EXP
LOGS="$EXP/jobs"
PROJECT=miaai
QUEUE=${QUEUE:-gpu_b300}
NM=${NM:-4}
PRED_IOU=${PRED_IOU:-}      # empty = the labeller's gate (0.7)
AMG=${AMG:-}                # extra generator settings, space-separated key=value (e.g. "prefer=part")
TAG=${TAG:-}                # output-directory suffix for such a variant (required with AMG)
[[ -z "$AMG" || -n "$TAG" ]] || { echo "AMG needs TAG=<name> for the output directory" >&2; exit 2; }
read -r -a GT_VOLUMES <<< "${VOLUMES:-kasthuri15_ac3 zebrafish_fish2_quadcube1 liconn_mouse_dg hemibrain_ellipsoid_body}"

ARM=${1:?arm name, e.g. arm1_4nm}
case "$NM" in
  8) GT_CONFIG="experiments/lmd_ssl_v1/lmd_finetune_singlescale.yaml"; BLOCK=512; FLOOR=512 ;;
  4) GT_CONFIG="experiments/sam_lmd_v1/data/lmd_finetune_singlescale_4nm.yaml"; BLOCK=1024; FLOOR=4096 ;;
  *) echo "NM must be 8 or 4" >&2; exit 2 ;;
esac

run=$(ls -dt "$RUNS"/sam1__${ARM}_r0_*/ 2>/dev/null | head -1) || true
[[ -n "${run:-}" ]] || { echo "no run matching sam1__${ARM}_r0_*" >&2; exit 1; }
run=${run%/}
NEWEST=$(ls -d "$run"/checkpoints/step_* | sed 's|.*step_||' | sort -n | tail -1)
STEP=${STEP:-$NEWEST}
[[ -d "$run/checkpoints/step_$STEP" ]] || { echo "no checkpoints/step_$STEP in $run" >&2; exit 1; }
SUFFIX="${PRED_IOU:+_iou$PRED_IOU}${TAG:+_$TAG}"
OUT="$STAGE/viz/${ARM}_step${STEP}${SUFFIX}"
GATE_FLAG=${PRED_IOU:+--pred-iou-thresh $PRED_IOU}
for kv in $AMG; do GATE_FLAG+=" --amg $kv"; done
mkdir -p "$OUT" "$LOGS" "$STAGE/cmd"
echo "teacher $run step $STEP -> $OUT"

WORKER="$STAGE/cmd/gallery_${ARM}_step${STEP}${SUFFIX}.sh"
{ echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "GTS=($(printf '%q ' "${GT_VOLUMES[@]}"))"
  echo "V=\${GTS[\${LSB_JOBINDEX:-1}-1]}"
  printf 'cd %q\n' "$REPO"
  echo "echo \"element \${LSB_JOBINDEX:-1}: volume \$V on \$(hostname)\""
  echo "nvidia-smi --query-gpu=name --format=csv,noheader | head -1"
  echo "$VENV/bin/python experiments/sam_lmd_v1/figures/labelling_gallery.py label '$run' --step $STEP \\"
  echo "  --volume \"\$V\" --out '$OUT' --gt-config '$GT_CONFIG' --block $BLOCK \\"
  echo "  --min-mask-voxels $FLOOR --min-truth-voxels $FLOOR $GATE_FLAG"
} > "$WORKER"
FINAL="$STAGE/cmd/gallery_${ARM}_step${STEP}${SUFFIX}_final.sh"
{ echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
  printf 'cd %q\n' "$REPO"
  echo "$VENV/bin/python experiments/sam_lmd_v1/figures/labelling_gallery.py render '$OUT'"
} > "$FINAL"

jobid () { sed -n 's/^Job <\([0-9]*\)>.*/\1/p'; }
arr=$(bsub -P "$PROJECT" -q "$QUEUE" -gpu "num=1" -n 12 -W 4:00 -J "sam1_gallery_${ARM}_${STEP}${SUFFIX}[1-${#GT_VOLUMES[@]}]" \
      -cwd "$REPO" -o "$LOGS/sam1_gallery_${ARM}_%J_%I.log" -e "$LOGS/sam1_gallery_${ARM}_%J_%I.err" \
      "bash '$WORKER'" | jobid)
fin=$(bsub -P "$PROJECT" -q local -n 4 -W 0:30 -J "sam1_gallery_${ARM}_${STEP}${SUFFIX}_final" -w "done($arr)" \
      -cwd "$REPO" -o "$LOGS/sam1_gallery_${ARM}_final_%J.log" -e "$LOGS/sam1_gallery_${ARM}_final_%J.err" \
      "bash '$FINAL'" | jobid)
echo "  array $arr (4 volumes) -> overview $fin;  figures: $OUT/gallery_*.png"
