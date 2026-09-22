#!/usr/bin/env bash
# Pictures of a PERFECT labeller: every window's masks are the ground truth's own objects on the
# mask grid, glued by the labeller's consensus rule -- so what the pictures show is the cost of the
# mask grid and of the gluing alone, to set beside an arm's gallery of the same blocks.
#
#   bash experiments/sam_lmd_v1/figures/oracle_gallery.sh 256          # arms 1-8's window -> viz/oracle_w256/
#   bash experiments/sam_lmd_v1/figures/oracle_gallery.sh 128          # arm 9's window     -> viz/oracle_w128/
#   MIN_SUPPORT=2 bash .../oracle_gallery.sh 256                        # two windows must agree -> viz/oracle_w256_support2/
#
# No model, so CPU only: one `local` job per GT volume, then a CPU job draws gallery_overview.png.
# Output: $STAGE/viz/oracle_w<window>[_support<N>]/{gallery_*.png, <volume>_{base,assembled}.npz, <volume>.json}.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/sam_lmd_v1
STAGE=$EXP
LOGS="$EXP/jobs"
PROJECT=miaai
MIN_SUPPORT=${MIN_SUPPORT:-1}
AGREE=${AGREE:-0.5}
read -r -a GT_VOLUMES <<< "${VOLUMES:-kasthuri15_ac3 zebrafish_fish2_quadcube1 liconn_mouse_dg hemibrain_ellipsoid_body}"

W=${1:?window edge in voxels, 256 or 128}
# The split copy whose patch_size is the window: VolumeGrid derives what it reads per tile from the
# config's patch, so a 128 window on the 256 config silently reads a 2x coarser lattice (found
# 2026-09-22: 27 windows and a 256^3 output on a 512^3 block).
if [[ "$W" == 256 ]]; then GT_CONFIG="experiments/lmd_ssl_v1/lmd_finetune_singlescale.yaml"
else GT_CONFIG="experiments/sam_lmd_v1/data/lmd_finetune_singlescale_crop${W}.yaml"; [[ -f "$GT_CONFIG" ]] || { echo "missing $GT_CONFIG: run make_configs.py" >&2; exit 1; }; fi
SUFFIX=""; [[ "$MIN_SUPPORT" != 1 ]] && SUFFIX="_support$MIN_SUPPORT"; [[ "$AGREE" != 0.5 ]] && SUFFIX+="_agree$AGREE"
OUT="$STAGE/viz/oracle_w${W}${SUFFIX}"
mkdir -p "$OUT" "$LOGS" "$STAGE/cmd"
echo "oracle at window $W, min_support $MIN_SUPPORT, agree $AGREE -> $OUT"

WORKER="$STAGE/cmd/oracle_gallery_w${W}${SUFFIX}.sh"
{ echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8"
  echo "GTS=($(printf '%q ' "${GT_VOLUMES[@]}"))"
  echo "V=\${GTS[\${LSB_JOBINDEX:-1}-1]}"
  printf 'cd %q\n' "$REPO"
  echo "echo \"element \${LSB_JOBINDEX:-1}: oracle volume \$V on \$(hostname)\""
  echo "$VENV/bin/python experiments/sam_lmd_v1/figures/labelling_gallery.py oracle --volume \"\$V\" --out '$OUT' \\"
  echo "  --gt-config '$GT_CONFIG' --block 512 --patch $W --min-mask-voxels 512 --min-truth-voxels 512 \\"
  echo "  --min-support $MIN_SUPPORT --agree-thresh $AGREE"
} > "$WORKER"
FINAL="$STAGE/cmd/oracle_gallery_w${W}${SUFFIX}_final.sh"
{ echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
  printf 'cd %q\n' "$REPO"
  echo "$VENV/bin/python experiments/sam_lmd_v1/figures/labelling_gallery.py render '$OUT'"
} > "$FINAL"

jobid () { sed -n 's/^Job <\([0-9]*\)>.*/\1/p'; }
arr=$(bsub -P "$PROJECT" -q local -n 8 -W 2:00 -J "sam1_oracle_gallery_w${W}${SUFFIX}[1-${#GT_VOLUMES[@]}]" \
      -cwd "$REPO" -o "$LOGS/sam1_oracle_gallery_w${W}${SUFFIX}_%J_%I.log" -e "$LOGS/sam1_oracle_gallery_w${W}${SUFFIX}_%J_%I.err" \
      "bash '$WORKER'" | jobid)
fin=$(bsub -P "$PROJECT" -q local -n 4 -W 0:30 -J "sam1_oracle_gallery_w${W}${SUFFIX}_final" -w "done($arr)" \
      -cwd "$REPO" -o "$LOGS/sam1_oracle_gallery_w${W}${SUFFIX}_final_%J.log" -e "$LOGS/sam1_oracle_gallery_w${W}${SUFFIX}_final_%J.err" \
      "bash '$FINAL'" | jobid)
echo "  array $arr (${#GT_VOLUMES[@]} volumes) -> overview $fin;  figures: $OUT/gallery_*.png"
