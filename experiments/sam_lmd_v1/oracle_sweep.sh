#!/usr/bin/env bash
# Would a PERFECT model's windows assemble into a precise labelling? The control for the sweep.
#
#   bash experiments/sam_lmd_v1/oracle_sweep.sh                 # every setting
#   SETTINGS_ONLY="oracle_consensus oracle_r50_consensus" bash experiments/sam_lmd_v1/oracle_sweep.sh
#   bash experiments/sam_lmd_v1/oracle_sweep.sh --dry-run
#
# `pseudolabel.py oracle` replaces the model with the ground truth inside every window: one perfect
# mask per object component of at least 512 voxels, at the mask stride, and pushes those through
# the same tile-assembly rules `assembly_sweep.sh` measured on the real model, on the SAME 512^3
# blocks (and the same single-tile 288^3 reference). `_r50` settings keep each window's masks with
# probability 0.5, independently per window: perfect masks, but an object found in one window may
# be missed in the next, which is what the real model does about half the time. `propagate` needs
# the model and is not offered.
#
# CPU: no model runs. Results: $STAGE/assembly_sweep/oracle/<setting>/..., table.txt beside them.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
STAGE=/nrs/scicompsoft/orhane/mia-train-scratch/sam_lmd_v1
LOGS="$RUNS/jobs"
PROJECT=miaai
GT_VOLUMES=(kasthuri15_ac3 zebrafish_fish2_quadcube1 liconn_mouse_dg hemibrain_ellipsoid_body)

DRY=0; [[ "${1:-}" == --dry-run ]] && DRY=1

declare -A FLAGS=(
  [oracle_single_tile]="--tile-merge none --block 288 --blocks 8"
  [oracle_canvas]="--tile-merge canvas --block 512 --blocks 1"
  [oracle_none]="--tile-merge none --block 512 --blocks 1"
  [oracle_edge_canvas]="--edge-discard 1 --tile-merge canvas --block 512 --blocks 1"
  [oracle_consensus]="--tile-merge consensus --agree-thresh 0.5 --block 512 --blocks 1"
  [oracle_canvas_s64]="--tile-merge canvas --block 512 --blocks 1 --tile-step 64"
  [oracle_consensus_s64]="--tile-merge consensus --agree-thresh 0.5 --block 512 --blocks 1 --tile-step 64"
  [oracle_r50_canvas]="--recall 0.5 --tile-merge canvas --block 512 --blocks 1"
  [oracle_r50_consensus]="--recall 0.5 --tile-merge consensus --agree-thresh 0.5 --block 512 --blocks 1"
  [oracle_r50_consensus_s64]="--recall 0.5 --tile-merge consensus --agree-thresh 0.5 --block 512 --blocks 1 --tile-step 64"
  [oracle_r50_single_tile]="--recall 0.5 --tile-merge none --block 288 --blocks 8"
)
ORDER=(oracle_single_tile oracle_canvas oracle_none oracle_edge_canvas oracle_consensus oracle_canvas_s64
       oracle_consensus_s64 oracle_r50_single_tile oracle_r50_canvas oracle_r50_consensus oracle_r50_consensus_s64)
read -r -a RUN <<< "${SETTINGS_ONLY:-${ORDER[*]}}"
for name in "${RUN[@]}"; do [[ -n "${FLAGS[$name]:-}" ]] || { echo "unknown setting $name" >&2; exit 2; }; done

OUT="$STAGE/assembly_sweep/oracle"
mkdir -p "$OUT" "$LOGS" "$STAGE/cmd"
echo "oracle assembly -> $OUT"
echo "settings: ${RUN[*]}"

STAMP=$(date +%H%M%S)
WORKER="$STAGE/cmd/oracle_sweep_${STAMP}.sh"
{ echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
  echo "I=\${LSB_JOBINDEX:-1}"
  echo "SETTINGS=($(printf '%q ' "${RUN[@]}"))"
  echo "GTS=($(printf '%q ' "${GT_VOLUMES[@]}"))"
  echo "S=\${SETTINGS[(I-1)/${#GT_VOLUMES[@]}]}; V=\${GTS[(I-1)%${#GT_VOLUMES[@]}]}"
  echo "declare -A FLAGS=("
  for name in "${RUN[@]}"; do printf '  [%s]=%q\n' "$name" "${FLAGS[$name]}"; done
  echo ")"
  printf 'cd %q\n' "$REPO"
  echo "echo \"element \$I: setting \$S volume \$V on \$(hostname)\""
  echo "mkdir -p '$OUT'/\"\$S\""
  echo "$VENV/bin/python experiments/sam_lmd_v1/pseudolabel.py oracle --volume \"\$V\" \\"
  echo "  --out '$OUT'/\"\$S\"/\"\$V\".json \${FLAGS[\$S]}"
} > "$WORKER"
FINAL="$STAGE/cmd/oracle_sweep_final.sh"
{ echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  printf 'cd %q\n' "$REPO"
  echo "for S in ${ORDER[*]}; do [[ -d '$OUT'/\$S ]] && $VENV/bin/python experiments/sam_lmd_v1/pseudolabel.py summarize '$OUT'/\$S > /dev/null; done"
  echo "$VENV/bin/python - '$OUT' ${ORDER[*]} <<'PY' | tee '$OUT/table.txt'"
  cat <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
names = [n for n in sys.argv[2:] if (out / n / "summary.json").exists()]
print(f"{'setting':26s} {'pseudo':>6s} {'truth':>6s} {'prec@.5':>8s} {'recall':>7s} {'merges':>6s} {'merge%':>7s} "
      f"{'frags':>6s} {'frag%':>6s} {'purity':>7s} {'share':>6s} {'bestIoU':>8s} {'claimed':>8s} {'tiles':>6s} {'s/blk':>6s}")
for name in names:
    s = json.load(open(out / name / "summary.json"))
    p = s["pooled"]; vols = s["volumes"]
    reports = [json.load(open(out / name / f"{v['volume']}.json")) for v in vols]
    blocks = [b for r in reports for b in r["blocks"]]
    best = sum(b["mean_best_iou"] * b["pseudo_instances"] for b in blocks) / max(p["pseudo_instances"], 1)
    claimed = sum(b["claimed_fraction"] for b in blocks) / len(blocks)
    secs = sum(b["seconds"] for b in blocks) / len(blocks)
    tiles = sum(b["tiles"] for b in blocks) / len(blocks)
    print(f"{name:26s} {p['pseudo_instances']:6d} {p['truth_instances']:6d} {p['precision']:8.3f} {p['recall']:7.3f} "
          f"{p['merges']:6d} {100*p['merge_rate']:6.1f}% {p['fragments']:6d} {100*p['fragment_rate']:5.1f}% "
          f"{p['pseudo_purity']:7.3f} {p['truth_best_share']:6.3f} {best:8.3f} {100*claimed:7.1f}% {tiles:6.0f} {secs:6.0f}")
print()
print("per volume: precision@0.5 / merges / fragments")
vols = [v["volume"] for v in json.load(open(out / names[0] / "summary.json"))["volumes"]]
print(f"{'setting':26s} " + " ".join(f"{v[:24]:>24s}" for v in vols))
for name in names:
    by = {v["volume"]: v for v in json.load(open(out / name / "summary.json"))["volumes"]}
    print(f"{name:26s} " + " ".join(
        f"{by[v]['precision']:.3f} / {by[v]['merges']:3d} / {by[v]['fragments']:3d}".rjust(24) for v in vols))
PY
} > "$FINAL"
chmod +x "$WORKER" "$FINAL"

TOTAL=$(( ${#RUN[@]} * ${#GT_VOLUMES[@]} ))
AARGS=(-P "$PROJECT" -q local -n 8 -W 2:00 -J "sam1_oracle[1-$TOTAL]" -cwd "$REPO"
       -o "$LOGS/sam1_oracle_%J_%I.log" -e "$LOGS/sam1_oracle_%J_%I.err")
if (( DRY )); then printf 'bsub %s bash %q\n' "${AARGS[*]}" "$WORKER"; cat "$WORKER"; exit 0; fi
arr=$(bsub "${AARGS[@]}" "bash '$WORKER'" | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
fin=$(bsub -P "$PROJECT" -q local -n 2 -W 0:30 -J "sam1_oracle_final" -w "ended($arr)" -cwd "$REPO" \
      -o "$LOGS/sam1_oracle_final_%J.log" -e "$LOGS/sam1_oracle_final_%J.err" "bash '$FINAL'" \
      | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
echo "array $arr [1-$TOTAL]  ->  table job $fin  ->  $OUT/table.txt"
