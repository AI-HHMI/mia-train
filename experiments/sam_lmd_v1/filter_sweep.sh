#!/usr/bin/env bash
# Does a merge-aware filter raise pseudo-label precision? Score the labeller on the GT blocks under
# several filter settings, with the SAME teacher and the SAME blocks as the round's own diagnostic.
#
#   bash experiments/sam_lmd_v1/filter_sweep.sh feat64            # teacher = newest sam1__feat64_r0 run
#   bash experiments/sam_lmd_v1/filter_sweep.sh feat64 --dry-run
#
# One LSF array element per (setting, GT volume): `pseudolabel.py diagnose` on one 512^3 block of
# each of the four finetune volumes, exactly the block the L1 diagnostic used (block choice is a
# deterministic per-volume permutation), so the `baseline` row must reproduce that diagnostic and
# every other row differs from it only in the filter. A CPU job then pools each setting and prints
# one table. Results: $STAGE/filter_sweep/<arm>/<setting>/{<volume>.json,summary.json} and
# $STAGE/filter_sweep/<arm>/table.txt.
#
# The GT volumes trained the teacher, so every number here is an upper bound on what the filter
# does on unlabeled data; the comparison between rows is what this measures, not the level.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
STAGE=/nrs/scicompsoft/orhane/mia-train-scratch/sam_lmd_v1
LOGS="$RUNS/jobs"
PROJECT=miaai
QUEUE=${QUEUE:-gpu_b300}

ARM="${1:?arm, e.g. feat64}"; shift || true
DRY=0; [[ "${1:-}" == --dry-run ]] && DRY=1
BLOCK=${BLOCK:-512}
PRED_IOU=${PRED_IOU:-0.7}       # the labelling gates, as submit.sh uses them
STABILITY=${STABILITY:-0.8}
GT_VOLUMES=(kasthuri15_ac3 zebrafish_fish2_quadcube1 liconn_mouse_dg hemibrain_ellipsoid_body)

# name -> extra pseudolabel.py flags. `baseline` is the labeller exactly as the L1 pass ran it.
declare -A SETTINGS=(
  [baseline]=""
  [tiled]="--split-tiled-wholes 1"
  [cons_top3_t05]="--consistency-clicks 3 --consistency-thresh 0.5 --consistency-pick top"
  [cons_top3_t07]="--consistency-clicks 3 --consistency-thresh 0.7 --consistency-pick top"
  [cons_best3_t05]="--consistency-clicks 3 --consistency-thresh 0.5 --consistency-pick best"
  [both_t05]="--split-tiled-wholes 1 --consistency-clicks 3 --consistency-thresh 0.5 --consistency-pick top"
)
ORDER=(baseline tiled cons_top3_t05 cons_top3_t07 cons_best3_t05 both_t05)

run=$(ls -dt "$RUNS"/sam1__${ARM}_r0_*/ 2>/dev/null | head -1) || true
[[ -n "${run:-}" ]] || { echo "no run matching sam1__${ARM}_r0_*" >&2; exit 1; }
run=${run%/}
STEP=$(ls -d "$run"/checkpoints/step_* | sed 's|.*step_||' | sort -n | tail -1)
OUT="$STAGE/filter_sweep/$ARM"
mkdir -p "$OUT" "$LOGS" "$STAGE/cmd"
echo "teacher $run step $STEP -> $OUT"

WORKER="$STAGE/cmd/filter_sweep_${ARM}.sh"
{ echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "I=\${LSB_JOBINDEX:-1}"
  echo "SETTINGS=($(printf '%q ' "${ORDER[@]}"))"
  echo "GTS=($(printf '%q ' "${GT_VOLUMES[@]}"))"
  echo "S=\${SETTINGS[(I-1)/${#GT_VOLUMES[@]}]}; V=\${GTS[(I-1)%${#GT_VOLUMES[@]}]}"
  echo "declare -A FLAGS=("
  for name in "${ORDER[@]}"; do printf '  [%s]=%q\n' "$name" "${SETTINGS[$name]}"; done
  echo ")"
  printf 'cd %q\n' "$REPO"
  echo "echo \"element \$I: setting \$S volume \$V\""
  echo "mkdir -p '$OUT'/\"\$S\""
  echo "$VENV/bin/python experiments/sam_lmd_v1/pseudolabel.py diagnose '$run' --step $STEP --volume \"\$V\" \\"
  echo "  --out '$OUT'/\"\$S\"/\"\$V\".json --blocks 1 --block $BLOCK --pred-iou $PRED_IOU --stability $STABILITY \${FLAGS[\$S]}"
} > "$WORKER"
FINAL="$STAGE/cmd/filter_sweep_${ARM}_final.sh"
{ echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  printf 'cd %q\n' "$REPO"
  echo "for S in ${ORDER[*]}; do $VENV/bin/python experiments/sam_lmd_v1/pseudolabel.py summarize '$OUT'/\$S > /dev/null; done"
  echo "$VENV/bin/python - '$OUT' ${ORDER[*]} <<'PY' | tee '$OUT/table.txt'"
  cat <<'PY'
import json, sys
from pathlib import Path
out, names = Path(sys.argv[1]), sys.argv[2:]
print(f"{'setting':16s} {'pseudo':>7s} {'prec@.5':>8s} {'recall':>7s} {'merges':>7s} {'merge%':>7s} {'bestIoU':>8s} {'claimed':>8s} {'s/block':>8s}")
for name in names:
    s = json.load(open(out / name / "summary.json"))
    p = s["pooled"]; vols = s["volumes"]
    reports = [json.load(open(out / name / f"{v['volume']}.json")) for v in vols]
    blocks = [b for r in reports for b in r["blocks"]]
    best = sum(b["mean_best_iou"] * b["pseudo_instances"] for b in blocks) / max(p["pseudo_instances"], 1)
    claimed = sum(b["claimed_fraction"] for b in blocks) / len(blocks)
    secs = sum(b["seconds"] for b in blocks) / len(blocks)
    print(f"{name:16s} {p['pseudo_instances']:7d} {p['precision']:8.3f} {p['recall']:7.3f} {p['merges']:7d} {100*p['merge_rate']:6.1f}% {best:8.3f} {100*claimed:7.1f}% {secs:8.0f}")
print()
print("per volume, precision@0.5 / merges:")
vols = [v["volume"] for v in json.load(open(out / names[0] / "summary.json"))["volumes"]]
print(f"{'setting':16s} " + " ".join(f"{v[:22]:>22s}" for v in vols))
for name in names:
    by = {v["volume"]: v for v in json.load(open(out / name / "summary.json"))["volumes"]}
    print(f"{name:16s} " + " ".join(f"{by[v]['precision']:.3f} / {by[v]['merges']:3d}".rjust(22) for v in vols))
PY
} > "$FINAL"
chmod +x "$WORKER" "$FINAL"

TOTAL=$(( ${#ORDER[@]} * ${#GT_VOLUMES[@]} ))
AARGS=(-P "$PROJECT" -q "$QUEUE" -gpu "num=1" -n 12 -W 2:00 -J "sam1_fsweep_${ARM}[1-$TOTAL]" -cwd "$REPO"
       -o "$LOGS/sam1_fsweep_${ARM}_%J_%I.log" -e "$LOGS/sam1_fsweep_${ARM}_%J_%I.err")
if (( DRY )); then printf 'bsub %s bash %q\n' "${AARGS[*]}" "$WORKER"; exit 0; fi
arr=$(bsub "${AARGS[@]}" "bash '$WORKER'" | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
fin=$(bsub -P "$PROJECT" -q local -n 2 -W 1:00 -J "sam1_fsweep_${ARM}_final" -w "ended($arr)" -cwd "$REPO" \
      -o "$LOGS/sam1_fsweep_${ARM}_final_%J.log" -e "$LOGS/sam1_fsweep_${ARM}_final_%J.err" "bash '$FINAL'" \
      | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
echo "array $arr [1-$TOTAL]  ->  table job $fin  ->  $OUT/table.txt"
