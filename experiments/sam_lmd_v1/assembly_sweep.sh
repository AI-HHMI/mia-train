#!/usr/bin/env bash
# Where does the labeller's precision go: in the tiles, or in putting the tiles together?
#
#   bash experiments/sam_lmd_v1/assembly_sweep.sh deep4            # every setting, teacher = newest sam1__deep4_r0
#   SETTINGS_ONLY="propagate_s64 canvas_s64" bash experiments/sam_lmd_v1/assembly_sweep.sh deep4
#   STEP=50000 SETTINGS_ONLY="single_tile consensus" bash experiments/sam_lmd_v1/assembly_sweep.sh feat64
#   NM=4 STEP=200000 SETTINGS_ONLY="single_tile consensus" bash experiments/sam_lmd_v1/assembly_sweep.sh arm1_4nm
#   bash experiments/sam_lmd_v1/assembly_sweep.sh deep4 --dry-run
#
# `pseudolabel.py diagnose` on the SAME 512^3 block of each of the four finetune volumes that the
# round's L1 diagnostic scored, once per tile-assembly setting, plus a no-assembly reference: eight
# 288^3 blocks per volume, each exactly one 256^3 tile, so nothing is ever joined. Every row is the
# same gates (0.7 / 0.8), the same grid, the same teacher; only what happens after the gates
# changes. The `_s64` settings advance the window by 64 output voxels instead of 128 (a quarter
# window instead of half: 125 tiles per 512 block instead of 27). The `consensus` settings join
# masks only where two windows agree in their shared region and leave disputed cells unlabelled
# (`amg.consensus_labelling`); `_t07` raises the agreement bar. The per-block scores carry
# `merges` (two true objects under one pseudo id) AND `fragments` (one true object under two
# pseudo ids), `pseudo_purity` and `truth_best_share`, which is what separates wrong joins from
# failed joins.
#
# Results: $STAGE/assembly_sweep/<arm>_step<N>/<setting>/{<volume>.json,summary.json}, table in
# $STAGE/assembly_sweep/<arm>_step<N>/table.txt (every setting with results, whichever run produced
# it). The deep4 version-2 results predate the step suffix and live in assembly_sweep/deep4/.
# Default queue gpu_b300 (the version-2 rows were scored on H100 while B300 was admin-closed; a GPU
# generation moves instance counts by ~2%, so the `propagate` row is expected near, not at, the
# B300 L1 diagnostic.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/sam_lmd_v1   # this experiment's home on /nrs: runs/ jobs/ eval/ probes/ (layout of 2026-09-16)
RUNS=$EXP/runs
STAGE=$EXP
LOGS="$EXP/jobs"
PROJECT=miaai
QUEUE=${QUEUE:-gpu_b300}   # the arms train here; H100 was used only while B300 was admin-closed
PRED_IOU=${PRED_IOU:-0.7}
STABILITY=${STABILITY:-0.8}
GT_VOLUMES=(kasthuri15_ac3 zebrafish_fish2_quadcube1 liconn_mouse_dg hemibrain_ellipsoid_body)
# NM=4 scores a model trained on the 4 nm lattice on the SAME physical blocks: the GT config is
# the 4 nm split copy, a 1024-voxel block is the 512-voxel block of the 8 nm tables, the
# single-window blocks are 320 voxels (one 256-voxel window of 1 um; 32 of them, about the tissue
# of the eight 2 um windows), and every voxel floor is 8x (4096 = 512 at 8 nm).
NM=${NM:-8}
case "$NM" in
  8) GT_CONFIG="experiments/lmd_ssl_v1/lmd_finetune_singlescale.yaml"; BLOCK=512; SINGLE="--block 288 --blocks 8"; FLOOR=512 ;;
  4) GT_CONFIG="experiments/sam_lmd_v1/data/lmd_finetune_singlescale_4nm.yaml"; BLOCK=1024; SINGLE="--block 320 --blocks 32"; FLOOR=4096 ;;
  *) echo "NM must be 8 or 4, got $NM" >&2; exit 2 ;;
esac
COMMON="--gt-config $GT_CONFIG --min-truth-voxels $FLOOR --min-mask-voxels $FLOOR"

ARM="${1:?arm, e.g. deep4}"; shift || true
DRY=0; [[ "${1:-}" == --dry-run ]] && DRY=1

# setting -> labeller flags. `propagate` is the L1 labeller exactly; `single_tile` is the reference.
declare -A FLAGS=(
  [propagate]="--tile-merge propagate --block $BLOCK --blocks 1"
  [canvas]="--tile-merge canvas --block $BLOCK --blocks 1"
  [none]="--tile-merge none --block $BLOCK --blocks 1"
  [edge_canvas]="--edge-discard 1 --tile-merge canvas --block $BLOCK --blocks 1"
  [edge_none]="--edge-discard 1 --tile-merge none --block $BLOCK --blocks 1"
  [single_tile]="--tile-merge none $SINGLE"
  [propagate_s64]="--tile-merge propagate --block $BLOCK --blocks 1 --tile-step 64"
  [canvas_s64]="--tile-merge canvas --block $BLOCK --blocks 1 --tile-step 64"
  [edge_canvas_s64]="--edge-discard 1 --tile-merge canvas --block $BLOCK --blocks 1 --tile-step 64"
  [consensus]="--tile-merge consensus --agree-thresh 0.5 --block $BLOCK --blocks 1"
  [consensus_t07]="--tile-merge consensus --agree-thresh 0.7 --block $BLOCK --blocks 1"
  [consensus_s64]="--tile-merge consensus --agree-thresh 0.5 --block $BLOCK --blocks 1 --tile-step 64"
)
ORDER=(propagate canvas none edge_canvas edge_none single_tile propagate_s64 canvas_s64 edge_canvas_s64
       consensus consensus_t07 consensus_s64)
read -r -a RUN <<< "${SETTINGS_ONLY:-${ORDER[*]}}"
for name in "${RUN[@]}"; do [[ -n "${FLAGS[$name]:-}" ]] || { echo "unknown setting $name" >&2; exit 2; }; done

run=$(ls -dt "$RUNS"/sam1__${ARM}_r0_*/ 2>/dev/null | head -1) || true
[[ -n "${run:-}" ]] || { echo "no run matching sam1__${ARM}_r0_*" >&2; exit 1; }
run=${run%/}
NEWEST=$(ls -d "$run"/checkpoints/step_* | sed 's|.*step_||' | sort -n | tail -1)
STEP=${STEP:-$NEWEST}                     # STEP=50000 scores an older checkpoint of the same run
[[ -d "$run/checkpoints/step_$STEP" ]] || { echo "no checkpoints/step_$STEP in $run" >&2; exit 1; }
# The window the model trained at decides the GT split copy and the click grid: RoPE normalises
# coordinates by the runtime grid, so a model MUST be labelled with its training crop. The
# labeller's defaults are a 256-voxel window and 14 clicks per side; another crop keeps the click
# spacing (256/14 = 18.3 voxels): 7 per side at 128, 19 at 352.
CROP=$("$VENV/bin/python" -c "import json,sys; print(int(json.load(open(sys.argv[1]))['model']['kwargs'].get('img_size', 256)))" "$run/resolved_config.json")
if [[ "$CROP" != 256 ]]; then
  [[ "$NM" == 8 ]] || { echo "a $CROP-voxel window is only generated at 8 nm" >&2; exit 2; }
  GT_CONFIG="experiments/sam_lmd_v1/data/lmd_finetune_singlescale_crop${CROP}.yaml"
  [[ -f "$GT_CONFIG" ]] || { echo "missing $GT_CONFIG: run make_configs.py" >&2; exit 1; }
  PPS=$(( (14 * CROP + 128) / 256 ))
  COMMON="--gt-config $GT_CONFIG --min-truth-voxels $FLOOR --min-mask-voxels $FLOOR --points-per-side $PPS"
  FLAGS[single_tile]="${FLAGS[single_tile]//--block 288/--block $((CROP + 32))}"   # one window (+32) per single-window block
  echo "window $CROP: GT config $GT_CONFIG, $PPS clicks per side, single-window blocks of $((CROP + 32))"
fi
OUT="$STAGE/assembly_sweep/${ARM}_step${STEP}"
mkdir -p "$OUT" "$LOGS" "$STAGE/cmd"
echo "teacher $run step $STEP -> $OUT"
echo "settings: ${RUN[*]}"

STAMP=$(date +%H%M%S)
WORKER="$STAGE/cmd/assembly_sweep_${ARM}_step${STEP}_${STAMP}.sh"
{ echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
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
  echo "$VENV/bin/python experiments/sam_lmd_v1/pseudolabel.py diagnose '$run' --step $STEP --volume \"\$V\" \\"
  echo "  --out '$OUT'/\"\$S\"/\"\$V\".json --pred-iou $PRED_IOU --stability $STABILITY $COMMON \${FLAGS[\$S]}"
} > "$WORKER"
FINAL="$STAGE/cmd/assembly_sweep_${ARM}_step${STEP}_final.sh"
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
print(f"{'setting':16s} {'pseudo':>6s} {'truth':>6s} {'prec@.5':>8s} {'recall':>7s} {'merges':>6s} {'merge%':>7s} "
      f"{'frags':>6s} {'frag%':>6s} {'swallow':>8s} {'purity':>7s} {'share':>6s} {'bestIoU':>8s} {'claimed':>8s} {'tiles':>6s} {'s/blk':>6s}")
for name in names:
    s = json.load(open(out / name / "summary.json"))
    p = s["pooled"]; vols = s["volumes"]
    reports = [json.load(open(out / name / f"{v['volume']}.json")) for v in vols]
    blocks = [b for r in reports for b in r["blocks"]]
    best = sum(b["mean_best_iou"] * b["pseudo_instances"] for b in blocks) / max(p["pseudo_instances"], 1)
    claimed = sum(b["claimed_fraction"] for b in blocks) / len(blocks)
    secs = sum(b["seconds"] for b in blocks) / len(blocks)
    tiles = sum(b["tiles"] for b in blocks) / len(blocks)
    print(f"{name:16s} {p['pseudo_instances']:6d} {p['truth_instances']:6d} {p['precision']:8.3f} {p['recall']:7.3f} "
          f"{p['merges']:6d} {100*p['merge_rate']:6.1f}% {p['fragments']:6d} {100*p['fragment_rate']:5.1f}% "
          f"{p.get('swallowed', 0):8d} {p['pseudo_purity']:7.3f} {p['truth_best_share']:6.3f} {best:8.3f} {100*claimed:7.1f}% {tiles:6.0f} {secs:6.0f}")
print()
print("per volume: precision@0.5 / merges / fragments / swallowed (0 in reports written before the field existed)")
vols = [v["volume"] for v in json.load(open(out / names[0] / "summary.json"))["volumes"]]
print(f"{'setting':16s} " + " ".join(f"{v[:28]:>28s}" for v in vols))
for name in names:
    by = {v["volume"]: v for v in json.load(open(out / name / "summary.json"))["volumes"]}
    print(f"{name:16s} " + " ".join(
        f"{by[v]['precision']:.3f} / {by[v]['merges']:3d} / {by[v]['fragments']:3d} / {by[v].get('swallowed', 0):3d}".rjust(28)
        for v in vols))
PY
} > "$FINAL"
chmod +x "$WORKER" "$FINAL"

TOTAL=$(( ${#RUN[@]} * ${#GT_VOLUMES[@]} ))
# 12 slots = one GPU's share on gpu_h100 (12 slots/GPU, 40 GB each), so a one-GPU job strands nothing;
# 480 GB also covers the 4 nm scoring (a 1024^3 int64 labelling, its truth, and the scoring's temporaries).
AARGS=(-P "$PROJECT" -q "$QUEUE" -gpu "num=1" -n 12 -W 4:00 -J "sam1_asweep_${ARM}_${STEP}[1-$TOTAL]" -cwd "$REPO"
       -o "$LOGS/sam1_asweep_${ARM}_%J_%I.log" -e "$LOGS/sam1_asweep_${ARM}_%J_%I.err")
if (( DRY )); then printf 'bsub %s bash %q\n' "${AARGS[*]}" "$WORKER"; cat "$WORKER"; exit 0; fi
arr=$(bsub "${AARGS[@]}" "bash '$WORKER'" | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
fin=$(bsub -P "$PROJECT" -q local -n 2 -W 0:30 -J "sam1_asweep_${ARM}_${STEP}_final" -w "ended($arr)" -cwd "$REPO" \
      -o "$LOGS/sam1_asweep_${ARM}_final_%J.log" -e "$LOGS/sam1_asweep_${ARM}_final_%J.err" "bash '$FINAL'" \
      | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
echo "array $arr [1-$TOTAL]  ->  table job $fin  ->  $OUT/table.txt"
