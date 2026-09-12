#!/usr/bin/env bash
# Per-candidate calibration of one or more teachers on the GT blocks the L1 diagnostic scored.
#
#   bash experiments/sam_lmd_v1/calibration_probe.sh deep4 feat64          # newest r0 run of each arm
#   QUEUE=gpu_b300 bash experiments/sam_lmd_v1/calibration_probe.sh deep4
#   bash experiments/sam_lmd_v1/calibration_probe.sh deep4 --dry-run
#
# One LSF array element per (arm, GT volume) runs `calibration_probe.py probe` over every tile of
# the volume's diagnostic block; a CPU job per arm then pools the four records into a table.
# Results: $STAGE/probe/<arm>/{<volume>.npz,summary.json,table.txt}.
#
# Default queue is gpu_h100: the probe compares stages of one pipeline against each other, and a
# GPU generation shifts the decoded masks by well under the effects it looks for (memory note:
# 520 vs 530 instances on one block). The eval and labelling passes stay pinned to B300.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
STAGE=/nrs/scicompsoft/orhane/mia-train-scratch/sam_lmd_v1
LOGS="$RUNS/jobs"
PROJECT=miaai
QUEUE=${QUEUE:-gpu_h100}
ROUND=${ROUND:-0}
BLOCK=${BLOCK:-512}
MAX_TILES=${MAX_TILES:-0}
GT_VOLUMES=(kasthuri15_ac3 zebrafish_fish2_quadcube1 liconn_mouse_dg hemibrain_ellipsoid_body)

ARMS=(); DRY=0
for arg in "$@"; do
  case "$arg" in --dry-run) DRY=1 ;; --*) echo "unknown flag $arg" >&2; exit 2 ;; *) ARMS+=("$arg") ;; esac
done
[[ ${#ARMS[@]} -gt 0 ]] || { sed -n '2,10p' "$0"; exit 2; }
mkdir -p "$LOGS" "$STAGE/cmd" "$STAGE/probe"

jobid () { sed -n 's/^Job <\([0-9]*\)>.*/\1/p'; }

for ARM in "${ARMS[@]}"; do
  run=$(ls -dt "$RUNS"/sam1__${ARM}_r${ROUND}_*/ 2>/dev/null | head -1) || true
  [[ -n "${run:-}" ]] || { echo "no run matching sam1__${ARM}_r${ROUND}_*" >&2; exit 1; }
  run=${run%/}
  STEP=$(ls -d "$run"/checkpoints/step_* | sed 's|.*step_||' | sort -n | tail -1)
  OUT="$STAGE/probe/${ARM}_r${ROUND}"
  mkdir -p "$OUT"
  echo "teacher $run step $STEP -> $OUT"

  WORKER="$STAGE/cmd/probe_${ARM}_r${ROUND}.sh"
  { echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    echo "GTS=($(printf '%q ' "${GT_VOLUMES[@]}"))"
    echo "V=\${GTS[\${LSB_JOBINDEX:-1}-1]}"
    printf 'cd %q\n' "$REPO"
    echo "echo \"element \${LSB_JOBINDEX:-1}: volume \$V on \$(hostname)\""
    echo "nvidia-smi --query-gpu=name --format=csv,noheader | head -1"
    echo "$VENV/bin/python experiments/sam_lmd_v1/calibration_probe.py probe '$run' --step $STEP --volume \"\$V\" \\"
    echo "  --out '$OUT'/\"\$V\".npz --block $BLOCK --max-tiles $MAX_TILES"
  } > "$WORKER"
  FINAL="$STAGE/cmd/probe_${ARM}_r${ROUND}_final.sh"
  { echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf 'cd %q\n' "$REPO"
    echo "$VENV/bin/python experiments/sam_lmd_v1/calibration_probe.py summarize '$OUT' | tee '$OUT/table.txt'"
  } > "$FINAL"

  if (( DRY )); then
    echo "  would submit: array [1-${#GT_VOLUMES[@]}] of $WORKER on $QUEUE, then $FINAL"
    continue
  fi
  arr=$(bsub -P "$PROJECT" -q "$QUEUE" -gpu "num=1" -n 12 -W 2:00 -J "sam1_probe_${ARM}[1-${#GT_VOLUMES[@]}]" \
        -cwd "$REPO" -o "$LOGS/sam1_probe_${ARM}_%J_%I.log" -e "$LOGS/sam1_probe_${ARM}_%J_%I.err" \
        "bash '$WORKER'" | jobid)
  fin=$(bsub -P "$PROJECT" -q local -n 4 -W 0:30 -J "sam1_probe_${ARM}_final" -w "done($arr)" \
        -cwd "$REPO" -o "$LOGS/sam1_probe_${ARM}_final_%J.log" -e "$LOGS/sam1_probe_${ARM}_final_%J.err" \
        "bash '$FINAL'" | jobid)
  echo "  array $arr (4 volumes) -> summarize $fin;  table: $OUT/table.txt"
done
