#!/usr/bin/env bash
# Queue an arm's label-quality scoring so it starts the moment its training job ends.
#
#   bash experiments/sam_lmd_v1/score_when_done.sh <arm> <training job id> <nm: 4|8> [step]
#   bash experiments/sam_lmd_v1/score_when_done.sh arm2_4nm_gb16 154281188 4
#   bash experiments/sam_lmd_v1/score_when_done.sh arm4_8nm_gb16 154281613 8
#
# Submits one small CPU job held on `ended(<training job>)`. When it runs it waits for the
# checkpoint directory to appear on /nrs (the file system lags the scheduler by seconds), then
# runs the three scoring scripts exactly as they were run for arm 1:
#   assembly_sweep.sh     single_tile + consensus on the four scored blocks  -> assembly_sweep/<arm>_step<N>/table.txt
#   calibration_probe.sh  every grid candidate on the same blocks             -> probe/<arm>_r0_step<N>/table.txt
#   labelling_gallery.sh  the pictures and the saved window labellings         -> viz/<arm>_step<N>/gallery_*.png
# Each of those submits its own GPU array on gpu_b300 and a CPU job that writes the table.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/sam_lmd_v1   # this experiment's home on /nrs: runs/ jobs/ eval/ probes/ (layout of 2026-09-16)
RUNS=$EXP/runs
STAGE=$EXP
LOGS="$EXP/jobs"
PROJECT=miaai

ARM=${1:?arm name}
TRAIN_JOB=${2:?training job id}
NM=${3:?4 or 8}
STEP=${4:-200000}
case "$NM" in
  4) PROBE_ENV="GT_CONFIG=experiments/sam_lmd_v1/data/lmd_finetune_singlescale_4nm.yaml BLOCK=1024 MIN_MASK=4096" ;;
  8) PROBE_ENV="" ;;
  *) echo "nm must be 4 or 8" >&2; exit 2 ;;
esac
mkdir -p "$LOGS" "$STAGE/cmd"

WORKER="$STAGE/cmd/score_when_done_${ARM}_step${STEP}.sh"
cat > "$WORKER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $REPO
echo "training job $TRAIN_JOB ended; \$(date)"
run=\$(ls -dt $RUNS/sam1__${ARM}_r0_*/ | head -1); run=\${run%/}
for i in \$(seq 1 60); do
  [[ -d "\$run/checkpoints/step_$STEP" ]] && break
  echo "waiting for \$run/checkpoints/step_$STEP (\$i/60)"; sleep 30
done
[[ -d "\$run/checkpoints/step_$STEP" ]] || { echo "no checkpoints/step_$STEP after 30 min in \$run" >&2; exit 1; }
echo "checkpoint present: \$run/checkpoints/step_$STEP"
NM=$NM STEP=$STEP SETTINGS_ONLY="single_tile consensus" bash experiments/sam_lmd_v1/assembly_sweep.sh $ARM
env STEP=$STEP $PROBE_ENV bash experiments/sam_lmd_v1/calibration_probe.sh $ARM
NM=$NM STEP=$STEP bash experiments/sam_lmd_v1/figures/labelling_gallery.sh $ARM
EOF

out=$(bsub -P "$PROJECT" -q local -n 1 -W 1:00 -J "sam1_score_when_done_${ARM}" -w "ended($TRAIN_JOB)" \
      -cwd "$REPO" -o "$LOGS/sam1_score_when_done_${ARM}_%J.log" -e "$LOGS/sam1_score_when_done_${ARM}_%J.err" \
      "bash '$WORKER'")
echo "$out" | grep -v 'billed to'
echo "  held on ended($TRAIN_JOB); log: $LOGS/sam1_score_when_done_${ARM}_<jobid>.log"
