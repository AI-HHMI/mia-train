#!/usr/bin/env bash
# gary_comparison: submit one arm to gpu_b300, gated on a 20-step smoke run.
#
#   bash experiments/gary_comparison/submit.sh 1a_dinov3_axial_subpixel             # smoke, then the real run chained on `done(smoke)`
#   bash experiments/gary_comparison/submit.sh --dry-run 1a_dinov3_axial_subpixel   # write the job scripts, print the bsub lines
#   bash experiments/gary_comparison/submit.sh --smoke 1a_dinov3_axial_subpixel     # smoke only
#   bash experiments/gary_comparison/submit.sh --no-smoke 1a_dinov3_axial_subpixel  # real run only; resubmitting after a
#                                                                                   # wall-time kill continues via --resume
#
# Everything this writes lands in this experiment's directory on /nrs, in the two places the layout
# of 2026-09-16 provides: jobs/ (LSF logs and the generated job scripts) and runs/ (run directories
# via --output-root). Smoke artifacts -- the derived 20-step config, its job script, its LSF logs and
# its run directory -- go under jobs/smoke/ and runs/smoke/, never beside the production runs.
# Nothing else is created at the experiment level. If the smoke fails, the chained real job stays
# PEND forever: `bkill` it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
PROJECT=miaai
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/gary_comparison   # this experiment's home on /nrs
RUNS=$EXP/runs
LOGS=$EXP/jobs                      # LSF logs AND the generated job scripts
SMOKE_LOGS=$LOGS/smoke              # the smoke's config, script and logs
SMOKE_RUNS=$RUNS/smoke              # the smoke's run directories
QUEUE=${QUEUE:-gpu_b300}
GPUS=8
SLOTS=96                            # 12 slots/GPU on gpu_b300
# 500k steps at the expected 0.15-0.22 s/step (2c's 0.27 s on Hopper eager, B300 compiled measured
# 0.12 s single-GPU, ~1.6x that under 8-rank FSDP) is 21-31 h; 96 h tolerates a 3x slower step.
# `-r` plus `--resume` turns a node failure into a requeue from the last checkpoint.
WALL=${WALL:-96:00}
THREADS="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
# inductor's layout optimisation plus the head's 3-wide voxel-resolution convolutions kills compiled
# runs at step 1 with a cuDNN SDPA stride error; this keeps compile and cuDNN SDPA together.
COMPILE_ENV="export TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0"

DRY=0 SMOKE=1 REAL=1
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --smoke) REAL=0 ;;
    --no-smoke) SMOKE=0 ;;
    -*) echo "unknown flag $1" >&2; exit 2 ;;
    *) NAME=$1 ;;
  esac
  shift
done
: "${NAME:?usage: submit.sh [--dry-run] [--smoke|--no-smoke] <config name without .toml>}"
CONFIG="$HERE/$NAME.toml"
[[ -f $CONFIG ]] || { echo "no such config: $CONFIG" >&2; exit 2; }
mkdir -p "$RUNS" "$LOGS" "$SMOKE_RUNS" "$SMOKE_LOGS"

jobid () { grep -o 'Job <[0-9]*>' | head -1 | tr -dc 0-9; }

# write_cmd <dir> <tag> <config> <nproc> <tail args...>: the script bsub runs, written into <dir>
write_cmd () {
  local dir=$1 tag=$2 cfg=$3 procs=$4; shift 4
  local cmd="$dir/$tag.sh"
  { echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "$THREADS"
    echo "$COMPILE_ENV"
    echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    printf 'cd %q\n' "$REPO"
    printf '%s --standalone --nproc_per_node=%s src/train.py --config %q %s\n' \
      "$VENV/bin/torchrun" "$procs" "$cfg" "$*"
  } > "$cmd"
  chmod +x "$cmd"
  echo "$cmd"
}

# submit <logdir> <tag> <cmd> <nproc> <slots> <wall> [dependency job id]  -> job id (or DRYRUN)
submit () {
  local logdir=$1 tag=$2 cmd=$3 procs=$4 slots=$5 wall=$6 dep=${7:-}
  local args=(-P "$PROJECT" -q "$QUEUE" -gpu "num=$procs" -n "$slots" -W "$wall" -r
              -J "gary_$tag" -cwd "$REPO" -o "$logdir/gary_${tag}_%J.log" -e "$logdir/gary_${tag}_%J.err")
  [[ -n "$dep" && "$dep" != DRYRUN ]] && args+=(-w "done($dep)")
  if [[ $DRY -eq 1 ]]; then
    printf 'bsub %s bash %q\n' "${args[*]}" "$cmd" >&2; echo DRYRUN
  else
    bsub "${args[@]}" "bash '$cmd'" | jobid
  fi
}

SMOKE_ID=""
if [[ $SMOKE -eq 1 ]]; then
  # 20 steps on one GPU, straight from the real config, so the smoke exercises what actually runs
  smoke_cfg="$SMOKE_LOGS/smoke_$NAME.toml"
  sed -e "s/^experiment_name = .*/experiment_name = \"smoke_gary__$NAME\"/" \
      -e 's/^max_steps = .*/max_steps = 20/'     -e 's/^warmup_steps = .*/warmup_steps = 2/' \
      -e 's/^val_every = .*/val_every = 10/'     -e 's/^checkpoint_every = .*/checkpoint_every = 20/' \
      -e 's/^samples_per_epoch = .*/samples_per_epoch = 20/' -e 's/^dp_shard = .*/dp_shard = 1/' \
      -e 's/^num_workers = .*/num_workers = 2/'  -e 's/^log_every = .*/log_every = 5/' \
      "$CONFIG" > "$smoke_cfg"
  cmd=$(write_cmd "$SMOKE_LOGS" "smoke_$NAME" "$smoke_cfg" 1 --output-root "$SMOKE_RUNS")
  SMOKE_ID=$(submit "$SMOKE_LOGS" "smoke_$NAME" "$cmd" 1 12 1:00)
  echo "smoke  $NAME: job $SMOKE_ID  ($cmd)"
fi
if [[ $REAL -eq 1 ]]; then
  cmd=$(write_cmd "$LOGS" "$NAME" "$CONFIG" "$GPUS" --output-root "$RUNS" --resume)
  REAL_ID=$(submit "$LOGS" "$NAME" "$cmd" "$GPUS" "$SLOTS" "$WALL" "$SMOKE_ID")
  echo "real   $NAME: job $REAL_ID  ($cmd)${SMOKE_ID:+, starts after done($SMOKE_ID)}"
fi
