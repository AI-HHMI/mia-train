#!/usr/bin/env bash
# large_inputs_nisb: submit arms to gpu_b300, one full node each, each gated on a 20-step smoke run.
#
#   bash experiments/large_inputs_nisb/submit.sh w256_1cube w256_3cube w256_5cube   # smoke, then the real run on done(smoke)
#   bash experiments/large_inputs_nisb/submit.sh --dry-run w256_1cube               # write the job scripts, print the bsub lines
#   bash experiments/large_inputs_nisb/submit.sh --smoke w512_5cube                 # smoke only
#   bash experiments/large_inputs_nisb/submit.sh --no-smoke w256_1cube              # real run only; after a wall-time kill this
#                                                                                   # continues from the last checkpoint (--resume)
#
# gary_comparison/submit.sh, with several arms per call and the wall time set by the window.
# Everything lands in this experiment's home on /nrs: jobs/ (LSF logs and the generated job
# scripts) and runs/ (run directories, via --output-root); smoke artifacts under jobs/smoke/ and
# runs/smoke/. If a smoke fails, its real job stays PEND forever: `bkill` it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
PROJECT=miaai
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/large_inputs_nisb   # this experiment's home on /nrs
RUNS=$EXP/runs
LOGS=$EXP/jobs                      # LSF logs AND the generated job scripts
SMOKE_LOGS=$LOGS/smoke
SMOKE_RUNS=$RUNS/smoke
QUEUE=${QUEUE:-gpu_b300}
GPUS=8                              # the configs' dp_shard
SLOTS=96                            # 12 slots per GPU on gpu_b300
SMOKE_GPUS=${SMOKE_GPUS:-1}         # the smoke's derived config gets dp_shard = SMOKE_GPUS
# Wall time by window: 500k steps at the measured s/step (README.md) with ~2x margin, so 0.21 s -> 29 h
# and 1.24 s -> 7.2 days; the latter is the queue's 14-day maximum. `-r` plus `--resume` turns a node
# failure into a requeue from the last checkpoint; a wall-time kill does not requeue, so it needs
# `--no-smoke <arm>` again.
WALL_W256=${WALL_W256:-72:00}
WALL_W512=${WALL_W512:-336:00}
THREADS="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
# inductor's layout optimisation plus the head's 3-wide voxel-resolution convolutions kills compiled
# runs at step 1 with a cuDNN SDPA stride error; this keeps compile and cuDNN SDPA together.
COMPILE_ENV="export TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0"

DRY=0 SMOKE=1 REAL=1 ARMS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --smoke) REAL=0 ;;
    --no-smoke) SMOKE=0 ;;
    -*) echo "unknown flag $1" >&2; exit 2 ;;
    *) ARMS+=("$1") ;;
  esac
  shift
done
[[ ${#ARMS[@]} -gt 0 ]] || { echo "usage: submit.sh [--dry-run] [--smoke|--no-smoke] <arm> [<arm> ...]" >&2; exit 2; }
for arm in "${ARMS[@]}"; do
  [[ -f $HERE/$arm.toml ]] || { echo "no such config: $HERE/$arm.toml" >&2; exit 2; }
done
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
              -J "linisb_$tag" -cwd "$REPO" -o "$logdir/linisb_${tag}_%J.log" -e "$logdir/linisb_${tag}_%J.err")
  [[ -n "$dep" && "$dep" != DRYRUN ]] && args+=(-w "done($dep)")
  if [[ $DRY -eq 1 ]]; then
    printf 'bsub %s bash %q\n' "${args[*]}" "$cmd" >&2; echo DRYRUN
  else
    bsub "${args[@]}" "bash '$cmd'" | jobid
  fi
}

for NAME in "${ARMS[@]}"; do
  CONFIG="$HERE/$NAME.toml"
  case $NAME in w256_*) WALL=$WALL_W256 ;; w512_*) WALL=$WALL_W512 ;; esac
  SMOKE_ID=""
  if [[ $SMOKE -eq 1 ]]; then
    # 20 steps on SMOKE_GPUS GPUs (default 1), straight from the real config. Batch 1 per rank, so
    # one GPU holds what each of the real run's eight holds.
    smoke_cfg="$SMOKE_LOGS/smoke_$NAME.toml"
    sed -e "s/^experiment_name = .*/experiment_name = \"smoke_linisb__$NAME\"/" \
        -e 's/^max_steps = .*/max_steps = 20/'     -e 's/^warmup_steps = .*/warmup_steps = 2/' \
        -e 's/^val_every = .*/val_every = 10/'     -e 's/^checkpoint_every = .*/checkpoint_every = 20/' \
        -e 's/^samples_per_epoch = .*/samples_per_epoch = 20/' -e "s/^dp_shard = .*/dp_shard = $SMOKE_GPUS/" \
        -e 's/^num_workers = .*/num_workers = 2/'  -e 's/^log_every = .*/log_every = 5/' \
        "$CONFIG" > "$smoke_cfg"
    cmd=$(write_cmd "$SMOKE_LOGS" "smoke_$NAME" "$smoke_cfg" "$SMOKE_GPUS" --output-root "$SMOKE_RUNS")
    SMOKE_ID=$(submit "$SMOKE_LOGS" "smoke_$NAME" "$cmd" "$SMOKE_GPUS" $((12 * SMOKE_GPUS)) 2:00)
    echo "smoke  $NAME: job $SMOKE_ID  ($cmd)"
  fi
  if [[ $REAL -eq 1 ]]; then
    cmd=$(write_cmd "$LOGS" "$NAME" "$CONFIG" "$GPUS" --output-root "$RUNS" --resume)
    REAL_ID=$(submit "$LOGS" "$NAME" "$cmd" "$GPUS" "$SLOTS" "$WALL" "$SMOKE_ID")
    echo "real   $NAME: job $REAL_ID  ($cmd, -W $WALL)${SMOKE_ID:+, starts after done($SMOKE_ID)}"
  fi
done
