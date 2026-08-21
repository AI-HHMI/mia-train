#!/usr/bin/env bash
# Submit the data-scaling arms. Each arm is one full node, and two stages held one behind the next.
#
#   bash experiments/data_scaling/submit.sh                 # all three arms
#   bash experiments/data_scaling/submit.sh 1 5             # just those arms
#   bash experiments/data_scaling/submit.sh --dry-run       # print the bsub lines, submit nothing
#   bash experiments/data_scaling/submit.sh --smoke         # 20 steps of every stage, one GPU
#
#   arm 1  seed0                          -> interp 200k -> subpixel 100k
#   arm 3  seed0,1,2                      -> interp 200k -> subpixel 100k
#   arm 5  seed0,1,2,3,4                  -> interp 200k -> subpixel 100k
#
# **One arm per node, one queue per arm** -- deliberately not the twin-queue pattern the earlier
# experiments used. Those twinned because they wanted whichever generation freed up first; here the
# ask is three whole nodes, and the three that exist are 2x H100 + 1x H200, so each arm is pinned to
# a queue and the pinning is the allocation. Override with ARM_QUEUE_5=gpu_h100 etc.
#
# H100 and H200 differ in throughput, not in arithmetic -- both run bf16 and both arms are 8 ranks
# of batch 1 -- so which arm lands where changes when a number arrives, never what it is. The
# 5-cube arm takes the H200 because it is the control the other two are read against, and the one
# that also re-derives `banis_parity`'s 200k checkpoint as a sanity anchor.
#
# Stage B cannot name its predecessor's checkpoint at submission time: the run directory does not
# exist yet. Each B config carries a `PREV_CHECKPOINT` placeholder that the job resolves for
# itself, from *its own arm's* stage A -- which is what keeps the arms independent.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
PROJECT=miaai
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
STAGE=/nrs/scicompsoft/orhane/mia-train-scratch/data_scaling   # NOT /tmp: that is node-local
LOGS="$RUNS/jobs"
mkdir -p "$LOGS" "$STAGE/cmd" "$STAGE/smoke"

GPUS=8
SLOTS=96                  # the whole node; 8 GPUs x 12 slots/GPU on h100/h200
THREADS="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"

# Wall clock. Measured on 8 GPUs: 0.41 s/step interpolating, 0.47 s/step sub-pixel -> 22.8 h and
# 13.1 h. Requested with ~1.5x margin, which covers the H100/H200 spread and any queue-side
# slowdown; `-r` plus `--resume` means an over-run is a requeue from the last checkpoint, not a
# loss. Do not trim these to the estimate: a wall-clock kill at step 190k costs 22 h.
WALL_A=34:00
WALL_B=20:00

# Only used by --smoke, where no predecessor exists yet: a trained ViT-L of exactly stage B's
# expected shape (3D, interpolating head), so the `prefix = "model."` load path is really exercised.
SMOKE_ENCODER=/nrs/scicompsoft/orhane/mia-train-runs/banis_parity__finetune_256_long_20260810_123308/checkpoints/step_200000

# Per-arm queue. 2 whole H100 nodes and 1 whole H200 node were free at submission time.
declare -A ARM_QUEUE=( [1]=gpu_h100 [3]=gpu_h100 [5]=gpu_h200 )
for n in 1 3 5; do
  var="ARM_QUEUE_$n"; [[ -n "${!var:-}" ]] && ARM_QUEUE[$n]="${!var}"
done

SMOKE=0 DRY=0
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --smoke)   SMOKE=1 ;;
    --dry-run) DRY=1 ;;
    *) echo "unknown flag $1" >&2; exit 2 ;;
  esac
  shift
done
ARMS=("$@"); [[ ${#ARMS[@]} -eq 0 ]] && ARMS=(1 3 5)

jobid () { sed -n 's/^Job <\([0-9]*\)>.*/\1/p'; }

# stage <arm> <config> <queue> <wall> [<predecessor experiment_name>] [<dependency job id>]
# Prints the submitted job id.
stage () {
  local arm=$1 config=$2 queue=$3 wall=$4 prev=${5:-} dep=${6:-}
  local name; name=$(basename "$config" .toml)
  local tag="ds${arm}_$name" cfg="$config" prologue="" procs=$GPUS slots=$SLOTS

  if [[ $SMOKE -eq 1 ]]; then
    wall=0:30; procs=1; slots=12; queue=gpu_short
    cfg="$STAGE/smoke_$name.toml"
    sed -e "s/^experiment_name = .*/experiment_name = \"smoke_ds_$name\"/" \
        -e 's/^max_steps = .*/max_steps = 20/'          -e 's/^warmup_steps = .*/warmup_steps = 2/' \
        -e 's/^val_every = .*/val_every = 10/'          -e 's/^checkpoint_every = .*/checkpoint_every = 20/' \
        -e 's/^samples_per_epoch = .*/samples_per_epoch = 20/' -e 's/^dp_shard = .*/dp_shard = 1/' \
        -e 's/^num_workers = .*/num_workers = 2/' \
        "$config" > "$cfg"
    [[ -n "$prev" ]] && sed -i "s|PREV_CHECKPOINT|$SMOKE_ENCODER|" "$cfg"
  elif [[ -n "$prev" ]]; then
    local resolved="$STAGE/${name}_resolved.toml"
    # Resolved inside the job, after the predecessor has written checkpoints. Highest step picked
    # numerically -- `ls -t` would order by write time, which is only coincidentally the same, and
    # `sort` without -n puts step_90000 after step_200000.
    prologue="RUN=\$(ls -dt $RUNS/${prev}_*/ | head -1)
STEP=\$(ls -d \${RUN}checkpoints/step_* | sed 's|.*step_||' | sort -n | tail -1)
echo \"[stage B] initialising from \${RUN}checkpoints/step_\$STEP\"
test \"\$STEP\" -ge 200000   # stage A must have finished, not merely have written something
sed \"s|PREV_CHECKPOINT|\${RUN}checkpoints/step_\$STEP|\" '$cfg' > '$resolved'"
    cfg="$resolved"
  fi

  # The stage's work as its own script, so nothing has to survive several layers of shell quoting.
  local cmd="$STAGE/cmd/${tag}.sh"
  { echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "$THREADS"
    echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    [[ -n "$prologue" ]] && echo "$prologue"
    printf '%s --standalone --nproc_per_node=%s src/train.py --config %q %s\n' \
      "$VENV/bin/torchrun" "$procs" "$cfg" \
      "$([[ $SMOKE -eq 1 ]] && printf -- "--output-root %q" "$STAGE/smoke" || echo "--resume")"
  } > "$cmd"

  local bsub_args=(-P "$PROJECT" -q "$queue" -gpu "num=$procs" -n "$slots" -W "$wall" -r
                   -J "$tag" -cwd "$REPO"
                   -o "$LOGS/${tag}_%J.log" -e "$LOGS/${tag}_%J.err")
  [[ -n "$dep" ]] && bsub_args+=(-w "done($dep)")

  if [[ $DRY -eq 1 ]]; then
    printf 'bsub %s bash %q\n' "${bsub_args[*]}" "$cmd" >&2
    echo "DRYRUN"
  else
    bsub "${bsub_args[@]}" "bash '$cmd'" | jobid
  fi
}

for arm in "${ARMS[@]}"; do
  q=${ARM_QUEUE[$arm]:?no queue for arm $arm}
  a=$(stage "$arm" "$HERE/${arm}a_interp.toml"   "$q" "$WALL_A")
  b=$(stage "$arm" "$HERE/${arm}b_subpixel.toml" "$q" "$WALL_B" "ds__${arm}cube_interp" "$a")
  printf "arm %s cube(s)  %-9s  A(interp 200k)=%s  ->  B(subpixel 100k)=%s\n" "$arm" "$q" "$a" "$b"
done
