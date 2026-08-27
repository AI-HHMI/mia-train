#!/usr/bin/env bash
# Submit the arms of lmd_ssl_v1. Each arm is a chain of stages, held one behind the next.
#
#   bash experiments/lmd_ssl_v1/submit.sh              # every arm
#   bash experiments/lmd_ssl_v1/submit.sh 1 3          # just those arms
#   bash experiments/lmd_ssl_v1/submit.sh --dry-run    # print the bsub lines, submit nothing
#   bash experiments/lmd_ssl_v1/submit.sh --smoke      # 20 steps of every stage, one GPU
#
#   arm 1  DINOv3 random init  -> SimMIM 100k    -> finetune interp 50k -> subpixel 50k
#   arm 2  DINOv3 <- LVD ckpt  -> (no SSL)       -> finetune interp 50k -> subpixel 50k
#   arm 3  MuViT random init   -> MuViT-MAE 100k -> finetune interp 50k -> subpixel 50k
#
# One full node per stage (8 GPUs, 96 slots, dp_shard 8, batch 1/rank -> global batch 8).
#
# Arm 3 goes to gpu_h200 and the others to gpu_h100, for a reason beyond availability: arm 3 pins
# `attention_backend = "flash4"`, whose CuTeDSL kernels need Hopper or newer, and its 3-level
# sequence is 12288 tokens against 4096 -- roughly 9x the attention work. The H200's larger memory
# gives that the most headroom. Both queues satisfy the Hopper requirement, so H100 also works if
# H200 is busy: ARM_QUEUE_3=gpu_h100.
#
# A stage after the first cannot name its predecessor's checkpoint at submission time -- the run
# directory does not exist yet -- so each dependent config carries a `PREV_CHECKPOINT` placeholder
# that the job resolves for itself.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
PROJECT=miaai
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
STAGE=/nrs/scicompsoft/orhane/mia-train-scratch/lmd_ssl_v1   # NOT /tmp: that is node-local
LOGS="$RUNS/jobs"
mkdir -p "$LOGS" "$STAGE/cmd" "$STAGE/smoke"

GPUS=8
SLOTS=96
THREADS="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"

# Wall clock, all MEASURED on this experiment's own first steps (2026-08-24, 8 GPUs, global
# batch 8):
#
#   arm 1 SimMIM        52.3 samples/s   0.153 s/step   100k -> 4.3 h    (mfu 8.2%)
#   arm 2 finetune      27.4 samples/s   0.292 s/step    50k -> 4.1 h    (mfu 8.5%)
#   arm 3 MuViT-MAE     17.8 samples/s   0.448 s/step   100k -> 12.4 h   (mfu 2.7%)
#
# Arm 3 is 2.9x arm 1's per-step cost, not the ~9x its token count suggests: FlashAttention-4 is
# doing the work, and MAE encodes only the visible 25% of tokens, so the effective sequence is far
# shorter than the nominal 12288. Its low MFU is the thing left to understand, not its wall clock.
#
# The limits below keep 2-3x headroom over those figures deliberately: `-r` plus `--resume` makes an
# over-run a requeue from the last checkpoint, but a `-W` kill does NOT requeue, and losing 12 h of
# arm 3 to a tight limit costs far more than the scheduling priority a generous one gives up.
# The SSL walls are set from a MEASURED throughput, and it is not the one above. At global batch
# 256 over 16 ranks arm 1 ran at 80.1 samples/s with `data_wait_frac = 0.766` -- the GPUs sat idle
# three quarters of every step waiting on the loader, so this pipeline is I/O bound, not compute
# bound, and halving the ranks does not halve the per-step time. Expect ~40 samples/s at 8 ranks:
#
#   arm 1  100k steps x 128 samples = 12.8M samples / ~40 per s  ~= 89 h
#   arm 3  unmeasured at this batch, and it reads THREE levels per sample against arm 1's one, so
#          its I/O per sample is ~3x and it is very unlikely to be faster
#
# 168 h therefore, not 48. A `-W` kill does not requeue, and checkpoints land only every 10k steps,
# so a tight limit trades scheduling priority for lost work. Note the step budget itself is the
# thing worth revisiting: at this batch 100k steps is 16x the data the original batch-8 plan
# budgeted, and ~27k steps would give a ~24 h turnaround with 3.5M samples still seen.
WALL_SSL_D=168:00
WALL_FT_D=10:00
WALL_SSL_M=168:00
WALL_FT_M=20:00

declare -A ARM_QUEUE=( [1]=gpu_h100 [2]=gpu_h100 [3]=gpu_h200 )

# The SSL stage needs a bigger card than the finetune stages, so it gets its own queue. Both SSL
# arms carry 16 samples per rank, and both measured over an H100's 80 GiB at that batch:
#
#   arm 1  SimMIM      peaks at 80.0 GiB of 139.8 on an H200 -- an H100 card is 79.2 GiB usable
#   arm 3  MuViT-MAE   OOMs on an H100 at 16/rank (max 12 there without checkpointing)
#
# The finetune stages run at global batch 8 and are nowhere near either limit, so they stay on
# whatever ARM_QUEUE says and leave the scarcer H200s free.
declare -A ARM_SSL_QUEUE=( [1]=gpu_h200 [3]=gpu_h200 )
for n in 1 3; do
  var="ARM_SSL_QUEUE_$n"; [[ -n "${!var:-}" ]] && ARM_SSL_QUEUE[$n]="${!var}"
done

# No stage is multi-node any more. Both SSL arms run at global batch 128 (16/rank x 8 ranks) on ONE
# node, which takes an ordinary queue rather than a `*_parallel` one and schedules far sooner.
# The machinery is kept because it is the only thing that would have to come back if the batch is
# raised again; set MULTINODE_STAGES to e.g. "1a 3a" to re-enable it.
MULTINODE_STAGES=""
MN_QUEUE=${MN_QUEUE:-gpu_h200_parallel}
MN_NODES=2
for n in 1 2 3; do
  var="ARM_QUEUE_$n"; [[ -n "${!var:-}" ]] && ARM_QUEUE[$n]="${!var}"
done

# A trained encoder of each shape, so --smoke can exercise the PREV_CHECKPOINT load path before any
# predecessor exists. Only the DINOv3 one exists today; arm 3's smoke stages skip [init].
SMOKE_DINOV3=/nrs/scicompsoft/orhane/mia-train-runs/banis_parity__finetune_256_long_20260810_123308/checkpoints/step_200000

SMOKE=0 DRY=0
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --smoke)   SMOKE=1 ;;
    --dry-run) DRY=1 ;;
    *) echo "unknown flag $1" >&2; exit 2 ;;
  esac
  shift
done
ARMS=("$@"); [[ ${#ARMS[@]} -eq 0 ]] && ARMS=(1 2 3)

jobid () { sed -n 's/^Job <\([0-9]*\)>.*/\1/p'; }

# stage <config> <queue> <wall> [<predecessor experiment_name>] [<dependency job id>]
stage () {
  local config=$1 queue=$2 wall=$3 prev=${4:-} dep=${5:-}
  local name; name=$(basename "$config" .toml)
  local cfg="$config" prologue="" procs=$GPUS slots=$SLOTS

  # Decided up front: both the command written into $cmd and the bsub arguments depend on it.
  # Multi-node stages go through deploy/lsf/launch_multinode.sh, which reads LSF's host list and
  # starts one torchrun per node under a c10d rendezvous. `blaunch` gives each node a FRESH shell,
  # so that launcher forwards the environment explicitly -- nothing exported here survives on its
  # own.
  local multinode=0
  [[ " $MULTINODE_STAGES " == *" ${name:0:2} "* ]] && multinode=1
  # A smoke run is one GPU on one node, so it never takes the multi-node path.
  [[ $SMOKE -eq 1 ]] && multinode=0

  if [[ $SMOKE -eq 1 ]]; then
    wall=0:30; procs=1; slots=12
    cfg="$STAGE/smoke_$name.toml"
    sed -e "s/^experiment_name = .*/experiment_name = \"smoke_$name\"/" \
        -e 's/^max_steps = .*/max_steps = 20/'   -e 's/^warmup_steps = .*/warmup_steps = 2/' \
        -e 's/^val_every = .*/val_every = 10/'   -e 's/^checkpoint_every = .*/checkpoint_every = 20/' \
        -e 's/^samples_per_epoch = .*/samples_per_epoch = 20/' -e 's/^dp_shard = .*/dp_shard = 1/' \
        -e 's/^num_workers = .*/num_workers = 2/' \
        "$config" > "$cfg"
    if [[ -n "$prev" ]]; then
      if [[ "$name" == 1* || "$name" == 2* ]]; then
        sed -i "s|PREV_CHECKPOINT|$SMOKE_DINOV3|" "$cfg"
      else
        # No MuViT checkpoint of this shape exists yet; drop [init] so the smoke run still
        # exercises the model, algorithm and data wiring.
        sed -i '/^path = "PREV_CHECKPOINT"$/d; /^prefix = "model."$/d' "$cfg"
      fi
    fi
  elif [[ -n "$prev" ]]; then
    local resolved="$STAGE/${name}_resolved.toml"
    # Resolved inside the job, after the predecessor has written checkpoints. Highest step chosen
    # numerically: `sort` without -n puts step_90000 after step_100000.
    prologue="RUN=\$(ls -dt $RUNS/${prev}_*/ | head -1)
STEP=\$(ls -d \${RUN}checkpoints/step_* | sed 's|.*step_||' | sort -n | tail -1)
echo \"initialising from \${RUN}checkpoints/step_\$STEP\"
sed \"s|PREV_CHECKPOINT|\${RUN}checkpoints/step_\$STEP|\" '$cfg' > '$resolved'"
    cfg="$resolved"
  fi

  local cmd="$STAGE/cmd/${name}.sh"
  { echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "$THREADS"
    echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    [[ -n "$prologue" ]] && echo "$prologue"
    local tail_args
    tail_args="$([[ $SMOKE -eq 1 ]] && printf -- "--output-root %q" "$STAGE/smoke" || echo "--resume")"
    if [[ $multinode -eq 1 ]]; then
      printf 'MIA_TRAIN=%q VENV=%q %q %s %q %s\n' \
        "$REPO" "$VENV" "$REPO/deploy/lsf/launch_multinode.sh" "$procs" "$cfg" "$tail_args"
    else
      printf '%s --standalone --nproc_per_node=%s src/train.py --config %q %s\n' \
        "$VENV/bin/torchrun" "$procs" "$cfg" "$tail_args"
    fi
  } > "$cmd"

  local args
  if [[ $multinode -eq 1 ]]; then
    args=(-P "$PROJECT" -q "$MN_QUEUE" -app "parallel-96" -gpu "num=$procs:mode=shared"
          -n "$((slots * MN_NODES))" -W "$wall" -r
          -J "lmd1_$name" -cwd "$REPO"
          -o "$LOGS/lmd1_${name}_%J.log" -e "$LOGS/lmd1_${name}_%J.err")
  else
    args=(-P "$PROJECT" -q "$queue" -gpu "num=$procs" -n "$slots" -W "$wall" -r
          -J "lmd1_$name" -cwd "$REPO"
          -o "$LOGS/lmd1_${name}_%J.log" -e "$LOGS/lmd1_${name}_%J.err")
  fi
  [[ -n "$dep" && "$dep" != "DRYRUN" ]] && args+=(-w "done($dep)")

  if [[ $DRY -eq 1 ]]; then
    printf 'bsub %s bash %q\n' "${args[*]}" "$cmd" >&2
    echo "DRYRUN"
  else
    bsub "${args[@]}" "bash '$cmd'" | jobid
  fi
}

for arm in "${ARMS[@]}"; do
  q=${ARM_QUEUE[$arm]:?no queue for arm $arm}
  case "$arm" in
    1) tag=dinov3_simmim; wssl=$WALL_SSL_D; wft=$WALL_FT_D ;;
    2) tag=dinov3_lvd;    wssl="";          wft=$WALL_FT_D ;;
    3) tag=muvit_mae;     wssl=$WALL_SSL_M; wft=$WALL_FT_M ;;
    *) echo "unknown arm $arm" >&2; exit 2 ;;
  esac

  prev_name="" prev_job=""
  if [[ -n "$wssl" ]]; then
    a=$(stage "$HERE/${arm}a_${tag}_pretrain.toml" "${ARM_SSL_QUEUE[$arm]:-$q}" "$wssl")
    prev_name="lmd1__${arm}a_${tag}_pretrain"; prev_job="$a"
    printf "arm %s  %-9s  A(ssl 100k)=%s" "$arm" "${ARM_SSL_QUEUE[$arm]:-$q}" "$a"
  else
    a="-"
    printf "arm %s  %-9s  A(none)     =%s" "$arm" "$q" "$a"
  fi

  b=$(stage "$HERE/${arm}b_${tag}_finetune_interpolate.toml" "$q" "$wft" "$prev_name" "$prev_job")
  c=$(stage "$HERE/${arm}c_${tag}_finetune_subpixel.toml" "$q" "$wft" \
            "lmd1__${arm}b_${tag}_ft_interpolate" "$b")
  printf "  ->  B(interp 50k)=%s  ->  C(subpixel 50k)=%s\n" "$b" "$c"
done
