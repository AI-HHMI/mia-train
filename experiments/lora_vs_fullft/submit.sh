#!/usr/bin/env bash
# Submit the arms of lora_vs_fullft. Each arm is two stages, the second held behind the first.
#
#   bash experiments/lora_vs_fullft/submit.sh              # both arms
#   bash experiments/lora_vs_fullft/submit.sh 2            # just the control
#   bash experiments/lora_vs_fullft/submit.sh --smoke 1    # 20 steps of arm 1's stages, one GPU
#   bash experiments/lora_vs_fullft/submit.sh --stage 1b_lora_subpixel   # one stage, see below
#
#   1  LoRA : DINOv3 ViT-L/16 -> interpolating 50k -> sub-pixel 50k, encoder adapted at rank 16
#   2  full : DINOv3 ViT-L/16 -> interpolating 50k -> sub-pixel 50k, all 306.6M params trained
#
# The two arms are the same four files with one section added, so 1 vs 2 isolates the fine-tuning
# method and nothing else. See README.md, including the learning-rate caveat -- both arms run at the
# same rate, which is the only way the comparison stays single-variable and is also a known handicap
# for the LoRA arm.
#
# **Queue-agnostic by default.** Every stage is submitted twice, once to gpu_h100 and once to
# gpu_h200, so it starts on whichever generation frees up first rather than waiting on one. The two
# twins share an `experiment_name` and both pass `--resume`, which makes them interchangeable --
# either can continue the same run directory from its last checkpoint -- so the only thing that has
# to be prevented is both running at once. `claim.sh` does that with an atomic lock; the loser exits
# 42 and releases its node within seconds.
#
# Neither tidier mechanism exists here: a queue list (`-q "gpu_h100 gpu_h200"`) is rejected by the
# site's esub, and the only queues spanning both host groups (`gpu_short`, `short`) cap at 60
# minutes. Set QUEUE=gpu_h200 to force a single queue instead of a pair.
#
# A second stage cannot name its predecessor's checkpoint at submission time -- the run directory
# does not exist yet -- so each dependent config carries a `PREV_CHECKPOINT` placeholder that the
# job resolves for itself once the dependency has fired.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
PROJECT=miaai
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
STAGE=/nrs/scicompsoft/orhane/mia-train-scratch/lora_vs_fullft
LOGS="$RUNS/jobs"
CLAIM="$STAGE/claim.sh"
mkdir -p "$LOGS" "$STAGE/locks" "$STAGE/cmd"

# The twin lock takes a lock directory and a command script and knows nothing about either
# experiment. Copied rather than referenced across experiments, so this one keeps working if another
# experiment's scratch directory is ever cleaned.
if [[ ! -f "$CLAIM" ]]; then
  for src in /nrs/scicompsoft/orhane/mia-train-scratch/{new_ssl_recipe,init_comparison}/claim.sh; do
    [[ -f "$src" ]] && { cp "$src" "$CLAIM"; break; }
  done
  [[ -f "$CLAIM" ]] || { echo "missing claim.sh; expected one to copy under mia-train-scratch" >&2; exit 2; }
fi

GPUS=8
SLOTS=96          # 12 slots per GPU, the H100/H200 queues' ratio -- see .claude/rules/cluster.md
THREADS="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"

# miao is an editable install pointing at a working tree, so a `git checkout` there rewrites the
# dataset code under every running job -- on 2026-08-13 that killed four live runs in 60 seconds.
# These stages pin no miao commit: rotation is `[augment] rotate` in the run's own .toml and the
# augmentations live in src/data/augment.py, so the only miao surface left is the dataset reader.

QUEUES=(gpu_h100 gpu_h200)
[[ -n "${QUEUE:-}" && "${QUEUE:-}" != "auto" ]] && QUEUES=("$QUEUE")

SMOKE=0
[[ "${1:-}" == "--smoke" ]] && { SMOKE=1; shift; }
ONE_STAGE=""
[[ "${1:-}" == "--stage" ]] && { ONE_STAGE=$2; shift 2; }

jobid () { sed -n 's/^Job <\([0-9]*\)>.*/\1/p'; }

# `done()` requires a zero exit, which is why a losing twin exits 42: an `||` of the pair fires only
# when the twin that actually trained finishes.
dep_of () { local d=""; for id in "$@"; do d+="${d:+ || }done($id)"; done; echo "$d"; }

# stage <config> <wall clock> [<predecessor experiment_name suffix> <dependency expression>]
# Prints the job ids of the twins it submitted.
stage () {
  local config=$1 wall=$2 prev=${3:-} dep=${4:-}
  local name; name=$(basename "$config" .toml)
  local cfg="$config" prologue="" procs=$GPUS slots=$SLOTS
  # Names the lock, the command script, the job and its logs. Smoke runs get their own, so a
  # 20-step check submitted while the real chain is queued cannot take the real stage's lock and
  # send it home with exit 42.
  local tag=$name

  if [[ $SMOKE -eq 1 ]]; then
    tag="smoke_$name"
    wall=0:30; procs=1; slots=12
    cfg="$STAGE/$tag.toml"
    # `dp_shard = 1` for the single GPU, and the cadences scaled so a 20-step run still validates
    # a checkpoint write and a validation pass. Nothing about [lora] is rewritten -- rank and alpha
    # are what the real run uses, which is the point of smoking it.
    sed -e "s/^experiment_name = .*/experiment_name = \"$tag\"/" \
        -e 's/^max_steps = .*/max_steps = 20/' -e 's/^warmup_steps = .*/warmup_steps = 2/' \
        -e 's/^val_every = .*/val_every = 10/' -e 's/^checkpoint_every = .*/checkpoint_every = 20/' \
        -e 's/^samples_per_epoch = .*/samples_per_epoch = 20/' -e 's/^dp_shard = .*/dp_shard = 1/' \
        "$config" > "$cfg"
  fi

  if [[ -n "$prev" ]]; then
    local resolved="$STAGE/${tag}_resolved.toml"
    local glob="$RUNS/lvf__${prev}_*/"
    # A smoke run resolves against its own chain, not a borrowed encoder: the handoff is exactly the
    # step a smoke run exists to exercise, and for arm 1 it is also where the adapter has to survive
    # a save/load round trip. The `done()` dependency makes it safe -- the predecessor has written
    # its step-20 checkpoint before this job starts.
    [[ $SMOKE -eq 1 ]] && glob="$STAGE/smoke/smoke_${prev}_*/"
    # Resolved in the job, after the predecessor has written checkpoints. Highest step chosen
    # numerically, not by modification time, which would only be a coincidence of write order.
    prologue="RUN=\$(ls -dt $glob | head -1)
STEP=\$(ls -d \${RUN}checkpoints/step_* | sed 's|.*step_||' | sort -n | tail -1)
echo \"initialising from \${RUN}checkpoints/step_\$STEP\"
sed \"s|PREV_CHECKPOINT|\${RUN}checkpoints/step_\$STEP|\" '$cfg' > '$resolved'"
    cfg="$resolved"
  fi

  # The stage's work as its own script, so the twins share one definition and nothing has to survive
  # several layers of shell quoting.
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

  local ids=() q id
  for q in "${QUEUES[@]}"; do
    id=$(bsub -P "$PROJECT" -q "$q" -gpu "num=$procs" -n "$slots" -W "$wall" -r \
      ${dep:+-w "$dep"} -J "lvf_$tag" -cwd "$REPO" \
      -o "$LOGS/lvf_${tag}_%J.log" -e "$LOGS/lvf_${tag}_%J.err" \
      "bash '$CLAIM' '$STAGE/locks/$tag' '$cmd'" | jobid)
    ids+=("$id")
  done
  echo "${ids[@]}"
}

report () { printf "%-24s jobs %s\n" "$1" "$2"; }

# One stage on its own, e.g. to resubmit a stage that lost both twins to a node failure. DEPENDS_ON
# is a dependency expression or a bare job id; PREV_STAGE names the predecessor to resolve against.
if [[ -n "$ONE_STAGE" ]]; then
  cfg="$HERE/${ONE_STAGE}.toml"
  [[ -f "$cfg" ]] || { echo "no config $cfg" >&2; exit 2; }
  dep=${DEPENDS_ON:-}
  [[ "$dep" =~ ^[0-9]+$ ]] && dep="done($dep)"
  report "$ONE_STAGE" "$(stage "$cfg" "${WALL:-10:00}" "${PREV_STAGE:-}" "$dep")"
  exit 0
fi

# Wall times from the slowest per-step rate this repo's ViT-L runs have measured for each stage type
# -- 0.41 s/step interpolating, 0.47 s/step sub-pixel -- times 50k steps: ~5.7 h and ~6.5 h. LoRA
# adds about 2% (two thin matmuls per adapted projection against a base matmul 48x their size) and
# saves no compute at all, so both arms get the same request. 10 h keeps a wide margin, and the
# margin is cheap both ways: `-r` marks the job requeuable and `--resume` makes the requeued job
# continue from its last checkpoint, so hitting the wall costs a requeue rather than the run, while
# an overestimate only makes the stage harder to backfill.
ARMS=("$@"); [[ ${#ARMS[@]} -gt 0 ]] || ARMS=(1 2)
for arm in "${ARMS[@]}"; do
  case $arm in
    1) a=$(stage "$HERE/1a_lora_interp.toml" 10:00); report 1a_lora_interp "$a"
       b=$(stage "$HERE/1b_lora_subpixel.toml" 10:00 1a_lora_interp "$(dep_of $a)"); report 1b_lora_subpixel "$b" ;;
    2) a=$(stage "$HERE/2a_full_interp.toml" 10:00); report 2a_full_interp "$a"
       b=$(stage "$HERE/2b_full_subpixel.toml" 10:00 2a_full_interp "$(dep_of $a)"); report 2b_full_subpixel "$b" ;;
    *) echo "unknown arm: $arm (expected 1 or 2)" >&2; exit 2 ;;
  esac
done
