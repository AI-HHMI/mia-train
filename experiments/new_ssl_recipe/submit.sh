#!/usr/bin/env bash
# Submit the arms of new_ssl_recipe. Each arm is a chain of stages, held one behind the next.
#
#   bash experiments/new_ssl_recipe/submit.sh              # every arm
#   bash experiments/new_ssl_recipe/submit.sh 2 3          # just those arms
#   bash experiments/new_ssl_recipe/submit.sh --smoke 2    # 20 steps of arm 2's stages, one GPU
#   bash experiments/new_ssl_recipe/submit.sh --stage 2b_ssl_twophase_interp   # one stage, see below
#
#   1  control  : finetune interpolating 100k -> sub-pixel 100k
#   2  two-phase: SimMIM 200k with a 20k frozen warm-up -> interpolating 100k -> sub-pixel 100k
#   3  joint    : SimMIM 200k, no freeze             -> interpolating 100k -> sub-pixel 100k
#   4  joint@32 : SimMIM 100k at global batch 32     -> interpolating 100k -> sub-pixel 100k
#
# All four start from the released DINOv3 **ViT-L/16** checkpoint, so 2 vs 3 isolates the frozen
# warm-up and 3 vs 1 isolates the SSL stage itself. See README.md.
#
# Arm 4 differs from the others in two ways at once and is not a drop-in comparison. It runs
# **every** stage at global batch 32 (local 2 x 16 ranks, i.e. 2 whole nodes) where arms 1-3 run
# global batch 8, and its SSL stage is 100k steps rather than 200k -- 3.2M samples against arm 3's
# 1.6M. Arms 2 and 3 were stopped after their SSL stages plateaued inside ~10k steps with val loss
# drifting up; arm 4 is the response. Because its finetune batch also differs from arm 1's, a
# batch-32 version of arm 1 is still owed before arm 4's result can be read as "SSL helped".
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
# A stage after the first cannot name its predecessor's checkpoint at submission time -- the run
# directory does not exist yet -- so each dependent config carries a `PREV_CHECKPOINT` placeholder
# that the job resolves for itself.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
PROJECT=miaai
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
STAGE=/nrs/scicompsoft/orhane/mia-train-scratch/new_ssl_recipe
LOGS="$RUNS/jobs"
CLAIM="$STAGE/claim.sh"
mkdir -p "$LOGS" "$STAGE/locks" "$STAGE/cmd"

# The twin lock is `init_comparison`'s, unmodified: it takes a lock directory and a command script
# and knows nothing about either experiment. Copied once rather than referenced across experiments,
# so this one keeps working if init_comparison's scratch directory is ever cleaned.
if [[ ! -f "$CLAIM" ]]; then
  SRC_CLAIM=/nrs/scicompsoft/orhane/mia-train-scratch/init_comparison/claim.sh
  [[ -f "$SRC_CLAIM" ]] || { echo "missing claim.sh; expected one at $SRC_CLAIM to copy" >&2; exit 2; }
  cp "$SRC_CLAIM" "$CLAIM"
fi

GPUS=8
SLOTS=96
THREADS="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"

# miao is an editable install pointing at a working tree, so a `git checkout` there rewrites the
# dataset code under every running job. That is not hypothetical: on 2026-08-13 a switch away from
# `feature/aug-rot-anisotropic` killed all four live runs within 60 seconds.
#
# These stages no longer pin a miao commit. They used to, because the data configs asked for
# `aug_rot: "inplane"`, a per-volume key only that branch had. Rotation is now `[augment] rotate`
# in the run's own .toml and the augmentations are implemented in this repo (src/data/augment.py),
# so the only miao surface left is the dataset reader -- `MiaoConfig`, `load_config`,
# `VolumeDataset` -- which master provides. A checkout in miao can still disturb a running job,
# but no longer changes what augmentation a run applies.

QUEUES=(gpu_h100 gpu_h200)
[[ -n "${QUEUE:-}" && "${QUEUE:-}" != "auto" ]] && QUEUES=("$QUEUE")

SMOKE=0
[[ "${1:-}" == "--smoke" ]] && { SMOKE=1; shift; }
ONE_STAGE=""
[[ "${1:-}" == "--stage" ]] && { ONE_STAGE=$2; shift 2; }

jobid () { sed -n 's/^Job <\([0-9]*\)>.*/\1/p'; }

# `done()` requires a zero exit, which is why a losing twin exits 42: an `||` of the pair then
# fires only when the twin that actually trained finishes.
dep_of () { local d=""; for id in "$@"; do d+="${d:+ || }done($id)"; done; echo "$d"; }

# stage <config> <wall clock> [<predecessor experiment_name> <dependency expression>]
# Prints the job ids of the twins it submitted.
# Set MULTINODE=1 before a `stage` call to run it across two whole nodes instead of one. Arm 4 is
# the only user: its stages carry `dp_shard = 16` at local batch 2, which needs 16 ranks and so
# cannot fit the single-node `--standalone` path the other arms use. Everything else about a stage
# -- the twin lock, PREV_CHECKPOINT resolution, the cmd script -- is identical either way.
#
# Two constraints that are not negotiable, both from deploy/lsf/README.md:
#   * `*_parallel` queues allocate WHOLE nodes; a partial request there wastes the remainder and
#     disables CPU fencing. 2 H100 nodes is `-n 192` at 96 slots each, `-app parallel-96`.
#   * H100 and H200 sit on SEPARATE InfiniBand fabrics with no path between them, so a job must
#     stay inside one generation. The twins still work -- each is a self-contained 2-node job on
#     one queue -- but a single job may never span both.
MULTINODE=${MULTINODE:-0}
NODES=${NODES:-2}

stage () {
  local config=$1 wall=$2 prev=${3:-} dep=${4:-}
  local name; name=$(basename "$config" .toml)
  local cfg="$config" prologue="" procs=$GPUS slots=$SLOTS
  local stage_multinode=$MULTINODE
  MULTINODE=0   # one-shot: set it immediately before each multi-node stage, never sticky
  # Names the lock, the command script, the job and its logs. Smoke runs get their own, so a
  # 20-step check submitted while the real chain is queued cannot take the real stage's lock and
  # send it home with exit 42.
  local tag=$name

  if [[ $SMOKE -eq 1 ]]; then
    tag="smoke_$name"
    wall=0:30; procs=1; slots=12
    cfg="$STAGE/$tag.toml"
    # `freeze_backbone_steps` is rewritten too, and not only for tidiness: the trainer rejects a
    # config whose freeze outlasts the run, so arm 2's 20000 against max_steps 20 would abort at
    # startup. 2 of 20 keeps the smoke run's 10% proportion, and exercises the unfreeze boundary.
    sed -e "s/^experiment_name = .*/experiment_name = \"$tag\"/" \
        -e 's/^max_steps = .*/max_steps = 20/' -e 's/^warmup_steps = .*/warmup_steps = 2/' \
        -e 's/^freeze_backbone_steps = .*/freeze_backbone_steps = 2/' \
        -e 's/^unfreeze_warmup_steps = .*/unfreeze_warmup_steps = 2/' \
        -e 's/^val_every = .*/val_every = 10/' -e 's/^checkpoint_every = .*/checkpoint_every = 20/' \
        -e 's/^samples_per_epoch = .*/samples_per_epoch = 20/' -e 's/^dp_shard = .*/dp_shard = 1/' \
        "$config" > "$cfg"
  fi

  if [[ -n "$prev" ]]; then
    local resolved="$STAGE/${tag}_resolved.toml"
    local glob="$RUNS/sslrec__${prev}_*/"
    # A smoke run resolves against its own chain rather than against a stand-in encoder the way
    # init_comparison did. Borrowing one of its ViT-L encoders would now load -- this experiment is
    # ViT-L too -- but it would test the wrong thing: a smoke run exists to check that *this*
    # chain's own handoff works, and a borrowed checkpoint is exactly the step that would not be
    # exercised. The chain's `done()` dependency is what makes it safe -- the predecessor has
    # written its step-20 checkpoint before this job starts.
    [[ $SMOKE -eq 1 ]] && glob="$STAGE/smoke/smoke_${prev}_*/"
    # Resolved in the job, after the predecessor has written checkpoints. Highest step chosen
    # numerically, not by modification time, which would only be a coincidence of write order.
    prologue="RUN=\$(ls -dt $glob | head -1)
STEP=\$(ls -d \${RUN}checkpoints/step_* | sed 's|.*step_||' | sort -n | tail -1)
echo \"initialising from \${RUN}checkpoints/step_\$STEP\"
sed \"s|PREV_CHECKPOINT|\${RUN}checkpoints/step_\$STEP|\" '$cfg' > '$resolved'"
    cfg="$resolved"
  fi

  # The stage's work as its own script, so the twins share one definition and nothing has to
  # survive several layers of shell quoting.
  local cmd="$STAGE/cmd/${tag}.sh"
  { echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "$THREADS"
    echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    [[ -n "$prologue" ]] && echo "$prologue"
    if [[ ${stage_multinode:-0} -eq 1 ]]; then
      # launch_multinode.sh reads LSB_MCPU_HOSTS itself and blaunches one torchrun per node, so
      # this passes GPUs-per-node rather than a total.
      printf 'MIA_TRAIN=%q VENV=%q %q/deploy/lsf/launch_multinode.sh %s %q %s\n' \
        "$REPO" "$VENV" "$REPO" "$procs" "$cfg" \
        "$([[ $SMOKE -eq 1 ]] && printf -- "--output-root %q" "$STAGE/smoke" || echo "--resume")"
    else
      printf '%s --standalone --nproc_per_node=%s src/train.py --config %q %s\n' \
        "$VENV/bin/torchrun" "$procs" "$cfg" \
        "$([[ $SMOKE -eq 1 ]] && printf -- "--output-root %q" "$STAGE/smoke" || echo "--resume")"
    fi
  } > "$cmd"

  local ids=() q id
  local queues=("${QUEUES[@]}") extra=()
  if [[ ${stage_multinode:-0} -eq 1 ]]; then
    # Whole-node queues, whole-node slot count, and `mode=shared` so all 8 GPUs on a node are
    # visible to the one torchrun launched there.
    queues=(gpu_h100_parallel gpu_h200_parallel)
    extra=(-app "parallel-$SLOTS" -gpu "num=$procs:mode=shared")
    slots=$((SLOTS * NODES))
  else
    extra=(-gpu "num=$procs")
  fi
  for q in "${queues[@]}"; do
    id=$(bsub -P "$PROJECT" -q "$q" "${extra[@]}" -n "$slots" -W "$wall" -r \
      ${dep:+-w "$dep"} -J "sslrec_$tag" -cwd "$REPO" \
      -o "$LOGS/sslrec_${tag}_%J.log" -e "$LOGS/sslrec_${tag}_%J.err" \
      "bash '$CLAIM' '$STAGE/locks/$tag' '$cmd'" | jobid)
    ids+=("$id")
  done
  echo "${ids[@]}"
}

report () { printf "%-26s jobs %s\n" "$1" "$2"; }

# One stage on its own, e.g. to convert an already-queued stage to a twin pair. DEPENDS_ON is a
# dependency expression or a bare job id.
if [[ -n "$ONE_STAGE" ]]; then
  cfg="$HERE/${ONE_STAGE}.toml"
  [[ -f "$cfg" ]] || { echo "no config $cfg" >&2; exit 2; }
  prev=${PREV_STAGE:-}
  dep=${DEPENDS_ON:-}
  [[ "$dep" =~ ^[0-9]+$ ]] && dep="done($dep)"
  report "$ONE_STAGE" "$(stage "$cfg" "${WALL:-20:00}" "$prev" "$dep")"
  exit 0
fi

# Wall times come from the slowest per-step rate init_comparison's own ViT-L runs measured for each
# stage type -- 0.41 s/step interpolating (1a/2a/3b/4a spanned 0.401-0.408), 0.47 s/step sub-pixel
# (3c/1b/2b/4b spanned 0.440-0.468), 0.17 s/step SimMIM (3a) -- times this experiment's step counts:
# ~9.4 h for a 200k SSL stage, ~11.4 h for a 100k interpolating finetune, ~13.0 h for a 100k
# sub-pixel one. Taking the slowest rather than the mean is the point: these spreads are node and
# input-pipeline variation, not model differences, so a run can land anywhere in the range. Every request keeps a wide margin on that, and the margin is
# cheap in both directions: `-r` below marks the job requeuable and `--resume` makes the requeued
# job continue from its last checkpoint, so hitting the wall costs a requeue rather than the run,
# while an overestimate only makes the stage harder to backfill.
ARMS=("$@"); [[ ${#ARMS[@]} -gt 0 ]] || ARMS=(1 2 3)
for arm in "${ARMS[@]}"; do
  case $arm in
    1) a=$(stage "$HERE/1a_dinov3_interp.toml" 20:00); report 1a_dinov3_interp "$a"
       b=$(stage "$HERE/1b_dinov3_subpixel.toml" 20:00 1a_dinov3_interp "$(dep_of $a)"); report 1b_dinov3_subpixel "$b" ;;
    2) a=$(stage "$HERE/2a_ssl_twophase.toml" 18:00); report 2a_ssl_twophase "$a"
       b=$(stage "$HERE/2b_ssl_twophase_interp.toml" 20:00 2a_ssl_twophase "$(dep_of $a)"); report 2b_ssl_twophase_interp "$b"
       c=$(stage "$HERE/2c_ssl_twophase_subpixel.toml" 20:00 2b_ssl_twophase_interp "$(dep_of $b)"); report 2c_ssl_twophase_subpixel "$c" ;;
    3) a=$(stage "$HERE/3a_ssl_joint.toml" 18:00); report 3a_ssl_joint "$a"
       b=$(stage "$HERE/3b_ssl_joint_interp.toml" 20:00 3a_ssl_joint "$(dep_of $a)"); report 3b_ssl_joint_interp "$b"
       c=$(stage "$HERE/3c_ssl_joint_subpixel.toml" 20:00 3b_ssl_joint_interp "$(dep_of $b)"); report 3c_ssl_joint_subpixel "$c" ;;
    4) # Every stage of arm 4 is 2 nodes x 8 GPUs = 16 ranks at local batch 2 (global 32). The
       # finetune stages match 4a's batch rather than arm 1's, so arm 4 is internally consistent
       # but NOT directly comparable to arm 1 -- a batch-32 control is still owed. Walls are arm
       # 1-3's halved and rounded up, since 16 ranks at batch 2 do 4x the samples per step.
       MULTINODE=1; a=$(stage "$HERE/4a_ssl_joint_b32.toml" 12:00); report 4a_ssl_joint_b32 "$a"
       MULTINODE=1; b=$(stage "$HERE/4b_ssl_b32_interp.toml" 16:00 4a_ssl_joint_b32 "$(dep_of $a)"); report 4b_ssl_b32_interp "$b"
       MULTINODE=1; c=$(stage "$HERE/4c_ssl_b32_subpixel.toml" 16:00 4b_ssl_b32_interp "$(dep_of $b)"); report 4c_ssl_b32_subpixel "$c" ;;
    *) echo "unknown arm: $arm (expected 1, 2, 3 or 4)" >&2; exit 2 ;;
  esac
done
