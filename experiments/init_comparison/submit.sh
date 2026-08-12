#!/usr/bin/env bash
# Submit the arms of init_comparison. Each arm is a chain of stages, held one behind the next.
#
#   bash experiments/init_comparison/submit.sh            # every arm
#   bash experiments/init_comparison/submit.sh 2 4        # just those arms
#   bash experiments/init_comparison/submit.sh --smoke 1  # 20 steps of each of arm 1's stages
#
#   1  scratch : interpolating 150k -> sub-pixel 100k
#   2  dinov3  : interpolating 150k -> sub-pixel 100k
#   3  simmim  : SimMIM 100k -> interpolating 150k -> sub-pixel 100k
#   4  dinov3 + augmentation : interpolating 150k -> sub-pixel 100k
#
# A stage after the first cannot name its predecessor's checkpoint at submission time -- the run
# directory does not exist yet -- so each dependent config carries a `PREV_CHECKPOINT` placeholder
# that the job resolves for itself. That is what makes a whole chain safe to submit in one command.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
PROJECT=miaai
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
STAGE=/nrs/scicompsoft/orhane/mia-train-scratch/init_comparison
LOGS="$RUNS/jobs"
mkdir -p "$LOGS" "$STAGE"

# Either GPU generation serves this experiment: single-node jobs, so the separate H100/H200
# InfiniBand fabrics never come into it. `auto` picks whichever queue has the shorter backlog, since
# which one is congested changes hour to hour. A queue list (`-q "a b"`) is rejected by the site's
# esub, and the only queue spanning both host groups is `gpu_short`, capped at one hour.
QUEUE=${QUEUE:-auto}
GPUS=8
SLOTS=96
THREADS="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4;"
THREADS="$THREADS export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;"

# A trained ViT-L encoder of this exact shape, used only to exercise the phase-2 load path in a
# smoke run, where none of the run's own predecessors exists yet.
SMOKE_ENCODER=/nrs/scicompsoft/orhane/mia-train-runs/banis_parity__finetune_256_long_20260810_123308/checkpoints/step_300000

# Which queue can start an 8-GPU job *now*. The number of pending jobs is a poor guide -- a queue
# with a long backlog can still have an idle node, and one with no backlog can be completely full --
# so this asks the host groups how many slots are actually free on their emptiest host. A job needs
# 96 on a single host, and picking by backlog once left five jobs queued behind 480 others while a
# whole node of the other generation sat idle.
pick_queue () {
  local best="" best_free=-1 q grp free
  for pair in "gpu_h100:h100s" "gpu_h200:h200s"; do
    q=${pair%%:*}; grp=${pair##*:}
    free=$(bhosts "$grp" 2>/dev/null |
           awk 'NR>1 && $4 ~ /^[0-9]+$/ {d = $4 - $5; if (d > m) m = d} END {print m + 0}')
    [[ "$free" =~ ^[0-9]+$ ]] || continue
    if (( free > best_free )); then best=$q; best_free=$free; fi
  done
  [[ -n "$best" ]] || best=gpu_h100
  echo "$best $best_free"
}

SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then SMOKE=1; shift; fi
ARMS=("$@")
[[ ${#ARMS[@]} -gt 0 ]] || ARMS=(1 2 3 4)

if [[ "$QUEUE" == "auto" ]]; then
  read -r QUEUE FREE_SLOTS <<<"$(pick_queue)"
  echo "queue: $QUEUE (emptiest host has $FREE_SLOTS/96 slots free; a job needs 96)"
fi

jobid () { sed -n 's/^Job <\([0-9]*\)>.*/\1/p'; }

# stage <config> <wall clock> [<predecessor experiment_name> <job id to wait for>]
#
# With a predecessor, the job first resolves that run's newest directory and its highest checkpoint
# -- highest *numerically*, not by modification time, which would only be a coincidence of write
# order -- substitutes it for PREV_CHECKPOINT, and trains from the patched config.
stage () {
  local config=$1 wall=$2 prev=${3:-} after=${4:-}
  local name; name=$(basename "$config" .toml)
  local wait_arg=() cfg="$config" prologue=""

  if [[ $SMOKE -eq 1 ]]; then
    wall=0:30
    cfg="$STAGE/smoke_$name.toml"
    sed -e "s/^experiment_name = .*/experiment_name = \"smoke_$name\"/" \
        -e 's/^max_steps = .*/max_steps = 20/' -e 's/^warmup_steps = .*/warmup_steps = 2/' \
        -e 's/^val_every = .*/val_every = 10/' -e 's/^checkpoint_every = .*/checkpoint_every = 20/' \
        -e 's/^samples_per_epoch = .*/samples_per_epoch = 20/' -e 's/^dp_shard = .*/dp_shard = 1/' \
        "$config" > "$cfg"
  fi

  if [[ -n "$prev" && $SMOKE -eq 1 ]]; then
    # No predecessor exists in a smoke run, so point the placeholder at an already trained ViT-L
    # encoder of the same shape. The load path -- prefix filtering, the predecessor's decoder keys
    # being dropped, a fresh head -- is what needs exercising, and that does not depend on which
    # trained encoder it came from.
    sed -i "s|PREV_CHECKPOINT|$SMOKE_ENCODER|" "$cfg"
  elif [[ -n "$prev" ]]; then
    [[ -n "$after" ]] && wait_arg=(-w "done($after)")
    local resolved="$STAGE/${name}_resolved.toml"
    prologue="RUN=\$(ls -dt $RUNS/init__${prev}_*/ | head -1); \
      STEP=\$(ls -d \${RUN}checkpoints/step_* | sed 's|.*step_||' | sort -n | tail -1); \
      echo \"initialising from \${RUN}checkpoints/step_\$STEP\"; \
      sed \"s|PREV_CHECKPOINT|\${RUN}checkpoints/step_\$STEP|\" '$cfg' > '$resolved'; "
    cfg="$resolved"
  fi

  local procs=$GPUS
  [[ $SMOKE -eq 1 ]] && procs=1
  bsub -P "$PROJECT" -q "$QUEUE" -gpu "num=$procs" -n $((procs == 1 ? 12 : SLOTS)) -W "$wall" -r \
    "${wait_arg[@]+"${wait_arg[@]}"}" -J "init_$name" -cwd "$REPO" \
    -o "$LOGS/init_${name}_%J.log" -e "$LOGS/init_${name}_%J.err" \
    "$THREADS $prologue \
     $VENV/bin/torchrun --standalone --nproc_per_node=$procs src/train.py \
       --config '$cfg' $([[ $SMOKE -eq 1 ]] && echo "--output-root '$STAGE/smoke'" || echo "--resume")" \
    | jobid
}

smoke_note () { [[ $SMOKE -eq 1 ]] && echo "  (smoke: unchained, stand-in encoder)"; }

for arm in "${ARMS[@]}"; do
  case $arm in
    1)
      a=$(stage "$HERE/1a_scratch_interp.toml" 36:00);  echo "1a scratch interp      job $a"
      b=$(stage "$HERE/1b_scratch_subpixel.toml" 18:00 1a_scratch_interp "$a"); echo "1b scratch subpixel    job $b$(smoke_note)"
      ;;
    2)
      a=$(stage "$HERE/2a_dinov3_interp.toml" 36:00);   echo "2a dinov3 interp       job $a"
      b=$(stage "$HERE/2b_dinov3_subpixel.toml" 18:00 2a_dinov3_interp "$a"); echo "2b dinov3 subpixel     job $b$(smoke_note)"
      ;;
    3)
      a=$(stage "$HERE/3a_simmim_pretrain.toml" 16:00); echo "3a simmim pretrain     job $a"
      b=$(stage "$HERE/3b_simmim_interp.toml" 36:00 3a_simmim_pretrain "$a"); echo "3b simmim interp       job $b"
      c=$(stage "$HERE/3c_simmim_subpixel.toml" 18:00 3b_simmim_interp "$b"); echo "3c simmim subpixel     job $c$(smoke_note)"
      ;;
    4)
      a=$(stage "$HERE/4a_dinov3_aug_interp.toml" 36:00); echo "4a dinov3+aug interp   job $a"
      b=$(stage "$HERE/4b_dinov3_aug_subpixel.toml" 18:00 4a_dinov3_aug_interp "$a"); echo "4b dinov3+aug subpixel job $b$(smoke_note)"
      ;;
    *) echo "unknown arm: $arm (expected 1, 2, 3 or 4)" >&2; exit 2 ;;
  esac
done
