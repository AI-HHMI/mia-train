#!/usr/bin/env bash
# Submit the whole comparison. Arm A's fine-tune waits on its pretraining via `bsub -w`, so all
# four can go in at once; B and C have no dependency and start immediately.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
PROJECT=miaai
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
LOGS="$RUNS/jobs"

QUEUE=gpu_h200
GPUS=8
SLOTS=96
mkdir -p "$LOGS"

# With no arguments every arm is submitted. Named arms submit only those: e.g. `bash submit.sh D` 
# adds one arm without relaunching the rest.
WANTED=("$@")
wanted () {
  [[ ${#WANTED[@]} -eq 0 ]] && return 0
  local arm
  for arm in "${WANTED[@]}"; do [[ "$arm" == "$1" ]] && return 0; done
  return 1
}

launch () {          # launch <name> <config> <hours> [dependency]
  local name=$1 config=$2 hours=$3 dep=${4:-}
  local wait_arg=()
  [[ -n "$dep" ]] && wait_arg=(-w "ended($dep)")
  bsub -P "$PROJECT" -q "$QUEUE" -gpu "num=$GPUS" -n "$SLOTS" -W "$hours:00" \
       -J "$name" -cwd "$REPO" \
       -o "$LOGS/${name}_%J.log" -e "$LOGS/${name}_%J.err" \
       "${wait_arg[@]}" \
       "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; \
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
        $VENV/bin/torchrun --standalone --nproc_per_node=$GPUS src/train.py --config $config"
}

if wanted A1; then
echo "arm A stage 1 -- SimMIM pretraining"
launch simmim_A1 "$HERE/1_simmim_pretrain.toml" 12
fi

# Run directories are timestamped, so arm A stage 2's [init] path is not knowable at submission
# time. It is resolved inside A2's own job instead, which `-w ended(simmim_A1)` guarantees runs 
# only after stage 1 has finished. Resolved config is written beside the run rather than edited
# in place, so the file in git keeps saying what the arm means and two launches cannot race on it
if wanted A2; then
echo "arm A stage 2 -- fine-tune from the SSL checkpoint (waits on A1)"
A1_STEPS=$(grep -oP 'max_steps = \K\d+' "$HERE/1_simmim_pretrain.toml")
RESOLVED="$RUNS/2A_resolved.toml"
bsub -P "$PROJECT" -q "$QUEUE" -gpu "num=$GPUS" -n "$SLOTS" -W 24:00 \
     -J simmim_A2 -cwd "$REPO" -w "ended(simmim_A1)" \
     -o "$LOGS/simmim_A2_%J.log" -e "$LOGS/simmim_A2_%J.err" \
     "set -euo pipefail; \
      export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; \
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
      A1_DIR=\$(ls -dt $RUNS/simmim_vs_direct__A1_simmim_pretrain_*/ | head -1); \
      CKPT=\"\${A1_DIR%/}/checkpoints/step_$A1_STEPS\"; \
      test -d \"\$CKPT\" || { echo \"no stage-1 checkpoint at \$CKPT\" >&2; exit 1; }; \
      sed \"s|^path = \\\"/nrs/.*A1_simmim_pretrain.*\\\"|path = \\\"\$CKPT\\\"|\" \
          '$HERE/2A_finetune_from_simmim.toml' > '$RESOLVED'; \
      echo \"arm A2 initialising from \$CKPT\"; \
      $VENV/bin/torchrun --standalone --nproc_per_node=$GPUS src/train.py --config '$RESOLVED'"
fi

if wanted B; then
echo "arm B -- fine-tune straight from the Meta checkpoint (the control)"
launch simmim_B  "$HERE/2B_finetune_from_dinov3.toml" 24
fi

if wanted C; then
echo "arm C -- the same architecture from random weights (the floor)"
launch simmim_C  "$HERE/2C_finetune_from_scratch.toml" 24
fi

# Arm D is the only multi-node arm: 2 whole nodes on the *_parallel queue, one torchrun per node.
# It needs its own bsub rather than `launch`, which assumes a single node and --standalone.
if wanted D; then
echo "arm D -- arm B at twice the batch, across 2 nodes (16 ranks)"
bsub -P "$PROJECT" -q gpu_h200_parallel -app parallel-96 -gpu "num=8:mode=shared" \
     -n 192 -W 24:00 -J simmim_D -cwd "$REPO" \
     -o "$LOGS/simmim_D_%J.log" -e "$LOGS/simmim_D_%J.err" \
     "MIA_TRAIN='$REPO' VENV='$VENV' \
      '$REPO/deploy/lsf/launch_multinode.sh' 8 '$HERE/2D_finetune_dinov3_batch16.toml'"
fi
