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

# H200, not H100. A 512-cube at patch 16 peaks at 112.7 GiB per GPU even with activation
# checkpointing on, which does not fit an 80 GiB H100 -- and the obvious way to make it fit,
# running the head's upsampling in bf16, was measured to cost 1.4x the interpolation accuracy and
# rejected. 8 GPUs is a whole node (8 per node, 12 slots each), so the batch grows with GPUs.
QUEUE=gpu_h200
GPUS=8
SLOTS=96
mkdir -p "$LOGS"

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

# `expandable_segments` above is not optional at this size: the head allocates 16 GiB tensors and
# frees them every step, and the default caching allocator strands enough between segments to OOM
# a run whose live set fits comfortably.

echo "arm A stage 1 -- SimMIM pretraining"
launch simmim_A1 "$HERE/1_simmim_pretrain.toml" 12

# Run directories are timestamped, so arm A stage 2's [init] path is not knowable at submission
# time -- stage 1's directory does not exist until stage 1 starts. It is resolved inside A2's own
# job instead, which `-w ended(simmim_A1)` guarantees runs only after stage 1 has finished. The
# resolved config is written beside the run rather than edited in place, so the file in git keeps
# saying what the arm means and two launches cannot race on it.
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

echo "arm B -- fine-tune straight from the Meta checkpoint (the control)"
launch simmim_B  "$HERE/2B_finetune_from_dinov3.toml" 24

echo "arm C -- the same architecture from random weights (the floor)"
launch simmim_C  "$HERE/2C_finetune_from_scratch.toml" 24
