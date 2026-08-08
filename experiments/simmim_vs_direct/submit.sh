#!/usr/bin/env bash
# Submit the whole comparison. Arm A's fine-tune waits on its pretraining via `bsub -w`, so all
# three can go in at once.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
PROJECT=miaai
QUEUE=gpu_h100
GPUS=2
SLOTS=24            # 12 slots/GPU on H100, so 2 GPUs' worth and no stranding
LOGS=/nrs/scicompsoft/orhane/mia-train-runs/jobs
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
        $VENV/bin/torchrun --standalone --nproc_per_node=$GPUS src/train.py --config $config"
}

echo "arm A stage 1 -- SimMIM pretraining"
launch simmim_A1 "$HERE/1_simmim_pretrain.toml" 12

# Run directories are timestamped, so arm A stage 2's [init] path cannot be written by hand ahead
# of time -- it is resolved here, after stage 1's directory exists, and patched into the config.
echo "arm A stage 2 -- fine-tune from the SSL checkpoint (waits on A1)"
A1_DIR=$(ls -dt /nrs/scicompsoft/orhane/mia-train-runs/simmim_vs_direct__A1_simmim_pretrain_*/ 2>/dev/null | head -1)
if [[ -n "$A1_DIR" ]]; then
  A1_STEPS=$(grep -oP 'max_steps = \K\d+' "$HERE/1_simmim_pretrain.toml")
  sed -i "s|^path = \"/nrs/.*A1_simmim_pretrain.*/checkpoints/step_[0-9]*\"|path = \"${A1_DIR%/}/checkpoints/step_${A1_STEPS}\"|" \
      "$HERE/2A_finetune_from_simmim.toml"
  echo "  init path -> ${A1_DIR%/}/checkpoints/step_${A1_STEPS}"
else
  echo "  WARNING: no stage-1 run directory yet; check [init].path in 2A before it starts" >&2
fi
launch simmim_A2 "$HERE/2A_finetune_from_simmim.toml" 8 simmim_A1

echo "arm B -- fine-tune straight from the Meta checkpoint (the control)"
launch simmim_B  "$HERE/2B_finetune_from_dinov3.toml" 8
