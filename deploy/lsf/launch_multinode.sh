#!/usr/bin/env bash
# One `torchrun` per node, for a job on a `*_parallel` queue.
#
#   MIA_TRAIN=... VENV=... launch_multinode.sh <gpus_per_node> <config.toml> [extra train.py args]
#
# Lives in deploy/ rather than src/ because it is scheduler-specific: it reads LSF's host list.
# `src/` stays scheduler-agnostic (DESIGN.md §6) and learns the topology only through the
# environment torchrun sets.
#
# Rendezvous is `c10d` rather than the static `--node_rank`/`--master_addr` pair the older recipe
# used. With c10d every node runs the *identical* command and the backend assigns ranks, so there
# is no per-host bookkeeping to get wrong -- and getting it wrong is quiet: two nodes claiming
# rank 0 hang in `init_process_group` until the wall clock runs out, with no error to read.
#
# The rendezvous port is derived from the job id so two of these can run concurrently on the same
# node without colliding on a fixed port.
set -euo pipefail

GPUS_PER_NODE=${1:?usage: launch_multinode.sh <gpus_per_node> <config.toml> [extra args]}
CONFIG=${2:?usage: launch_multinode.sh <gpus_per_node> <config.toml> [extra args]}
shift 2

: "${MIA_TRAIN:?set MIA_TRAIN to the repo root}"
: "${VENV:?set VENV to the python virtualenv}"
: "${LSB_MCPU_HOSTS:?not inside an LSF job, or the queue does not allocate whole nodes}"
: "${LSB_JOBID:?not inside an LSF job}"

# LSB_MCPU_HOSTS is "host1 slots1 host2 slots2 ...", one entry per host. Take every other field.
read -r -a MCPU_FIELDS <<< "$LSB_MCPU_HOSTS"
HOSTS=()
for ((i = 0; i < ${#MCPU_FIELDS[@]}; i += 2)); do
  HOSTS+=("${MCPU_FIELDS[i]}")
done

NNODES=${#HOSTS[@]}
MASTER=${HOSTS[0]}
PORT=$((29400 + LSB_JOBID % 1000))

echo "multi-node launch: $NNODES node(s) x $GPUS_PER_NODE GPU(s) = $((NNODES * GPUS_PER_NODE)) ranks"
echo "  hosts      : ${HOSTS[*]}"
echo "  rendezvous : $MASTER:$PORT (c10d, id $LSB_JOBID)"
echo "  config     : $CONFIG"

# Threads are per *process*, and there is one process per GPU, so this is slots-per-GPU budget
# rather than the node's core count -- see the hint sheet on layered threading.
# NCCL_DEBUG defaults to WARN so a fabric problem is visible without INFO's per-rank flood; set
# NCCL_DEBUG=INFO in the environment to confirm which transport was selected.
REMOTE_CMD=$(cat <<EOF
cd '$MIA_TRAIN' || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# Forwarded explicitly: \`blaunch\` starts a fresh shell on every node, so anything the submitting
# environment set is gone unless it is written into this command. PYTHONPATH is the one that
# matters -- the NISB data configs need a pinned \`miao\` (they set \`aug_rot\`, which the default
# build rejects outright), and single-node \`submit.sh\` pins it the same way. Without this the
# workers die in \`MiaoConfig\` validation before the first step, which reads as a config error
# rather than as a missing environment.
export PYTHONPATH='${PYTHONPATH:-}'
exec '$VENV/bin/torchrun' \
  --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE \
  --rdzv_backend=c10d --rdzv_id=$LSB_JOBID --rdzv_endpoint=$MASTER:$PORT \
  src/train.py --config '$CONFIG' $*
EOF
)

# One task per host named in -z. blaunch blocks until every task exits and returns non-zero if
# any of them failed, so LSF sees the real outcome.
exec blaunch -z "${HOSTS[*]}" bash -c "$REMOTE_CMD"
