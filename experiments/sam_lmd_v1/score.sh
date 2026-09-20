#!/usr/bin/env bash
# Score one SAM arm's round on the lmd_ssl_v1_neuron_instance leaderboard, beside the MWS rows.
#
#   bash experiments/sam_lmd_v1/score.sh base 2 [--step N]
#
# The artifacts are `instances`, so they go through the task files the mutex-watershed entries use
# (`lmd_ssl_v1_neuron_instance_mws_{fit,test}.toml`): a `size_filter` swept over
# [0, 500, 5000, 50000] on the finetune half and applied once to the reported half, panoptic
# quality ranking, same region. That is the whole of the comparison: the SAM row and the MWS row
# differ only in what wrote the labelling.
#
# CPU queue: scoring needs no GPU, and mia-evals lives in the banis environment.
set -euo pipefail

ARM="${1:?arm}"; ROUND="${2:?round}"; shift 2
STEP=""
while [[ "${1:-}" == --* ]]; do
  case "$1" in --step) STEP="$2"; shift ;; *) echo "unknown flag $1" >&2; exit 2 ;; esac; shift
done

EVALS=/groups/scicompsoft/home/orhane/projects/mia-evals
PY=/groups/scicompsoft/home/orhane/banisvenv/bin/python
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/sam_lmd_v1   # this experiment's home on /nrs: runs/ jobs/ eval/ probes/ (layout of 2026-09-16)
RUNS=$EXP/runs
STAGE=$EXP
LOGS="$EXP/jobs"

run=$(ls -dt "$RUNS"/sam1__${ARM}_r${ROUND}_*/ 2>/dev/null | head -1) || true
[[ -n "${run:-}" ]] || { echo "no run matching sam1__${ARM}_r${ROUND}_*" >&2; exit 1; }
run=${run%/}
[[ -n "$STEP" ]] || STEP=$(ls -d "$run"/checkpoints/step_* | sed 's|.*step_||' | sort -n | tail -1)
ART="$STAGE/eval/${ARM}_r${ROUND}"
for half in test finetune; do
  n=$(ls -d "$ART/$half"/*.zarr 2>/dev/null | grep -vc '\.gt\.zarr$' || true)
  [[ "$n" -eq 4 ]] || { echo "$ART/$half holds $n prediction artifacts, expected 4 -- run predict_eval.sh first" >&2; exit 1; }
done
LABEL="sam1_${ARM}_r${ROUND}_step${STEP}"
SCRATCH="$STAGE/score/${ARM}_r${ROUND}"
mkdir -p "$SCRATCH" "$LOGS"

cmd="export OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONPATH=$EVALS/src; cd $EVALS && \
$PY -m evaluate score configs/lmd_ssl_v1_neuron_instance/mws.toml \
  --test '$ART/test' --val '$ART/finetune' \
  --run-dir '$run' --scratch '$SCRATCH'"
# (2026-09-18) one scoring config carries both splits and the record is named <run>.step<N>.<route>
# by mia-evals itself; the old --val-config and --label flags no longer exist. NOTE: that config's
# route is `mws`, which names these SAM labellings wrongly -- score SAM arms through a config whose
# route is `size_filter` before the next one is recorded.
id=$(bsub -P miaai -q local -n 20 -W 12:00 -J "sc_$LABEL" -o "$LOGS/sc_${LABEL}_%J.log" -e "$LOGS/sc_${LABEL}_%J.err" "$cmd" \
     | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
echo "  $LABEL  job=$id  -> $EVALS/leaderboard/lmd_ssl_v1_neuron_instance/records/$LABEL.json"
