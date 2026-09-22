#!/usr/bin/env bash
# Score one SAM arm's round on the lmd_ssl_v1_neuron_instance leaderboard, beside the MWS rows.
#
#   bash experiments/sam_lmd_v1/score.sh base 2 [--step N]
#
# The artifacts are `instances`, so they go through mia-evals' `size_filter` route
# (`configs/lmd_ssl_v1_neuron_instance/size_filter.toml`): the same size filter the mutex-watershed
# rows get (`mws.toml`), swept over [0, 500, 5000, 50000] on the finetune half and applied once to
# the reported half, panoptic quality ranking, same region. That is the whole of the comparison:
# the SAM row and the MWS row differ only in what wrote the labelling. mia-evals names the record
# `<run dir>.step<N>.size_filter` and keeps the post-processed labellings under the scratch dir.
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
ART="${ART_DIR:-$STAGE/eval/${ARM}_r${ROUND}}"        # ART_DIR: score another artifact directory of this run,
                                                        # e.g. the 256-lattice crop of a 128-window arm (crop_artifacts.py)
for half in test finetune; do
  n=$(ls -d "$ART/$half"/*.zarr 2>/dev/null | grep -vc '\.gt\.zarr$' || true)
  [[ "$n" -eq 4 ]] || { echo "$ART/$half holds $n prediction artifacts, expected 4 -- run predict_eval.sh first" >&2; exit 1; }
done
LABEL="sam1_${ARM}_r${ROUND}_step${STEP}"
SCRATCH="${SCRATCH_DIR:-$STAGE/score/${ARM}_r${ROUND}}"
mkdir -p "$SCRATCH" "$LOGS"

cmd="export OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONPATH=$EVALS/src; cd $EVALS && \
$PY -m evaluate score configs/lmd_ssl_v1_neuron_instance/size_filter.toml \
  --test '$ART/test' --val '$ART/finetune' \
  --run-dir '$run' --scratch '$SCRATCH'"
id=$(bsub -P miaai -q local -n 20 -W 12:00 -J "sc_$LABEL" -o "$LOGS/sc_${LABEL}_%J.log" -e "$LOGS/sc_${LABEL}_%J.err" "$cmd" \
     | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
echo "  $LABEL  job=$id  -> $EVALS/leaderboard/lmd_ssl_v1_neuron_instance/records/$(basename "$run").step${STEP}.size_filter.json"
