#!/usr/bin/env bash
# Predict, score and visualise one checkpoint of one arm of this experiment.
#
#   bash experiments/lsd_aux_v1/score_checkpoint.sh 15000 mtlsd_s80
#   bash experiments/lsd_aux_v1/score_checkpoint.sh 15000 mtlsd_s80 --reuse --no-score
#
# A wrapper over experiments/banis_parity/score_checkpoint.sh, exactly as the control's
# experiments/subpixel_decoder/score_checkpoint.sh is: it resolves the arm's newest run directory
# and a tag that keeps its artifacts from colliding with the control's, and passes every other
# option through (run the shared script with no arguments for the list).
#
# Nothing here knows the head predicts descriptors: prediction rebuilds the algorithm from the
# run's own `resolved_config.json` and writes the six affinity channels, which is all scoring reads.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED="$HERE/../banis_parity/score_checkpoint.sh"
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/lsd_aux_v1   # runs/ jobs/ eval/ probes/ (layout of 2026-09-16)
RUNS=$EXP/runs

[[ $# -ge 2 ]] || { echo "usage: $(basename "$0") <step> <arm> [shared-script flags]" >&2; exec bash "$SHARED"; }
STEP=$1
ARM=$2

RUN=$(ls -dt "$RUNS"/lsd_aux_v1__${ARM}_*/ 2>/dev/null | head -1 || true)
[[ -n "$RUN" ]] || { echo "no run of arm $ARM under $RUNS" >&2; exit 1; }

# The step first, then this experiment's defaults, then the caller's flags, so a caller's `--tag`
# or `--run` still wins.
exec bash "$SHARED" "$STEP" --run "${RUN%/}" --tag "$ARM" "${@:3}"
