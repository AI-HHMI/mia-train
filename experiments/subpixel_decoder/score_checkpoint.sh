#!/usr/bin/env bash
# Predict, score and visualise one checkpoint of *this* experiment.
#
#   bash experiments/subpixel_decoder/score_checkpoint.sh 15000
#   bash experiments/subpixel_decoder/score_checkpoint.sh 15000 --reuse --no-score
#
# A wrapper, not a copy. The work is `banis_parity/score_checkpoint.sh`, which was written to take
# `--run` and `--tag` precisely so a second experiment could reuse it; all this adds is resolving
# this experiment's newest run directory and a tag that keeps its artifacts from colliding with the
# control's. Every option of the underlying script passes through -- run it with no arguments for
# the list.
#
# Nothing here needs to know that the run uses a sub-pixel decoder: prediction rebuilds the
# algorithm from the run's own `resolved_config.json`, so the head follows the checkpoint.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED="$HERE/../banis_parity/score_checkpoint.sh"
RUNS=/nrs/scicompsoft/orhane/mia-train-runs

[[ $# -gt 0 ]] || exec bash "$SHARED"      # no step given: let the shared script print its usage

RUN=$(ls -dt "$RUNS"/subpixel_decoder__subpixel_256_*/ 2>/dev/null | head -1 || true)
[[ -n "$RUN" ]] || { echo "no subpixel_decoder run found under $RUNS" >&2; exit 1; }

# The step first, then this experiment's defaults, then the caller's flags -- so a caller can still
# override `--tag` or `--run` and have their value win.
exec bash "$SHARED" "$1" --run "${RUN%/}" --tag sp "${@:2}"
