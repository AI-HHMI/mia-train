#!/usr/bin/env bash
# Predict, score and visualise one checkpoint of one arm of this experiment.
#
#   bash experiments/init_comparison_comparison/score_checkpoint.sh 2_dinov3 100000
#   bash experiments/init_comparison_comparison/score_checkpoint.sh 1_scratch 50000 --reuse --no-score
#
# A wrapper over banis_parity/score_checkpoint.sh, which takes --run/--tag for exactly this. The
# arm name selects the run directory and the artifact tag, so arms cannot overwrite each other's
# affinities. Every option of the underlying script passes through.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED="$HERE/../banis_parity/score_checkpoint.sh"
RUNS=/nrs/scicompsoft/orhane/mia-train-runs

if [[ $# -lt 2 ]]; then
  echo "usage: $(basename "$0") <arm> <step> [options passed to the shared script]" >&2
  echo "  stages: 1a_scratch_interp 1b_scratch_subpixel 2a_dinov3_interp 2b_dinov3_subpixel\n          3a_simmim_pretrain 3b_simmim_interp 3c_simmim_subpixel\n          4a_dinov3_aug_interp 4b_dinov3_aug_subpixel" >&2
  exit 2
fi
ARM=$1 STEP=$2; shift 2

RUN=$(ls -dt "$RUNS/init__${ARM}"_*/ 2>/dev/null | head -1 || true)
[[ -n "$RUN" ]] || { echo "no run found for arm '$ARM' under $RUNS" >&2; exit 1; }

exec bash "$SHARED" "$STEP" --run "${RUN%/}" --tag "vb_${ARM}" "$@"
