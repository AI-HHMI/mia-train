#!/usr/bin/env bash
# Submit the arms of init_comparison. Each arm is a chain of stages, held one behind the next.
#
#   bash experiments/init_comparison/submit.sh              # every arm
#   bash experiments/init_comparison/submit.sh 2 4          # just those arms
#   bash experiments/init_comparison/submit.sh --smoke 1    # 20 steps of arm 1's stages, one GPU
#   bash experiments/init_comparison/submit.sh --stage 2b_dinov3_subpixel   # one stage, see below
#
#   1  scratch : interpolating 150k -> sub-pixel 100k
#   2  dinov3  : interpolating 150k -> sub-pixel 100k
#   3  simmim  : SimMIM 100k -> interpolating 150k -> sub-pixel 100k
#   4  dinov3 + augmentation : interpolating 150k -> sub-pixel 100k
#
# **Queue-agnostic by default.** Every stage is submitted twice, once to gpu_h100 and once to
# gpu_h200, so it starts on whichever generation frees up first rather than waiting on one. The two
# twins share an `experiment_name` and both pass `--resume`, which makes them interchangeable --
# either can continue the same run directory from its last checkpoint -- so the only thing that has
# to be prevented is both running at once. `claim.sh` does that with an atomic lock; the loser exits
# 42 and releases its node within seconds.
#
# Neither tidier mechanism exists here: a queue list (`-q "gpu_h100 gpu_h200"`) is rejected by the
# site's esub, and the only queues spanning both host groups (`gpu_short`, `short`) cap at 60
# minutes. Set QUEUE=gpu_h200 to force a single queue instead of a pair.
#
# A stage after the first cannot name its predecessor's checkpoint at submission time -- the run
# directory does not exist yet -- so each dependent config carries a `PREV_CHECKPOINT` placeholder
# that the job resolves for itself.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
PROJECT=miaai
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
STAGE=/nrs/scicompsoft/orhane/mia-train-scratch/init_comparison
LOGS="$RUNS/jobs"
CLAIM="$STAGE/claim.sh"
mkdir -p "$LOGS" "$STAGE/locks" "$STAGE/cmd"

GPUS=8
SLOTS=96
THREADS="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"

# miao is an editable install pointing at a working tree, so a `git checkout` there rewrites the
# dataset code under every running job. That is not hypothetical: on 2026-08-13 a switch away from
# `feature/aug-rot-anisotropic` killed all four live runs within 60 seconds, because these configs
# ask for `aug_rot: "inplane"` -- a mode only that branch has -- and master reads the string as a
# plain truthy boolean and then rejects NISB's anisotropic 9x9x20 voxels.
#
# So pin the commit instead of trusting the checkout. A `.pth` editable install is an ordinary
# sys.path entry appended during site-packages processing, so PYTHONPATH precedes it and wins.
MIAO_PIN=/nrs/scicompsoft/orhane/mia-train-scratch/miao-pinned/8d41638/src
[[ -d "$MIAO_PIN" ]] || { echo "missing pinned miao at $MIAO_PIN" >&2; exit 2; }
PIN="export PYTHONPATH=$MIAO_PIN\${PYTHONPATH:+:\$PYTHONPATH}"

# A trained ViT-L encoder of this exact shape, used only to exercise the phase-2 load path in a
# smoke run, where none of the run's own predecessors exists yet.
SMOKE_ENCODER=/nrs/scicompsoft/orhane/mia-train-runs/banis_parity__finetune_256_long_20260810_123308/checkpoints/step_300000

QUEUES=(gpu_h100 gpu_h200)
[[ -n "${QUEUE:-}" && "${QUEUE:-}" != "auto" ]] && QUEUES=("$QUEUE")

SMOKE=0
[[ "${1:-}" == "--smoke" ]] && { SMOKE=1; shift; }
ONE_STAGE=""
[[ "${1:-}" == "--stage" ]] && { ONE_STAGE=$2; shift 2; }

jobid () { sed -n 's/^Job <\([0-9]*\)>.*/\1/p'; }

# `done()` requires a zero exit, which is why a losing twin exits 42: an `|| ` of the pair then
# fires only when the twin that actually trained finishes.
dep_of () { local d=""; for id in "$@"; do d+="${d:+ || }done($id)"; done; echo "$d"; }

# stage <config> <wall clock> [<predecessor experiment_name> <dependency expression>]
# Prints the job ids of the twins it submitted.
stage () {
  local config=$1 wall=$2 prev=${3:-} dep=${4:-}
  local name; name=$(basename "$config" .toml)
  local cfg="$config" prologue="" procs=$GPUS slots=$SLOTS

  if [[ $SMOKE -eq 1 ]]; then
    wall=0:30; procs=1; slots=12
    cfg="$STAGE/smoke_$name.toml"
    sed -e "s/^experiment_name = .*/experiment_name = \"smoke_$name\"/" \
        -e 's/^max_steps = .*/max_steps = 20/' -e 's/^warmup_steps = .*/warmup_steps = 2/' \
        -e 's/^val_every = .*/val_every = 10/' -e 's/^checkpoint_every = .*/checkpoint_every = 20/' \
        -e 's/^samples_per_epoch = .*/samples_per_epoch = 20/' -e 's/^dp_shard = .*/dp_shard = 1/' \
        "$config" > "$cfg"
  fi

  if [[ -n "$prev" && $SMOKE -eq 1 ]]; then
    # No predecessor exists in a smoke run, so point the placeholder at an already trained encoder
    # of the same shape. `skip = ["rope_embed."]` is what lets that work for either RoPE variant.
    sed -i "s|PREV_CHECKPOINT|$SMOKE_ENCODER|" "$cfg"
  elif [[ -n "$prev" ]]; then
    local resolved="$STAGE/${name}_resolved.toml"
    # Resolved in the job, after the predecessor has written checkpoints. Highest step chosen
    # numerically, not by modification time, which would only be a coincidence of write order.
    prologue="RUN=\$(ls -dt $RUNS/init__${prev}_*/ | head -1)
STEP=\$(ls -d \${RUN}checkpoints/step_* | sed 's|.*step_||' | sort -n | tail -1)
echo \"initialising from \${RUN}checkpoints/step_\$STEP\"
sed \"s|PREV_CHECKPOINT|\${RUN}checkpoints/step_\$STEP|\" '$cfg' > '$resolved'"
    cfg="$resolved"
  fi

  # The stage's work as its own script, so the twins share one definition and nothing has to
  # survive several layers of shell quoting.
  local cmd="$STAGE/cmd/${name}.sh"
  { echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "$THREADS"
    echo "$PIN"
    echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    [[ -n "$prologue" ]] && echo "$prologue"
    printf '%s --standalone --nproc_per_node=%s src/train.py --config %q %s\n' \
      "$VENV/bin/torchrun" "$procs" "$cfg" \
      "$([[ $SMOKE -eq 1 ]] && printf -- "--output-root %q" "$STAGE/smoke" || echo "--resume")"
  } > "$cmd"

  local ids=() q id
  for q in "${QUEUES[@]}"; do
    id=$(bsub -P "$PROJECT" -q "$q" -gpu "num=$procs" -n "$slots" -W "$wall" -r \
      ${dep:+-w "$dep"} -J "init_$name" -cwd "$REPO" \
      -o "$LOGS/init_${name}_%J.log" -e "$LOGS/init_${name}_%J.err" \
      "bash '$CLAIM' '$STAGE/locks/$name' '$cmd'" | jobid)
    ids+=("$id")
  done
  echo "${ids[@]}"
}

report () { printf "%-24s jobs %s\n" "$1" "$2"; }

# One stage on its own, e.g. to convert an already-queued stage to a twin pair. DEPENDS_ON is a
# dependency expression or a bare job id.
if [[ -n "$ONE_STAGE" ]]; then
  cfg="$HERE/${ONE_STAGE}.toml"
  [[ -f "$cfg" ]] || { echo "no config $cfg" >&2; exit 2; }
  prev=${PREV_STAGE:-}
  dep=${DEPENDS_ON:-}
  [[ "$dep" =~ ^[0-9]+$ ]] && dep="done($dep)"
  report "$ONE_STAGE" "$(stage "$cfg" "${WALL:-18:00}" "$prev" "$dep")"
  exit 0
fi

ARMS=("$@"); [[ ${#ARMS[@]} -gt 0 ]] || ARMS=(1 2 3 4)
for arm in "${ARMS[@]}"; do
  case $arm in
    1) a=$(stage "$HERE/1a_scratch_interp.toml" 36:00); report 1a_scratch_interp "$a"
       b=$(stage "$HERE/1b_scratch_subpixel.toml" 18:00 1a_scratch_interp "$(dep_of $a)"); report 1b_scratch_subpixel "$b" ;;
    2) a=$(stage "$HERE/2a_dinov3_interp.toml" 36:00); report 2a_dinov3_interp "$a"
       b=$(stage "$HERE/2b_dinov3_subpixel.toml" 18:00 2a_dinov3_interp "$(dep_of $a)"); report 2b_dinov3_subpixel "$b" ;;
    3) a=$(stage "$HERE/3a_simmim_pretrain.toml" 16:00); report 3a_simmim_pretrain "$a"
       b=$(stage "$HERE/3b_simmim_interp.toml" 36:00 3a_simmim_pretrain "$(dep_of $a)"); report 3b_simmim_interp "$b"
       c=$(stage "$HERE/3c_simmim_subpixel.toml" 18:00 3b_simmim_interp "$(dep_of $b)"); report 3c_simmim_subpixel "$c" ;;
    4) a=$(stage "$HERE/4a_dinov3_aug_interp.toml" 36:00); report 4a_dinov3_aug_interp "$a"
       b=$(stage "$HERE/4b_dinov3_aug_subpixel.toml" 18:00 4a_dinov3_aug_interp "$(dep_of $a)"); report 4b_dinov3_aug_subpixel "$b" ;;
    *) echo "unknown arm: $arm (expected 1, 2, 3 or 4)" >&2; exit 2 ;;
  esac
done
