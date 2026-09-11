#!/usr/bin/env bash
# Submit sam_lmd_v1: per arm, round 0 -> label -> round 1 -> label -> round 2, each held behind the last.
#
#   bash experiments/sam_lmd_v1/submit.sh                    # every arm
#   bash experiments/sam_lmd_v1/submit.sh base stride2       # just these arms
#   bash experiments/sam_lmd_v1/submit.sh --dry-run          # write the job scripts, print the bsub lines
#   bash experiments/sam_lmd_v1/submit.sh --smoke base       # 20-step rounds, one tiny block, one GPU
#   bash experiments/sam_lmd_v1/submit.sh --rounds 1 base    # round 0 and one data-engine round only
#   bash experiments/sam_lmd_v1/submit.sh --skip-r0 base     # round 0 already ran; start at L1
#
# The chain, per arm:
#
#   r0     promptable_seg from the DINOv3 LVD checkpoint on the 4 GT finetune volumes, 100k steps
#   L1     r0's model labels 1 block (512^3 at 8 nm) of each of the 87 unlabeled volumes -- an LSF
#          array, one element per volume, plus 4 elements that run the same labeller over blocks of
#          the GT volumes and score it against their truth (the diagnostic); then a CPU job writes
#          the round's data config and the diagnostic table
#   r1     warm start from r0 (whole model), GT + L1's blocks, 50k steps
#   L2     r1 labels 4 blocks per volume (a superset of L1's cells)
#   r2     warm start from r1, GT + L2's blocks, 50k steps
#
# Training takes one whole B300 node (8 GPUs, 96 slots, dp_shard 8, batch 1/rank -> global batch 8,
# as every finetune stage of lmd_ssl_v1). Labelling takes one GPU per element. Everything on
# `gpu_b300`: compiled, H100 and B300 are within 4% on this algorithm (promptable_seg_v1/RESULTS.md),
# the `stride1` arm's decoder tensors need the B300's memory, and every SAM arm is then predicted
# on one architecture -- the same code gives different instance counts on a different GPU
# generation, so the arms must not be spread across two.
#
# A stage after the first cannot name its predecessor's checkpoint or its round's data config at
# submission time -- neither exists yet -- so the configs carry `PREV_CHECKPOINT` / `ROUND_CONFIG`
# placeholders that each job resolves for itself in a prologue, like lmd_ssl_v1 and pseudo_labeling.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV=/groups/scicompsoft/home/orhane/myvenv
PROJECT=miaai
RUNS=/nrs/scicompsoft/orhane/mia-train-runs
STAGE=/nrs/scicompsoft/orhane/mia-train-scratch/sam_lmd_v1     # NOT /tmp: that is node-local
LOGS="$RUNS/jobs"

ALL_ARMS=(base stride2 stride1 stride2_small feat64 refine4 deep4 wide512)
PREFIX=sam1__                                                    # matches make_configs.PREFIX

QUEUE=${QUEUE:-gpu_b300}
GPUS=8
SLOTS=96                                                         # 12 slots/GPU on gpu_b300, 40 GB each
LABEL_QUEUE=${LABEL_QUEUE:-gpu_b300}
LABEL_SLOTS=12
# How many labelling elements run at once. Each holds one GPU; 24 is three nodes' worth, which
# leaves the rest of the queue to the training jobs of the other arms.
LABEL_CONCURRENT=${LABEL_CONCURRENT:-24}
THREADS="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
# Inductor's channels-last rewrite of conv graphs, OFF for every training job. With it on, the
# `stride1` arm -- whose mask head runs 3D convolutions over 256-cubes and so trips the size
# heuristic that turns the rewrite on -- died at its first compiled step inside cuDNN's attention
# ("stride for the last dimension ... should be 1 for Q"): the rewrite propagates a permuted
# layout into an SDPA input the cuDNN fallback does not constrain (torch 2.13). Measured: the same
# job passes with the variable set. Applied to all arms rather than the one that needs it, so no
# arm's kernels differ from another's by a layout decision inductor made for it.
COMPILE_ENV="export TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0"

# Wall clocks. The 150k-step reference run of this head on 8 B300s (promptable_lmd_v2) averaged
# ~0.48 s/step compiled, so round 0 is ~14 h and a 50k round ~7 h at the base head. The limits keep
# 3-5x headroom: `-r` plus `--resume` turns a node failure into a requeue from the last checkpoint,
# but a `-W` kill does NOT requeue. The stride arms multiply the decoder's output tensors by 8x
# (stride2) and 64x (stride1) and are given proportionally more; their real step time is one of
# the things this experiment measures (read `samples_per_s` in TensorBoard and tighten these).
WALL_R0=${WALL_R0:-72:00}
WALL_ROUND=${WALL_ROUND:-48:00}
WALL_LABEL=${WALL_LABEL:-8:00}
wall_scale () {                        # arm -> multiplier applied to the walls above
  case "$1" in stride1) echo 2 ;; stride2|stride2_small) echo 1.5 ;; *) echo 1 ;; esac
}
scaled () {                            # H:MM x factor -> H:MM
  local h=${1%%:*} f=$2; printf '%d:00' "$(awk -v h="$h" -v f="$f" 'BEGIN{printf "%d", h*f+0.5}')"
}

# The data engine's knobs. Environment-overridable rather than edited, so a run's job scripts and
# manifests record what was used.
BLOCK=${BLOCK:-512}                    # block edge in voxels of the 8 nm lattice
BLOCKS_R1=${BLOCKS_R1:-1}              # blocks per volume labelled for round 1 ...
BLOCKS_R2=${BLOCKS_R2:-4}              # ... and for round 2 (a superset: the same cells plus more)
GT_WEIGHT=${GT_WEIGHT:-0.5}            # share of training samples drawn from the GT volumes
PRED_IOU=${PRED_IOU:-0.7}              # labelling gates, stricter than eval's 0.5/0.5 -- precision
STABILITY=${STABILITY:-0.8}            #   matters more than coverage for a training target
CORPUS="$HERE/unlabeled_corpus.yaml"
GT_VOLUMES=(kasthuri15_ac3 zebrafish_fish2_quadcube1 liconn_mouse_dg hemibrain_ellipsoid_body)

# --smoke: 20-step rounds on one GPU, one 320-voxel block of one unlabeled volume and one GT volume,
# and the finished 150k-step promptable run as the labelling teacher so the smoke labels are real
# masks rather than the empty output of a 20-step model. The r1/r2 smoke stages still warm-start
# from the smoke r0/r1 of their own arm, so the `target = "algorithm"` load path is exercised on
# the arm's own architecture.
SMOKE_TEACHER=${SMOKE_TEACHER:-$RUNS/promptable_lmd_v2_20260910_104156}
SMOKE_VOLUME=${SMOKE_VOLUME:-em-zebrafish-fish2/crop-005_quadcube3_x9638_y10314_z0}
SMOKE_GT=${SMOKE_GT:-kasthuri15_ac3}

SMOKE=0 DRY=0 ROUNDS=2 SKIP_R0=0
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --smoke)   SMOKE=1 ;;
    --dry-run) DRY=1 ;;
    --rounds)  ROUNDS=$2; shift ;;
    --skip-r0) SKIP_R0=1 ;;
    *) echo "unknown flag $1" >&2; exit 2 ;;
  esac
  shift
done
ARMS=("$@"); [[ ${#ARMS[@]} -eq 0 ]] && ARMS=("${ALL_ARMS[@]}")
for arm in "${ARMS[@]}"; do
  [[ -f "$HERE/${arm}_r0.toml" ]] || { echo "unknown arm $arm (no ${arm}_r0.toml)" >&2; exit 2; }
done

ROOT=$STAGE; [[ $SMOKE -eq 1 ]] && ROOT=$STAGE/smoke
mkdir -p "$LOGS" "$ROOT/cmd" "$ROOT/rounds" "$ROOT/pseudo" "$ROOT/diag" "$ROOT/resolved"

# The scratch tree records how to regenerate itself (the /nrs retention rule).
[[ -f "$STAGE/README.md" ]] || cat > "$STAGE/README.md" <<EOF
# sam_lmd_v1 scratch

Everything here is regenerable from mia-train's \`experiments/sam_lmd_v1/\`:

- \`pseudo/<arm>/*.zarr\`   pseudo-label sidecars (\`raw\` symlinked, \`labels/sam_rN\` ours) --
                              \`bash experiments/sam_lmd_v1/submit.sh <arm>\` regenerates them from the
                              round's teacher checkpoint under $RUNS/sam1__<arm>_r<N-1>_*
- \`rounds/<arm>_rN.yaml\`   the round's miao config -- \`make_round_config.py\` over \`pseudo/<arm>\`
- \`diag/<arm>/sam_rN/\`      the labeller scored on the GT volumes -- \`pseudolabel.py diagnose\`
- \`eval/<arm>_rN/\`          whole-volume \`instances\` artifacts -- \`predict_eval.sh <arm> <N>\`
- \`cmd/\`, \`resolved/\`      the job scripts and placeholder-resolved TOMLs each job ran
- \`smoke/\`                  the same tree for \`--smoke\`; delete freely

Delete a round's sidecars once the round trained on them has been scored; the scored artifacts
under \`eval/\` are the largest thing here and can go once their leaderboard record exists.
EOF

jobid () { sed -n 's/^Job <\([0-9]*\)>.*/\1/p'; }
newest_run () {                        # experiment name -> newest run dir under $1's root
  echo "RUN=\$(ls -dt $1/${2}_*/ | head -1); RUN=\${RUN%/}"
}
latest_step () {                       # emits shell that sets STEP from \$RUN
  echo "STEP=\$(ls -d \$RUN/checkpoints/step_* | sed 's|.*step_||' | sort -n | tail -1)"
}

# stage <config> <round> <arm> [<predecessor experiment name>] [<dependency job id>]
stage () {
  local config=$1 round=$2 arm=$3 prev=${4:-} dep=${5:-}
  local name; name=$(basename "$config" .toml)
  local exp="${PREFIX}${name}" tag="$name" procs=$GPUS slots=$SLOTS
  local wall; wall=$(scaled "$([[ $round -eq 0 ]] && echo "$WALL_R0" || echo "$WALL_ROUND")" "$(wall_scale "$arm")")
  local cfg="$config" runs_root=$RUNS tail_args="--resume"

  if [[ $SMOKE -eq 1 ]]; then
    exp="smoke_${PREFIX}${name}"; tag="smoke_$name"; wall=1:00; procs=1; slots=12
    runs_root=$ROOT; tail_args="--output-root $ROOT"
    cfg="$ROOT/cmd/$tag.toml"
    sed -e "s/^experiment_name = .*/experiment_name = \"$exp\"/" \
        -e 's/^max_steps = .*/max_steps = 20/'   -e 's/^warmup_steps = .*/warmup_steps = 2/' \
        -e 's/^val_every = .*/val_every = 10/'   -e 's/^checkpoint_every = .*/checkpoint_every = 20/' \
        -e 's/^samples_per_epoch = .*/samples_per_epoch = 20/' -e 's/^dp_shard = .*/dp_shard = 1/' \
        -e 's/^num_workers = .*/num_workers = 2/' \
        "$config" > "$cfg"
  fi

  local resolved="$ROOT/resolved/$tag.toml" prologue=""
  local roundcfg="$ROOT/rounds/${arm}_r${round}.yaml"
  if [[ $round -gt 0 ]]; then
    local prev_exp="$prev"; [[ $SMOKE -eq 1 ]] && prev_exp="smoke_$prev"
    prologue="$(newest_run "$runs_root" "$prev_exp")
$(latest_step)
echo \"warm start from \$RUN/checkpoints/step_\$STEP; data \$(head -3 '$roundcfg' | tail -2 | tr '\n' ' ')\"
sed -e \"s|ROUND_CONFIG|$roundcfg|\" -e \"s|PREV_CHECKPOINT|\$RUN/checkpoints/step_\$STEP|\" '$cfg' > '$resolved'"
  else
    prologue="cp '$cfg' '$resolved'"
  fi

  local cmd="$ROOT/cmd/$tag.sh"
  { echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "$THREADS"
    echo "$COMPILE_ENV"
    echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    echo "$prologue"
    printf 'cd %q\n' "$REPO"
    printf '%s --standalone --nproc_per_node=%s src/train.py --config %q %s\n' \
      "$VENV/bin/torchrun" "$procs" "$resolved" "$tail_args"
  } > "$cmd"
  chmod +x "$cmd"

  local args=(-P "$PROJECT" -q "$QUEUE" -gpu "num=$procs" -n "$slots" -W "$wall" -r
              -J "sam1_$tag" -cwd "$REPO" -o "$LOGS/sam1_${tag}_%J.log" -e "$LOGS/sam1_${tag}_%J.err")
  [[ -n "$dep" && "$dep" != DRYRUN ]] && args+=(-w "done($dep)")
  if [[ $DRY -eq 1 ]]; then
    printf 'bsub %s bash %q\n' "${args[*]}" "$cmd" >&2; echo DRYRUN
  else
    bsub "${args[@]}" "bash '$cmd'" | jobid
  fi
}

# label <arm> <round> [<dependency job id>] -> the finalise job's id
#
# Two jobs. An ARRAY over the volumes: element i labels the i-th unlabeled volume with the previous
# round's model (or, for the last 4 elements, labels blocks of a GT volume and scores them against
# truth). Elements are idempotent -- a block already in the volume's manifest is skipped -- so a
# requeue after a wall-clock kill resumes rather than restarts. A FINALISE job on the CPU queue
# then writes the round's miao config from the manifests and pools the diagnostics; it waits on
# done(<array>), which LSF satisfies only once every element has finished, and the next training
# round depends on IT -- the config has to exist, not merely the labels.
label () {
  local arm=$1 round=$2 dep=${3:-}
  local label_name="sam_r$round" tag="L${round}_${arm}"
  local blocks; blocks=$([[ $round -eq 1 ]] && echo "$BLOCKS_R1" || echo "$BLOCKS_R2")
  local block=$BLOCK wall; wall=$(scaled "$WALL_LABEL" "$(wall_scale "$arm")")
  local teacher_exp="${PREFIX}${arm}_r$((round - 1))" runs_root=$RUNS
  local sidecars="$ROOT/pseudo/$arm" diag="$ROOT/diag/$arm/$label_name"
  local roundcfg="$ROOT/rounds/${arm}_r${round}.yaml"

  local volumes gts
  mapfile -t volumes < <(grep '^- name: ' "$CORPUS" | sed 's/^- name: //')
  gts=("${GT_VOLUMES[@]}")
  local teacher_prologue
  if [[ $SMOKE -eq 1 ]]; then
    volumes=("$SMOKE_VOLUME"); gts=("$SMOKE_GT"); blocks=1; block=320; wall=1:00; tag="smoke_$tag"
    teacher_prologue="RUN=$SMOKE_TEACHER
$(latest_step)"
  else
    teacher_prologue="$(newest_run "$runs_root" "$teacher_exp")
$(latest_step)"
  fi
  local n=${#volumes[@]} total=$(( ${#volumes[@]} + ${#gts[@]} ))
  mkdir -p "$sidecars" "$diag"

  local worker="$ROOT/cmd/$tag.sh" final="$ROOT/cmd/${tag}_final.sh"
  { echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "$THREADS"
    echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    echo "$teacher_prologue"
    echo "I=\${LSB_JOBINDEX:-1}"
    echo "VOLUMES=($(printf '%q ' "${volumes[@]}"))"
    echo "GTS=($(printf '%q ' "${gts[@]}"))"
    printf 'cd %q\n' "$REPO"
    echo "if (( I <= $n )); then"
    echo "  V=\${VOLUMES[I-1]}; echo \"element \$I/$total: label \$V with \$RUN step \$STEP\""
    echo "  $VENV/bin/python experiments/sam_lmd_v1/pseudolabel.py label \"\$RUN\" --step \"\$STEP\" \\"
    echo "    --volume \"\$V\" --out '$sidecars' --label-name $label_name --blocks $blocks --block $block \\"
    echo "    --pred-iou $PRED_IOU --stability $STABILITY"
    echo "else"
    echo "  V=\${GTS[I-$n-1]}; echo \"element \$I/$total: diagnose \$V with \$RUN step \$STEP\""
    echo "  $VENV/bin/python experiments/sam_lmd_v1/pseudolabel.py diagnose \"\$RUN\" --step \"\$STEP\" \\"
    echo "    --volume \"\$V\" --out '$diag'/\"\$V\".json --blocks $blocks --block $block \\"
    echo "    --pred-iou $PRED_IOU --stability $STABILITY"
    echo "fi"
  } > "$worker"
  { echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "$THREADS"
    printf 'cd %q\n' "$REPO"
    echo "$VENV/bin/python experiments/sam_lmd_v1/make_round_config.py --sidecar-root '$sidecars' \\"
    echo "    --label-name $label_name --out '$roundcfg' --gt-weight $GT_WEIGHT --expect $n --verify"
    echo "$VENV/bin/python experiments/sam_lmd_v1/pseudolabel.py summarize '$diag' || echo 'diagnostics summary failed (non-fatal)'"
  } > "$final"
  chmod +x "$worker" "$final"

  local aargs=(-P "$PROJECT" -q "$LABEL_QUEUE" -gpu "num=1" -n "$LABEL_SLOTS" -W "$wall" -r
               -J "sam1_${tag}[1-$total]%$LABEL_CONCURRENT" -cwd "$REPO"
               -o "$LOGS/sam1_${tag}_%J_%I.log" -e "$LOGS/sam1_${tag}_%J_%I.err")
  [[ -n "$dep" && "$dep" != DRYRUN ]] && aargs+=(-w "done($dep)")
  local arr fin
  if [[ $DRY -eq 1 ]]; then
    printf 'bsub %s bash %q\n' "${aargs[*]}" "$worker" >&2; arr=DRYRUN
  else
    arr=$(bsub "${aargs[@]}" "bash '$worker'" | jobid)
  fi
  local fargs=(-P "$PROJECT" -q local -n 4 -W 2:00 -J "sam1_${tag}_final" -cwd "$REPO"
               -o "$LOGS/sam1_${tag}_final_%J.log" -e "$LOGS/sam1_${tag}_final_%J.err")
  [[ "$arr" != DRYRUN ]] && fargs+=(-w "done($arr)")
  if [[ $DRY -eq 1 ]]; then
    printf 'bsub %s bash %q\n' "${fargs[*]}" "$final" >&2; fin=DRYRUN
  else
    fin=$(bsub "${fargs[@]}" "bash '$final'" | jobid)
  fi
  echo "    $tag: array $arr [1-$total]%$LABEL_CONCURRENT -> finalise $fin" >&2
  echo "$fin"
}

summary=""
for arm in "${ARMS[@]}"; do
  if [[ $SKIP_R0 -eq 1 ]]; then
    # Round 0's run directory must already exist; the labelling job resolves it by name.
    r=""; line=$(printf '%-14s r0=(existing)' "$arm")
  else
    r=$(stage "$HERE/${arm}_r0.toml" 0 "$arm")
    line=$(printf '%-14s r0=%s' "$arm" "$r")
  fi
  prev_exp="${PREFIX}${arm}_r0"
  for (( round = 1; round <= ROUNDS; round++ )); do
    l=$(label "$arm" "$round" "$r")
    r=$(stage "$HERE/${arm}_r${round}.toml" "$round" "$arm" "$prev_exp" "$l")
    line+=$(printf '  L%s=%s  r%s=%s' "$round" "$l" "$round" "$r")
    prev_exp="${PREFIX}${arm}_r${round}"
  done
  summary+="$line"$'\n'
done
echo
printf '%s' "$summary"
echo
echo "logs: $LOGS/sam1_*   scratch: $ROOT"
