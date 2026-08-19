#!/usr/bin/env bash
# Mutual exclusion between the H100 and H200 twins of one training stage.
#
# A stage is submitted twice, once to each GPU queue, so whichever generation frees up first runs
# it. Both twins carry the same `experiment_name` and pass `--resume`, so they are interchangeable:
# either can continue the same run directory from its last checkpoint. The only thing that must
# never happen is both running at once, which is what this lock is for.
#
#   claim.sh <lock dir> <command script>
#
# The loser exits **non-zero on purpose**. A dependent stage waits on `done(A) || done(B)`, and
# `done()` requires a zero exit -- so a loser that exited 0 would satisfy the dependency and release
# the next stage while the winner was still training.
set -uo pipefail

LOCK=$1
CMD=$2
mkdir -p "$(dirname "$LOCK")"

holder_alive () {
  local id
  id=$(cat "$LOCK/jobid" 2>/dev/null) || return 1
  [[ -n "$id" ]] || return 1
  # STAT is the third column of bjobs' second line; RUN or PEND means the holder is still ours.
  [[ "$(bjobs "$id" 2>/dev/null | awk 'NR==2 {print $3}')" =~ ^(RUN|PEND)$ ]]
}

# Cancel the other twin as soon as this one wins, rather than leaving it queued to be dispatched
# later, hold a whole 96-slot node for the few seconds it takes to read this lock, and exit 42.
# The dependency is `done(a) || done(b)`, so a killed twin (EXIT) still leaves the pair satisfiable
# by the winner. Only PEND twins are touched: a RUNNING one means the lock logic already failed and
# killing it would be the wrong response.
release_twin () {
  local other
  for other in $(bjobs -J "$LSB_JOBNAME" -noheader -o "jobid stat" 2>/dev/null \
                 | awk '$2=="PEND" {print $1}'); do
    [[ "$other" == "$LSB_JOBID" ]] && continue
    echo "[twin] releasing queued twin $other so it does not take a node to read this lock"
    bkill "$other" >/dev/null 2>&1 || true
  done
}

if mkdir "$LOCK" 2>/dev/null; then
  echo "$LSB_JOBID" > "$LOCK/jobid"
  echo "[twin] claimed by $LSB_JOBID on $(hostname)"
  release_twin
else
  if holder_alive; then
    echo "[twin] already claimed by job $(cat "$LOCK/jobid"); exiting so this node is released"
    exit 42
  fi
  # The holder is gone -- a node failure, or a kill. Take over, but pause first: if both twins
  # discover the stale claim together, the wait plus the re-check keeps one of them out.
  sleep $(( (RANDOM % 20) + 5 ))
  if holder_alive; then
    echo "[twin] claimed during takeover by $(cat "$LOCK/jobid"); exiting"
    exit 42
  fi
  echo "$LSB_JOBID" > "$LOCK/jobid"
  echo "[twin] took over a stale claim from a job that is no longer running"
fi

# Released on any exit, including a wall-time kill, so a rerun or the other twin can pick the stage
# up and resume from the last checkpoint.
#
# The command is **not** `exec`ed. `exec` replaces this shell with the training process, which
# discards the EXIT trap along with it, so the lock would outlive every job that ever held it.
# That is exactly what happened up to 2026-08-14: all six stage locks were still present, each
# naming a job that had finished hours earlier, and the takeover branch above was silently doing
# all the work -- paying a 5-24s sleep every time and leaning on a re-check that only resolves a
# simultaneous double-takeover probabilistically. Running the command as a child costs one extra
# shell process and makes the release path the normal one.
trap 'rm -rf "$LOCK"' EXIT
bash "$CMD"
exit $?
