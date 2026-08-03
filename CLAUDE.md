# `mia-train` Development Guide

## Core Principles

1. **PyTorch-native training techniques.** Core training infrastructure and parallelism 
code must not depend on non-PyTorch libraries. Don't use external training libraries 
like `accelerate` or `lightning`. Techniques with moderate-to-large complexity belong 
in their proper upstream repo (pytorch/pytorch for parallelisms, pytorch/data for data 
loaders, etc.).

2. **Investigate root cause before patching.** Don't land band-aid fixes. Understand
*why* something fails before proposing a solution. If a change seems to help but you 
can't explain why, dig deeper.

3. **Reuse over duplication.** Before writing new code, check if existing implementations
already handle the case. Unify similar code paths across models rather than creating
per-model wrappers. If upstream (torchao, PyTorch) already provides functionality, use it.

4. **Don't leak experiments into core.** Use the `mia-train/experiments/` folder for any 
experiments. Don't modify core code to accommodate experiment-specific needs (e.g. don't add 
`if experiment_x:` branches to core files). Deprecated files should be removed, not updated.

5. **Protect battle-tested code paths.** Be cautious changing converged behavior. Flag 
potential silent breakage of existing user code or checkpoints. When in doubt, ask.

6. **Audit all callsites.** When changing shared code (common model components, config fields, distributed utilities), check and update every callsite. This includes all model variants.

7. **No speculative defensive checks.** Don't add checks, casts, fallbacks, or conversions 
"just in case." Only validate explicit contracts, user-facing configuration, or invariants 
whose failure would otherwise be silent or unclear.

## Cluster & Execution Rules

0. **Read the cluster hint sheet first.** `.claude/rules/cluster.md` is the admin-authored
Janelia HPC hint sheet (LSF queues, the slot model, storage tiers, and a list of mistakes AI
agents commonly make on this cluster). Read it before submitting jobs or advising on job scripts.
Repo-specific submission recipes live in `deploy/lsf/README.md`.

1. **Environment & Paths**
- **Local Environment Config:** Read `configs/cluster/active.toml` for dataset paths, checkpoint directories, and virtual environment paths. Never hardcode absolute cluster paths in `src/`.
- **Python Executable:** When instructed to run scripts or tests on the cluster, use the Python binary defined in `configs/cluster/active.toml` or the active virtual environment.
- **`/tmp` is node-local.** Anything a submitted job must read (scripts, configs) or write (logs,
checkpoints) has to live on shared storage — `/groups/...` (PRFS) or `/nrs/...` (NRS). Staging a
job script in `/tmp` fails silently-ish: the job lands on another host, the path doesn't exist,
and LSF reports a bare non-zero exit.

2. **Job Submission & Deployment**
- **Scheduler Instructions:** Refer to `deploy/lsf/README.md` (or `deploy/slurm/README.md`) for job submission commands (`bsub`/`sbatch`), queue names, and interactive debug commands.
- **Submitting Jobs:** When generating batch submission scripts, base them on templates in `deploy/lsf/` and verify that execution commands invoke `torchrun` pointing to `src/train.py`.
- **Always pass `-P`.** Members of the `scicompsoft` group must specify the billing project
explicitly, using `[scheduler].project` from `configs/cluster/active.toml`; the default would
charge the wrong group. `lsfgroup $USER` shows the default.

3. **Running the Test Suite**
- **Don't run the multi-process tiers inline.** An interactive VSCode/dev session is typically a
1-slot allocation (`LSB_DJOB_NUMPROC=1`). `tests/unit/` is single-process and fine to run
directly, but `tests/distributed/` spawns up to 4 Gloo processes and must be submitted with
enough slots (`bsub -P <project> -n 4 -W 0:20 ...`), or it oversubscribes the allocation and risks
LSF's memory kill. See `deploy/lsf/README.md` for the full command.
- **Set thread limits.** Export `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` in the
submitted command; unset, each torch process threads to the node's full physical core count
rather than the slot allocation.