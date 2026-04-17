# GPU POC Follow-Up

## Current Status

The mainline repository now contains the `Chm02` CUDA correctness-first bring-up
 work merged via PR `#84`, but it does **not** contain the standalone batched
 GPU constructor proof-of-concept or its detailed experimental ledgers.

The branch of record for that ongoing work is:

- `gpu-batched-peeling-poc`

That branch is based on `main` and carries the standalone POC, benchmark
 scaffolding, and the detailed experiment history.

## Best Known POC Baseline

The most recent best-known standalone GPU constructor configuration from the
 `gpu-batched-peeling-poc` branch was:

- `solve-mode=device-serial`
- `device-serial-peel-geometry=block-shared-vertex`
- `assignment-backend=cpu`
- `allocation-mode=explicit-device`
- `hash-function=Mulshrolate4RX`
- `batch=16`
- `threads=1024`
- `build-threads=1024`

On the first large key set that was pushed far enough (`24,810,562` keys), that
 configuration reached roughly:

- `7343.363 ms` GPU wall time for `32` attempts
- about `9.34 ns/attempted-key`

This was better than the pinned 10-core CPU baseline that had previously landed
 around `12.6-12.9 ns/attempted-key` on the same `32`-attempt budget.

## What Was Tried Recently

The recent successful changes on the POC branch were:

- `block-shared-vertex`
  - keep the same `block-shared` round semantics
  - store only the frontier vertex
  - recompute the edge in the processing phase
- a build-kernel retile
  - 2D graph-tiled launch
  - shared staging of per-graph seeds
  - optional `--build-threads` override

The recent *rejected* ideas were:

- fully parallel worklist peel
  - fast, but not correctness-aligned
- sequential worklist GPU baselines
  - correct, but too serial to beat `block-shared`
- several broad CUB/warp-aggregation attempts in the peel path
  - either slower or too risky relative to the gains

## Recommended Next Steps

If this work is picked up again, the recommended order is:

1. Resume on `gpu-batched-peeling-poc`, not on `main`.
2. Reconfirm the current baseline:
   - `block-shared-vertex`
   - `batch=16`
   - `threads=1024`
   - `build-threads=1024`
3. Move up to the next safe large key-size tier, likely around `43.9M`, using
   the same one-job-at-a-time approach.
4. Continue with **low-risk** peel refinements only:
   - preserve the current round semantics
   - prefer small traffic reductions over semantic changes
5. After constructor measurements are stable, build the lookup/probe harness for
   the dimension/fact-table use case.

## Practical Reminder

Before resuming work, update the local branch with current `main` first:

- `git checkout gpu-batched-peeling-poc`
- `git fetch origin --prune`
- `git merge origin/main`

That ensures the standalone POC branch keeps the latest merged `Chm02` CUDA
 fixes and test infrastructure.
