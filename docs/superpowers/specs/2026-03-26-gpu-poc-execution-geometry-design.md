# GPU POC Execution Geometry Design

**Date:** 2026-03-26

## Goal

Improve the standalone batched GPU POC by adding execution-geometry
hyperparameters and measurements before changing the kernel memory model.

This is an intentionally narrow stage-1 slice. It exists to answer which kernel
geometry is worth optimizing further.

## Context

The current POC has mixed execution shapes:

- build, frontier collection, frontier peel, and verify are grid-stride kernels
  parameterized by `--threads`
- assignment is one thread per graph
- `device-serial` peel is also one thread per graph
- there is no shared-memory usage
- there is no CUB block primitive usage
- there is no `ITEMS_PER_THREAD` parameterization

The user wants to explore:

- one-thread, one-warp, and one-block per graph variants
- geometry hyperparameters
- shared memory and CUB later

## Approaches Considered

### Approach 1: Jump straight to shared-memory/CUB refactor

Pros:

- moves directly toward a more GPU-native implementation

Cons:

- mixes two variables at once:
  - execution geometry
  - memory movement strategy
- makes it harder to explain wins or regressions

### Approach 2: Geometry sweep first, then memory-model refactor

Pros:

- isolates the first question cleanly
- gives a performance baseline for later shared-memory/CUB work
- lower risk and easier to validate

Cons:

- adds an intermediate step before deeper optimization

### Approach 3: Benchmark-only sweep with no code changes

Pros:

- fastest to run

Cons:

- current POC does not expose the alternative per-graph geometries we care about
- does not answer warp-per-graph or block-per-graph questions

## Chosen Approach

Use approach 2.

Stage 1 will add execution-geometry modes and measurement only. In practice,
the first real implementation target is `device-serial` peel geometry. Shared-
memory staging, CUB block primitives, `ITEMS_PER_THREAD` bulk-loading, and
cooperative assignment are explicitly deferred to later work.

## Scope

### In Scope

- Add explicit per-graph execution geometry modes for the kernels that are
  currently scalar per graph:
  - `device-serial` peel
- Keep `assign_geometry` as a surfaced configuration/reporting field for future
  work, but do not require stage 1 to make assignment cooperative
- Keep current build/collect/verify kernels intact except for any minimal
  parameter exposure needed for measurement
- Extend JSON/human output so runs record the selected geometry
- Extend the benchmark runner/config if needed so the new geometry modes can be
  exercised safely
- Run a bounded local sweep on safe datasets only

### Out of Scope

- shared-memory caching
- CUB `BlockLoad` / `BlockStore`
- `ITEMS_PER_THREAD` tuning
- changing hash formulas
- changing seed-generation semantics
- folding the POC into `PerfectHashCreate`

## Design

### New geometry knobs

Add separate geometry selection for the kernels that operate per graph:

- `assign-geometry`
- `device-serial-peel-geometry`

Initial legal values:

- `thread`
- `warp`
- `block`

The default should remain behaviorally equivalent to today:

- assignment: `thread`
- device-serial peel: `thread`

The existing `--threads` option remains the CUDA block size for stage 1.
Geometry selection changes how threads within that block are partitioned across
graphs; it does not introduce a second block-size parameter yet.

### Kernel strategy

#### Assignment

`assign_geometry` remains a surfaced field in stage 1, but assignment execution
stays on the current scalar/thread path for now.

Reason:

- ordered assignment is currently a reverse-peel dependency chain
- each step reads `Assigned[other]` before writing `Assigned[owner]`
- so simple warp/block-per-graph parallelization is not correct without
  additional peel-layer or ready-state metadata

#### Device-serial peel

Provide the same three implementations:

- `thread`
- `warp`
- `block`

For stage 1, these may still use global-memory degree/XOR state. The only
change is who cooperates on a graph and how work is partitioned.

Validation rules for stage 1:

- `thread`:
  - any positive `--threads` value already accepted by the POC remains valid
- `warp`:
  - reject non-multiples of 32 at argument-validation time
- `block`:
  - keep using `--threads` as the block size
  - require `--threads >= 32` so the block mode being measured is not
    degenerate

### Measurement

Record the selected geometries in output:

- human output
- JSON output

JSON should expose explicit fields so the benchmark runner does not need to
infer them from free-form text:

- `assign_geometry`
- `device_serial_peel_geometry`

Keep existing stage timings and add no new timing buckets yet. The point is to
compare geometry choices within the current timing surface.

### Safety constraints

- reuse the existing batch memory guard and auto-scaling
- keep the first sweep bounded to repo-local or already-approved tiny datasets
- do not expand the benchmark runner allowlist aggressively in this stage

## Verification

Stage 1 is complete when:

1. the POC supports selecting/reporting geometry for assignment and executing
   real geometry variants for `device-serial` peel
2. GPU/CPU correctness remains intact on the existing safe regression cases
3. output clearly identifies the selected geometry
4. at least one bounded sweep shows whether `thread`, `warp`, or `block` is the
   better follow-on target for shared-memory/CUB work

## Stage 2 Preview

If stage 1 identifies a promising geometry, the next optimization pass will
target that geometry for:

- shared-memory staging
- CUB block primitives
- possible `ITEMS_PER_THREAD` parameterization

Cooperative assignment should only be revisited after the solver records enough
metadata to support reverse-layer assignment safely.

That work is intentionally deferred until after stage 1 data exists.
