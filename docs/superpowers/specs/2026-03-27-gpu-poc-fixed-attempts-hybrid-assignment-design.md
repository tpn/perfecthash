# GPU POC Fixed-Attempts And Hybrid Assignment Design

**Date:** 2026-03-27

## Goal

Extend the standalone batched GPU POC so it can model the CPU solver's
attempt-budget semantics more faithfully and so it can evaluate a hybrid
GPU-peel / CPU-assign execution path.

## Why This Slice

The current POC still collapses two concepts into one:

- `batch`
- total number of attempts evaluated

That is fine for smoke tests, but it is wrong for hard cases where a realistic
run may need tens of thousands or millions of attempts.

Separately, recent measurements showed:

- GPU peel is now becoming competitive or better on some batched cases
- current scalar GPU assignment and verify are still much more expensive than
  the CPU equivalents

On GB10, unified RAM/VRAM makes a hybrid design realistic:

- GPU build/peel many attempts
- compact solved survivors
- CPU assign/verify those survivors

## Approaches Considered

### Approach 1: Exact fixed-attempt enforcement

Run exactly `N` logical attempts by masking off the tail of the final batch.

Pros:

- cleaner accounting

Cons:

- adds awkward tail logic into the hot path
- not necessary for the user's stated needs

### Approach 2: Batch-boundary fixed-attempt enforcement

Run full batches until the requested attempt budget is met or exceeded, then
report both requested and actual attempts.

Pros:

- simpler control flow
- better matches batched GPU throughput
- explicitly approved by the user

Cons:

- actual attempts can exceed requested attempts slightly

### Approach 3: Keep assignment on GPU and optimize it next

Pros:

- preserves a fully device-resident pipeline

Cons:

- assignment is currently the wrong optimization target
- ordered assignment still has a dependency chain

### Approach 4: Hybrid GPU-peel / CPU-assign now

Pros:

- directly addresses the current bottleneck profile
- fits GB10 well
- avoids pretending assignment is ready for GPU parallelization

Cons:

- adds a backend split
- sacrifices “all on GPU” purity for now

## Chosen Approach

Combine approach 2 and approach 4.

Implement:

1. a controller-level `fixed_attempts` loop with batch-boundary overshoot
2. deterministic global attempt numbering across batches
3. a hybrid CPU assignment/verify backend for solved survivors

Defer:

- exact tail masking
- best-coverage / predicate search
- cooperative GPU assignment

## Design

### New control-plane knobs

Add:

- `--fixed-attempts <n>`
  - total attempt budget across batches
- `--first-solution-wins`
  - stop after the first batch that contains at least one solved graph
- `--assignment-backend <gpu|cpu>`
  - `gpu` means current scalar GPU assignment/verify path
  - `cpu` means GPU build/peel followed by CPU assignment/verify for solved
    survivors only

Defaults:

- `fixed_attempts = 0`
  - means current one-batch behavior
- `first_solution_wins = false`
- `assignment_backend = gpu`

### Deterministic attempt numbering

Each attempt gets a global attempt id:

- `attempt_base + local_batch_index`

Seed generation must be derived from this global attempt id, not from a
batch-local `0..batch-1` index, so repeated runs stay reproducible regardless of
how the work is chunked into batches.

### Batch loop semantics

If `fixed_attempts == 0`:

- current one-batch behavior remains

If `fixed_attempts > 0`:

- run full batches of size `batch`
- stop at the first batch boundary where:
  - `actual_attempts_tried >= fixed_attempts`, or
  - `first_solution_wins` and the batch contains one or more solved graphs

Report:

- `requested_fixed_attempts`
- `actual_attempts_tried`
- `batches_run`
- `first_solved_attempt`
- `first_solved_batch`

### Hybrid CPU assignment backend

For `assignment_backend=cpu`:

- GPU still does build and peel
- solved survivors are identified after peel
- CPU assignment/verify is performed only for survivors
- assignment/verify timings should reflect only those solved survivors

This backend should still preserve ordered indexing semantics and produce the
same pass/fail result as the current CPU reference.

### Result reporting

JSON and human output should make the controller-level semantics explicit:

- `assignment_backend`
- `requested_fixed_attempts`
- `actual_attempts_tried`
- `batches_run`
- `first_solution_wins`
- `first_solved_attempt` if any

CPU timing output already has all-attempt and solved-only views; keep those and
reuse them for the new hybrid comparisons.

## Verification

This slice is complete when:

1. fixed-attempt runs evaluate multiple batches deterministically
2. requested vs actual attempts are reported clearly
3. first-solution-wins stops at a batch boundary and reports the winning global
   attempt id
4. `assignment_backend=cpu` works on the same solved survivors as the GPU path
5. bounded local runs can compare:
   - GPU peel + GPU assignment
   - GPU peel + CPU assignment

## Deferred Work

- best-coverage / predicate search
- exact tail masking for fixed attempts
- cooperative GPU assignment
- peel-layer metadata and reverse-layer assignment
