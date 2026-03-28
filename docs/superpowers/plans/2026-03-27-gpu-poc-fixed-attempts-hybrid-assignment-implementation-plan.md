# GPU POC Fixed-Attempts And Hybrid Assignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add controller-level fixed-attempt semantics and a hybrid CPU assignment backend to the standalone batched GPU POC so hard-case batched experiments become meaningful.

**Architecture:** Keep GPU build/peel batched, but move global attempt accounting out to a host-side controller loop. Add deterministic global attempt numbering across batches, allow batch-boundary overshoot for fixed-attempt runs, and introduce an `assignment_backend` switch so solved survivors can use CPU assignment/verify instead of the current scalar GPU assignment. Best-coverage search remains deferred.

**Tech Stack:** CUDA C++, existing `experiments/gpu_batched_peeling_poc` binary, existing benchmark runner/tests, local GB10 bounded runs.

---

## File Map

- Modify: `experiments/gpu_batched_peeling_poc/main.cu`
  - add fixed-attempt and assignment-backend options
  - change seed generation to use global attempt ids
  - add host-side multi-batch controller loop
  - add hybrid CPU assignment/verify path for solved survivors
  - extend output/reporting
- Modify: `experiments/gpu_batched_peeling_poc/README.md`
  - document new options and semantics
- Modify: `tests/run_gpu_poc_geometry_smoke_test.cmake`
  - extend or add checks for the new output fields if they affect the default smoke shape
- Create: `docs/superpowers/reports/2026-03-27-gpu-poc-fixed-attempts-hybrid-baseline.md`
  - capture the first bounded comparison of GPU assignment vs CPU assignment backend
- Modify: `agents/PERFECTHASH-GPU-SOLVING-NOTES.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-LOG.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-TODO.md`

## Task 1: Add Fixed-Attempts Control Plane

**Files:**
- Modify: `experiments/gpu_batched_peeling_poc/main.cu`
- Modify: `experiments/gpu_batched_peeling_poc/README.md`

- [ ] **Step 1: Write the failing behavior check**

Use a tiny smoke run with a new `--fixed-attempts` flag and confirm it currently
fails because the option does not exist.

- [ ] **Step 2: Add new options**

Add:

- `--fixed-attempts <n>`
- `--first-solution-wins`

Defaults:

- `fixed_attempts = 0`
- `first_solution_wins = false`

- [ ] **Step 3: Make seed generation use global attempt ids**

Refactor seed generation so each attempt uses:

- `attempt_base + batch_local_index`

instead of a batch-local graph index only.

- [ ] **Step 4: Add host-side batch loop**

If `fixed_attempts == 0`:

- preserve current one-batch behavior

If `fixed_attempts > 0`:

- run repeated batches
- stop at a batch boundary once:
  - `actual_attempts_tried >= fixed_attempts`, or
  - `first_solution_wins` and a batch contains one or more solved attempts

- [ ] **Step 5: Report controller-level fields**

Add to JSON/human output:

- `requested_fixed_attempts`
- `actual_attempts_tried`
- `batches_run`
- `first_solution_wins`
- `first_solved_attempt`

- [ ] **Step 6: Verify**

Run a tiny bounded case and confirm:

- multiple batches execute
- actual attempts can overshoot the request at a batch boundary
- output fields are present

- [ ] **Step 7: Commit**

```bash
git add experiments/gpu_batched_peeling_poc/main.cu \
        experiments/gpu_batched_peeling_poc/README.md
git commit -m "Add fixed-attempt control plane to GPU POC"
```

## Task 2: Add Hybrid CPU Assignment Backend

**Files:**
- Modify: `experiments/gpu_batched_peeling_poc/main.cu`
- Modify: `experiments/gpu_batched_peeling_poc/README.md`

- [ ] **Step 1: Write the failing backend check**

Run with `--assignment-backend cpu` and confirm the option does not exist yet.

- [ ] **Step 2: Add backend option**

Add:

- `--assignment-backend <gpu|cpu>`

Default:

- `gpu`

- [ ] **Step 3: Implement hybrid CPU assignment**

For `assignment_backend=cpu`:

- GPU still performs build/peel
- solved survivors are identified after peel
- CPU assignment/verify runs only on those survivors

Keep ordered-index semantics unchanged.

- [ ] **Step 4: Preserve timing/reporting clarity**

Update output to record:

- `assignment_backend`

Reuse the existing CPU stage timing reporting so the hybrid path is measurable.

- [ ] **Step 5: Verify**

Run paired bounded cases on the same input:

- `assignment_backend=gpu`
- `assignment_backend=cpu`

Expected:

- correctness remains aligned
- output clearly distinguishes the backend

- [ ] **Step 6: Commit**

```bash
git add experiments/gpu_batched_peeling_poc/main.cu \
        experiments/gpu_batched_peeling_poc/README.md
git commit -m "Add hybrid CPU assignment backend to GPU POC"
```

## Task 3: Refresh Smoke Coverage If Needed

**Files:**
- Modify: `tests/run_gpu_poc_geometry_smoke_test.cmake`

- [ ] **Step 1: Extend smoke expectations only if defaults changed**

If the new output fields affect the default smoke contract, assert them here.

- [ ] **Step 2: Verify**

Run:

```bash
ctest --test-dir build-cuda --output-on-failure -R 'perfecthash\.gpu\.poc\.geometry\.smoke'
```

- [ ] **Step 3: Commit**

```bash
git add tests/run_gpu_poc_geometry_smoke_test.cmake
git commit -m "Refresh GPU POC smoke expectations"
```

## Task 4: Run A Bounded Hybrid Baseline

**Files:**
- Create: `docs/superpowers/reports/2026-03-27-gpu-poc-fixed-attempts-hybrid-baseline.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-NOTES.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-LOG.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-TODO.md`

- [ ] **Step 1: Run bounded local cases**

Use small, safe runs only. Suggested first matrix:

- generated `8193`
- `batch=128`
- `threads=128`
- `device-serial`
- `device_serial_peel_geometry=block`
- compare:
  - `assignment_backend=gpu`
  - `assignment_backend=cpu`

Then run one real-key bounded case, likely:

- `HologramWorld-31016.keys`
- conservative batch

- [ ] **Step 2: Record fixed-attempt behavior**

Run at least one bounded `--fixed-attempts` case and record:

- requested attempts
- actual attempts
- batches run
- whether first-solution mode triggered

- [ ] **Step 3: Write the report**

Create:

- `docs/superpowers/reports/2026-03-27-gpu-poc-fixed-attempts-hybrid-baseline.md`

- [ ] **Step 4: Update ledgers**

Update:

- `agents/PERFECTHASH-GPU-SOLVING-NOTES.md`
- `agents/PERFECTHASH-GPU-SOLVING-LOG.md`
- `agents/PERFECTHASH-GPU-SOLVING-TODO.md`

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/reports/2026-03-27-gpu-poc-fixed-attempts-hybrid-baseline.md \
        agents/PERFECTHASH-GPU-SOLVING-NOTES.md \
        agents/PERFECTHASH-GPU-SOLVING-LOG.md \
        agents/PERFECTHASH-GPU-SOLVING-TODO.md
git commit -m "Record fixed-attempt and hybrid assignment baseline"
```
