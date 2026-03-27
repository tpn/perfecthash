# GPU POC Execution Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stage-1 execution-geometry modes to the batched GPU POC, verify correctness stays intact, and collect a bounded baseline showing whether `thread`, `warp`, or `block` is the better follow-on target.

**Architecture:** Keep the current POC execution model intact and only change how per-graph work is partitioned where it is technically valid. In practice, this plan now implements real geometry variants for `device-serial` peel, while `assign_geometry` remains surfaced but non-operative until the solver records enough metadata to support reverse-layer assignment safely. Keep `--threads` as the CUDA block size, reject invalid geometry/thread combinations early, and extend the benchmark surface so geometry choices are recorded in JSON and human output. No shared-memory staging, CUB block primitives, or `ITEMS_PER_THREAD` changes in this plan.

**Tech Stack:** CUDA C++, existing `experiments/gpu_batched_peeling_poc` binary, Python benchmark runner, CMake/CTest, local GB10 safe-smoke runs.

---

## File Map

- Modify: `experiments/gpu_batched_peeling_poc/main.cu`
  - add new geometry enums/options
  - add argument validation rules
  - add real warp/block variants for `device-serial` peel
  - keep assignment geometry surfaced for future work
  - include selected geometries in output
- Modify: `experiments/gpu_batched_peeling_poc/README.md`
  - document new flags and validation rules
- Modify: `scripts/benchmark_gpu_solver.py`
  - pass new geometry flags through to the POC when configured
- Modify: `scripts/benchmark_gpu_solver_config.json`
  - add one tiny-safe geometry-aware variant or geometry fields for the existing POC variant
- Modify: `tests/test_benchmark_gpu_solver.py`
  - extend config/argv tests for the new geometry fields
- Create: `tests/run_gpu_poc_geometry_smoke_test.cmake`
  - tiny smoke test for one non-default geometry mode
- Modify: `tests/CMakeLists.txt`
  - register the new POC geometry smoke test if the experiment target exists
- Create: `docs/superpowers/reports/2026-03-26-gpu-poc-execution-geometry-baseline.md`
  - capture the bounded stage-1 measurement results
- Modify: `agents/PERFECTHASH-GPU-SOLVING-LOG.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-NOTES.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-TODO.md`

## Task 1: Add Geometry Options and Validation

**Files:**
- Modify: `experiments/gpu_batched_peeling_poc/main.cu`
- Modify: `experiments/gpu_batched_peeling_poc/README.md`

- [ ] **Step 1: Write the failing option-surface expectation**

Add a tiny smoke command to the plan notes and verify it currently fails because
the flags do not exist:

```bash
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc \
  --edges 16 \
  --batch 1 \
  --threads 64 \
  --solve-mode device-serial \
  --assign-geometry warp \
  --device-serial-peel-geometry warp \
  --output-format json
```

Expected:
- non-zero exit
- unknown argument error

- [ ] **Step 2: Add new enums and defaults**

In `main.cu`, add:

```cpp
enum class GraphGeometry {
    Thread,
    Warp,
    Block,
};
```

and new `Options` fields:

```cpp
GraphGeometry AssignGeometry = GraphGeometry::Thread;
GraphGeometry DeviceSerialPeelGeometry = GraphGeometry::Thread;
```

- [ ] **Step 3: Add parsing helpers and CLI flags**

Add:

- `ParseGraphGeometry()`
- `GraphGeometryToString()`
- `--assign-geometry`
- `--device-serial-peel-geometry`

Also update `--help` text in `ParseOptions()`.

- [ ] **Step 4: Add validation rules**

Reject invalid combinations during option validation:

- `warp` requires `--threads % 32 == 0`
- `block` for stage 1 requires `--threads >= 32`
- geometry flags only apply to `--solve-mode device-serial` for the peel kernel

Expected behavior:
- reject at argument-parse/validation time
- print a clear error message

- [ ] **Step 5: Include geometry in output**

Add the selected geometry names to:

- human output
- JSON output

Required JSON fields:

- `assign_geometry`
- `device_serial_peel_geometry`

- [ ] **Step 6: Update README**

Document:

- new flags
- defaults
- validation rules
- that stage 1 changes execution geometry only, not memory staging

- [ ] **Step 7: Run the new option smoke**

Run:

```bash
cmake --build build/gpu-batched-peeling-poc -j2
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc \
  --edges 16 \
  --batch 1 \
  --threads 64 \
  --solve-mode device-serial \
  --assign-geometry warp \
  --device-serial-peel-geometry warp \
  --output-format json
```

Expected:
- exit `0`
- JSON contains the two geometry fields

- [ ] **Step 8: Commit**

```bash
git add experiments/gpu_batched_peeling_poc/main.cu \
        experiments/gpu_batched_peeling_poc/README.md
git commit -m "Add GPU POC execution geometry options"
```

## Task 2: Assignment Geometry Blocker Note

**Files:**
- Modify: `experiments/gpu_batched_peeling_poc/main.cu`

- [ ] **Step 1: Record the blocker**

Document the reason assignment geometry is deferred:

- reverse ordered assignment currently has a loop-carried dependency chain
- each step reads `Assigned[other]` before writing `Assigned[owner]`
- simple warp/block wrappers are not a valid implementation

- [ ] **Step 2: Keep `assign_geometry` surfaced but non-operative**

Do not reintroduce fake warp/block assignment execution.

- [ ] **Step 3: Revisit only after peel-layer metadata exists**

Future assignment work should be based on reverse-layer assignment, not direct
parallelization of the current scalar loop.

## Task 3: Implement Device-Serial Peel Geometry Variants

**Files:**
- Modify: `experiments/gpu_batched_peeling_poc/main.cu`

- [ ] **Step 1: Write the failing peel-geometry check**

Use the same tiny safe run shape, but vary only
`--device-serial-peel-geometry=warp`.

Expected before implementation:
- non-zero exit or no distinct path yet

- [ ] **Step 2: Split `device-serial` peel into geometry-specific kernels**

Refactor:

- current scalar implementation into `PeelGraphsDeviceSerialThreadKernel`
- add `PeelGraphsDeviceSerialWarpKernel`
- add `PeelGraphsDeviceSerialBlockKernel`

All three should preserve:

- `OwnerVertex`
- `PeelOrder`
- `PeeledCount`
- `GraphRounds`

- [ ] **Step 3: Add host-side dispatch**

Dispatch peel by:

```cpp
switch (Opts.DeviceSerialPeelGeometry) {
    case GraphGeometry::Thread: ...
    case GraphGeometry::Warp: ...
    case GraphGeometry::Block: ...
}
```

- [ ] **Step 4: Run tiny correctness sweeps**

Run all combinations relevant to the current truthful implementation:

- peel `thread`, assign `thread`
- peel `warp`, assign `thread`
- peel `block`, assign `thread`

Expected:
- exit `0`
- `mismatches == 0`

- [ ] **Step 5: Commit**

```bash
git add experiments/gpu_batched_peeling_poc/main.cu
git commit -m "Add GPU POC device-serial peel geometry variants"
```

## Task 4: Extend Runner and Add Geometry Smoke Coverage

**Files:**
- Modify: `scripts/benchmark_gpu_solver.py`
- Modify: `scripts/benchmark_gpu_solver_config.json`
- Modify: `tests/test_benchmark_gpu_solver.py`
- Create: `tests/run_gpu_poc_geometry_smoke_test.cmake`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Add failing runner tests**

Extend `tests/test_benchmark_gpu_solver.py` with config cases that include:

```json
{
  "assign_geometry": "warp",
  "device_serial_peel_geometry": "warp"
}
```

Expected before implementation:
- runner ignores or rejects unknown fields

- [ ] **Step 2: Thread geometry fields through the runner**

Update the POC variant handling so these fields map to:

- `--assign-geometry`
- `--device-serial-peel-geometry`

- [ ] **Step 3: Update config**

Add one tiny-safe variant or extend the existing tiny-safe POC variant with
explicit geometry fields. Keep the execution allowlist bounded.

- [ ] **Step 4: Add a geometry smoke test**

Create `tests/run_gpu_poc_geometry_smoke_test.cmake` that runs a tiny POC case:

- `--edges 16`
- `--batch 1`
- `--threads 64`
- `--solve-mode device-serial`
- `--assign-geometry warp`
- `--device-serial-peel-geometry warp`
- `--output-format json`

Validate:

- exit `0`
- JSON contains geometry fields

- [ ] **Step 5: Register and run tests**

Run:

```bash
python -m unittest discover -s tests -p 'test_benchmark_gpu_solver.py' -v
ctest --test-dir build-cuda --output-on-failure -R 'gpu.*geometry|perfecthash\\.cuda\\.chm02'
```

Expected:
- Python runner tests pass
- new geometry smoke passes
- existing CUDA `Chm02` tests stay green

- [ ] **Step 6: Commit**

```bash
git add scripts/benchmark_gpu_solver.py \
        scripts/benchmark_gpu_solver_config.json \
        tests/test_benchmark_gpu_solver.py \
        tests/run_gpu_poc_geometry_smoke_test.cmake \
        tests/CMakeLists.txt
git commit -m "Add GPU POC geometry benchmark coverage"
```

## Task 5: Run a Bounded Stage-1 Baseline and Record It

**Files:**
- Create: `docs/superpowers/reports/2026-03-26-gpu-poc-execution-geometry-baseline.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-NOTES.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-LOG.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-TODO.md`

- [ ] **Step 1: Define the bounded sweep**

Use only safe local runs such as:

- generated `8193`
- `HologramWorld-31016.keys` with a conservative batch

Keep:

- current memory guard enabled
- current tiny-run discipline

- [ ] **Step 2: Run geometry sweeps**

Run the POC with `device-serial` for:

- assign `thread`
- peel `thread|warp|block`

Keep the matrix small. Record:

- `gpu_ms`
- `cpu_ms`
- `solved`
- `peel_rounds`
- geometry fields

- [ ] **Step 3: Write the baseline report**

Create:

- `docs/superpowers/reports/2026-03-26-gpu-poc-execution-geometry-baseline.md`

Include:

- datasets used
- exact commands
- results table
- recommendation for stage 2 target geometry

- [ ] **Step 4: Update ledgers**

Update:

- `agents/PERFECTHASH-GPU-SOLVING-NOTES.md`
- `agents/PERFECTHASH-GPU-SOLVING-LOG.md`
- `agents/PERFECTHASH-GPU-SOLVING-TODO.md`

Make the next step explicit:

- which geometry won
- whether stage 2 should target shared memory, CUB, or both

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/reports/2026-03-26-gpu-poc-execution-geometry-baseline.md \
        agents/PERFECTHASH-GPU-SOLVING-NOTES.md \
        agents/PERFECTHASH-GPU-SOLVING-LOG.md \
        agents/PERFECTHASH-GPU-SOLVING-TODO.md
git commit -m "Record GPU POC execution geometry baseline"
```

## Final Verification

- [ ] **Step 1: Rebuild the POC**

```bash
cmake --build build/gpu-batched-peeling-poc -j2
```

Expected:
- build succeeds

- [ ] **Step 2: Run the runner tests**

```bash
python -m unittest discover -s tests -p 'test_benchmark_gpu_solver.py' -v
```

Expected:
- all tests pass

- [ ] **Step 3: Run the geometry smoke**

```bash
ctest --test-dir build-cuda --output-on-failure -R 'gpu.*geometry'
```

Expected:
- geometry smoke passes

- [ ] **Step 4: Run one final tiny POC JSON command**

```bash
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc \
  --edges 16 \
  --batch 1 \
  --threads 64 \
  --solve-mode device-serial \
  --assign-geometry warp \
  --device-serial-peel-geometry warp \
  --output-format json
```

Expected:
- exit `0`
- JSON includes geometry fields
- `mismatches == 0`

- [ ] **Step 5: Confirm clean worktree**

```bash
git status --short --branch
```

Expected:
- no unintended changes
