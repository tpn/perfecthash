# Graph.cu Chm02 Correctness Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the legacy single-graph `Graph.cu` / `Graph.cuh` CUDA path produce a correct GPU-owned peel/order result for `PerfectHashCreate ... Chm02 Mulshrolate3RX ...` on a known-good seed set.

**Architecture:** Keep scope narrow. Fix the correctness hazards in `src/PerfectHashCuda/Graph.cu` so the GPU path can hash, build, peel, and capture a real `Order[]` for one graph. Use the existing CPU fallback for assignment/verify as a bring-up oracle through the `PerfectHashCreate` CLI. Do not optimize for batching or throughput in this pass.

**Tech Stack:** C, C++, CUDA, existing PerfectHash CLI (`PerfectHashCreate`), local CMake build, local CUDA runtime.

---

### Task 1: Add A Repro Harness For Chm02 CUDA

**Files:**
- Create: `tests/run_cli_chm02_cuda_known_seed_test.cmake`

- [ ] **Step 1: Write the failing harness**

Create a CMake script that:
- runs `PerfectHashCreate`
- uses:
  - `keys/HologramWorld-31016.keys`
  - `Chm02`
  - `Mulshrolate3RX`
  - `And`
  - maximum concurrency `1`
  - `--CuConcurrency=1`
  - `--FixedAttempts=1`
  - `--Seeds=0xF0192B55,0xD9C83970,0x0C1E0D10,0xD11A5847`
- fails if stderr contains `PerfectHashTableCreate failed`

- [ ] **Step 2: Run the harness and verify it fails**

Run:

```bash
cmake -DTEST_EXE=/home/trent/src/perfecthash/build/bin/Release/PerfectHashCreate \
      -DTEST_KEYS=/home/trent/src/perfecthash/keys/HologramWorld-31016.keys \
      -DTEST_OUTPUT=/tmp/ph-chm02-cuda-known-seed \
      -P /home/trent/src/perfecthash/tests/run_cli_chm02_cuda_known_seed_test.cmake
```

Expected:
- failure
- output contains `PerfectHashTableCreate failed with error: 0x80000001`

- [ ] **Step 3: Commit**

```bash
git add tests/run_cli_chm02_cuda_known_seed_test.cmake
git commit -m "Add Chm02 CUDA known-seed repro harness"
```

### Task 2: Fix GraphCuAddEdge1 Degree Handling

**Files:**
- Modify: `src/PerfectHashCuda/Graph.cu`

- [ ] **Step 1: Remove the double increment bug in `GraphCuAddEdge1()`**

Change the degree update so it increments exactly once.

- [ ] **Step 2: Rebuild**

Run:

```bash
cmake --build /home/trent/src/perfecthash/build -j
```

Expected:
- build succeeds

- [ ] **Step 3: Re-run the repro harness**

Run the Task 1 harness again.

Expected:
- likely still fails, but with one known data-corruption bug removed

- [ ] **Step 4: Commit**

```bash
git add src/PerfectHashCuda/Graph.cu
git commit -m "Fix degree increment in GraphCuAddEdge1"
```

### Task 3: Make GPU Peel Capture Real Order

**Files:**
- Modify: `src/PerfectHashCuda/Graph.cu`

- [ ] **Step 1: Re-enable `Order[]` recording in `GraphCuRemoveVertex()`**

Implement atomic order capture so every successful peel stores the edge at the decremented `OrderIndex`.

- [ ] **Step 2: Rebuild**

Run:

```bash
cmake --build /home/trent/src/perfecthash/build -j
```

- [ ] **Step 3: Re-run the repro harness**

Expected:
- may still fail, but the GPU path now owns the peel order instead of only the count

- [ ] **Step 4: Commit**

```bash
git add src/PerfectHashCuda/Graph.cu
git commit -m "Capture peeled edge order in GraphCuRemoveVertex"
```

### Task 4: Revalidate Degree-1 Selection Under Mutation

**Files:**
- Modify: `src/PerfectHashCuda/Graph.cu`

- [ ] **Step 1: Eliminate stale degree/edge selection in `GraphCuRemoveVertex()`**

Refactor so the edge chosen for removal is revalidated under the same ownership/locking scheme used for the actual remove.

- [ ] **Step 2: Rebuild**

Run:

```bash
cmake --build /home/trent/src/perfecthash/build -j
```

- [ ] **Step 3: Re-run the repro harness**

Expected:
- the known-good CLI seed should succeed, or at minimum the failure mode should change in a way that proves the stale-read issue was real

- [ ] **Step 4: Commit**

```bash
git add src/PerfectHashCuda/Graph.cu
git commit -m "Revalidate degree-1 selection in GraphCuRemoveVertex"
```

### Task 5: Verify Chm02 CPU Fallback Uses GPU Order

**Files:**
- Modify: `src/PerfectHash/GraphCu.c`

- [ ] **Step 1: Verify the CPU fallback no longer recomputes order**

Inspect `GraphCuIsAcyclic()` and ensure the bring-up path consumes GPU-generated `Order[]` instead of recomputing CPU `AddKeys()` + CPU `IsAcyclic()` when possible.

- [ ] **Step 2: Make the minimal change needed for bring-up**

Keep CPU assignment/verify if needed, but avoid overwriting GPU peel order once the GPU path is coherent enough to provide it.

- [ ] **Step 3: Rebuild**

Run:

```bash
cmake --build /home/trent/src/perfecthash/build -j
```

- [ ] **Step 4: Re-run the repro harness**

Expected:
- known-good `Chm02` CUDA CLI repro succeeds

- [ ] **Step 5: Commit**

```bash
git add src/PerfectHash/GraphCu.c
git commit -m "Prefer GPU peel order in Chm02 CUDA bring-up path"
```

### Task 6: Record Outcome And Deferred Path

**Files:**
- Modify: `agents/PERFECTHASH-GPU-SOLVING-NOTES.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-LOG.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-TODO.md`

- [ ] **Step 1: Record what worked**

Document:
- whether the known-good `Chm02` CLI case now succeeds
- which `Graph.cu` bug(s) were essential

- [ ] **Step 2: Record what was explicitly not done**

Document the deferred path:
- batching
- cooperative-groups/global frontier convergence
- performance work

- [ ] **Step 3: Commit**

```bash
git add agents/PERFECTHASH-GPU-SOLVING-NOTES.md agents/PERFECTHASH-GPU-SOLVING-LOG.md agents/PERFECTHASH-GPU-SOLVING-TODO.md
git commit -m "Record Graph.cu Chm02 correctness findings"
```
