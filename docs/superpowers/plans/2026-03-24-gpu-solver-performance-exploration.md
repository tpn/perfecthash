# GPU Solver Performance Exploration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fair, repeatable performance exploration workflow that compares CPU solving, legacy single-graph `Chm02` CUDA solving, and the batched GPU POC on the local GB10 system and `nv1`.

**Architecture:** Treat performance exploration as a measurement problem first, not an optimization problem. Separate solver modes by what they actually optimize: `Chm01`/`Chm02` for single-table attempt latency and fixed-attempt yield, and the batched POC for attempt throughput / solved-tables-per-second. Standardize datasets, metrics, memory-placement modes, and output formats before making any kernel changes so later optimizations can be judged against stable baselines. For GB10 specifically, treat unified memory as a capacity/coordination feature to be measured explicitly, not as an assumption that all placements are equivalent.

**Tech Stack:** CMake/CTest, existing PerfectHash CLI (`PerfectHashCreate`), existing `experiments/gpu_batched_peeling_poc`, CUDA 13.x, Python 3 for orchestration/report collation, local GB10, remote `nv1`.

---

## Review Summary

What exists today:
- Legacy CUDA path:
  - `PerfectHashCreate ... Chm02 ...`
  - now has GPU add-keys, GPU acyclic / peel-order capture, GPU assignment, GPU verify
  - has first-class `ctest` coverage for:
    - `perfecthash.cuda.chm02.hologram.nofileio`
    - `perfecthash.cuda.chm02.hologram.fileio`
    - `perfecthash.cuda.chm02.generated33000.nofileio`
- Batched POC:
  - `experiments/gpu_batched_peeling_poc`
  - already measures overall GPU and CPU milliseconds
  - already supports `host-roundtrip` vs `device-serial`
  - already has some benchmark notes for HologramWorld / Hydrogen and local GB10 vs `nv1`

What is missing:
- a single benchmark harness that compares CPU CLI, CUDA CLI, and batched POC consistently
- stable benchmark datasets covering:
  - easy assigned16
  - hard assigned16
  - easy non-assigned16
  - hard non-assigned16
- phase-level timing for the POC
- GPU-specific phase timing in the legacy `Chm02` path
- explicit reporting of:
  - attempts/sec
  - solves/sec
  - fixed-attempt yield
  - phase breakdown
  - memory mode / batch size / rounds

Important fairness rule:
- Do **not** compare raw wall-clock numbers across fundamentally different execution models without labeling the metric.
- We need at least two benchmark families:
  - `single_graph_latency`: one table create run, fixed attempts, minimal I/O
  - `batched_attempt_throughput`: many attempts in one run, report attempts/sec and solves/sec

## GB10 Memory-Model Hypotheses

The performance work should explicitly test these hypotheses instead of treating GB10 as “just a smaller discrete GPU”:

- Hypothesis 1:
  - GB10 benefits more from reduced CPU↔GPU transfer overhead and larger effective batch size than from raw per-kernel bandwidth.
- Hypothesis 2:
  - For the batched POC, the best GB10 architecture will likely use unified memory as a capacity valve, while still preferring GPU-resident or GPU-prefetched placement for hot reused arrays.
- Hypothesis 3:
  - For one-touch or control-plane data, mapped/pinned/zero-copy style access may be better than migrating full buffers.
- Hypothesis 4:
  - The legacy single-graph `Chm02` path may benefit modestly from unified memory, but not enough to overcome its lack of batching.
- Hypothesis 5:
  - Discrete `nv1`-style GPUs will likely keep an advantage on bandwidth-dominated phases even when GB10 wins on copy avoidance or capacity.

These hypotheses imply that the plan must compare at least:
- `allocation_mode`
  - `explicit-device`
  - `managed-default`
  - `managed-prefetch-gpu`
  - `mapped-pinned` or nearest practical zero-copy mode
- `machine`
  - `gb10`
  - `nv1`

## File Map

Planned benchmark/report files:
- Create: `scripts/benchmark_gpu_solver.py`
  - single entrypoint for running benchmark matrices and writing JSON/CSV summaries
- Create: `scripts/benchmark_gpu_solver_config.json`
  - dataset list, variant list, machine labels, and default benchmark matrix
- Create: `docs/superpowers/reports/2026-03-24-gpu-solver-performance-baseline.md`
  - human-readable summary of initial measurement results

Planned POC instrumentation files:
- Modify: `experiments/gpu_batched_peeling_poc/main.cu`
  - add per-stage timings, memory-placement flags, and machine-readable output mode
- Modify: `experiments/gpu_batched_peeling_poc/README.md`
  - document benchmark flags / output schema / memory-placement modes

Planned legacy CLI instrumentation files:
- Modify: `src/PerfectHash/Graph.h`
  - add any minimal per-phase CUDA timing fields if needed
- Modify: `src/PerfectHash/GraphCu.c`
  - capture GPU add-keys / acyclic / assign / verify timing if current table CSV fields are insufficient
- Modify: `src/PerfectHash/TableCreateCsv.h`
  - expose any new GPU-specific timing fields in table-create CSV output

Planned test/fixture files:
- Create: `tests/run_cli_chm02_cuda_perf_benchmark.cmake`
  - minimal wrapper for performance-style CLI runs
- Create: `tests/run_cli_generated_perf_keys_test.cmake`
  - deterministic generated-key performance fixture(s) if needed
- Modify: `tests/CMakeLists.txt`
  - only if we decide any performance smoke tests belong in `ctest`

Planned ledger/report updates:
- Modify: `agents/PERFECTHASH-GPU-SOLVING-NOTES.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-LOG.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-TODO.md`

## Benchmark Questions

The exploration must answer these questions explicitly:

1. Is the legacy single-graph CUDA `Chm02` path ever faster than CPU `Chm01` for solving a single table under fixed-attempt budgets?
2. At what batch size does the batched POC beat CPU multi-attempt throughput?
3. Which phase dominates each solver mode?
   - add-keys / hash
   - peel / acyclic
   - assign
   - verify
4. How much do dataset hardness and the `Assigned16` boundary matter?
5. How much do local GB10 vs `nv1` differ?
6. For GPU modes, does throughput scale with:
   - batch size
   - threads per block
   - solve mode
   - storage width
7. On GB10, which allocation mode is best for:
   - hot graph state
   - key input
   - result summaries
8. At what batch size does GB10 start trading capacity wins for locality losses?
9. Does `nv1` prefer explicit device placement even when GB10 prefers unified-memory-style placement?

## Dataset Matrix

Keep the first measurement matrix small and representative.

Repo-local deterministic datasets:
- `keys/HologramWorld-31016.keys`
  - hard-ish assigned16
- generated `33000` keys via deterministic generator
  - easy non-assigned16
- generated `8193` keys via deterministic generator
  - easy assigned16 baseline

External datasets if `/home/trent/src/perfecthash-keys` exists:
- `/home/trent/src/perfecthash-keys/hard/Hydrogen-40147.keys`
  - hard non-assigned16
- `/home/trent/src/perfecthash-keys/hard/CoreUIComponents-7995.keys`
  - smaller/easier real-world baseline

Initial matrix:
- `easy_assigned16`
- `hard_assigned16`
- `easy_non_assigned16`
- `hard_non_assigned16` (optional if external checkout available)

## Metrics

Every run should record:
- machine label
- machine memory model
- solver family
  - `cpu-cli`
  - `cuda-chm02`
  - `gpu-poc`
- dataset
- hash function
- fixed attempts or batch size
- solve mode
- storage bits
- threads per block
- allocation mode
- total wall-clock milliseconds
- attempts/sec
- solves/sec
- solved/attempted ratio
- per-stage timing where available
- peel rounds for the POC
- any auto-scaled batch change due to memory limits
- any observed managed-memory fallback or placement adjustment

## Task 1: Create a Single Benchmark Runner

**Files:**
- Create: `scripts/benchmark_gpu_solver.py`
- Create: `scripts/benchmark_gpu_solver_config.json`

- [ ] **Step 1: Write the failing smoke invocation**

Write a thin Python runner that accepts:
- `--config`
- `--machine-label`
- `--output`

and fails with a clear error if the config file is missing required sections:
- `datasets`
- `variants`

- [ ] **Step 2: Run it to verify it fails cleanly**

Run:

```bash
python scripts/benchmark_gpu_solver.py --config /tmp/missing.json --machine-label local --output /tmp/out.json
```

Expected:
- non-zero exit
- clear config-missing error

- [ ] **Step 3: Implement minimal config loading**

Support JSON config with:
- datasets
- variants
- output options

- [ ] **Step 4: Run smoke test to verify it passes**

Run with a tiny config that has empty dataset/variant arrays.

Expected:
- exit `0`
- output JSON written

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark_gpu_solver.py scripts/benchmark_gpu_solver_config.json
git commit -m "Add GPU solver benchmark runner scaffold"
```

## Task 2: Add POC Stage Timing, Memory-Placement Modes, and Machine-Readable Output

**Files:**
- Modify: `experiments/gpu_batched_peeling_poc/main.cu`
- Modify: `experiments/gpu_batched_peeling_poc/README.md`

- [ ] **Step 1: Write the failing parser expectation**

Extend the benchmark runner to expect POC output fields:
- `gpu_ms`
- `cpu_ms`
- `peel_rounds`
- `solved`
- `batch`

Run the runner against current POC output and confirm parse failure.

- [ ] **Step 2: Add minimal POC structured output mode**

Add a flag such as:

```text
--output-format json
```

that prints one JSON object containing:
- dataset name
- batch
- solve mode
- threads
- storage bits
- hash function
- gpu milliseconds
- cpu milliseconds
- solved count
- peel rounds

- [ ] **Step 3: Add stage timing**

Split total GPU timing into:
- add/build
- peel
- assign
- verify

Use `cudaEvent` timing around each stage.

- [ ] **Step 4: Add allocation-mode support**

Add a flag such as:

```text
--allocation-mode explicit-device|managed-default|managed-prefetch-gpu|mapped-pinned
```

and make the allocation path visible in structured output.

The first implementation can keep scope narrow:
- only apply allocation modes to the POC
- only switch the large graph-state arrays
- use whichever modes are actually practical on both GB10 and `nv1`

- [ ] **Step 5: Verify structured output**

Run:

```bash
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc \
  --keys-file keys/HologramWorld-31016.keys \
  --batch 128 \
  --hash-function Mulshrolate3RX \
  --allocation-mode managed-default \
  --output-format json
```

Expected:
- JSON output
- stage timing fields present
- allocation mode field present

- [ ] **Step 6: Commit**

```bash
git add experiments/gpu_batched_peeling_poc/main.cu experiments/gpu_batched_peeling_poc/README.md
git commit -m "Add structured timing and allocation modes to GPU batched peeling POC"
```

## Task 3: Add Legacy `Chm02` GPU Perf Capture

**Files:**
- Modify: `src/PerfectHash/Graph.h`
- Modify: `src/PerfectHash/GraphCu.c`
- Modify: `src/PerfectHash/TableCreateCsv.h`

- [ ] **Step 1: Determine whether current table-create CSV is sufficient**

Run one `Chm01` and one `Chm02` table-create with CSV enabled and inspect:
- `AddKeysElapsedMicroseconds`
- `SolveMicroseconds`
- `VerifyMicroseconds`

If these are enough for initial comparison, keep scope narrow.
If not, add explicit CUDA phase timing fields:
- `CuAddKeysMicroseconds`
- `CuIsAcyclicMicroseconds`
- `CuAssignMicroseconds`
- `CuVerifyMicroseconds`

- [ ] **Step 2: Add only the missing fields**

Capture the minimal additional timing needed to separate:
- CPU CLI solve time
- GPU `Chm02` phase times

- [ ] **Step 3: Verify CSV output**

Run:

```bash
./build-cuda/bin/PerfectHashCreate \
  keys/HologramWorld-31016.keys \
  /tmp/ph-chm02-perf \
  Chm02 \
  Mulshrolate3RX \
  And \
  1 \
  --CuConcurrency=1 \
  --FixedAttempts=2 \
  --Seeds=0xF0192B55,0xD9C83970,0x0C1E0D10,0xD11A5847 \
  --NoFileIo
```

Expected:
- CSV generated
- required timing fields present

- [ ] **Step 4: Commit**

```bash
git add src/PerfectHash/Graph.h src/PerfectHash/GraphCu.c src/PerfectHash/TableCreateCsv.h
git commit -m "Add legacy Chm02 GPU timing capture"
```

## Task 4: Standardize Repo-Local Dataset Fixtures

**Files:**
- Create: `tests/run_cli_generated_perf_keys_test.cmake`
- Modify: `tests/CMakeLists.txt` (only if a smoke test is justified)

- [ ] **Step 1: Reuse the deterministic 33,000-key generator**

Lift the generator pattern from the existing generated-33000 CUDA regression into a reusable helper for performance harnesses.

- [ ] **Step 2: Add a deterministic 8193-key generator**

Create an easy assigned16 dataset generator with a stable salt.

- [ ] **Step 3: Verify both generators**

Run the helper and confirm:
- 8193-key fixture created
- 33000-key fixture created

- [ ] **Step 4: Commit**

```bash
git add tests/run_cli_generated_perf_keys_test.cmake
git commit -m "Add deterministic generated-key perf fixtures"
```

## Task 5: Add GB10-vs-Discrete Memory-Model Experiments

**Files:**
- Modify: `scripts/benchmark_gpu_solver.py`
- Modify: `scripts/benchmark_gpu_solver_config.json`

- [ ] **Step 1: Add allocation-mode dimensions to the config**

Support per-variant configuration for:
- `allocation_mode`
- `prefetch`
- `batch_headroom_policy`

- [ ] **Step 2: Restrict the first memory-model matrix**

Keep it small:
- datasets:
  - HologramWorld `31016`
  - generated `33000`
- machines:
  - `gb10`
  - `nv1`
- allocation modes:
  - `explicit-device`
  - `managed-default`
  - `managed-prefetch-gpu`

- [ ] **Step 3: Verify dry run**

Run:

```bash
python scripts/benchmark_gpu_solver.py \
  --config scripts/benchmark_gpu_solver_config.json \
  --machine-label gb10 \
  --output /tmp/gpu-solver-memmodel.json \
  --dry-run
```

Expected:
- the planned runs show allocation mode as an explicit dimension

- [ ] **Step 4: Commit**

```bash
git add scripts/benchmark_gpu_solver.py scripts/benchmark_gpu_solver_config.json
git commit -m "Add GB10 memory-model benchmark variants"
```

## Task 6: Implement the Benchmark Matrix

**Files:**
- Modify: `scripts/benchmark_gpu_solver.py`
- Modify: `scripts/benchmark_gpu_solver_config.json`

- [ ] **Step 1: Add CPU CLI benchmark variants**

Support:
- `cpu-cli-chm01-single`
- `cpu-cli-chm01-threaded`

Use:
- `--NoFileIo`
- fixed attempts
- CSV parsing for timing extraction

- [ ] **Step 2: Add legacy CUDA `Chm02` variants**

Support:
- `cuda-chm02-single`

Use:
- `--NoFileIo`
- fixed attempts
- known-good seed mode for smoke/per-phase checks
- fixed-attempt mode for throughput/yield checks

- [ ] **Step 3: Add batched POC variants**

Support:
- `gpu-poc-host-roundtrip`
- `gpu-poc-device-serial`

Record:
- batch
- threads
- storage bits
- hash function

- [ ] **Step 4: Encode the initial benchmark matrix**

Use:
- HologramWorld `31016`
- generated `8193`
- generated `33000`
- Hydrogen `40147` if external checkout exists

For POC:
- batch sizes `128`, `512`, `2048`
- threads `128`, `256`
- solve modes `host-roundtrip`, `device-serial`

For CLI:
- fixed attempts `128`, `2048`
- `MaximumConcurrency=1`
- CPU threaded baseline at machine max concurrency if supported by CLI flags/config

- [ ] **Step 5: Verify dry run**

Run:

```bash
python scripts/benchmark_gpu_solver.py \
  --config scripts/benchmark_gpu_solver_config.json \
  --machine-label gb10 \
  --output /tmp/gpu-solver-bench.json \
  --dry-run
```

Expected:
- no execution
- full benchmark plan emitted

- [ ] **Step 6: Commit**

```bash
git add scripts/benchmark_gpu_solver.py scripts/benchmark_gpu_solver_config.json
git commit -m "Encode GPU solver benchmark matrix"
```

## Task 7: Capture Local GB10 Baselines

**Files:**
- Create: `docs/superpowers/reports/2026-03-24-gpu-solver-performance-baseline.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-NOTES.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-LOG.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-TODO.md`

- [ ] **Step 1: Run the reduced local matrix**

Run the full runner on the local machine with:
- generated `8193`
- HologramWorld `31016`
- generated `33000`
- Hydrogen if available

- [ ] **Step 2: Summarize by benchmark family**

Document separately:
- `single_graph_latency`
- `fixed_attempt_yield`
- `batched_attempt_throughput`

- [ ] **Step 3: Record stage-dominance conclusions**

For each solver family, identify whether time is dominated by:
- add-keys
- peel
- assign
- verify
- file I/O

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/reports/2026-03-24-gpu-solver-performance-baseline.md \
        agents/PERFECTHASH-GPU-SOLVING-NOTES.md \
        agents/PERFECTHASH-GPU-SOLVING-LOG.md \
        agents/PERFECTHASH-GPU-SOLVING-TODO.md
git commit -m "Record local GPU solver performance baseline"
```

## Task 8: Capture `nv1` Comparison

**Files:**
- Modify: `docs/superpowers/reports/2026-03-24-gpu-solver-performance-baseline.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-NOTES.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-LOG.md`

- [ ] **Step 1: Sync branch to `nv1`**

Use the existing branch-safe workflow already used in this session; do not touch remote `main`.

- [ ] **Step 2: Run the reduced matrix on `nv1`**

Use the same config with:
- `--machine-label nv1`

- [ ] **Step 3: Compare GB10 vs `nv1`**

Highlight:
- CPU scaling difference
- GPU scaling difference
- batch-size sensitivity
- whether the legacy `Chm02` path is ever compelling relative to the batched POC

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/reports/2026-03-24-gpu-solver-performance-baseline.md \
        agents/PERFECTHASH-GPU-SOLVING-NOTES.md \
        agents/PERFECTHASH-GPU-SOLVING-LOG.md
git commit -m "Add nv1 GPU solver performance comparison"
```

## Task 9: Decide the Optimization Target

**Files:**
- Modify: `docs/superpowers/reports/2026-03-24-gpu-solver-performance-baseline.md`
- Modify: `agents/PERFECTHASH-GPU-SOLVING-TODO.md`

- [ ] **Step 1: Rank the solver paths**

Decide based on the collected data:
- keep investing in legacy `Chm02`
- keep `Chm02` as correctness reference only
- move primary perf work to the batched POC

- [ ] **Step 2: Write the next optimization plan**

The result should answer:
- whether the next work item is POC batching/throughput
- legacy `Chm02` cleanup only
- or a shared instrumentation/refactor layer

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/reports/2026-03-24-gpu-solver-performance-baseline.md \
        agents/PERFECTHASH-GPU-SOLVING-TODO.md
git commit -m "Choose next GPU solver optimization target"
```
