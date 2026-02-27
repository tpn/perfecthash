# sqlite-online-jit Plan

## Status

- Phase 1 (vendoring + scaffold) implemented.
- Phase 2 (virtual table core) implemented.
- Phase 3 (A/B benchmark harness) expanded with full permutation matrix mode.
- Phase 4 (CI expansion) implemented for matrix + single smoke coverage.
- Phase 5 (multi-run creation-cost analysis + notebook visualization) in progress.

## Inputs Reviewed

sqlite source components reviewed for integration fit:

- `sqlite/src/where.c`
- `sqlite/src/vtab.c`
- `sqlite/ext/misc/series.c`

These confirm that virtual-table `xBestIndex` is a practical low-risk seam for
query planner participation without patching sqlite core in v1.

## Scope

Build an example project that demonstrates runtime generation and use of a
PerfectHash index inside sqlite query execution for 32-bit key joins/lookups.

## Design

### 1. Integration Boundary

Implement a sqlite virtual table module, `perfecthash_vtab`.

- Module is registered at sqlite startup in the example binary.
- A virtual table instance references a source dimension table and key column.
- During initialization, keys are materialized and a PerfectHash table is built
  via `PerfectHashOnlineJit.h`.
- Runtime lookups in `xFilter` use `PhOnlineJitIndex32()` to resolve rows.

### 2. Planner Cooperation

Use `xBestIndex` to prefer equality constraints on the key column and advertise
low estimated cost for point lookups.

Expected benefit: sqlite nested-loop joins can use very fast key probes into
the PerfectHash-backed virtual table.

### 3. A/B + Matrix Benchmark Harness

Include benchmark modes in one executable:

1. `baseline-btree`:
   join fact table to ordinary dimension table with B-tree index.
2. `perfecthash-rawdog-jit`:
   join fact table to PerfectHash virtual table with RawDog JIT backend.
3. `perfecthash-llvm-jit`:
   join fact table to PerfectHash virtual table with LLVM JIT backend.

Matrix mode expands this across:

- backend: `rawdog-jit`, `llvm-jit`
- hash: curated good-hash set
- vector width: `1`, `2`, `4`, `8`, `16`
- build repetitions: `--build-runs`

### 4. Build-Cost Instrumentation

Build/report phases include:

- source extraction from sqlite table
- `PhOnlineJitCreateTable32()` time
- compile time (`PhOnlineJitCompileTableEx()`)
- map materialization time
- external CREATE VIRTUAL TABLE wall time

This allows both query-only and end-to-end comparisons plus break-even
query-count estimates.

### 5. Notebook-Based Analysis

A committed notebook (`notebooks/sqlite_online_jit_matrix_analysis.ipynb`) will
visualize:

- speedup heatmaps by backend/hash/vector
- creation-time distributions across repeated runs
- tradeoff scatter plots (speedup vs build cost)
- top-ranked configurations

## Risks and Mitigations

- Planner does not pick virtual table as expected:
  tune `xBestIndex` cost estimates and inspect `EXPLAIN QUERY PLAN`.
- JIT backend availability differences by host:
  keep explicit mode selection and log fallback behavior.
- Large dataset build overhead:
  isolate table-build time from query-execution time in benchmark reports.

## Success Criteria (v1)

- Example builds on Linux/macOS/Windows.
- Both PerfectHash backend modes execute successfully where available.
- A/B harness runs in CI and locally with consistent output format.
- Multi-run matrix outputs can be visualized quickly in notebook form.
- At least one dataset/query shape shows measurable query-time improvement in
  PerfectHash mode compared with baseline B-tree mode.
