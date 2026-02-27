# sqlite-online-jit Plan

## Status

- Phase 1 (vendoring + scaffold) implemented.
- Phase 2 (virtual table core) implemented.
- Phase 3 (A/B benchmark harness) expanded with full permutation matrix mode.
- Phase 4 (CI expansion) implemented for single-mode smoke runs.

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

Implement a sqlite virtual table module, tentatively `perfecthash_vtab`.

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

### 3. A/B Benchmark Harness

Include benchmark modes in one executable:

1. `baseline-btree`:
   join fact table to ordinary dimension table with B-tree index.
2. `perfecthash-rawdog-jit`:
   join fact table to PerfectHash virtual table with RawDog JIT backend.
3. `perfecthash-llvm-jit`:
   join fact table to PerfectHash virtual table with LLVM JIT backend.

Benchmark output:

- elapsed wall time,
- rows processed / throughput,
- `EXPLAIN QUERY PLAN` text for each mode.

### 4. Key Handling

v1 target is strict 32-bit keys.

- Source column must fit unsigned 32-bit range.
- Rows outside range are rejected during virtual-table build.
- Future extension: optional 64-bit downsize policy with explicit collision
  handling rules (not in v1).

## Implementation Phases

1. Vendoring decision and scaffold:
   choose sqlite import format (`amalgamation` preferred for lean builds) and
   create `examples/sqlite-online-jit/CMakeLists.txt`.
2. Virtual table core:
   implement module lifecycle and lookup path with PerfectHash build/compile.
3. Benchmark wiring:
   deterministic dataset generator, query runner, timing/reporting.
4. CI:
   add platform matrix execution with bounded dataset size/runtime.
5. Documentation:
   quickstart, architecture notes, and result interpretation guidance.

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
- At least one dataset/query shape shows measurable query-time improvement in
  PerfectHash mode compared with baseline B-tree mode.
