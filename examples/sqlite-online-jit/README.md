# sqlite-online-jit

Planned example showing how to integrate PerfectHash online table generation
into sqlite for fast 32-bit key lookups, with a reproducible A/B benchmark.

## Status

Planning complete for v1 integration approach. Implementation is next.

## Goal

Provide a real-world sqlite integration example that is:

- portable across Windows/Linux/macOS,
- easy for humans and LLMs to read,
- easy to benchmark in A/B mode.

## Selected v1 Approach

Use a sqlite virtual table module backed by PerfectHash online JIT.

- Base mode: regular sqlite join/index behavior.
- PerfectHash mode: join through a PerfectHash-backed virtual table.
- Backend toggle in PerfectHash mode: `rawdog-jit` or `llvm-jit`.

This avoids invasive sqlite planner patches in the first iteration while still
using a planner-aware integration seam (`xBestIndex`).

## A/B Benchmark Concept

Initial benchmark will compare:

1. Baseline sqlite join path (B-tree indexed dimension table).
2. PerfectHash virtual-table join path (`rawdog-jit` backend).
3. PerfectHash virtual-table join path (`llvm-jit` backend).

Benchmark harness will report total runtime, rows/s, and query plan text so
speedups can be interpreted with planner context.

## Next Deliverables

- finalize sqlite vendoring strategy for this repo,
- scaffold CMake build for sqlite + PerfectHash integration,
- implement virtual-table module and benchmark runner,
- add CI job coverage for the sqlite example.
