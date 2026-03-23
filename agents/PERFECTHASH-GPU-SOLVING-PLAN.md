# PerfectHash GPU Solving Plan

## Objective
Understand the current CPU and CUDA graph-solving implementations in PerfectHash and identify a viable modern GPU design for solving many perfect-hash table construction requests efficiently on NVIDIA hardware.

## Scope
- Reverse engineer current `GraphImpl1/2/3` and `PerfectHashCuda` solving flows.
- Identify why the current CUDA path is not robust or performant.
- Compare current implementation characteristics against modern GPU techniques and libraries.
- Produce a concrete design space and recommended next experiments for a batched GPU solver.

## Non-goals
- Shipping a production GPU solver in this session.
- Rewriting unrelated PerfectHash components.
- Benchmarking on hardware in this session unless lightweight local inspection is enough.

## Current Status
- Session started.
- No prior ledgers existed for this subproject in `agents/`.
- Code and repository reconnaissance completed.
- Active CUDA backend understood as a hybrid GPU+CPU path.
- External GPU MPHF / AMQ literature survey completed.
- Current chosen implementation direction:
  - device-resident per-graph solve loop (`device-serial`)
- Explicitly deferred alternative:
  - cooperative-groups / global frontier device-side convergence across the full batch
  - deferred because it is more invasive and harder to validate quickly against the current baseline

## Phases
1. Inspect existing CPU and CUDA solver implementations.
2. Identify structural bottlenecks, branchiness, serial regions, and memory layout issues.
3. Research modern GPU approaches relevant to batched graph peeling, assignment, and filtering.
4. Synthesize a recommended design and staged prototype plan.

## Dependencies
- Local repo sources under `src/PerfectHash/` and `src/PerfectHashCuda/`.
- External research and current NVIDIA/CUDA ecosystem material as needed.

## Risks / Unknowns
- Existing CUDA path may reflect experimental ideas rather than a coherent execution model.
- The right GPU strategy may require changing the unit of parallelism from “one table” to “many tables”.
- Performance viability may depend on batching, frontier compaction, and memory layout more than kernel micro-optimizations.
- If CHM semantics must remain fixed, the amount of restructuring needed may approach a fresh solver implementation anyway.
- If the end goal is really a GPU-native solved-table representation competitive with cuCollections-like structures, CHM may be the wrong construction target.

## Exit Criteria
- Clear understanding of current implementations.
- Clear explanation of why the current CUDA approach struggles.
- Ranked set of viable GPU approaches for PerfectHash on modern NVIDIA hardware.
- Concrete next experiments captured in `TODO.md`.
