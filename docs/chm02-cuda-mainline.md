# Chm02 CUDA Mainline Note

## Summary

This note captures the intent of the `issue-79-chm02-cuda-mainline` branch.

The branch promotes the legacy `Chm02` CUDA path from a CPU-assisted bring-up
 state toward a first-class correctness path by moving the major solve phases
 (`IsAcyclic`, `Assign`, `Verify`) onto the GPU while keeping CPU-oracle-style
 validation and debugging support available during bring-up.

## Goals

- Fix correctness blockers in the existing `Chm02` CUDA path.
- Make known-seed CLI runs succeed on Linux in both no-file-io and file-io
  configurations.
- Add regression coverage for:
  - known-seed `Chm02` CUDA runs
  - a generated non-`Assigned16` case
  - timing-field presence
- Expose explicit per-phase CUDA timing fields for measurement.

## Non-Goals

- High-throughput GPU solving.
- Batched multi-attempt GPU construction.
- Replacing the standalone GPU peeling POC.
- Eliminating all CPU-oracle/debug-only code from the branch.

The current `Chm02` CUDA implementation remains correctness-first, not
 throughput-first.

## Supported Scope

- Algorithm: `Chm02`
- Hash path: known-good seeded hash families already supported by the repo
- CUDA path: single-graph bring-up / validation
- Platform focus:
  - Linux with CUDA enabled
  - existing regression coverage on the configured CUDA host

The following supporting code changes are considered in-scope for this branch:

- Linux file-work compatibility fixes needed for the `Chm02Compat` path
- CSV/timing schema updates needed to surface CUDA phase timing
- the Linux `QueryPerformanceFrequency()` correction that makes those timings
  sane on non-Windows builds

## Fallback / Debugging Policy

- Normal operation should use the GPU path for add-keys, acyclic detection,
  assignment, and verify.
- CPU-oracle and order-validation logic is intended as bring-up/debug support.
- `PH_DEBUG_CUDA_CHM02` enables extra logging and validation details for
  troubleshooting.

## Timing Contract

The following CSV fields are emitted:

- `CuAddKeysMicroseconds`
- `CuIsAcyclicMicroseconds`
- `CuAssignMicroseconds`
- `CuVerifyMicroseconds`

These are synchronized phase timings around the CUDA-backed phase wrappers, not
 raw kernel-only device timings.

Compatibility note:

- this branch preserves the historical
  `GpuIsAcyclicButCpuIsCyclicFailures` column as a zero-valued compatibility
  stub in order to keep downstream CSV column positions stable
- this branch intentionally adds the four `Cu*` timing fields above
- the existing non-CUDA timing fields should continue to use the same timing
  base; the Linux `QueryPerformanceFrequency()` fix is included specifically so
  those timings remain coherent on this platform as well as for the new CUDA
  timing fields

## Failure-Path Expectations

- Cyclic graphs are expected to return normal non-success solve results; they
  are not considered internal errors.
- CUDA-disabled builds are expected to continue using the non-CUDA code paths.
- GPU order-validation and extra CPU-oracle diagnostics are debug-only aids,
  controlled by `PH_DEBUG_CUDA_CHM02`.
- Non-debug runs are expected to surface failure through the normal `HRESULT`
  and verification paths, not through verbose stderr diagnostics.
- The current serial CUDA kernels are correctness-first and must not be treated
  as throughput-optimized production behavior.

## Debug Surface

The following debug surface is intentionally supported for this bring-up phase:

- `PH_DEBUG_CUDA_CHM02`
- stderr logging from the CUDA `Chm02` path
- stable debug tokens used by the known-seed regression harnesses:
  - `PH_CHM02_CUDA_ORDER_OK`
  - `PH_CHM02_CUDA_ASSIGN_OK`
  - `PH_CHM02_CUDA_VERIFY_OK`

This surface is explicitly considered temporary bring-up instrumentation, not a
 long-term stable user-facing API.

## Staged Task List

1. Fix correctness blockers in the legacy CUDA `Chm02` path.
2. Establish known-seed Linux no-file-io parity.
3. Establish Linux file-io parity.
4. Move assignment and verify onto the GPU.
5. Expose explicit per-phase CUDA timing fields for measurement.
6. Add focused CUDA regression coverage:
   - known-seed path
   - non-`Assigned16` generated path
   - timing-field presence

## Acceptance

- The focused CUDA `Chm02` regression tests pass when CUDA is enabled.
- Known-seed `Chm02` CUDA runs succeed on Linux.
- File-io and no-file-io paths both work in the covered scenarios.
- Timing fields are present and non-negative in CSV output.
