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

## Acceptance

- The focused CUDA `Chm02` regression tests pass when CUDA is enabled.
- Known-seed `Chm02` CUDA runs succeed on Linux.
- File-io and no-file-io paths both work in the covered scenarios.
- Timing fields are present and non-negative in CSV output.
