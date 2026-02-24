# IOCP TODO

This is the prioritized, session-ready backlog for the IOCP server/client work.

## P0: Correctness + Completion Guarantees

- [ ] Re-verify bulk completion accounting under stress (single token must always signal exactly once):
  - `OutstandingWorkItems`
  - `PendingCompletions`
  - per-node decrement paths
  - final request completion transition
- [ ] Add/assert invariants in debug builds for async counters:
  - `Job->ActiveGraphs`
  - `Context->RemainingSolverLoops`
  - `Job->Async.Outstanding`
- [ ] Audit all failure/requeue paths again to ensure every submitted work item has exactly one completion/decrement.
- [ ] Add targeted tests for:
  - `PerfectHashAsyncRequeueWork()` failure path
  - graph submit failure rollback
  - bulk finalization when one node completes last.
- [ ] Confirm server/client connection lifecycle robustness:
  - `--WaitForServer`
  - `--ConnectTimeout=<ms>`
  - ping/pong readiness before bulk submission.

## P0: IOCP-Only Runtime Hygiene

- [ ] Ensure no legacy threadpool work submission is used by IOCP backend paths (except unavoidable OS-internal TP activity).
- [ ] Keep server silent by default; output only when `--Verbose` is specified.
- [ ] Re-verify `--NoFileIo` on server for fast stress loops and CI-style regression runs.

## P1: Buffer Pool Redesign Completion (Lookaside/NUMA Style)

- [ ] Finalize transition to per-NUMA global size-class pools (4KB..16MB default classes).
- [ ] Ensure oversize buffers are pooled/reusable (not leaked one-offs), with guarded-list ownership for teardown.
- [ ] Validate size-class mapping and payload offsets:
  - header precedes payload
  - `File->BaseAddress` points to payload only
  - overrun should fail fast (`PH_RAISE`).
- [ ] Add pool diagnostics (debug-only):
  - allocation count
  - in-use count
  - exhaustion count
  - per-class hit/miss.
- [ ] Harden teardown/rundown:
  - safe list flush
  - safe free after server stop
  - no outstanding buffer references.

## P1: File I/O Pipeline Hardening (Overlapped)

- [ ] Audit all CHM01 save callbacks for large-write safety with bounded, chunked writes.
- [ ] Add explicit handling for payloads that exceed current buffer capacity:
  - predictable flush-and-continue logic
  - no undefined pointer arithmetic on overflow.
- [ ] Keep OG memory-mapped path untouched and validated.
- [ ] Re-check ReFS/ADS-related edge cases (table-size stream behavior) and preserve `ADS.md` notes.

## P1: Tests (Unit + E2E)

- [ ] Expand unit tests around IOCP buffer pool:
  - guard-page mode on/off
  - oversize class reuse
  - multi-thread pop/push behavior.
- [ ] Add IOCP file-write completion unit tests:
  - success path (event signal + buffer return)
  - error path (propagated HRESULT + release semantics).
- [ ] Add repeatable E2E matrix scripts:
  - `test1`, `hard`, `sys32-200`, `sys32-1000`
  - `NoFileIo` and file-I/O modes
  - standard ramp presets.

## P2: Performance Characterization + Tuning

- [ ] Re-run and capture normalized OG vs IOCP timings (same algo/flags/output drive):
  - `Mulshrolate4RX`
  - concurrency sets: `4`, `32`, `64`.
- [ ] Sweep per-file ramp knobs:
  - `--InitialPerFileConcurrency`
  - `--MaxPerFileConcurrency`
  - `--IncreaseConcurrencyAfterMilliseconds`
- [ ] Profile system-vs-user overhead and memory footprint at high concurrency.
- [ ] Tune default knobs for balanced throughput and memory use.

## P2: API/CLI Polish

- [ ] Keep/confirm naming split:
  - `--IocpConcurrency`
  - `--MaxThreads` (default `IocpConcurrency * 2`)
  - `--MaxPerFileConcurrency` (must be `<= IocpConcurrency`).
- [ ] Validate NUMA targeting options and document examples (all nodes, single node, mask).
- [ ] Decide whether `BulkCreate=` request form should also block on token like `BulkCreateDirectory`.

## Regression Commands (Baseline)

- IOCP smoke:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\iocp-smoke.ps1 -BuildDir build-win -Config Release`
- IOCP stress (NoFileIo):
  - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stress-sys32-iocp.ps1 -BuildDir build-win -Config Release -IocpConcurrency 32 -MaxThreads 64 -NoFileIo`
- OG stress:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stress-sys32.ps1 -BuildDir build-win -Config Release -MaximumConcurrency 32`

