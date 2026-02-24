# IOCP Backend Handoff Prompt (Start Here)

Continue IOCP server/client development for PerfectHash with the current state below.

## Goal

Make bulk perfect-hash creation saturate CPU effectively by decoupling work from workers with a NUMA-aware IOCP pipeline, while keeping OG executables/paths intact.

## What Is Implemented

- Additive IOCP architecture (new server/client executables; OG untouched).
- Per-NUMA node IOCP runtime:
  - one completion port per node
  - manually created worker threads
  - node/group affinity wiring.
- Named-pipe request/response state machine with:
  - ping/pong readiness
  - shutdown
  - table-create and bulk-directory requests.
- Bulk directory request flow:
  - server walks `.keys` files
  - dispatches per-file work across configured nodes
  - returns a single completion token (event/mapping) to client.
- CHM01 async state-machine path present with per-file ramp controls:
  - `--InitialPerFileConcurrency`
  - `--MaxPerFileConcurrency`
  - `--IncreaseConcurrencyAfterMilliseconds`.
- IOCP file-work path has overlapped save I/O and keys-load overlapped reads.
- IOCP buffer pool infrastructure exists and is being reworked toward NUMA lookaside-like size classes with optional guard pages.

## Important Fixes Already Landed

- Lifetime fixes:
  - retained command-line/argv buffers for bulk request lifetime
  - fixed use-after-free in bulk callback path.
- Async accounting fixes:
  - requeue failure now decrements outstanding
  - graph-submit failure rollback fixed (`ActiveGraphs`, loop counters, cleanup).
- Crash fixes:
  - guarded legacy threadpool callback usage (`TpIsTimerSet` crash source)
  - fixed stale `NumberOfBytesWritten`/sizing bugs that produced oversized files
  - fixed IOCP file sizing index bug (`FileId` vs `FileWorkId`).
- Added crash diagnostics/minidump improvements and server/client wait-for-ready behavior.

## Observed Performance Snapshot (Recent)

- On some file-I/O workloads, IOCP outperformed OG.
- On some no-file-I/O workloads, OG remained faster.
- High concurrency (`IocpConcurrency=64`) exposed memory pressure/over-allocation; pool policy still needs tuning.

## Current Risks / Gaps

- Buffer pool policy can overprovision at high concurrency.
- Need stricter invariants around async completion counters to prevent latent hangs.
- Large-payload file writes need explicit chunk/flush strategy validation.
- Need broader repeatable E2E coverage (test1/hard/sys32 subsets/full).

## Next Execution Order

1. Lock correctness first:
   - counter invariants
   - completion-once guarantees
   - failure-path decrements.
2. Finish buffer-pool redesign:
   - per-NUMA size-class global pools
   - oversize reuse pools
   - safe guarded-list rundown.
3. Harden overlapped file-I/O:
   - large write chunking
   - strict bounds/fail-fast behavior
   - OG path unchanged.
4. Re-run standardized perf matrix and tune defaults.

## Key Files

- IOCP runtime:
  - `src/PerfectHash/PerfectHashContextIocp.c`
  - `src/PerfectHash/PerfectHashContextIocp.h`
- Server/client core:
  - `src/PerfectHash/PerfectHashServer.c`
  - `src/PerfectHash/PerfectHashClient.c`
- Async engine:
  - `src/PerfectHash/PerfectHashAsync.c`
  - `src/PerfectHash/Chm01Async.c`
- IOCP file work/buffer pool:
  - `src/PerfectHash/Chm01FileWork.c`
  - `src/PerfectHash/Chm01FileWorkIocp.c`
  - `src/PerfectHash/PerfectHashIocpBufferPool.c`
  - `src/PerfectHash/PerfectHashIocpBufferPool.h`
- Ledgers:
  - `IOCP-LOGS.md`
  - `IOCP-TOOD.md`
  - `IOCP-README.md`

## Quick Commands

- Build:
  - `cmake --build build-win --config Release --target PerfectHashServerExe PerfectHashClientExe PerfectHashBulkCreateExe`
- IOCP smoke:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\iocp-smoke.ps1 -BuildDir build-win -Config Release`
- IOCP stress:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stress-sys32-iocp.ps1 -BuildDir build-win -Config Release -IocpConcurrency 32 -MaxThreads 64`
- OG stress:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stress-sys32.ps1 -BuildDir build-win -Config Release -MaximumConcurrency 32`

