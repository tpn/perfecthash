# IOCP Backend Status (PerfectHash)

This prompt summarizes the current IOCP backend work for the perfecthash server/client, including what is implemented, recent fixes, and next steps.

## Current State

The IOCP backend exists as a parallel, additive path (does not replace existing executables). It provides:
- A NUMA-aware IOCP runtime with one completion port per node and manually created worker threads pinned to node affinity.
- A named-pipe server with an IOCP state machine for request/response.
- A client that submits requests and optionally waits on bulk completion tokens.
- A bulk-create directory request that fans out per-file work items to IOCP nodes.

The core IOCP pipeline runs on Windows with IOCP worker threads (no Windows threadpools for IOCP workers). Per-file perfect hash work still uses legacy `PERFECT_HASH_CONTEXT` / table-create logic, including its internal threadpool usage.

## Key Components / Files

IOCP runtime + NUMA:
- `src/PerfectHash/PerfectHashContextIocp.c`
- `src/PerfectHash/PerfectHashContextIocp.h`

Server / pipe IOCP:
- `src/PerfectHash/PerfectHashServer.c`
- `src/PerfectHash/PerfectHashServer.h`

Client:
- `src/PerfectHash/PerfectHashClient.c`
- `src/PerfectHash/PerfectHashClient.h`

Server exe:
- `src/PerfectHashServerExe/PerfectHashServerExe.c`

Client exe:
- `src/PerfectHashClientExe/PerfectHashClientExe.c`

Protocol constants and bulk result struct:
- `include/PerfectHash.h`

Scripts:
- `scripts/iocp-smoke.ps1` (table create over IOCP)
- `scripts/stress-sys32-iocp.ps1` (bulk-create directory request)
- `scripts/stress-sys32.ps1` (baseline bulk create)

Logs / TODOs:
- `IOCP-LOGS.md`
- `IOCP-TOOD.md`

## Implemented Features (IOCP Backend)

- NUMA-aware IOCP runtime:
  - Enumerates NUMA nodes.
  - Creates one IOCP per node.
  - Spawns worker threads per node with group affinity.
  - Configurable max concurrency and NUMA node mask.
  - Implemented in `src/PerfectHash/PerfectHashContextIocp.c`.

- Server named-pipe IOCP transport:
  - Per-node pipe instances tied to the node IOCP.
  - IOCP state machine for connect, read header/payload, write response.
  - Local-only vs allow-remote configuration.
  - Implemented in `src/PerfectHash/PerfectHashServer.c`.

- Client request handling:
  - Supports `--TableCreate`, `--BulkCreate`, `--BulkCreateDirectory`, `--Shutdown`.
  - Waits on bulk-create completion tokens via event + mapping.
  - Implemented in `src/PerfectHashClientExe/PerfectHashClientExe.c` and `src/PerfectHash/PerfectHashClient.c`.

- Bulk-create directory request:
  - Server walks a directory and queues per-file IOCP work items.
  - Uses round-robin dispatch across NUMA nodes.
  - Completion logic tracks per-node counts and outstanding totals to signal a single completion token.
  - Implemented in `src/PerfectHash/PerfectHashServer.c`.

## Recent Fixes / Diagnostics

- Fixed a crash in IOCP bulk-create caused by `Context->CommandLineW` being NULL when `PrepareCHeaderFileChm01` emits the command line.
  - Bulk request now keeps a copy of the command line and assigns it to each per-file context.
  - Files: `src/PerfectHash/PerfectHashServer.c`

- Fixed use-after-free in bulk work callback (request freed before work item cleanup).
  - Files: `src/PerfectHash/PerfectHashServer.c`

- Fixed `CommandLineToArgvW` lifetime for bulk directory requests:
  - `ArgvW` is now retained for the request lifetime, and freed on completion.
  - Files: `src/PerfectHash/PerfectHashServer.c`

- Added richer crash logging for server process:
  - Records exception address, module base, offset, and thread ID before minidump attempt.
  - Files: `src/PerfectHashServerExe/PerfectHashServerExe.c`
  - Use env var `PH_LOG_SERVER_CRASH=1` to enable.
  - `PH_SERVER_MINIDUMP_FORCE_FALLBACK=1` forces fallback minidump behavior.

- Adjusted IOCP stress script to accept success exit code:
  - `PH_S_SERVER_BULK_CREATE_ALL_SUCCEEDED` (0x2004000F).
  - Files: `scripts/stress-sys32-iocp.ps1`

## What Works Now (Observed)

- `scripts/iocp-smoke.ps1` succeeds (table create via IOCP server/client).
- `scripts/stress-sys32-iocp.ps1` succeeds on a small directory (single keys file) after crash fixes.

## Known Gaps / Risks

- IOCP bulk-create directory with full `..\perfecthash-keys\sys32` has not been rerun post-fix; expected to be next.
- Per-file work still uses legacy table-create which relies on its internal threadpool and console hooks; IOCP orchestration does not yet replace that internal compute pipeline.
- Backpressure/queueing policies are still “post everything immediately” (no high-water mark).
- Linux/`io_uring` path is not implemented; current IOCP design is Windows-only.

## Next Steps (Plan)

1. Run full IOCP stress:
   - `scripts/stress-sys32-iocp.ps1` against `..\perfecthash-keys\sys32` with Release and desired concurrency.
2. Performance comparison:
   - Compare baseline `scripts/stress-sys32.ps1` vs IOCP run (same hash/mask/concurrency).
3. Native IOCP pipeline:
   - Replace legacy context delegation with explicit IOCP-driven phases (key load, graph solve, file work).
   - Add proper queue/backpressure per request and per node.
4. Multi-node tracking polish:
   - Expand per-node completion tracking if needed for more granular results.
5. Docs:
   - Draft `IOCP-README.md` once behavior and protocol stabilize.

## Suggested Commands

Build:
- `cmake --build build-win --config Release --target PerfectHashServerExe PerfectHashClientExe`

Smoke:
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\iocp-smoke.ps1 -BuildDir build-win -Config Release`

IOCP stress (sys32):
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stress-sys32-iocp.ps1 -BuildDir build-win -Config Release -MaximumConcurrency 32`

Baseline:
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stress-sys32.ps1 -BuildDir build-win -Config Release -MaximumConcurrency 32`

