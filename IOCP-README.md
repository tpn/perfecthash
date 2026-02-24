# IOCP PerfectHash Backend

This document is the current high-level reference for the additive IOCP implementation.

## Scope

- New server/client path for table and bulk create requests.
- Keep OG `CreateExe` / `BulkCreateExe` behavior intact.
- Use manual Windows threads + `GetQueuedCompletionStatus()` loops.
- NUMA-aware design: one IOCP per node, affinity-aware workers.

## Design Summary

1. Client sends request over named pipe.
2. Server accepts request via IOCP pipe state machine.
3. For bulk-directory requests:
   - enumerate `.keys` files
   - dispatch per-file work items (round-robin across selected NUMA nodes)
   - track per-request counters
   - signal one completion token when all file work is complete.
4. Client waits on token and receives a final bulk status code.

## Concurrency Controls

- `--IocpConcurrency`:
  - completion-port concurrency level (also aliasable from older `--MaxConcurrency` usage in some paths/scripts).
- `--MaxThreads`:
  - max worker threads to create (default: `IocpConcurrency * 2`).
- Per-file async ramp:
  - `--InitialPerFileConcurrency`
  - `--MaxPerFileConcurrency`
  - `--IncreaseConcurrencyAfterMilliseconds`.

## File I/O Mode

- IOCP path:
  - overlapped writes for generated files
  - overlapped keys reads
  - pooled intermediate buffers.
- OG path:
  - existing memory-mapped behavior preserved.

## Buffering

- Direction: NUMA-aware lookaside-style pooling.
- Current state:
  - size-class pools (power-of-two) are in place and evolving
  - oversize buffer handling exists and needs further hardening/tuning
  - optional guard-page behavior is supported for safer debug/development runs.

## Operational Flags

- `--WaitForServer` and `--ConnectTimeout=<ms>` on client for robust startup coordination.
- `--Verbose` on server gates console output (silent by default).
- `--NoFileIo` supported for rapid compute-only stress loops.

## Current Status

- Core IOCP pipeline is functional.
- Small/medium dataset runs are stable.
- Full sys32 runs have succeeded in both `NoFileIo` and file-I/O modes under tested settings.
- Remaining work is mostly:
  - correctness hardening for edge/failure paths
  - buffer pool policy/memory scaling
  - deeper performance tuning and regression automation.

## See Also

- Execution log: `IOCP-LOGS.md`
- Active backlog: `IOCP-TOOD.md`
- Session handoff prompt: `IOCP-PROMPT.md`

