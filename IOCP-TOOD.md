# IOCP TODO

- Validate bulk-create directory request end-to-end (sys32 stress) and tune per-file concurrency defaults.
- Run IOCP sys32 stress pass (Release, max concurrency) after crash fixes.
- Investigate Release bulk-create crash in `PerfectHashBulkCreate.exe` (0xC0000005) during sys32 stress.
- Track down `FlushConsoleInputBuffer` failures during bulk create (seen on `main` and iocp-dev).
- Validate access-denied fallback for per-file context threadpool minimum failures at higher concurrency.
- Recheck named-pipe endpoint handling if `PerfectHashServer-StressSys32` continues to fail.
- Decide whether BulkCreateDirectory should accept a single-directory short form or keep output dir required.
- Exercise IOCP file work dispatch path (bulk create) to confirm outstanding event signaling and non-threadpool file work callbacks.
- Validate CHM01 async path with smaller keysets (e.g. `hard`) and record timing vs legacy.
- Investigate IOCP bulk-create async hang on `hard` keyset (client wait never completes).
- Validate context file work skip fix for `SetEndOfFile`/`PerfectHashFileTruncate` 1224 failures; decide whether to generate context files once per bulk request.
- Handle CHM01 async resize/try-larger-table-size path (currently returns `PH_E_NOT_IMPLEMENTED`).
- Wire IOCP server bulk path to CHM01 async jobs and request completion callbacks.
- Complete native IOCP table/bulk create workflows (replace legacy-context delegation).
- Add IOCP work item queueing/state to drive per-request pipelines.
- Integrate per-request concurrency caps and queueing policies.
- Draft IOCP-README.md once protocol and dispatch behavior settle.
