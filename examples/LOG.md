# Examples Work Log

This file is append-only. Add new entries at the end with an explicit date.

- 2026-02-27: Created branch `examples-dev` for examples-focused work.
- 2026-02-27: Created `examples/` ledger files (`NOTES.md`, `LOG.md`, `AGENTS.md`, `TODO.md`).
- 2026-02-27: Audited existing targets and build wiring for online/rawdog support; identified `PerfectHashOnlineCore` as the current minimal-runtime candidate to validate.
- 2026-02-27: Refined ledger content for readability/ingestion and aligned TODO phases for the first implementation pass.
- 2026-02-27: Added slim public online/rawdog header `include/PerfectHashOnlineRawdog.h` and wrapper implementation `src/PerfectHash/PerfectHashOnlineRawdog.c`.
- 2026-02-27: Wired wrapper API into `src/PerfectHash/CMakeLists.txt` and Visual Studio project files.
- 2026-02-27: Created example project `examples/cpp-console-online-rawdog-jit/` (CMake finder module, console app, README).
- 2026-02-27: Added static-link fallback dependencies in the example finder module (`PerfectHashAsm`, `Threads`, `rt`/`rpcrt4` as applicable).
- 2026-02-27: Enabled core-only online compile and JIT stubs to route RawDog backend requests instead of unconditional `PH_E_NOT_IMPLEMENTED`.
- 2026-02-27: Enabled RawDog compile definitions for `PerfectHashOnlineCore` targets.
- 2026-02-27: Validated Linux configure/build/run for the new example with static and shared discovery; observed expected RawDog capability fallback (`PH_E_NOT_IMPLEMENTED`) on this host and retained non-JIT verification path.
