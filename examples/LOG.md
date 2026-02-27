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
- 2026-02-27: Investigated `PH_E_NOT_IMPLEMENTED` on x64 AVX-512 host; confirmed CPU feature detection is correct and failure was due to hash/vector kernel availability (`mulshrolate4rx` vector path with default `PH_RAWDOG_VECTOR_VERSION=3`), not missing AVX-512 support.
- 2026-02-27: Updated slim RawDog JIT API compile wrapper to retry smaller vector widths (e.g., 16->8->4->1) before surfacing `PH_E_NOT_IMPLEMENTED`.
- 2026-02-27: Changed console example default hash to `multiplyshiftr` for a cross-platform RawDog JIT-success default path.
- 2026-02-27: Updated console example default hash to `mulshrolate2rx` per user direction.
- 2026-02-27: Fixed Windows build break in `PerfectHashOnlineRawdog.c` by using GUID pointer forms (`&CLSID`, `&IID`) for `PerfectHashDllGetClassObject()` / `CreateInstance()` arguments.
- 2026-02-27: Added slim dual-backend online API (`PerfectHashOnlineJit.h` + `PerfectHashOnlineJit.c`) supporting RawDog JIT and LLVM JIT backend selection.
- 2026-02-27: Added `examples/cpp-console-online-jit/` C++ CMake sample for runtime table generation with `rawdog-jit`, `llvm-jit`, or `auto`.
- 2026-02-27: Added GitHub Actions coverage to build/run both C++ online JIT examples on Linux/macOS/Windows matrices.
- 2026-02-27: Began sqlite integration discovery by ingesting sqlite upstream source and reviewing `src/where.c`, `src/vtab.c`, and `ext/misc/series.c`.
- 2026-02-27: Added sqlite example planning scaffold under `examples/sqlite-online-jit/` with integration and benchmark design notes.
- 2026-02-27: Vendored sqlite amalgamation snapshot `3.51.2` (`3510200`) into `examples/sqlite-online-jit/sqlite/`.
- 2026-02-27: Added `examples/sqlite-online-jit/CMakeLists.txt` and finder wiring for `PerfectHashOnlineJit`.
- 2026-02-27: Implemented `perfecthash` sqlite virtual table module (`src/perfecthash_vtab.cpp`) backed by runtime PerfectHash index generation.
- 2026-02-27: Implemented sqlite benchmark runner (`src/main.cpp`) with A/B comparison between baseline B-tree join and PerfectHash virtual-table join.
- 2026-02-27: Validated local runs for backends `rawdog-jit`, `llvm-jit`, and `auto` with matching results and reported speedups.
