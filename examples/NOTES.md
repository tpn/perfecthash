# Examples Project Notes

## Objective
Build a new example project rooted in `examples/` that demonstrates a minimal C++ CMake console application using PerfectHash online mode with RawDog JIT to generate a runtime perfect hash table for 32-bit keys.

## User Requirements (Captured)
- Use CMake to find and consume the online + RawDog runtime.
- Target the smallest shippable runtime artifact (`.so` / `.dylib` / `.dll` + `.lib`, or static `.a` / `.lib` where needed).
- Keep architecture-specific behavior clean: `x86_64` builds should only carry x86_64 hash/JIT routines, and `arm64` builds should only carry arm64 hash/JIT routines.
- Work across Windows, Linux, and macOS.
- Work across mainstream compilers (MSVC, clang-cl, Clang, GCC, AppleClang).
- Treat this as a human-readable and LLM-ingestable reference integration.

## Current Technical Observations
- Existing online-related CMake targets already exist: `PerfectHashOnline`, `PerfectHashOnlineStatic`, `PerfectHashOnlineCore`, and `PerfectHashOnlineCoreStatic`.
- `PerfectHashOnlineCore` excludes `ChmOnline01.c` (LLVM-heavy path), which makes it the leading candidate for the smallest RawDog-focused runtime.
- RawDog availability and arch macros are already wired in the core build (`PH_HAS_RAWDOG_JIT`, `PH_RAWDOG_X64`, `PH_RAWDOG_ARM64`).
- The root project currently installs targets but does not yet export an installed CMake package config for downstream `find_package(...)`.

## Working Assumption
Start with `PerfectHashOnlineCore` / `PerfectHashOnlineCoreStatic` as the baseline runtime to ship for this example, and verify by:
- dependency surface (no LLVM runtime dependency),
- binary size comparisons,
- successful runtime JIT path using `PERFECT_HASH_TABLE_COMPILE_FLAGS` with `Jit=TRUE` + `JitBackendRawDog=TRUE`.

## Proposed Example Layout
- `examples/AGENTS.md`: local agent workflow guardrails for this subtree.
- `examples/NOTES.md`: evolving architecture/design decisions.
- `examples/LOG.md`: append-only execution log.
- `examples/TODO.md`: actionable task tracker.
- `examples/cpp-console-online-rawdog-jit/`: standalone C++ CMake project.

## Planned Implementation Phases
1. Ledger setup and planning docs.
2. Define downstream CMake discovery contract for online/RawDog-JIT runtime.
3. Create minimal console example that loads/provides a 32-bit key set, creates a table via online API, compiles the table with the RawDog JIT backend, and validates lookup uniqueness.
4. Add platform-aware CMake logic for Windows/Linux/macOS and x86_64/arm64 routing.
5. Add verification notes and copy/paste build instructions for each platform/compiler family.

## Open Questions To Resolve During Implementation
- Preferred external `find_package` UX: `find_package(PerfectHash CONFIG REQUIRED)` (if we add package export), or `find_package(PerfectHashOnline REQUIRED)` via a module in `examples/`.
- Whether the sample should default to shared or static linking for easiest cross-platform adoption.
- Whether to include a tiny generated key dataset in-tree or build keys at runtime in code.

## Current Direction (2026-02-27)
- Added a new slim public header: `include/PerfectHashOnlineRawdog.h`.
- Added a new wrapper implementation: `src/PerfectHash/PerfectHashOnlineRawdog.c`.
- Wrapper API hides internal COM-style interfaces and exposes a compact C API for:
- open/close online context,
- create 32-bit table,
- compile table with RawDog JIT flags,
- index 32-bit keys,
- release table.
- Enabled RawDog compile flow for `PH_ONLINE_CORE_ONLY` builds by routing
  core-only JIT compile stubs through RawDog paths when requested.
- Enabled RawDog compile defines for `PerfectHashOnlineCore` targets in CMake.
- Example project name is now fixed as `cpp-console-online-rawdog-jit`.
- The console example handles `PH_E_NOT_IMPLEMENTED` from RawDog JIT as a
  host/table capability fallback and still verifies index uniqueness via
  non-JIT indexing.

## PH_E_NOT_IMPLEMENTED Investigation (2026-02-27)
- Reproduced the issue on x64 AVX-512 host with `--hash mulshrolate4rx --vector-width 16`.
- Enabled RawDog JIT CPU tracing and confirmed host detection is healthy:
  `AVX=1`, `AVX2=1`, `AVX512F=1`, `OSXSAVE=1`, valid `XCR0`.
- Root cause is vector kernel availability for specific hash + vector-width
  combinations (notably `mulshrolate4rx` under default
  `PH_RAWDOG_VECTOR_VERSION=3`), not ISA detection failure.
- Mitigation added in slim API wrapper: compile retries smaller vector widths
  before returning `PH_E_NOT_IMPLEMENTED`, which keeps RawDog JIT enabled when
  at least scalar JIT codegen is available.

## Dual-Backend Console Example (2026-02-27)
- Added `include/PerfectHashOnlineJit.h` and
  `src/PerfectHash/PerfectHashOnlineJit.c` as a slim public online API that
  supports both RawDog JIT and LLVM JIT backends.
- Added a second C++ CMake sample:
  `examples/cpp-console-online-jit/`.
- New sample supports `--backend rawdog-jit|llvm-jit|auto`.
- Default hash for the console flow is `mulshrolate2rx`.
- Finder module (`FindPerfectHashOnlineJit.cmake`) resolves both:
- `PerfectHashOnline` runtime (main API surface),
- `PerfectHashLLVM` runtime (needed for LLVM JIT backend use).

## CI Matrix Coverage For Examples (2026-02-27)
- Linux, macOS, and Windows workflows now build and execute:
- `cpp-console-online-rawdog-jit`
- `cpp-console-online-jit` in both backend modes:
  `--backend rawdog-jit` and `--backend llvm-jit`.
- This gives platform-matrix coverage for the minimal RawDog-JIT runtime path
  and the dual-backend path.

## SQLite Online JIT Integration Direction (Draft, 2026-02-27)
- Reviewed upstream sqlite planner/virtual-table internals to identify the
  lowest-risk integration seam:
- `sqlite/src/where.c` (`xBestIndex` planning/loop costing hooks),
- `sqlite/src/vtab.c` (module lifecycle and registration),
- `sqlite/ext/misc/series.c` (reference virtual table implementation style).
- Chosen first strategy: integrate PerfectHash as a virtual table module
  backed by an in-memory perfect hash index for 32-bit keys, instead of
  patching sqlite core planner internals in v1.
- Planned A/B benchmark modes:
- baseline join (`fact` + `dim` with sqlite B-tree index),
- PerfectHash virtual-table join (`fact` + `dim_ph`),
- backend variant toggle (`rawdog-jit` vs `llvm-jit`) for the PerfectHash path.

## SQLite Online JIT Prototype Implemented (2026-02-27)
- Added `examples/sqlite-online-jit/` CMake project with vendored sqlite
  amalgamation (`3.51.2` / `3510200`).
- Implemented `perfecthash` sqlite virtual table module in
  `src/perfecthash_vtab.cpp`.
- Module builds a PerfectHash table from sqlite source table rows
  (`source_table`, `key_column`, `value_column`) and serves lookup rows via:
- `xBestIndex` equality planning on key,
- `xFilter`/`xColumn` point-lookup fast path backed by `PhOnlineJitIndex32()`.
- Implemented benchmark runner in `src/main.cpp` with direct A/B execution:
- baseline sqlite join against B-tree path,
- PerfectHash virtual-table join using selectable backend
  (`rawdog-jit`, `llvm-jit`, `auto`).
- Added result parity check (baseline sum must equal PerfectHash sum) and
  reported average runtime + speedup.

## SQLite Permutation Matrix Expansion (2026-02-27)
- Expanded sqlite benchmark runner to compare full permutations of:
- backend: `rawdog-jit`, `llvm-jit`,
- hash function: all curated 32-bit hashes,
- vector width: `1`, `2`, `4`, `8`, `16` (including AVX-512 width `16`).
- Matrix mode is now the default when no single-run override flags are passed.
- Added strict vector-width compile support to slim online API:
  `PhOnlineJitCompileTableEx(..., Flags, EffectiveBackend, EffectiveVectorWidth)`
  with `PH_ONLINE_JIT_COMPILE_FLAG_STRICT_VECTOR_WIDTH`.
- sqlite virtual-table integration now records and reports per-run compile
  metadata:
- requested/effective backend,
- requested/effective vector width,
- compile HRESULT,
- JIT enabled/disabled state.
