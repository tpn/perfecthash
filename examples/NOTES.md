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
- `examples/cpp-online-rawdog-console/` (planned): standalone C++ CMake project.

## Planned Implementation Phases
1. Ledger setup and planning docs.
2. Define downstream CMake discovery contract for online/rawdog runtime.
3. Create minimal console example that loads/provides a 32-bit key set, creates a table via online API, compiles the table with the RawDog JIT backend, and validates lookup uniqueness.
4. Add platform-aware CMake logic for Windows/Linux/macOS and x86_64/arm64 routing.
5. Add verification notes and copy/paste build instructions for each platform/compiler family.

## Open Questions To Resolve During Implementation
- Preferred external `find_package` UX: `find_package(PerfectHash CONFIG REQUIRED)` (if we add package export), or `find_package(PerfectHashOnline REQUIRED)` via a module in `examples/`.
- Whether the sample should default to shared or static linking for easiest cross-platform adoption.
- Whether to include a tiny generated key dataset in-tree or build keys at runtime in code.
