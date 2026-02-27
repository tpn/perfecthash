# Examples TODO

## Ledger Bootstrap
- [x] Create `examples/` ledgers: `NOTES.md`, `LOG.md`, `AGENTS.md`, `TODO.md`.
- [x] Capture initial technical assumptions for online/RawDog-JIT integration.
- [x] Confirm with user that ledger direction and scope are acceptable before code scaffolding.

## Discovery and Packaging Contract
- [x] Decide initial downstream discovery mechanism: module-based `find_package(PerfectHashOnlineRawdog)` in the example subtree.
- [ ] Verify minimal shippable runtime artifact choice (`PerfectHashOnlineCore` shared/static) with dependency and size checks.
- [x] Define stable imported target name for the example project: `PerfectHash::OnlineRawdog`.

## Example Project Scaffolding
- [x] Create `examples/cpp-console-online-rawdog-jit/` CMake project skeleton.
- [x] Add platform-aware find logic and runtime-copy/build-rpath handling in CMake.
- [x] Add 32-bit key runtime create/compile/query flow using online API and RawDog JIT flags.
- [x] Add a new slim public C API/header pair for online+RawDog usage.

## Validation
- [x] Test configure/build/run on Linux (`gcc`) for static and shared finder paths.
- [ ] Test configure/build/run on Linux (`clang`).
- [ ] Test configure/build/run on macOS (`appleclang`).
- [ ] Test configure/build/run on Windows (`msvc`, `clang-cl`).
- [ ] Verify x86_64 and arm64 builds only include their respective RawDog-JIT routines.

## Docs and Handoff
- [x] Add `README.md` inside the example with quickstart and troubleshooting.
- [ ] Add compact architecture diagram or bullet walkthrough for human + LLM ingestion.
- [ ] Record final results and residual follow-up ideas (e.g., UnrealEngine/sqlite3 integrations).

## Dual-Backend Example (`cpp-console-online-jit`)
- [x] Add slim public online API that supports RawDog JIT and LLVM JIT.
- [x] Add `examples/cpp-console-online-jit/` with CMake finder + console app.
- [x] Support explicit backend selection (`rawdog-jit`, `llvm-jit`, `auto`).
- [x] Ensure default hash is `mulshrolate2rx`.

## CI Coverage
- [x] Add GitHub Actions steps for Linux/macOS/Windows to build and run both
  C++ online JIT examples.
- [ ] Monitor PR checks and confirm all platform jobs are green after push.

## SQLite Online JIT Example
- [x] Ingest sqlite source tree for design review and identify integration seam.
- [x] Review sqlite planner/virtual-table internals (`where.c`, `vtab.c`,
  `ext/misc/series.c`).
- [x] Select initial integration strategy (virtual table module + PerfectHash
  online JIT).
- [x] Draft `examples/sqlite-online-jit/README.md` and `PLAN.md`.
- [x] Choose and document sqlite vendoring approach for this repo
  (`amalgamation` snapshot vs full mirror snapshot).
- [x] Scaffold `examples/sqlite-online-jit/` CMake project (sqlite + PerfectHash).
- [x] Implement PerfectHash-backed virtual table module.
- [x] Add benchmark harness with easy A/B toggles.
- [x] Add CI execution for sqlite example.
