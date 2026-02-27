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
