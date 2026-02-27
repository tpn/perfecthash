# Examples TODO

## Ledger Bootstrap
- [x] Create `examples/` ledgers: `NOTES.md`, `LOG.md`, `AGENTS.md`, `TODO.md`.
- [x] Capture initial technical assumptions for online/rawdog integration.
- [ ] Confirm with user that ledger direction and scope are acceptable before code scaffolding.

## Discovery and Packaging Contract
- [ ] Decide final downstream CMake discovery mechanism (`CONFIG` package export vs module-based `find_package` fallback).
- [ ] Verify minimal shippable runtime artifact choice (`PerfectHashOnlineCore` shared/static) with dependency and size checks.
- [ ] Define stable imported target name(s) for the example project to consume.

## Example Project Scaffolding
- [ ] Create `examples/cpp-online-rawdog-console/` CMake project skeleton.
- [ ] Add platform/compiler detection and clear diagnostics for unsupported toolchains/configurations.
- [ ] Add 32-bit key runtime create/compile/query flow using online API and RawDog JIT flags.

## Validation
- [ ] Test configure/build/run on Linux (`gcc`, `clang`).
- [ ] Test configure/build/run on macOS (`appleclang`).
- [ ] Test configure/build/run on Windows (`msvc`, `clang-cl`).
- [ ] Verify x86_64 and arm64 builds only include their respective rawdog routines.

## Docs and Handoff
- [ ] Add `README.md` inside the example with quickstart and troubleshooting.
- [ ] Add compact architecture diagram or bullet walkthrough for human + LLM ingestion.
- [ ] Record final results and residual follow-up ideas (e.g., UnrealEngine/sqlite3 integrations).
