---
name: improved-ui
description: Agent companion for the PerfectHash UI: command builder guidance, environment/bootstrap steps, build flows, tests, and analysis checkpoints.
---

# Improved UI Companion

## User / Create
- Generate a CLI command for a single keys file:
  - `PerfectHashCreate.exe <KeysPath> <OutputDir> Chm01 MultiplyShiftRX And <MaxConcurrency>`
- Bulk create across a keys directory:
  - `PerfectHashBulkCreate.exe <KeysDir> <OutputDir> Chm01 MultiplyShiftRX And <MaxConcurrency>`
- UI runner (local):
  - `cd ui && npm run server`
  - Or `cd ui && npm run dev:full` to run UI + runner together.
- Runner defaults to port 7071; override with `PERFECTHASH_UI_SERVER_PORT`.
- UI can target a custom endpoint via `VITE_RUNNER_URL`.
- Recommended hash set for exploration: MultiplyShiftR, MultiplyShiftLR, MultiplyShiftRMultiply, MultiplyShiftR2, MultiplyShiftRX, Mulshrolate1RX, Mulshrolate2RX, Mulshrolate3RX, Mulshrolate4RX.
- Common flags:
  - `--SkipTestAfterCreate` to skip verification/benchmarks.
  - `--Compile` to build generated outputs (msbuild must be on PATH).
  - `--IndexOnly` to omit table values array and associated projects.

## Developer / Environment
- Install OS dependencies:
  - `scripts/install-deps/ubuntu.sh`
  - `scripts/install-deps/fedora.sh`
  - `scripts/install-deps/arch.sh`
  - `scripts/install-deps/windows.ps1`
- Optional Rust/Cargo install:
  - `WITH_RUST=1 scripts/install-deps/ubuntu.sh`
  - `WITH_RUST=1 scripts/install-deps/fedora.sh`
  - `WITH_RUST=1 scripts/install-deps/arch.sh`
  - `WITH_RUST=1 powershell -ExecutionPolicy Bypass -File scripts/install-deps/windows.ps1`
- Linux conda environment:
  - `mamba env create -f conda/environments/dev-linux_os-linux_arch-x86_64_py-313_cuda-none_compiler-llvm.yaml`
  - `mamba activate dev-linux_os-linux_arch-x86_64_py-313_cuda-none_compiler-llvm`
- UI setup:
  - `cd ui && npm install`
- Environment UI bootstrap:
  - Use the Bootstrap button to run system prerequisites then conda create.
  - Conda create uses composable Python/CUDA/compiler selections and `conda run`.
  - If sudo is required, the UI prompts for the password and sends it to the session.

## Developer / Build
- Configure via `build.sh` (auto-selects Ninja when available):
  - `./build.sh`
  - `./build.sh -DCMAKE_BUILD_TYPE=Release`
- Build:
  - `cmake --build build --config Release`
- Windows:
  - `msbuild /nologo /m /t:Rebuild /p:Configuration=Release;Platform=x64 src/PerfectHash.sln`
- Build UI:
  - Use the CMake configure/build/install rows to run `cmake` commands with selected generators, build types, and PerfectHash options.

## Developer / Test
- CMake/CTest:
  - `ctest --test-dir build --output-on-failure`
- CLI regression:
  - `cmake -P tests/run_cli_test.cmake`
  - `cmake -P tests/run_cli_codegen_test.cmake`
- UI:
  - `cd ui && npm test`
  - `cd ui && npm run test:e2e`

## Analysis
- CSV outputs live in the table output directory; summarize coverage and timing deltas.
- Use the curated good hash set when benchmarking.
- Track output DLL size and Index() throughput across runs.
