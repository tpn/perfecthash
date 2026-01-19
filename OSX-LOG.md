# OSX Port Log

- 2026-01-19: Started macOS Apple Silicon bring-up; reviewed existing CMake + compat layer and prepared for first build attempt.
- 2026-01-19: Created `perfecthash-macos` mamba env; added macOS env recipes under `conda/environments/` and updated `dependencies.yaml`.
- 2026-01-19: Added macOS compile definitions and page-size detection in CMake; fixed compat layer for macOS (pthread barriers, SRW lock lazy init, file mapping/move/info/handle closures, TLS typing, VirtualAlloc/Protect/Free sizing, MAP_* fallbacks).
- 2026-01-19: Built `Release` via CMake/Ninja and ran full CTest suite successfully (14/14 passing).
- 2026-01-19: Rebuilt `build-macos` and re-ran full `ctest` suite (Release) successfully after cleanup.
- 2026-01-19: Installed `rapids-dependency-file-generator` in the macOS env and regenerated all dependency outputs.
- 2026-01-19: Added Intel macOS envs plus CMake preset for `ninja-multi-macos-x86_64` and build presets.
- 2026-01-19: Added GitHub Actions macOS workflow for arm64 (macos-14) and x86_64 (macos-13) using micromamba + CMake presets.
- 2026-01-19: Added Rust toolchain (cargo) to dependency sets and regenerated env files; updated README to note Rust requirement for codegen tests.
- 2026-01-19: Strengthened codegen test coverage to assert generated artifacts, require cargo when Rust outputs exist, and run a Unix Makefiles CMake build.
- 2026-01-19: Installed Rust toolchain into `perfecthash-macos` via mamba to enable cargo-backed codegen tests.
- 2026-01-19: Configured `build-tests-macos`, built with Ninja, and ran full CTest suite (14/14 passing) with cargo-enabled codegen checks.
