# Roborev Context For PerfectHash

This file provides review context for Roborev and the main Codex session in
this repo.

## What This Repo Covers

PerfectHash spans several surfaces that often move together:

- native library and CLI code under `src/**` and `include/**`,
- generated-file plumbing in `src/PerfectHash/**`,
- Python bindings and tests under `python/perfecthash/**` and `python/tests/**`,
- build, preset, packaging, and release automation under `CMakeLists.txt`,
  `CMakePresets.json`, `ci/**`, `conda/**`, and `scripts/**`.

## Review Boundary

When reviewing a diff:

- focus on correctness, regressions, missing targeted validation, and docs/API
  drift in the touched scope,
- prefer concrete findings tied to the actual files changed,
- avoid broad cleanup asks unless the current diff makes them necessary,
- verify added source files are mirrored in the relevant CMake and Visual
  Studio project files when required.

## Priority Review Areas

- generated-file ordering and callback wiring invariants,
- curated good-hash selection consistency in `include/PerfectHash.h`,
- CMake preset/build-profile regressions across default, online, and CUDA paths,
- Python CLI/API behavior drift relative to docs and tests,
- missing CTest or `pytest` coverage for changed user-facing behavior,
- release/versioning mistakes when release notes, packaging, or workflows are
  touched.

## Validation Expectations

Prefer targeted validation for the touched area, for example:

- `env -u PYTHONPATH uv run pre-commit run --files ...`
- `env -u PYTHONPATH uv run pytest`
- `cmake --build ...`
- `ctest --test-dir ... --output-on-failure`

If a finding depends on a platform-specific path, call out the exact platform,
build profile, or preset implicated.
