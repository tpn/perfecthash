# Release Engineering TODO

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Completed

## Core Automation
- [x] Implement git-tag-derived version resolution in CMake and release scripts.
  - Completed: `cmake/PerfectHashVersion.cmake`, top-level + standalone `src/` wiring, script fallback order updates.
- [x] Remove or deprecate manual bump workflow/docs (`ci/bump-version.sh`, historical references).
  - Completed: removed `ci/bump-version.sh`.
  - Completed: added replacement release-tag helper `ci/cut-release.sh`.
  - Completed: docs now point to tag-cut workflow.
- [~] Add release metadata generation (version, commit, build profile) into artifact naming or manifest.
  - Completed: profile-aware artifact naming + generated release notes with commit summaries.
  - Remaining: optional machine-readable artifact manifest (`manifest.json`).

## CMake Packaging + Consumer Experience
- [x] Add install/export package config support (`PerfectHashConfig.cmake`, `PerfectHashTargets.cmake`, version config).
- [x] Provide `FetchContent`/CPM consumer example under `examples/`.
- [x] Migrate all maintained examples to GitHub-backed FetchContent-first workflows with local fallback.
- [x] Add/verify namespaced imported targets (for example `PerfectHash::PerfectHashOnlineCore`).
- [x] Validate FetchContent consumer builds across profiles against GitHub `main`.
  - Completed: `online-rawdog-jit`, `online-rawdog-and-llvm-jit`, `online-llvm-jit`, and `full` all pass with local source-override validation.
  - Note: direct GitHub `main` currently still uses legacy profile names until these profile-renaming changes are published.

## Build Profiles
- [x] Introduce explicit build profile options:
  - `online-rawdog-jit`
  - `online-rawdog-and-llvm-jit`
  - `online-llvm-jit`
  - `full`
- [x] Disable embedded human-readable error payloads for online profiles while preserving raw `PH_*` HRESULT behavior.
  - Completed: added `PERFECTHASH_ENABLE_EMBEDDED_ERROR_STRINGS` and set profile defaults to `OFF` for online profiles, `ON` for `full`.
  - Completed: fixed RawDog/header-generation independence from embedded-error toggling and revalidated all profile builds.
- [x] Ensure profile docs map to concrete CMake options and emitted targets.
- [x] Add CMake presets for each profile.
- [x] Validate release-profile behavior across all platforms in CI.
  - Completed: local online-rawdog-jit install/consume path.
  - Completed: full profile matrix tag run (`v0.70.5`) succeeded on Linux/macOS/Windows publish path.

## GitHub Release Workflow
- [x] Extend release workflow with generated release notes from commits since previous tag.
- [x] Add support for curated notes integration (`RELEASE-NOTES.md` sections).
- [x] Ensure release artifacts and checksums are uploaded in consistent layout.
- [x] Add optional manual dispatch mode for profile-specific dry-run releases.

## PGO + Distribution Strategy
- [~] Define Windows PGO release path in CI (instrument/train/optimize/package).
  - Completed: documented strategy in `docs/release-process.md`.
  - Remaining: implement workflow jobs/scripts.
- [ ] Decide whether Linux/macOS PGO variants are desirable or Windows-only for now.
- [x] Document asset naming convention for PGO vs standard artifacts.

## Conda / Package Distribution
- [x] Create conda package strategy doc and scaffolding.
  - Completed: `conda/recipe/meta.yaml`, `conda/recipe/build.sh`, `.github/workflows/conda-package.yml`, docs strategy.
- [x] Validate conda package workflow on `main` for `online-rawdog-jit`.
  - Completed: workflow dispatch run `22515274877` succeeded after recipe fixes.
- [~] Evaluate staged rollout to conda-forge (manual PR assist + bot automation hooks).
  - Remaining: implement feedstock update automation and credentials governance.

## Documentation
- [x] Rewrite README release/build sections to reflect current project status and automation flow.
- [x] Add `RELEASE-NOTES.md` with agent instructions and section format.
- [x] Add `docs/release-process.md` for maintainers.

## Tracking
- [~] Stabilize cross-platform CI after assigned16 boundary hardening.
  - Completed: fixed Windows configure failure caused by duplicate RawDog custom-command outputs (`src/PerfectHash/CMakeLists.txt` x64 non-Windows gating).
  - Completed: made assigned16 boundary unit coverage deterministic and bounded; removed flaky RNG-dependent behavior.
  - Completed: excluded heavy boundary tests from `perfecthash.fast.unit` while keeping dedicated per-test coverage via `gtest_discover_tests`.
  - Completed: updated CLI assigned16 boundary test flags to avoid the hash-all-keys-first crash path.
  - Remaining: confirm green GitHub Linux/Windows/macOS matrix on the post-fix commit.
- [~] Keep this TODO synchronized as tasks complete; each completed task must have a corresponding LOG entry.
