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
- [x] Add/verify namespaced imported targets (for example `PerfectHash::PerfectHashOnlineCore`).

## Build Profiles
- [x] Introduce explicit build profile options:
  - `online-rawdog`
  - `online-rawdog-llvm`
  - `full`
- [x] Ensure profile docs map to concrete CMake options and emitted targets.
- [x] Add CMake presets for each profile.
- [~] Validate release-profile behavior across all platforms in CI.
  - Completed: local online-rawdog install/consume path.
  - Remaining: full matrix run on GitHub Actions after merge.

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
- [~] Evaluate staged rollout to conda-forge (manual PR assist + bot automation hooks).
  - Remaining: implement feedstock update automation and credentials governance.

## Documentation
- [x] Rewrite README release/build sections to reflect current project status and automation flow.
- [x] Add `RELEASE-NOTES.md` with agent instructions and section format.
- [x] Add `docs/release-process.md` for maintainers.

## Tracking
- [~] Keep this TODO synchronized as tasks complete; each completed task must have a corresponding LOG entry.
