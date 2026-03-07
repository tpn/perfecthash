# Release Notes

## Agent Instructions

- Update this file in every change set that affects release behavior, build output, packaging, or CI/CD.
- Add new bullets under `## Unreleased` in the most relevant section.
- Keep bullets concise, user-facing, and outcome-focused.
- Do not rewrite or delete historical release sections.
- When cutting a release tag `vX.Y.Z`, copy `Unreleased` entries into a new `## vX.Y.Z - YYYY-MM-DD` section, then reset `Unreleased` to empty templates.
- If a change is not shipped, remove it from `Unreleased` before release.

## Unreleased

### Added
- Repo-local release-engineering and conda-packaging skills now capture the maintainer workflows used to cut releases, validate conda packages, and publish org-channel artifacts.

### Changed
- Installed public headers now live under `include/PerfectHash/`, and public consumers use `<PerfectHash/...>` include paths instead of polluting the flat include directory.
- Conda packaging now propagates the package version into CMake/binaries via `PERFECTHASH_VERSION_OVERRIDE`, so org-channel builds report the shipped version instead of the latest git tag.

### Fixed
- Windows RawDog MASM header generation now resolves the new public include root correctly after the header-layout refactor.
- Conda profile variants now declare solver-level mutual exclusion, preventing co-install of conflicting profile packages.

### Docs
- Added `docs/packaging.md` and refreshed release/CI notes to reflect the current conda workflow, org-channel publishing, and release-versioning policy.

## v0.70.7 - 2026-02-28

### Added
- No new additions.

### Changed
- Dependency-mode defaults now disable `PERFECTHASH_BUILD_EXES` and `PERFECTHASH_ENABLE_TESTS` when PerfectHash is consumed as a subproject.

### Fixed
- FetchContent consumer `ALL` builds now avoid multi-target RawDog header generation races for `online-rawdog-llvm` and `full` profiles.

### Docs
- Release engineering ledgers now include the full FetchContent profile-matrix validation and `v0.70.7` release verification trail.

## v0.70.6 - 2026-02-28

### Added
- No new additions.

### Changed
- Conda recipe now relies on canonical `build.sh` behavior without duplicate `build.script` declaration.
- Conda recipe build requirements now include `nasm` for Linux/macOS assembly builds.

### Fixed
- Windows release packaging now uses strict error handling and consistent stage/package naming so `.zip` artifacts are generated reliably.

### Docs
- Updated release engineering ledgers with post-release validation outcomes and run references.

## v0.70.5 - 2026-02-28

### Added
- Tag-derived CMake version resolution with optional explicit override for release automation.
- CMake package export/config support for downstream `find_package(PerfectHash CONFIG REQUIRED)` usage.
- Build profiles: `full`, `online-rawdog`, and `online-rawdog-llvm`.
- New consumer example at `examples/cmake-fetchcontent-consumer` covering FetchContent and installed-package flows.
- Auto-generated release note script (`ci/generate-release-notes.sh`) for commit summaries since prior tag.
- New tag-cut helper `ci/cut-release.sh` with preflight checks and optional tag push.

### Changed
- Release scripts now resolve version from git tags before file-based fallbacks.
- Release scripts and GitHub release workflow now support profile-aware builds and profile-tagged artifact names.
- Release workflow now generates release body content from repository history plus curated notes.
- Release docs now direct maintainers to use `ci/cut-release.sh` for cutting tags.

### Fixed
- Installed CMake package now exports only profile-relevant targets, enabling slim-profile install + consume flows.
- Slim-profile FetchContent consumers no longer pull unnecessary ALL-target build edges that caused RawDog object race failures.

### Docs
- Release engineering ledgers added under `agents/RELEASE-ENGINEERING-*`.
- Removed references to the legacy manual version-bump flow.
