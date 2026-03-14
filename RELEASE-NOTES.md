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
- Release automation now builds bundled standalone Python wheels for the `full`
  profile and a matching source distribution on Linux x86_64.
- Conda packaging now includes a `perfecthash-python` output, and the top-level
  `perfecthash` package now installs both the native full profile and the
  Python package.
- Release automation now publishes bundled Python wheels to TestPyPI from the
  shared `release.yml` workflow before any real PyPI publish.

### Changed
- The supported Python package now resolves its version from the same tag/CI
  inputs as the native release path instead of using a hard-coded pre-release
  version string.
- Linux branch CI now validates the root Python package, bundled-wheel build,
  and runtime smoke path on the `linux-x86_64-py313` leg.
- The published Python distribution name is now `tpn-perfecthash`, while the
  import package and CLI entry point remain `perfecthash` and `ph`.

### Fixed
- Tag-triggered conda publication now expands the built package artifact list before invoking the Anaconda CLI, avoiding literal `**` glob failures in GitHub Actions.
- Standalone bundled Python wheels now report the installed package version
  correctly outside a git checkout by preferring wheel metadata at runtime.
- Python release publication now uses GitHub OIDC trusted publishing instead of
  long-lived index API tokens.

### Docs
- Release, packaging, CI, and Python docs now describe the root Python package,
  bundled-wheel release assets, and the legacy status of `python/`.
- Release docs now cover the GitHub environment approvals and TestPyPI/PyPI
  trusted publisher setup for Python releases.

## v0.72.2 - 2026-03-08

### Added
- No new additions.

### Changed
- No changes.

### Fixed
- Tag-triggered Linux conda packaging now installs and resolves NASM reliably on GitHub runners by provisioning `nasm` in the workflow environment and preferring PATH-based NASM discovery before BUILD_PREFIX fallback.

### Docs
- No documentation-only updates.

## v0.72.1 - 2026-03-08

### Added
- No new additions.

### Changed
- No changes.

### Fixed
- Improved tag-triggered conda packaging NASM discovery by allowing the recipe build to prefer a BUILD_PREFIX-provided assembler when available.

### Docs
- No documentation-only updates.

## v0.72.0 - 2026-03-07

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
