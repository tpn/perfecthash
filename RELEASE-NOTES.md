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

### Docs
- Release engineering ledgers added under `agents/RELEASE-ENGINEERING-*`.
- Removed references to the legacy manual version-bump flow.
