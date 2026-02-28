# Release Engineering Notes

## Mission
Establish a robust, low-friction, highly automated release process for PerfectHash using existing GitHub CI as the backbone.

## Initial State Snapshot (2026-02-28, pre-implementation)
- CI workflows already exist for Linux/macOS/Windows.
- A release workflow already exists (`.github/workflows/release.yml`) and publishes artifacts on tag pushes.
- Release scripts exist under `ci/` for Linux, macOS, Windows.
- Versioning was coupled to a hardcoded `project(VERSION 0.63.0)` and fallback parsing of `CMakeLists.txt`.
- Manual version bump flow existed (`ci/bump-version.sh`) and docs referenced it.
- CMake package export/config scaffolding was missing; downstream `find_package(PerfectHash)` consumption was not wired.
- No consumer example existed in `examples/` for FetchContent/CPM usage.
- README reflected historical context and needed release-process refresh.

## Release Engineering Principles
- Single source of truth for release version: git tags (`vX.Y.Z`) on release builds.
- No mandatory post-tag version bump commits.
- Reproducible release artifacts with consistent naming and checksums.
- CMake-first downstream consumption (`FetchContent`/CPM + `find_package`).
- Release notes generated automatically from commits since prior tag, with optional curated overlay.
- Keep workflow friction minimal: tag + push should be enough for a standard release.

## Desired Build Offerings
- `online-rawdog`: slim online/JIT-focused build (x64/arm64).
- `online-rawdog-llvm`: slim build plus LLVM JIT support.
- `full`: full build including executables and all enabled JIT paths.

## Open Questions / Follow-Up
- Conda-forge publish path will likely require a separate feedstock PR workflow and tokens; stage as blueprint + docs first.
- Windows PGO strategy should be codified in CI (matrix/preset/script) while preserving current non-PGO baseline release artifacts.
- Verify whether package consumers should default to shared or static online targets.

## Implemented Decisions (2026-02-28)
- Version source of truth now prioritizes git tags and supports explicit release override.
- Build profiles are first-class CMake options and are propagated through release scripts/workflows.
- CMake package export/config is now available and validated for `online-rawdog` installed consumption.
- Release notes now blend curated `RELEASE-NOTES.md` content with commit history since prior tag.
- Release artifact names include build profile for clearer distribution semantics.
- Conda packaging is staged with local recipe + CI build scaffold; conda-forge publication remains a follow-on.
- Legacy manual version-bump helper has been removed; release tag cutting is now handled by `ci/cut-release.sh`.

## Verification Updates (2026-02-28)
- Tag `v0.70.5` release workflow (`22515338522`) completed successfully across Linux/macOS/Windows and published assets.
- Conda packaging workflow (`22515274877`) now passes after recipe conflict and toolchain dependency fixes.
- Windows packaging path still needs one follow-on fix: `v0.70.5` published only `*.zip.sha256` for Windows, indicating a packaging script mismatch that must be corrected before the next tag.
