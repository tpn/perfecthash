# PerfectHash Release Process

## Goals

- Maximum release automation with minimal manual friction.
- Tag-driven versioning and publishing.
- Profile-aware artifacts for slim vs full distributions.
- Reproducible release notes from both curated and commit-derived inputs.

## Standard Release Flow

1. Ensure `RELEASE-NOTES.md` `Unreleased` section reflects shipped changes.
2. Create and push an annotated tag (`vX.Y.Z`), preferably via:
   - `ci/cut-release.sh --version X.Y.Z --push`
3. GitHub `release.yml` builds Linux/macOS/Windows artifacts.
4. Workflow publishes GitHub release assets and generated release notes.

## Versioning Model

- Source of truth: git tags matching `v*`.
- CMake resolves version automatically from:
  1. `PERFECTHASH_VERSION_OVERRIDE` (when set)
  2. exact tag on `HEAD`
  3. latest reachable `v*` tag
  4. fallback value in CMake
- Release scripts pass `PERFECTHASH_VERSION_OVERRIDE` to guarantee tagged builds
  embed the intended version.

## Build Profiles

- `full`
  - Includes CLI executables and full online/JIT set.
- `online-rawdog`
  - Slim online/rawdog package for x64/arm64 use cases.
- `online-rawdog-llvm`
  - Slim profile plus LLVM JIT support.

Artifacts include profile in names (for example,
`perfecthash-0.63.0-online-rawdog-linux-x86_64.tar.gz`).

## Release Notes Strategy

- Curated notes live in `RELEASE-NOTES.md`.
- `ci/generate-release-notes.sh` merges curated notes with commit history since
  the prior release tag.
- `release.yml` uses the generated markdown as the GitHub release body.

## Windows PGO Strategy (Planned)

- Keep baseline non-PGO assets for every release.
- Add a dedicated Windows PGO pipeline stage with:
  1. `PGInstrument` build
  2. deterministic training workload execution
  3. `PGOptimize` build and packaging
- Publish PGO assets alongside baseline assets with explicit suffixes
  (`-pgo-windows-x86_64.zip`).

## Conda / Conda-Forge Strategy (Staged)

1. Maintain a local recipe under `conda/recipe/` and validate it in CI.
2. Publish optional channel artifacts (if desired) via `ANACONDA_API_TOKEN`.
3. For conda-forge, automate feedstock PR updates from release tags.
   Note: conda-forge publication requires feedstock governance/bot integration
   and cannot be fully controlled from this repository alone.

## Operational Notes

- Keep release scripts dry-run-friendly for local validation.
- Prefer workflow dispatch for pre-release profile smoke tests.
- Keep `agents/RELEASE-ENGINEERING-TODO.md` and
  `agents/RELEASE-ENGINEERING-LOG.md` in sync with progress.
