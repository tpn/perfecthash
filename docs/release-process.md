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
- `online-rawdog-jit`
  - Slim online/rawdog package for x64/arm64 use cases.
- `online-rawdog-and-llvm-jit`
  - Slim profile plus LLVM JIT support.
- `online-llvm-jit`
  - LLVM-only online JIT profile with RawDog generation disabled.

Artifacts include profile in names (for example,
`perfecthash-0.63.0-online-rawdog-jit-linux-x86_64.tar.gz`).

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

1. Linux-first multi-output recipe is maintained under `conda/recipe/` and
   validated in CI.
2. Recipe outputs:
   - `perfecthash` (meta package, depends on `perfecthash-full`)
   - `perfecthash-full`
   - `perfecthash-online-rawdog-jit`
   - `perfecthash-online-rawdog-and-llvm-jit`
   - `perfecthash-online-llvm-jit`
3. Tag pushes trigger `.github/workflows/conda-package.yml`, which builds the
   Linux outputs and uploads artifacts.
   - Tag builds resolve GitHub release tarball URL + SHA256 and build from that
     source archive for feedstock-style validation.
4. Optional publish to a maintainer Anaconda channel remains available via
   `ANACONDA_API_TOKEN` (not conda-forge publication).
5. Conda-forge publication still requires one-time staged-recipes/feedstock
   bootstrap and feedstock bot governance; recurring publication is feedstock
   CI/bot-driven.

See `docs/packaging.md` for the operational checklist and maintainer actions.

## Operational Notes

- Keep release scripts dry-run-friendly for local validation.
- Prefer workflow dispatch for pre-release profile smoke tests.
- Keep `agents/RELEASE-ENGINEERING-TODO.md` and
  `agents/RELEASE-ENGINEERING-LOG.md` in sync with progress.
