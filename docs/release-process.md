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
3. GitHub `release.yml` builds Linux/macOS/Windows native artifacts.
4. For the `full` profile, the workflow also builds bundled standalone Python
   wheels per platform plus a source distribution.
5. Workflow publishes bundled Python wheels to TestPyPI.
6. Workflow publishes bundled Python wheels to PyPI after `pypi` environment
   approval.
7. Workflow publishes GitHub release assets and generated release notes.

## Versioning Model

- Source of truth: git tags matching `v*`.
- Reserve patch (`X.Y.Z`) bumps for release-oriented non-functional fixes only.
- Use a minor or major bump for shipped functional changes, new features, or
  other user-visible behavior changes.
- CMake resolves version automatically from:
  1. `PERFECTHASH_VERSION_OVERRIDE` (when set)
  2. exact tag on `HEAD`
  3. latest reachable `v*` tag
  4. fallback value in CMake
- Release scripts pass `PERFECTHASH_VERSION_OVERRIDE` to guarantee tagged builds
  embed the intended version.
- The root Python package resolves its build version from:
  1. `PERFECTHASH_PYTHON_VERSION`
  2. `PERFECTHASH_VERSION_OVERRIDE`
  3. `PERFECTHASH_CONDA_VERSION`
  4. exact git tag on `HEAD`
  5. latest reachable `v*` tag
  6. fallback value in `python/perfecthash/_version.py`
- Tagged release wheels and sdists must report the same release version as the
  native artifacts.

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

Python packaging currently targets the `full` profile only:

- published distribution name: `tpn-perfecthash`
- import package / CLI module name: `perfecthash`
- standalone release wheels bundle native assets under `perfecthash/_native/`
- slim native profiles do not currently produce Python wheels
- PyPI/TestPyPI publication is wheel-only for now; the sdist remains a GitHub
  release asset until source builds can produce a working native runtime

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
   - `perfecthash-python`
   - `perfecthash-online-rawdog-jit`
   - `perfecthash-online-rawdog-and-llvm-jit`
   - `perfecthash-online-llvm-jit`
3. Tag pushes trigger `.github/workflows/conda-package.yml`, which builds the
   Linux outputs and uploads artifacts.
   - Tag builds resolve GitHub release tarball URL + SHA256 and build from that
     source archive for feedstock-style validation.
   - The Python conda output installs the root Python package unbundled and
     depends on the matching native `perfecthash-full` output.
4. Optional publish to a maintainer Anaconda channel remains available via
   `ANACONDA_API_TOKEN` (not conda-forge publication).
5. Conda-forge publication still requires one-time staged-recipes/feedstock
   bootstrap and feedstock bot governance; recurring publication is feedstock
   CI/bot-driven.

See `docs/packaging.md` for the operational checklist and maintainer actions.

## PyPI / TestPyPI Strategy

1. Use Trusted Publishing via GitHub OIDC; do not add long-lived API tokens.
2. Publish to TestPyPI first from the same `release.yml` workflow.
3. Gate real PyPI publication behind the GitHub `pypi` environment.
4. Publish only bundled wheels until the sdist can build/package native
   artifacts on its own.

## Operational Notes

- Keep release scripts dry-run-friendly for local validation.
- Prefer workflow dispatch for pre-release profile smoke tests.
- Treat the Python wheel/sdist path as part of release validation, not as a
  post-release follow-up.
- Treat TestPyPI as the pre-PyPI release gate for Python artifacts.
- Keep `agents/RELEASE-ENGINEERING-TODO.md` and
  `agents/RELEASE-ENGINEERING-LOG.md` in sync with progress.
