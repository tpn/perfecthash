# Packaging Notes

## Mission
Deliver fully automated package distribution from release tags, with:
- Conda/mamba installs from conda-forge.
- Explicit package permutations for PerfectHash build profiles.
- A follow-on path to automated PyPI publishing.

## Initial State Snapshot (2026-03-01, pre-implementation)
- Release automation exists and is tag-driven via `ci/cut-release.sh` plus `.github/workflows/release.yml`.
- Local conda recipe scaffolding exists under `conda/recipe/` and is CI-validated in `.github/workflows/conda-package.yml`.
- Current recipe is Linux-only (`skip: true  # [win]`) and effectively single-profile per run (`PERFECTHASH_BUILD_PROFILE` env).
- There is no conda-forge staged-recipes submission or feedstock yet.
- Python package scaffolding exists under `python/` with `pyproject.toml`, but automated PyPI publishing is not wired.

## Conda-forge Reality and Constraints
- New conda-forge packages start in `staged-recipes`; once merged, conda-forge creates a dedicated feedstock repository.
- Feedstock CI and upload pipelines are controlled by conda-forge infrastructure, not by this repo directly.
- Ongoing updates are PR-driven in the feedstock; version update PRs are usually opened by `regro-cf-autotick-bot`.
- Feedstock-side bot automerge can be enabled (`bot.automerge: true`) with conda-forge-admin setup.
- Result: truly no-human recurring releases are feasible only after one-time bootstrap tasks are done.

## Recommended Automation Model
### Primary (recommended): Conda-forge bot-managed updates
1. Cut/push tag via `ci/cut-release.sh --version X.Y.Z --push`.
2. Existing release workflow publishes GitHub release/tag.
3. `regro-cf-autotick-bot` opens feedstock PR with new version/checksum.
4. Feedstock CI builds packages for configured platforms.
5. Bot automerges version PR when checks are green.
6. Conda-forge publishes artifacts; users can install from `-c conda-forge`.

Properties:
- No conda-forge auth secret is needed in this upstream repository for steady-state releases.
- End-to-end remains asynchronous (bot scheduling + feedstock CI latency).

### Optional accelerator: Upstream-driven feedstock PR
- Add a release-triggered GitHub Actions job in this repo to open/update a feedstock PR directly.
- Requires a GitHub App or PAT with permissions to create branches/PRs against the feedstock.
- Keep as fallback or latency reducer, not first-line architecture.

## Package Permutations Strategy
Use one feedstock with multi-output recipe, building each profile as a distinct package name.

Proposed outputs:
- `perfecthash-online-rawdog-jit`
- `perfecthash-online-rawdog-and-llvm-jit`
- `perfecthash-online-llvm-jit`
- `perfecthash-full`
- `perfecthash` (meta package depending on the default profile package)

Default recommendation:
- `perfecthash` -> depends on `perfecthash-full` (selected).
- profile-specific package names remain available for slim/runtime-targeted installs.

Why multi-output over separate feedstocks:
- Single source of truth for versioning and recipe logic.
- One bot PR per release instead of four.
- Easier synchronization of dependency and build-system changes.

## Build/Dependency Notes for Outputs
- All outputs compile from source with `-DPERFECTHASH_BUILD_PROFILE=<profile>`.
- LLVM profiles require LLVM toolchain/runtime alignment (`llvmdev >= 15` and any platform-specific runtime libs).
- Keep architecture matrix aligned with project support (`linux-64`, `linux-aarch64`, `osx-64`, `osx-arm64`, later `win-64` once stable in conda-forge recipe).
- Use output-level tests that validate:
  - installed headers,
  - installed `PerfectHashConfig.cmake`,
  - simple downstream `find_package(PerfectHash)` compile smoke test.

## Human Steps: One-Time vs Recurring
One-time bootstrap (human):
- Submit package to `conda-forge/staged-recipes`.
- Become/confirm feedstock maintainer.
- Enable bot automerge in feedstock (`conda-forge.yml` + admin command).
- Configure branch protections/automerge policy to permit zero-touch version PR merges.

Recurring release flow (target):
- Human runs `ci/cut-release.sh --version ... --push`.
- No additional human steps required unless feedstock CI fails or bot PR needs manual conflict resolution.

## Implemented Decisions (2026-03-01)
- Default package behavior set as requested:
  - `conda install -c conda-forge perfecthash` should resolve to `full` via
    `perfecthash` meta package -> `perfecthash-full`.
- Linux-first rollout selected:
  - recipe outputs are currently Linux-only.
  - CI packaging workflow currently builds Linux outputs only.
- Local recipe converted to multi-output profile packages with per-output
  profile-driven build scripts and install-surface tests.
- Local conda workflow now triggers on tag pushes, so release tags
  automatically build Linux conda artifacts in CI.
- Tag-triggered conda builds now resolve GitHub release tarball URL + SHA256
  and build from archive source for feedstock-style source validation.
- Real org-channel publication is now validated for Linux via `anaconda.org/perfecthash`.
- Public installed headers now use `include/PerfectHash/` as the namespace root
  instead of polluting the flat include directory.

## Authentication and Secrets
- For recommended bot-managed path:
  - No extra conda-forge auth setup is needed in this upstream repo.
  - Do not use `ANACONDA_API_TOKEN` for conda-forge publishing; conda-forge handles publication from feedstock CI.
- For optional upstream-driven PR automation:
  - Add GitHub credential scoped to the feedstock repo only.
  - Store in this repo's GitHub Actions secrets.

## Pip Roadmap (Follow-on)
### Phase A: Clean metadata and release discipline
- Align package metadata with repo truth (license consistency, version source tied to tags).
- Reserve and validate PyPI project naming (`perfecthash` currently appears available; re-check before publish).

### Phase B: Trusted publishing
- Use PyPI Trusted Publishing (GitHub OIDC), not long-lived API tokens.
- Trigger publish from signed/tagged release workflow only.

### Phase C: Wheel strategy
- Start with Python package publish from `python/` (pure/Python+existing Cython surface).
- Then decide whether native runtime profiles are represented as:
  - separate pip packages, or
  - optional extras that pull profile-specific runtime dependencies.
- If shipping native wheels, use `cibuildwheel` and align profile naming with conda outputs.

## Risks and Open Decisions
- Revisit default package mapping only if `full` footprint causes solver/adoption friction.
- Determine whether Windows conda-forge builds are phase 1 or phase 2.
- Decide whether to keep the current overlapping on-disk file layout for profile variants (with solver-level mutual exclusion) or redesign package contents further to eliminate conda-build overlap warnings entirely.
- Resolve Python metadata/license/version normalization before PyPI rollout.
- Define policy for bot PR failures (manual fallback SLA, release blocking vs non-blocking).

## Primary References
- https://conda-forge.org/docs/maintainer/adding_pkgs/
- https://conda-forge.org/docs/maintainer/understanding_conda_forge/feedstocks/
- https://conda-forge.org/docs/maintainer/infrastructure/
- https://conda-forge.org/docs/maintainer/updating_pkgs/
- https://conda-forge.org/docs/maintainer/conda_forge_yml/
- https://docs.conda.io/projects/conda-build/en/latest/resources/define-metadata.html
- https://conda.github.io/conda-build/resources/variants.html
- https://docs.pypi.org/trusted-publishers/
