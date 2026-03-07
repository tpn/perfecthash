# Packaging TODO

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Completed

## Planning and Baseline
- [x] Create packaging ledgers:
  - `agents/PACKAGING-NOTES.md`
  - `agents/PACKAGING-TODO.md`
  - `agents/PACKAGING-LOG.md`
- [x] Audit current repo packaging/release state (`conda/recipe`, release workflows, python packaging baseline).
- [x] Document conda-forge bootstrap + automation constraints and recommended architecture.

## Decisions Required
- [x] Choose default package behavior for `conda install -c conda-forge perfecthash`:
  - Selected: depend on `perfecthash-full`.
- [x] Confirm package naming for profile outputs:
  - `perfecthash-online-rawdog-jit`
  - `perfecthash-online-rawdog-and-llvm-jit`
  - `perfecthash-online-llvm-jit`
  - `perfecthash-full`
- [x] Decide platform rollout sequence:
  - Selected now: Phase 1 Linux.
  - Planned follow-on: macOS, then Windows.

## Conda-forge Bootstrap (One-Time Human Setup)
- [ ] Prepare staged-recipes submission for initial package creation.
- [ ] Get feedstock created and maintainer access confirmed.
- [ ] Enable feedstock bot automerge:
  - `bot.automerge: true` in `conda-forge.yml`.
  - run conda-forge-admin command to enable bot automerge.
- [ ] Settle feedstock branch protection/automerge policy for version-only bot PRs.

## Feedstock Recipe Architecture
- [x] Convert recipe to multi-output profile packages.
- [x] Implement per-output build scripts (`PERFECTHASH_BUILD_PROFILE=<profile>`).
- [~] Add output-level tests for headers, CMake config, and consumer smoke compile.
  - Completed: header and CMake config file checks.
  - Remaining: downstream consumer compile smoke test command.
- [x] Add LLVM-specific dependencies only to LLVM outputs.
- [x] Make profile packages mutually exclusive if file overlaps cause co-install conflicts.
  - Completed: added cross-variant `run_constrained` exclusions for the profile outputs.
- [x] Ensure package version and embedded/CMake-reported version stay aligned during conda builds.
  - Completed: recipe now exports `PERFECTHASH_VERSION_OVERRIDE={{ version }}` and local/org validation for `0.71.2` reports `PerfectHash version: 0.71.2`.
- [x] Add meta-package output `perfecthash` that depends on chosen default profile output.

## CI and Release Integration
- [ ] Keep local `conda/recipe` in sync with feedstock recipe source-of-truth policy.
- [x] Expand local conda workflow to validate all profiles on pull requests.
  - Completed via Linux multi-output recipe build in one run.
- [x] Add release-time validation check that expected source tarball URL/checksum pattern is bot-friendly.
  - Completed: tag-triggered conda workflow now resolves GitHub tarball URL + SHA256 and builds from archive source.
- [ ] Document expected end-to-end release path:
  - tag push -> bot PR -> feedstock CI -> automerge -> conda-forge availability.

## Optional Upstream-Driven Feedstock PR Automation
- [ ] Decide if we need a direct updater workflow from this repo to feedstock.
- [ ] If yes, provision minimal-scope GitHub credential for feedstock PR creation.
- [ ] Implement updater script/workflow and conflict-safe behavior.
- [ ] Keep this path disabled by default if bot-managed flow proves sufficient.

## Pip/PyPI Follow-On
- [ ] Normalize Python package metadata:
  - license consistency,
  - versioning strategy tied to release tags,
  - long description/readme sanity.
- [ ] Configure PyPI Trusted Publishing via GitHub OIDC.
- [ ] Add publish workflow gated on release tags.
- [ ] Decide wheel strategy for native runtime/profile distribution:
  - Python-only first, native wheels second; or
  - native wheels from day one via `cibuildwheel`.
- [ ] Align pip package naming/extras with conda profile model.

## Documentation and Operations
- [x] Add `docs/packaging.md` with:
  - conda-forge architecture,
  - bootstrap steps,
  - recurring release SLO/expectations,
  - failure playbook.
- [x] Update `docs/release-process.md` with packaging automation status and trigger mapping.
- [ ] Keep `agents/PACKAGING-LOG.md` updated for each completed TODO item.

## Tracking
- [~] Keep this TODO synchronized with NOTES/LOG as implementation progresses.
