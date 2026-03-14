# PerfectHash Packaging

## Current Scope
- Linux-first packaging implementation.
- Multi-output conda recipe under `conda/recipe/`.
- `perfecthash` defaults to full package behavior via dependency on
  `perfecthash-full` and `perfecthash-python`.
- Root Python packaging lives at repo root (`pyproject.toml` + `python_src/`).

## Conda Package Outputs
- `perfecthash` (meta package, depends on `perfecthash-full` and `perfecthash-python`)
- `perfecthash-full`
- `perfecthash-python`
- `perfecthash-online-rawdog-jit`
- `perfecthash-online-rawdog-and-llvm-jit`
- `perfecthash-online-llvm-jit`

All profile outputs are currently Linux-only.

## Local Automation in This Repository
- `ci/cut-release.sh --version X.Y.Z --push` creates and pushes release tags.
- `.github/workflows/conda-package.yml` now runs on:
  - tag push (`v*`)
  - workflow dispatch
  - packaging-related pull requests
- The conda workflow builds Linux multi-output packages and uploads artifacts.
- On tag builds, the workflow resolves GitHub release source tarball URL +
  SHA256 and builds from that archive (feedstock-style source validation).
- The Python conda output installs the root Python package via `pip` without
  bundling native artifacts, and depends on the matching `perfecthash-full`
  native package.
- Packaging-related PR validation now tracks root Python packaging files
  (`pyproject.toml`, `python_src/**`, `hatch_build.py`) in addition to native
  recipe inputs.

## GitHub Release Python Assets
- `release.yml` builds bundled standalone Python wheels for the `full` profile
  on each release platform.
- Linux x86_64 also emits a source distribution.
- Bundled wheels stage native assets under `perfecthash/_native/`.
- Wheel smoke tests currently verify:
  - `ph --version`
  - `ph create --dry-run`
  - `build_table()` from a clean virtual environment

Note:
- PyPI/TestPyPI publication is now wired via GitHub OIDC Trusted Publishing,
  not long-lived API tokens.
- The published Python distribution name is `tpn-perfecthash`; the import
  package remains `perfecthash`.
- Only bundled wheels are uploaded to PyPI/TestPyPI for now.
- The current sdist is still attached to GitHub releases, but it is not
  published to PyPI because it does not yet build or bundle the native runtime
  by itself.
- Upload to `anaconda.org` via `ANACONDA_API_TOKEN` remains optional and is not
  part of conda-forge publishing.
- CI upload defaults to `--user perfecthash --label main --skip-existing`.
- Override upload user by setting GitHub Actions variable `ANACONDA_UPLOAD_USER`.

## PyPI / TestPyPI Trusted Publishing

The release workflow now has two Python publication jobs:

1. `publish-testpypi`
2. `publish-pypi`

Behavior:

- `publish-testpypi` runs for release tags and for manual `workflow_dispatch`
  runs of the `full` profile.
- Manual `workflow_dispatch` TestPyPI runs should provide an explicit unique
  release version input.
- `publish-pypi` runs only for release tags, and only after `publish-testpypi`
  succeeds.
- Both jobs publish bundled wheels only.
- `publish-pypi` uses the GitHub `pypi` environment; require manual approval on
  that environment.
- `publish-testpypi` uses the GitHub `testpypi` environment; manual approval is
  usually unnecessary there.

One-time maintainer setup:

1. Create a TestPyPI account if you do not already have one.
2. In GitHub repository settings, create environments named `testpypi` and
   `pypi`.
3. Add protection rules to `pypi` that require manual approval by trusted
   maintainers.
4. On PyPI, configure the trusted publisher for:
   - owner: `tpn`
   - repository: `perfecthash`
   - workflow file: `.github/workflows/release.yml`
   - environment: `pypi`
5. On TestPyPI, configure the trusted publisher for:
   - owner: `tpn`
   - repository: `perfecthash`
   - workflow file: `.github/workflows/release.yml`
   - environment: `testpypi`

Notes:

- PyPI and TestPyPI are separate services; a PyPI account/project does not
  automatically exist on TestPyPI.
- Use `tpn-perfecthash` on both services for the published project name.
- If you want TestPyPI to create the project on first publish, use a pending
  publisher there.
- No `PYPI_TOKEN` or `TEST_PYPI_TOKEN` secret is required with this setup.

## Conda-forge Automation Model
Target recurring flow:
1. Tag release in this repo.
2. Conda-forge bot opens feedstock version update PR.
3. Feedstock CI builds.
4. Bot automerges when checks are green.
5. Packages become available via `conda install -c conda-forge ...`.

## When Maintainer Action Is Required
One-time conda-forge bootstrap (human-required):
1. Submit initial recipe to `conda-forge/staged-recipes`.
2. Merge and wait for feedstock creation.
3. Confirm maintainer access to the new feedstock repo.
4. Enable feedstock automerge:
   - set `bot.automerge: true` in feedstock `conda-forge.yml`
   - run the `@conda-forge-admin, please update automerge` command on a feedstock PR

After bootstrap:
- Routine releases should not need manual conda-forge steps unless bot/update
  PRs fail or require recipe fixes.

## Next Implementation Phases
1. Finalize staged-recipes-ready metadata (source URL + checksums from release tarballs).
2. Submit staged-recipes PR.
3. Mirror recipe into feedstock and enable automerge policy.
4. Expand platform matrix beyond Linux (macOS first, Windows last).
