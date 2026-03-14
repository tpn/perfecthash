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
- Upload to `anaconda.org` via `ANACONDA_API_TOKEN` remains optional and is not
  part of conda-forge publishing.
- CI upload defaults to `--user perfecthash --label main --skip-existing`.
- Override upload user by setting GitHub Actions variable `ANACONDA_UPLOAD_USER`.

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
