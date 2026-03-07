# PerfectHash Packaging

## Current Scope
- Linux-first packaging implementation.
- Multi-output conda recipe under `conda/recipe/`.
- `perfecthash` defaults to full package behavior via dependency on `perfecthash-full`.

## Conda Package Outputs
- `perfecthash` (meta package, depends on `perfecthash-full`)
- `perfecthash-full`
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
