---
name: perfecthash-conda-packaging
description: Build, smoke-test, upload, and debug PerfectHash conda packages and release automation. Use when working on `conda/recipe/**`, `.github/workflows/conda-package.yml`, Anaconda.org uploads for `anaconda.org/perfecthash`, or conda-forge bootstrap/feedstock planning.
---

# PerfectHash Conda Packaging

## Overview

Use this skill for the full PerfectHash conda lifecycle:
- local multi-output builds from `conda/recipe/`
- local `file://` smoke tests
- upload to `anaconda.org/perfecthash`
- remote smoke-test installs from the org channel
- conda-forge bootstrap and feedstock automation changes

Current repository assumptions:
- packaging is Linux-first
- `perfecthash` is a meta package that depends on `perfecthash-full` and `perfecthash-python`
- upload credentials come from `ANACONDA_API_TOKEN`
- default upload target comes from `ANACONDA_UPLOAD_USER` and currently defaults to `perfecthash`

Primary repo files:
- `conda/recipe/meta.yaml`
- `conda/recipe/build.sh`
- `.github/workflows/conda-package.yml`
- `docs/packaging.md`

## Preferred Workflow

1. Inspect the current recipe and workflow before editing.
2. Use the helper script for the normal local cycle:
   - `skills/perfecthash-conda-packaging/scripts/conda-cycle.sh`
3. If packaging files change, validate both:
   - `conda build conda/recipe --output`
   - workflow YAML parse
4. If upload is part of the task, verify the remote install path from `-c perfecthash -c conda-forge`.
5. If the task is conda-forge-related, keep upstream/local recipe logic aligned with `docs/packaging.md`.

## Local Cycle

Default local build plus local smoke test:

```bash
skills/perfecthash-conda-packaging/scripts/conda-cycle.sh --version X.Y.Z
```

Build, local smoke test, upload, and remote smoke test:

```bash
skills/perfecthash-conda-packaging/scripts/conda-cycle.sh \
  --version X.Y.Z \
  --upload \
  --remote-test
```

Preview commands without executing:

```bash
skills/perfecthash-conda-packaging/scripts/conda-cycle.sh \
  --version X.Y.Z \
  --upload \
  --remote-test \
  --dry-run
```

What the script does:
- resolves version from `--version`, `PERFECTHASH_CONDA_VERSION`, or latest `v*` tag
- runs `conda build ... --output`
- builds artifacts into `conda/out` by default
- runs `conda index` and installs `perfecthash` from the local `file://` channel
- uploads artifacts with `anaconda ... upload --user "$ANACONDA_UPLOAD_USER" --label main --skip-existing`
- optionally installs `perfecthash` back from the remote org channel

## Editing Rules

When changing `conda/recipe/meta.yaml`:
- keep the multi-output package names stable unless the user explicitly wants a package rename
- keep `perfecthash` as the meta package and `perfecthash-full` + `perfecthash-python` as the default dependency set unless the user explicitly changes policy
- keep LLVM requirements restricted to LLVM-backed outputs
- keep the Python output unbundled for conda and let it depend on the matching native package output
- remember that local tree builds may use `source.path`, while tag-triggered CI builds use release tarball URL + SHA256

When changing `.github/workflows/conda-package.yml`:
- keep tag-triggered builds aligned with release tags (`v*`)
- do not treat Anaconda.org upload as conda-forge publication
- keep upload target configurable via `ANACONDA_UPLOAD_USER`

When changing `.envrc` or auth flow:
- prefer `direnv` plus `gpg`-decrypted token files over hardcoded plaintext tokens
- avoid printing secrets in command output
- if a token is exposed in output, tell the user to rotate it

## Conda-forge Notes

Use `docs/packaging.md` as the source of truth for current bootstrap status.

Important constraints:
- conda-forge publication does not use this repo's `ANACONDA_API_TOKEN`
- staged-recipes bootstrap is a one-time maintainer step
- recurring updates happen in feedstock CI and usually via bot PRs
- new output package names added later may require extra conda-forge admin handling, so preserve package names unless there is a deliberate migration

## Validation

Minimum validation after packaging edits:

```bash
conda build conda/recipe --output
ruby -ryaml -e 'YAML.load_file(".github/workflows/conda-package.yml") && puts("ok")'
```

Preferred validation after meaningful recipe/workflow changes:

```bash
skills/perfecthash-conda-packaging/scripts/conda-cycle.sh --version X.Y.Z
```

If upload or channel behavior changed:

```bash
skills/perfecthash-conda-packaging/scripts/conda-cycle.sh \
  --version X.Y.Z \
  --upload \
  --remote-test
```
