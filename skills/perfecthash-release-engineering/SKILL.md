---
name: perfecthash-release-engineering
description: Cut, validate, publish, or document PerfectHash releases. Use when working on `ci/cut-release.sh`, `.github/workflows/release.yml`, `docs/release-process.md`, `ci/README.md`, `RELEASE-NOTES.md`, or `agents/RELEASE-ENGINEERING-*`, or when the user asks to prepare a release tag, dry-run a release, audit release automation, or update release-process documentation.
---

# PerfectHash Release Engineering

## Overview

Use this skill for tag-driven PerfectHash releases and the repo files that define them. The release source of truth is the git tag (`vX.Y.Z`), and the preferred maintainer entry point is `ci/cut-release.sh`.

Primary repo files:
- `ci/cut-release.sh`
- `ci/release-linux.sh`
- `ci/release-macos.sh`
- `ci/release-windows.ps1`
- `.github/workflows/release.yml`
- `docs/release-process.md`
- `ci/README.md`
- `RELEASE-NOTES.md`
- `agents/RELEASE-ENGINEERING-NOTES.md`
- `agents/RELEASE-ENGINEERING-TODO.md`
- `agents/RELEASE-ENGINEERING-LOG.md`

## Release Workflow

1. Check repository state first.
   - Prefer a clean working tree before cutting a release.
   - If the tree is dirty, do not cut a real tag unless the user explicitly wants a dirty-tree release.
2. Check `RELEASE-NOTES.md`.
   - Ensure `Unreleased` reflects what is actually being shipped.
3. Dry-run the cut helper when validating mechanics.

```bash
ci/cut-release.sh --version X.Y.Z --dry-run
```

4. Cut the tag for a real release.

```bash
ci/cut-release.sh --version X.Y.Z --push
```

5. After tag push, rely on `.github/workflows/release.yml` for the cross-platform build and GitHub release publication.
6. Record significant release-engineering changes in `agents/RELEASE-ENGINEERING-LOG.md` and keep `agents/RELEASE-ENGINEERING-TODO.md` synchronized.

## Guardrails

- Do not invent release versions. Base them on the existing tag series and explain the chosen increment.
- Reserve patch (`X.Y.Z`) bumps for release-oriented non-functional fixes only. Use a minor or major bump when the shipped change is functional, feature-bearing, or otherwise user-visible.
- Do not push a tag that already exists locally or remotely.
- Do not bypass the clean-tree guard with `--allow-dirty` unless the user explicitly approves that release policy.
- Keep version references tag-centric; avoid reintroducing manual version-bump workflows.
- Treat GitHub release publication and Anaconda/conda publication as related but separate flows.

## Validation

Use these checks after release-related edits:

```bash
bash -n ci/cut-release.sh
ruby -ryaml -e 'YAML.load_file(".github/workflows/release.yml") && puts("ok")'
```

If the task is release planning rather than an actual cut:

```bash
git describe --tags --abbrev=0 --match 'v[0-9]*'
ci/cut-release.sh --version X.Y.Z --dry-run
```

If release docs changed, re-read:
- `docs/release-process.md`
- `ci/README.md`
- `RELEASE-NOTES.md`

## Documentation Rules

- Update `RELEASE-NOTES.md` for any shipped release-behavior, CI/CD, packaging, or artifact-layout changes.
- Keep `agents/RELEASE-ENGINEERING-NOTES.md`, `agents/RELEASE-ENGINEERING-TODO.md`, and `agents/RELEASE-ENGINEERING-LOG.md` aligned with completed work.
- Keep examples in docs consistent with current profile names:
  - `full`
  - `online-rawdog-jit`
  - `online-rawdog-and-llvm-jit`
  - `online-llvm-jit`

## Conda Interaction

The release flow now overlaps with conda packaging:
- tag pushes can trigger `.github/workflows/conda-package.yml`
- tag-triggered conda builds use release tarball URL + SHA256
- conda-forge publication still requires separate feedstock governance

For conda build/upload/bootstrap work, use `skills/perfecthash-conda-packaging/SKILL.md` instead of expanding this skill.
