# Release Guide

This directory contains helper scripts that build, test, install, and package
PerfectHash for release. It is aimed at contributors doing a release for the
first time.

## Pre-flight Checklist

- Sync your branch and ensure submodules are present:
  `git submodule update --init --recursive`
- Decide the version number and update `CMakeLists.txt` if needed.
- Ensure the working tree is clean (or at least no uncommitted release changes).
- Confirm you can build and run tests locally (see OS sections below).

## What the Scripts Produce

Each script creates an artifact in:
`out/release/<version>/<platform>/dist/`

Examples:
- `perfecthash-0.63.0-linux-x86_64.tar.gz`
- `perfecthash-0.63.0-macos-arm64.tar.gz`
- `perfecthash-0.63.0-windows-x86_64.zip`

Each package includes the installed binaries and headers plus README/LICENSE.

## Version Bumping (Automation)

Use the helper script to bump `CMakeLists.txt` and optionally create a tag:

```
ci/bump-version.sh --version 0.63.1 --commit --tag
```

This creates a commit and a `v0.63.1` tag. Push the commit and tag to trigger
the release workflow.
On Windows, run this from Git Bash or WSL.

## Linux Release

Recommended environment (x86_64 example):

```
mamba env create -f conda/environments/dev-linux_os-linux_arch-x86_64_py-313_cuda-none_compiler-llvm.yaml
mamba activate dev-linux_os-linux_arch-x86_64_py-313_cuda-none_compiler-llvm
```

Run the release script:

```
RELEASE_VERSION=0.63.0 ci/release-linux.sh
```

Useful knobs:

- `--skip-tests` if tests cannot run (try to fix instead).
- `--clean` to delete the release output directory.
- `PERFECTHASH_ENABLE_NATIVE_ARCH=OFF` for portable binaries.
- `USE_CUDA=ON` and `CMAKE_CUDA_ARCHITECTURES=89` for CUDA builds.

## macOS Release

Recommended environment (arm64 example):

```
mamba env create -f conda/environments/dev-macos_os-macos_arch-arm64_py-313_cuda-none_compiler-llvm.yaml
mamba activate dev-macos_os-macos_arch-arm64_py-313_cuda-none_compiler-llvm
```

Run the release script for the native arch:

```
RELEASE_VERSION=0.63.0 ci/release-macos.sh
```

To build x86_64 on Apple Silicon, pass `--arch x86_64`:

```
RELEASE_VERSION=0.63.0 ci/release-macos.sh --arch x86_64
```

## Windows Release

Open a Developer PowerShell (VS 2022) and ensure Ninja/CMake are available.
If needed, use `scripts/install-deps/windows.ps1` to bootstrap.

```
pwsh -File ci/release-windows.ps1 -ReleaseVersion 0.63.0
```

Optional flags:

- `-SkipTests` to skip CTest.
- `-Clean` to remove the default release output directory.
- `-Generator "Ninja Multi-Config"` to force the generator.

## Troubleshooting

- Tests require `python` and `cargo`; the conda envs include these.
- If CTest fails due to missing keys, confirm `keys/HologramWorld-31016.keys`
  exists in the repo.
- Use `--dry-run` (Linux/macOS) or `-DryRun` (Windows) to preview commands.

## GitHub Actions Release Workflow

Tag pushes (e.g., `v0.63.0`) trigger the release workflow, which runs the
platform scripts, uploads artifacts, and publishes a GitHub release. If you
use the `ci/bump-version.sh` script with `--tag`, pushing the tag is enough to
kick off the automation.

For a dry run without publishing, use the workflow dispatch and supply a
version; it will build and upload artifacts only.

## Release Upload (Manual Fallback)

If you need to upload manually:

- Sanity-check the artifacts by extracting and running `PerfectHashCreateExe`.
- Create a GitHub release and upload the artifacts from `out/release/.../dist`.
