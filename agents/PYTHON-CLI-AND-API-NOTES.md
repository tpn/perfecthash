# PYTHON-CLI-AND-API Notes

## Scope

- Project covers the legacy `python/perfecthash` package, whether it should ship as part of the supported Python package, how to separate analysis-only functionality, and what a modern Python CLI/API should look like for table creation and online use.

## Current Packaging State

- The Python package lives under [`python/`](./python) with [`python/pyproject.toml`](./python/pyproject.toml).
- Package metadata currently names the project `perfecthash` at version `0.1.0`.
- Runtime dependencies currently include `numpy`, `pandas`, `psutil`, `tqdm`, and `pywin32` on Windows.
- Optional `analysis` extras already exist in `pyproject.toml` and include `bokeh`, `joypy`, `matplotlib`, `pyarrow`, `scipy`, and `seaborn`.
- The conda recipe at [`conda/recipe/meta.yaml`](./conda/recipe/meta.yaml) does not build or ship the Python package. Current outputs are C/C++ library/executable packages only.
- There are no modern packaging entry points defined in `pyproject.toml` for a Python CLI.

## Current Python CLI State

- Legacy CLI framework lives in [`python/perfecthash/cli.py`](./python/perfecthash/cli.py) and [`python/perfecthash/commands.py`](./python/perfecthash/commands.py).
- The historical in-process alias appears to be `ph` via [`python/perfecthash/exe.py`](./python/perfecthash/exe.py), but it is not wired into packaging as an installed console script.
- Command inventory is mostly research/maintenance oriented: CFG extraction, CSV/parquet conversion, ETW/xperf processing, scaffolding helpers, and a few experiment-specific table creation wrappers.
- There is no general-purpose supported `create` or `bulk-create` Python command that maps cleanly to the C CLI surface.
- [`python/perfecthash/__init__.py`](./python/perfecthash/__init__.py) is currently empty, while `cli.py` references `perfecthash.__version__`, which suggests the legacy CLI is not in a polished/packaged state.

## Current Python API State

- There is a Python/Cython graph implementation in [`python/perfecthash/_graph.pyx`](./python/perfecthash/_graph.pyx) and [`python/perfecthash/graph.py`](./python/perfecthash/graph.py).
- There is also a Windows-specific ctypes wrapper around the DLL in [`python/perfecthash/dll/PerfectHash.py`](./python/perfecthash/dll/PerfectHash.py).
- The DLL wrapper relies on config-driven local DLL paths from [`conf/perfecthash.conf`](./conf/perfecthash.conf) and expects workstation-specific Windows paths/tooling.
- The existing Python surface does not provide a supported programmatic API for “build an online table from keys and use it in-process in Python”.

## Existing C Runtime Hooks Relevant To Python

- The core library already has an in-memory online creation API:
  - [`include/PerfectHash/PerfectHash.h`](./include/PerfectHash/PerfectHash.h) exposes `PerfectHashOnlineCreateTableFromSortedKeys()` and `PerfectHashOnlineCreateTableFromUnsortedKeys()`.
  - [`src/PerfectHash/PerfectHashOnlineCreate.c`](./src/PerfectHash/PerfectHashOnlineCreate.c) implements creation from an in-memory key array with `NoFileIo = TRUE`.
- `PerfectHashKeysLoadFromArray()` in [`src/PerfectHash/PerfectHashKeysLoad.c`](./src/PerfectHash/PerfectHashKeysLoad.c) already supports loading 32-bit or 64-bit keys from an in-memory buffer, with optional sorting/verification flags.
- The returned `PERFECT_HASH_TABLE` still exposes table methods such as `Index`, `Lookup`, `Insert`, and `Delete`; the newer minimal public online wrappers currently focus on creation + indexing.
- There are newer small public C APIs for online/JIT use:
  - [`include/PerfectHash/PerfectHashOnlineRawdog.h`](./include/PerfectHash/PerfectHashOnlineRawdog.h)
  - [`include/PerfectHash/PerfectHashOnlineJit.h`](./include/PerfectHash/PerfectHashOnlineJit.h)
- Current tests in [`tests/PerfectHashOnlineTests.cpp`](./tests/PerfectHashOnlineTests.cpp) exercise the in-memory online creation path across sorted/unsorted inputs, 32-bit/64-bit keys, seed/graph parameters, assigned16 behavior, and JIT variants.

## Existing C File-Generation Hooks Relevant To Python

- The generator already knows how to emit pure-Python artifacts for created tables:
  - [`src/PerfectHash/Chm01FileWorkPythonFile.c`](./src/PerfectHash/Chm01FileWorkPythonFile.c)
  - [`src/PerfectHash/Chm01FileWorkPythonTestFile.c`](./src/PerfectHash/Chm01FileWorkPythonTestFile.c)
- Generated Python output includes metadata, keys, table data, and an `index()` implementation for the curated `And`-mask hash functions.
- There are really two distinct Python stories already present in the codebase:
  - offline/exported Python modules for generated tables
  - in-process runtime creation via the online C API

## Configuration/Portability Findings

- [`python/perfecthash/config.py`](./python/perfecthash/config.py) still assumes:
  - Windows-specific default output directories.
  - Visual Studio `link.exe` and `dumpbin.exe` paths.
  - In-repo DLL/EXE paths such as `src/x64/Release/PerfectHashCreate.exe`.
- This makes the current Python package unsuitable as a cross-platform supported public package without cleanup.

## Directional Conclusions

- Shipping the current `python/perfecthash` tree as-is would expose a large amount of stale, Windows-centric, analysis/research code as the public package surface.
- The existing `analysis` extra is a useful starting point, but analysis code likely needs a clearer namespace split so that the default package can stay lean.
- A modern Python CLI should likely be a thin wrapper over supported library APIs and/or the existing C CLI primitives, not a direct continuation of the legacy command framework.
- A modern Python API for online tables will likely need a dedicated extension/binding layer over the C runtime instead of relying on the old ctypes/config-based DLL wrapper.
- The underlying C support for the Python API you described largely already exists; the missing work is packaging, binding design, and deciding what counts as supported public surface.

## Recommended Direction

- Do not ship the legacy `python/perfecthash` tree wholesale as the supported Python package.
- Build a new supported runtime surface around the existing online C APIs and keep the legacy research/maintenance modules out of the default public surface.
- Prefer `ph` as the installed CLI name if a Python CLI is added; it is already the historical alias in the legacy code and is short enough for frequent use.
- Recommended package split:
  - `perfecthash` base package: supported runtime bindings, stable CLI, minimal dependencies.
  - `perfecthash[analysis]` if keeping analysis in one dist with optional extras.
  - `perfecthash-analysis` companion dist if the goal is to quarantine older/staler analysis code more aggressively.
- Recommended first supported Python API shape:
  - accept Python integer sequences plus efficient array/buffer inputs (`numpy`, `array`, memoryview-compatible inputs) for 32-bit and 64-bit keys
  - expose sorted vs unsorted input handling explicitly, mapping to the existing online C helpers and key-load flags
  - surface common table-create flags/parameters as Python keyword arguments instead of mirroring the raw CLI argument grammar
  - return a `Table` object that supports at least `index()` and batched indexing; value binding/lookup can layer on top
- Recommended first CLI scope:
  - `ph create`
  - `ph bulk-create`
  - possibly `ph export-python` for generated pure-Python table modules
  - leave experiment-specific and research commands out of the first supported CLI release

## Extraction Strategy

- Moving the legacy `python/` tree out of this repo is feasible and likely the cleanest long-term path.
- I would not do a literal one-shot `mv python ../perfecthash-python` as the very first migration commit.
- Better sequence:
  - create a new development line for the extraction work
  - scaffold the new package/tooling first
  - treat the old `python/` tree as reference material
  - migrate code in small, testable slices
  - remove or archive the old in-repo `python/` tree only after the replacement surface is real
- This avoids a period where:
  - packaging is broken
  - the old code has disappeared but no replacement exists
  - tests/CI have no stable target

## Worktree

- Created a dedicated extraction worktree:
  - branch: `python-dev`
  - path: `/home/trentn/src/perfecthash-python-dev`
- This is a good staging area for the extraction/bootstrap work before deciding whether the end state should remain in this repo or become a separate standalone repo.

## Proposed Migration Plan

- Phase 0: Bootstrap
  - create the new Python project skeleton in the `python-dev` worktree
  - choose the modern packaging/build stack up front
  - add lint/test/typecheck baseline before importing legacy code
- Phase 1: Runtime foundation
  - add the lowest-level binding layer to the online C API
  - prove in-memory table creation and indexing from Python with focused tests
- Phase 2: Public Python API
  - add a small stable `Table`/`Builder` API
  - support Python integer iterables plus buffer-friendly inputs
- Phase 3: Python CLI
  - add `ph create`
  - add `ph bulk-create`
  - keep the CLI thin and library-backed
- Phase 4: Optional analysis surface
  - migrate only the still-useful analysis pieces
  - decide whether they live behind `perfecthash[analysis]` or a separate `perfecthash-analysis` package
- Phase 5: Retirement
  - delete or archive the old legacy `python/` tree from the main repo once the new package is credible and tested

## Recommended Tooling Baseline

- Prefer a fresh `src/` layout with `pyproject.toml`.
- For a compiled extension/binding package, `scikit-build-core` is the strongest fit with the existing C/CMake build.
- Prefer `pytest` for tests, `ruff` for linting and formatting, and targeted `mypy` for the public API boundary.
- Keep CI minimal at first: import test, online-create test, CLI smoke test.

## Commit Cadence

- Each migration slice should be one small atomic commit with:
  - one narrow unit of functionality
  - tests for that slice
  - no opportunistic porting of unrelated legacy modules
- Examples of good slice boundaries:
  - project scaffold only
  - constants/enums only
  - minimal online binding only
  - `Table.index()` only
  - numpy input support only
  - `ph create` only
