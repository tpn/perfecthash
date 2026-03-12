# PYTHON-CLI-AND-API Notes

## Scope

- This worktree is the extraction/bootstrap line for a new Python CLI/API effort.
- The goal is to replace the legacy `python/` tree with a fresh, testable, modern package and migrate code over in small, verified slices.

## Worktree

- Active branch: `python-dev`
- Active worktree: `/home/trentn/src/perfecthash-python-dev`

## Strategy

- Do not rehabilitate the legacy `python/` tree in place.
- Bootstrap a new package first, keep the old tree as reference material, and migrate only the code we still want.
- Keep commits atomic:
  - scaffold only
  - low-level binding only
  - one API slice only
  - one CLI slice only
  - one analysis migration slice only

## Bootstrap Decisions

- New Python project config lives at the repo root in this worktree.
- Python source lives under `python_src/` instead of `src/` because the repo already has a C source tree at `src/`.
- Build backend for the bootstrap phase: `hatchling`
- Development workflow tool: `uv`
- Code quality tooling from the start:
  - `ruff` for linting
  - `black` for formatting
  - `pytest` for tests
  - `mypy` for type checking
  - `pre-commit` with local hooks invoking `uv run`

## Quality Tooling

- `pyproject.toml` defines:
  - project metadata
  - `ruff` config
  - `black` config
  - `pytest` config
  - `mypy` config
  - `uv` default group configuration
- `.pre-commit-config.yaml` uses local hooks so it does not depend on pinning external hook repos just to get started.
- Hook scope is limited to the new extraction scaffold files and does not target the legacy repo contents.
- Local hook commands explicitly unset `PYTHONPATH` before invoking tooling. This avoids accidentally importing the legacy `python/perfecthash` package from another shell session.
- Git hooks are isolated to this worktree via `core.hooksPath=.githooks` stored in the worktree-specific git config.

## Shell Environment Caveat

- On this machine, the active shell domain is `nvidia` via `~/.zsh/desired_domain`.
- `~/.zsh/zshrc` sources `~/.zsh/domains/<domain>/zshrc`.
- The active `PYTHONPATH` entries came from:
  - `~/.zsh/domains/nvidia/zshrc`
  - specifically the `_prepend_pythonpath ~/src/tpn/lib` and `_prepend_pythonpath ~/src/perfecthash/python` lines
- This is why clean-room Python commands in this worktree need `env -u PYTHONPATH`.

## First Package Slice

- Package name: `perfecthash`
- Console script name: `ph`
- Initial runtime scope is intentionally tiny:
  - package import
  - version reporting
  - Typer CLI scaffold
- No legacy Python modules have been migrated yet.
- Baseline validation completed successfully:
  - `pytest`
  - `ruff check`
  - `black --check`
  - `mypy`

## Curated Hash Functions

- The curated supported set is captured explicitly in the repo, not just inferred.
- Source of truth:
  - [`include/PerfectHash/PerfectHash.h`](./include/PerfectHash/PerfectHash.h)
  - `PERFECT_HASH_GOOD_HASH_FUNCTION_TABLE_ENTRY()`
  - `IsGoodPerfectHashHashFunctionId()`
- Current curated set:
  - `MultiplyShiftR`
  - `MultiplyShiftRX`
  - `Mulshrolate1RX`
  - `Mulshrolate2RX`
  - `Mulshrolate3RX`
  - `Mulshrolate4RX`
- Downstream generators already use this predicate to decide what is supported for emitted artifacts such as Python and Rust outputs.

## Current CLI Scaffold

- The public CLI now uses `Typer`.
- The current `ph create` command now follows the offline C creation path:
- The current `ph bulk-create` command now follows the offline C bulk-create path:
  - typed options
  - curated hash-function enum preserving exact C spellings
  - `Pydantic` request model
  - current C CLI argv renderer for a small supported subset
  - offline create-binary discovery
  - subprocess execution of the discovered/provided C create or bulk-create binary
- Current implemented mapping target is `PerfectHashCreate` with:
  - `<KeysPath> <OutputDirectory> <Algorithm> <HashFunction> <MaskFunction> <MaximumConcurrency>`
  - optional flags/parameters:
    - `--Compile`
    - `--DisableCsvOutputFile`
    - `--DoNotTryUseHash16Impl`
    - `--GraphImpl=<N>`
    - `--MaxSolveTimeInSeconds=<N>`
- Current `ph create` control options:
  - `--emit-c-argv`
  - `--dry-run`
  - `--create-binary`
- Current `ph bulk-create` control options:
  - `--emit-c-argv`
  - `--dry-run`
  - `--bulk-create-binary`
- `--emit-c-argv` prints the abstract translated C argv.
- `--dry-run` prints the resolved executable command that would be run.
- By default, `ph create` now shells out to the offline C create binary.
- By default, `ph bulk-create` now shells out to the offline C bulk-create binary.
- Typer wiring currently provides:
  - root help
  - command help
  - typed option validation
  - completion support via Typer's built-in completion flags
- Documentation wiring now includes:
  - root and command help text in the Typer command/option definitions
  - [`README-python.md`](./README-python.md) with user-facing CLI/API overview and examples
  - [`docs/python-cli-and-api.md`](./docs/python-cli-and-api.md) with a dedicated explanation of the offline CLI vs in-process API split

## Offline CLI vs In-Process API

- `ph create` should be treated as the offline table-generation workflow.
- That means its natural backend is the existing offline C creation path / C CLI contract, not the in-process online RawDog JIT runtime.
- The `rawdog_jit` / online path is better treated as a programmatic API for embedding inside Python code, e.g. `perfecthash.create(...)` or `build_table(...)`.
- In other words:
  - CLI `ph create`: offline/create-artifacts/reproducible-output semantics
  - Python API `build_table()` / future `create()`: in-process native table construction semantics
- This keeps the meaning of "create" aligned with the existing C ecosystem and avoids conflating offline generated outputs with transient online runtime tables.

## Offline Create Binary Discovery

- The offline CLI discovery/execution layer lives in [`python_src/perfecthash/c_cli.py`](./python_src/perfecthash/c_cli.py).
- Discovery order for the create binary:
  - `PERFECTHASH_CREATE_BINARY` / `PERFECTHASH_CREATE_EXE`
  - package-bundled native directories (e.g. `perfecthash/_native/bin`)
  - install-prefix oriented directories derived from `PERFECTHASH_PREFIX`, `PERFECTHASH_INSTALL_PREFIX`, `CONDA_PREFIX`, and `sys.prefix`
  - known `build*/bin/` outputs under the current worktree root
  - known `build*/bin/` outputs under the sibling C repo root
  - direct legacy fallback paths like `src/x64/Release/PerfectHashCreate`
- The same install-prefix-first policy now applies to native library discovery for the programmatic API.
- This is much better aligned with conda / packaged installs than the previous development-tree-first logic, though it will still need tightening once packaging/distribution decisions are settled.
- The same module now provides analogous discovery/execution support for the offline bulk-create binary as well.

## First Online Binding Slice

- The first real runtime binding target is the RawDog online C API, not the full internal interface.
- Current Python entry point:
  - `perfecthash.online.build_rawdog_jit_table()`
- Current implementation:
  - [`python_src/perfecthash/online/rawdog_jit.py`](./python_src/perfecthash/online/rawdog_jit.py)
- Current capabilities:
  - create an in-memory 32-bit table from Python integer sequences
  - accept sorted or unsorted keys
  - return a `RawdogJitTable` object with `index()` and `indexes()`
  - manage the native table/context lifetime with `close()` and context-manager support
- Current limitations:
  - 32-bit keys only
  - RawDog path only
  - library discovery is development-oriented and searches known build outputs plus `PERFECTHASH_LIBRARY_PATH` / `PERFECTHASH_LIB_PATH`
  - not yet wired into `ph create`
- Current library-discovery behavior:
  - first check `PERFECTHASH_LIBRARY_PATH` / `PERFECTHASH_LIB_PATH`
  - then search `build*/lib/` under the current worktree root
  - then search sibling `/home/trentn/src/perfecthash/build*/lib/`
- The current tests passed against the existing sibling build outputs from the C repo.

## Higher-Level Table API

- There is now a backend-neutral public `Table` layer on top of the low-level `rawdog_jit` binding.
- Public entry points:
  - `perfecthash.build_table()`
  - `perfecthash.Table`
  - `perfecthash.BuildTableOptions`
  - `perfecthash.TableBackend`
- Current backend behavior:
  - `TableBackend.Auto` currently resolves to `TableBackend.RawdogJit`
  - the internal implementation still delegates to the low-level `RawdogJitTable`
- Current public `Table` surface is intentionally small:
  - `index()`
  - `index_many()`
  - `close()`
  - context-manager support
  - metadata such as `backend`, `hash_function`, `key_count`, and `library_path`
- A compatibility shim remains at `perfecthash.online.rawdog` for the bootstrap slice, but the preferred naming is now `rawdog_jit`.
- This higher-level `Table` API is currently best understood as the foundation for the programmatic Python runtime story, not as the implementation target for `ph create`.

## Performance Direction

- The Python API should keep the actual table implementation native wherever possible.
- Python should ideally only:
  - normalize inputs
  - manage object lifetime
  - dispatch into native create/index routines
  - optionally bind external Python values to native indexes
- Avoid reimplementing table logic or storing table internals in Python for the fast path.
- The pure-Python emitted table modules are useful as offline/export/debug artifacts, but they are not the primary high-performance runtime direction for the new package.

## Packaging Direction For Native Runtime

- Requiring end users to have a local compiler is undesirable, especially on Windows.
- A Cython or CPython-extension path is still viable if we ship wheels; local compiler requirements mainly matter for source installs.
- Current bias:
  - bootstrap with thin ABI-level bindings over shipped shared libraries
  - prefer wheels that bundle the native library/runtime pieces needed by the Python package
  - only require local compilation for developer/source-build workflows
- This keeps the runtime fast without forcing most users to have MSVC/LLVM installed.

## Recommended End-User Layout

- Use one logical discovery contract across packaging modes, but allow different physical layouts:
  - wheels: bundle native artifacts inside the Python package under `perfecthash/_native/`
  - conda: install native artifacts into the environment prefix (`$PREFIX/bin`, `$PREFIX/lib`) and let Python discover them there
- Recommended wheel layout:
  - `perfecthash/_native/bin/PerfectHashCreate[.exe]`
  - `perfecthash/_native/bin/PerfectHashBulkCreate[.exe]`
  - `perfecthash/_native/lib/libPerfectHash*` and any dependent shared libraries needed by the selected profile
- Recommended conda layout:
  - binaries in `$PREFIX/bin`
  - shared libraries in `$PREFIX/lib` (or platform-equivalent)
  - Python package in `site-packages/perfecthash`
- Rationale:
  - wheels should be self-contained
  - conda should use normal prefix-managed binaries and libraries instead of duplicating them inside `site-packages`
  - the Python discovery code can support both cleanly
- Practical dependency recommendation:
  - if the Python package is meant to provide both `ph create` and programmatic runtime support, conda should depend on the profile package that includes both offline binaries and the needed libraries (today that is closest to `perfecthash-full`)
  - if a lighter runtime-only Python package is desired later, it can target the online-only native package outputs separately
- This implies the current discovery logic should continue to prefer:
  - explicit overrides
  - package-local `_native`
  - install prefixes / conda prefix
  - development-tree fallbacks last

## Editable Install Workflow

- `pip install -e .` (or the current `uv sync` editable-like workflow) should remain a supported developer path.
- For editable installs, the Python package should stay source-backed, but native artifacts should come from a local install prefix rather than from `site-packages`.
- Recommended editable/dev-native layout:
  - repo source checkout for Python code
  - repo-local native install prefix such as `.perfecthash-prefix/`
  - binaries in `.perfecthash-prefix/bin`
  - libraries in `.perfecthash-prefix/lib`
- Recommended developer workflow:
  1. build/install the native C artifacts into a repo-local prefix
  2. `pip install -e .` or `uv sync`
  3. point discovery at that prefix via `PERFECTHASH_PREFIX=$PWD/.perfecthash-prefix`
- This is cleaner than relying on arbitrary sibling build directories and is much closer to how packaged installs behave.
- A helper script now exists for this workflow:
  - [`scripts/install-python-native-prefix.sh`](./scripts/install-python-native-prefix.sh)

## Naming Direction

- When referring to the RawDog JIT path in Python-facing code, prefer `rawdog_jit` / `RawdogJit` naming where practical.
- The current first slice still uses `rawdog` names because it mirrors the existing C exports (`PhOnlineRawdog*`), but the public Python naming should be adjusted toward `rawdog_jit`.

## Near-Term Direction

- Next substantive slice should add the thinnest possible binding to the existing online C API and prove in-memory create/index from Python.
- After that:
  - `Table` object
  - sorted/unsorted key handling
  - buffer-friendly inputs
  - `ph create`
  - `ph bulk-create`

## CLI Framework Direction

- Replacing the legacy `cli.py` / `command.py` / `commands.py` / invariant stack is justified here because we are building a new CLI from scratch, not trying to preserve that framework.
- Current recommendation:
  - use `Typer` for the user-facing CLI layer
  - use `Pydantic` for validated option/config models where structured validation adds value
  - keep the core runtime and business logic independent of both
- Rationale:
  - Typer gives us modern typed command declarations, standard help, and shell completion without us maintaining custom parser infrastructure.
  - Pydantic is useful for grouped validated inputs, cross-field checks, config loading, and serialization, but it should not become the CLI framework itself.
  - The core implementation should accept normal Python objects and remain callable without CLI dependencies.
- Preferred architecture:
  - `perfecthash.cli`: Typer commands only
  - `perfecthash.models`: Pydantic input/config/result models
  - `perfecthash.api` / `perfecthash.online`: actual runtime-facing logic
- Important boundary:
  - parse CLI arguments with Typer
  - map them into Pydantic models if needed
  - call plain Python service functions
  - do not bury core logic inside Typer callbacks or Pydantic validators
- Use Typer/Pydantic where they provide real benefits, not as a blanket rewrite mandate for every old concept.
