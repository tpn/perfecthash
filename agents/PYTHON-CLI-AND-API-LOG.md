# PYTHON-CLI-AND-API Log

## 2026-03-10

- 00:00 PT: Started project reconnaissance. Checked repo root, Python package tree, and git status.
- 00:02 PT: Confirmed no existing `PYTHON-CLI-AND-API` ledgers were present.
- 00:04 PT: Inspected `python/pyproject.toml`, `python/setup.py`, `python/README.md`, `python/perfecthash/cli.py`, and `python/perfecthash/commands.py`.
- 00:06 PT: Confirmed current Python packaging has optional `analysis` extras but no console-script entry points.
- 00:08 PT: Confirmed legacy in-process alias `ph` exists in `python/perfecthash/exe.py`, but is not installed via package metadata.
- 00:10 PT: Inspected `python/perfecthash/config.py` and `conf/perfecthash.conf`; found strong Windows/local-workstation assumptions for DLLs, output dirs, and VS tool paths.
- 00:12 PT: Inspected `conda/recipe/meta.yaml`; confirmed current conda outputs package only C/C++ artifacts, not the Python package.
- 00:16 PT: Validated the legacy Python CLI via `PYTHONPATH=python python - <<'PY' ...`; `cli.run('ph', 'perfecthash', 'help')` works and confirms the historical command name `ph`, but `perfecthash.__version__` is currently unset.
- 00:20 PT: Inspected `src/PerfectHash/Chm01FileWorkPythonFile.c` and `src/PerfectHash/Chm01FileWorkPythonTestFile.c`; confirmed existing codegen support for generated pure-Python table modules/tests.
- 00:24 PT: Inspected `include/PerfectHash/PerfectHash.h`, `src/PerfectHash/PerfectHashOnlineCreate.c`, `src/PerfectHash/PerfectHashOnlineRawdog.c`, and `src/PerfectHash/PerfectHashOnlineJit.c`; confirmed an existing in-memory online creation path and newer minimal public C APIs for JIT/runtime table creation.
- 00:28 PT: Inspected `tests/PerfectHashOnlineTests.cpp` and `src/PerfectHash/PerfectHashKeysLoad.c`; confirmed the online creation path already supports sorted/unsorted in-memory keys, 32-bit/64-bit keys, and table-create parameters, with test coverage around many of those cases.
- 20:59 PT: Made a ledger pass to capture findings about online APIs, generated Python artifacts, and next-step design work.
- 21:01 PT: Added a recommended direction to `NOTES.md` covering package split, CLI naming (`ph`), and the first supported Python API/CLI scope.
- 21:07 PT: Inspected git branches/worktrees, confirmed no existing `python-dev` branch/worktree conflicted, and created new worktree `/home/trentn/src/perfecthash-python-dev` on branch `python-dev`.
- 21:09 PT: Updated `NOTES.md` and `TODO.md` with the extraction recommendation: bootstrap a fresh package first, migrate legacy Python code surgically in atomic tested slices, and defer deleting the old `python/` tree until the replacement exists.
