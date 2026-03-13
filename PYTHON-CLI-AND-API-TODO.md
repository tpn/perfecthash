# PYTHON-CLI-AND-API TODO

## In Progress

- [ ] Decide what the next programmatic `Table` API expansion should be after `index()` / `index_many()`.

## Next

- [ ] Decide whether the programmatic Python entry point should stay as `build_table()` or grow a `perfecthash.create()` alias/wrapper.
- [ ] Add key-input normalization beyond plain Python integer sequences.
- [ ] Decide the next runtime slice after RawDog: full online interface vs JIT wrapper vs 64-bit path.
- [ ] Decide whether the first production-quality native Python path stays ctypes/ABI-based or moves to a compiled extension once wheel packaging is in place.
- [ ] Extend the `ph create` C-argv translation coverage once the low-level binding/API shape is clearer.
- [ ] Decide whether `ph create` should keep executing the C create binary by default or require an explicit execution flag once more options land.
- [ ] Decide what packaged wheel/conda layout we want for bundled native binaries/libraries under the Python package.
- [ ] Decide whether the eventual Python package should target `perfecthash-full` semantics by default for conda, or split CLI/runtime support into separate Python distributions.
- [ ] Decide whether the editable-install helper should grow flags/options or stay a simple convention-driven script.
- [ ] Extend the `ph bulk-create` option coverage beyond the initial subset.
- [ ] Expand the user-facing Python/CLI docs once `ph bulk-create` and richer programmatic APIs exist.

## Later

- [ ] Decide whether analysis stays as `perfecthash[analysis]` or moves into a companion distribution.
- [ ] Migrate useful legacy analysis code surgically, with tests and cleanup per slice.
- [ ] Remove or archive the legacy in-repo `python/` tree after the replacement package is credible.

## Done

- [x] Create the `python-dev` branch/worktree at `/home/trentn/src/perfecthash-python-dev`.
- [x] Bootstrap the extracted Python package scaffold in the `python-dev` worktree.
- [x] Wire up lint/format/test/typecheck and a working worktree-local `pre-commit` hook from the start.
- [x] Generate `uv.lock` and validate the baseline developer workflow.
- [x] Mirror the project ledgers into the new package workflow and keep them current here.
- [x] Add initial `Typer` + `Pydantic` scaffolding for `ph create`.
- [x] Capture the curated supported hash-function set from the repo source of truth.
- [x] Add the first low-level online RawDog binding slice with focused tests.
- [x] Rename the Python-facing RawDog binding surface toward `rawdog_jit` / `RawdogJit` while preserving compatibility with the underlying C exports.
- [x] Add the first backend-neutral public `Table` API layer.
- [x] Keep `ph create` aligned with the offline C creation workflow and add its first executable path via the existing C CLI.
- [x] Add the initial supported `ph bulk-create` surface.
- [x] Refactor binary/library discovery so installed prefixes are preferred over development-tree fallbacks.
- [x] Add a documented repo-local native install-prefix workflow for editable installs, with a helper script.
- [x] Add user-facing Python/CLI documentation beyond inline Typer help.
- [x] Add worked `ph create` examples and document the offline CLI vs programmatic API split.
