# PYTHON-CLI-AND-API TODO

## In Progress

- [ ] Produce a recommendation for packaging strategy, CLI naming, and API layering.

## Next

- [ ] Seed the `python-dev` worktree with the project ledgers and a fresh Python package scaffold.
- [ ] Choose the new packaging/build stack for the extracted package.
- [ ] Define the first atomic migration slices and commit order.
- [ ] Draft the recommended supported package split:
  - default runtime package
  - optional or separate analysis package
  - migration path for legacy modules
- [ ] Draft a proposed Python CLI command surface based on supported library APIs.
- [ ] Draft a proposed Python binding shape for online tables and value lookup.

## Later

- [ ] Decide whether legacy analysis modules stay in-tree under a private namespace, move under a dedicated analysis namespace, or split into a separate distributable package.
- [ ] Design Python `create` and `bulk-create` commands against stable library APIs instead of experiment-specific wrappers.
- [ ] Design a public `Table`/`Builder` API for in-process perfect-hash creation and lookup.
- [ ] Add packaging/tests once the supported Python surface is defined.

## Done

- [x] Create canonical project ledgers for `PYTHON-CLI-AND-API`.
- [x] Inventory the current Python package, CLI, and analysis/runtime split.
- [x] Determine whether any existing Python-facing C scaffolding already exists for generated files or online/runtime hooks.
- [x] Validate the current Python CLI/import path enough to identify obvious breakage or unsupported assumptions.
- [x] Inspect C-side Python-related file generation hooks and online API entry points.
- [x] Create the `python-dev` branch/worktree at `/home/trentn/src/perfecthash-python-dev`.
