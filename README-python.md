# PerfectHash Python

This worktree hosts the fresh Python extraction/bootstrap effort for
PerfectHash.

The current goal is to build a modern, testable Python package and migrate
useful legacy functionality over in small, verifiable slices.

## Status

There are currently two distinct Python-facing workflows:

1. Offline CLI workflow via `ph create`
2. Offline CLI workflow via `ph bulk-create`
3. In-process programmatic workflow via `build_table()`

These are intentionally different:

- `ph create` maps to the existing offline C creation workflow and is meant
  to generate offline table artifacts.
- `ph bulk-create` maps to the existing offline C bulk-create workflow and is
  meant to generate offline tables for a directory of key files.
- `build_table()` is the in-process native runtime path for code that wants
  to create and use a table from within Python.

## CLI

Show the current CLI surface with:

```bash
env -u PYTHONPATH uv run python -m perfecthash --help
env -u PYTHONPATH uv run python -m perfecthash create --help
env -u PYTHONPATH uv run python -m perfecthash bulk-create --help
```

The current `ph create` command translates modern Python-style options into
the existing `PerfectHashCreate` contract, then shells out to the C create
binary.

Example dry run:

```bash
env -u PYTHONPATH uv run python -m perfecthash create \
  keys/example.keys \
  out \
  --hash-function MultiplyShiftR \
  --graph-impl 3 \
  --create-binary /home/trentn/src/perfecthash/build/bin/PerfectHashCreate \
  --dry-run
```

Current output:

```text
/home/trentn/src/perfecthash/build/bin/PerfectHashCreate keys/example.keys out Chm01 MultiplyShiftR And 0 --GraphImpl=3
```

Useful control flags:

- `--emit-c-argv` prints the abstract translated C argv.
- `--dry-run` prints the resolved executable command.
- `--create-binary` points explicitly at the offline create binary.

There is also an initial `ph bulk-create` command with the same execution
model for the offline bulk-create workflow.

Discovery now prefers installation-oriented locations first:

- package-bundled native dirs such as `perfecthash/_native/bin` and
  `perfecthash/_native/lib`
- explicit install-prefix env vars like `PERFECTHASH_PREFIX`
- the active Python / conda prefix (`sys.prefix`, `CONDA_PREFIX`)
- development-tree fallbacks last

For now, you may still need to pass `--create-binary` or set
`PERFECTHASH_CREATE_BINARY`, but the package no longer assumes a source-tree
build first.

## Programmatic API

The current in-process runtime entry point is `build_table()`:

```python
from perfecthash import build_table

with build_table([1, 3, 5, 7, 11, 13, 17, 19], hash_function="MultiplyShiftR") as table:
    print(table.backend)
    print(table.hash_function)
    print(table.key_count)
    print(table.index(13))
```

Current output from a local verification run:

```text
rawdog_jit
MultiplyShiftR
8
5
```

This path keeps the hot table implementation native and is the foundation for
the future programmatic Python API.

## Curated Hash Functions

The currently supported curated set is:

- `MultiplyShiftR`
- `MultiplyShiftRX`
- `Mulshrolate1RX`
- `Mulshrolate2RX`
- `Mulshrolate3RX`
- `Mulshrolate4RX`

These names are preserved exactly as-is in the Python CLI and API.

## More Detail

See [docs/python-cli-and-api.md](docs/python-cli-and-api.md) for a fuller
overview of the current Python package direction.

## Development

Install the baseline developer environment with:

```bash
env -u PYTHONPATH uv sync
```

For editable installs with native artifacts from this checkout, use the
repo-local install prefix helper:

```bash
./scripts/install-python-native-prefix.sh
export PERFECTHASH_PREFIX="$PWD/.perfecthash-prefix"
env -u PYTHONPATH uv sync
```

This keeps Python editable while making binary/library discovery behave more
like a packaged install.

Run the baseline checks with:

```bash
env -u PYTHONPATH uv run pytest
env -u PYTHONPATH uv run ruff check python_src python_tests
env -u PYTHONPATH uv run black --check python_src python_tests
env -u PYTHONPATH uv run mypy
```

Install the git hook with:

```bash
env -u PYTHONPATH uv run pre-commit install
```

The explicit `env -u PYTHONPATH` is there to avoid inheriting any legacy
PerfectHash Python path overrides from an existing shell session.
