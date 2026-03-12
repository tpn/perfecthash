# Python CLI And API

## Overview

The Python package currently has two separate tracks:

- Offline CLI generation via `ph create`
- Offline CLI generation via `ph bulk-create`
- In-process runtime creation via `build_table()`

That split is deliberate.

`ph create` stays aligned with the existing offline `PerfectHashCreate`
workflow. `ph bulk-create` stays aligned with the existing offline
`PerfectHashBulkCreate` workflow. `build_table()` is the start of the
programmatic Python runtime API.

## Offline CLI

The `ph create` command is the Python-facing wrapper around the existing
offline C create contract.

Conceptually:

1. Parse a modern Python CLI
2. Translate that into the C `PerfectHashCreate` argv shape
3. Execute the offline create binary

Current command shape:

```bash
ph create KEYS_PATH OUTPUT_DIR --hash-function HASH_FUNCTION [OPTIONS]
```

```bash
ph bulk-create KEYS_DIR OUTPUT_DIR --hash-function HASH_FUNCTION [OPTIONS]
```

Examples:

```bash
env -u PYTHONPATH uv run python -m perfecthash create \
  keys/example.keys \
  out \
  --hash-function MultiplyShiftR \
  --emit-c-argv
```

```bash
env -u PYTHONPATH uv run python -m perfecthash create \
  keys/example.keys \
  out \
  --hash-function MultiplyShiftR \
  --graph-impl 3 \
  --create-binary /home/trentn/src/perfecthash/build/bin/PerfectHashCreate \
  --dry-run
```

Current `ph create` supports a small but real subset of the C create surface:

- `--hash-function`
- `--maximum-concurrency`
- `--compile`
- `--disable-csv-output-file`
- `--do-not-try-use-hash16-impl`
- `--graph-impl`
- `--max-solve-time-in-seconds`

It also supports:

- `--emit-c-argv`
- `--dry-run`
- `--create-binary`

The initial `ph bulk-create` command supports the same overall execution model,
plus a first useful subset of bulk-specific flags such as:

- `--compile`
- `--skip-test-after-create`
- `--quiet`
- `--disable-csv-output-file`
- `--omit-csv-row-if-table-create-failed`
- `--omit-csv-row-if-table-create-succeeded`

## Programmatic API

The current programmatic API entry point is:

```python
from perfecthash import build_table
```

Example:

```python
from perfecthash import build_table

keys = [1, 3, 5, 7, 11, 13, 17, 19]

with build_table(keys, hash_function="MultiplyShiftR") as table:
    print(table.backend)
    print(table.hash_function)
    print(table.key_count)
    print(table.index(13))
    print(table.index_many(keys))
```

Current behavior:

- The fast path stays native.
- Python is mostly coordinating input normalization and object lifetime.
- The current implementation uses the `rawdog_jit` online runtime path.

Current `Table` surface is intentionally small:

- `index()`
- `index_many()`
- `close()`
- context-manager support
- metadata:
  - `backend`
  - `hash_function`
  - `key_count`
  - `library_path`

## Hash Functions

The currently curated supported set is:

- `MultiplyShiftR`
- `MultiplyShiftRX`
- `Mulshrolate1RX`
- `Mulshrolate2RX`
- `Mulshrolate3RX`
- `Mulshrolate4RX`

These names are preserved exactly as they appear in the C codebase.

## Current Limitations

Offline CLI:

- Not all C create options are exposed yet.
- Binary/library discovery is better aligned with installed prefixes now, but
  packaged distribution layout still needs to be finalized.

Programmatic API:

- Current backend is `rawdog_jit`.
- Current binding path is ABI-level `ctypes`.
- Current key support is focused on Python integer sequences and 32-bit keys.
- The API is still missing higher-level conveniences such as value binding and
  richer input normalization.

## Direction

Near-term:

- expand the programmatic `Table` API carefully without pulling table logic into
  Python
- extend `ph create` option coverage where it makes sense
- add `ph bulk-create`

Longer-term:

- decide whether the production native Python path remains ABI-based or grows a
  compiled extension once wheel packaging is in place
- improve packaged binary discovery instead of relying on development-tree
  heuristics

## Discovery Order

The current Python package now prefers installation-oriented discovery paths
before it falls back to source-tree builds.

For binaries and libraries, the search order is roughly:

1. Explicit binary/library env vars
2. Package-bundled native dirs under `perfecthash/_native/`
3. Explicit install-prefix env vars such as `PERFECTHASH_PREFIX`
4. The active Python / conda prefix (`sys.prefix`, `CONDA_PREFIX`)
5. Development-tree fallback paths

That is meant to make conda or wheel-style end-user installs the default case,
with source-tree builds as the fallback rather than the primary assumption.

## Editable Install Workflow

For developers working from a source checkout, the recommended workflow is:

```bash
./scripts/install-python-native-prefix.sh
export PERFECTHASH_PREFIX="$PWD/.perfecthash-prefix"
env -u PYTHONPATH uv sync
```

That gives you:

- editable Python code from the source tree
- native binaries/libraries installed to a repo-local prefix
- discovery behavior that is much closer to conda/wheel installs than ad hoc
  build-tree probing
