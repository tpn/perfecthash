# sqlite-online-jit

Real-world sqlite integration example that embeds PerfectHash online JIT in a
virtual table module and benchmarks A/B behavior.

## What This Example Demonstrates

- sqlite integration without sqlite core planner patches
- runtime PerfectHash table generation from a sqlite dimension table
- backend toggle for PerfectHash compile path:
  `rawdog-jit`, `llvm-jit`, `auto`
- direct A/B benchmark in one executable:
  baseline B-tree join vs PerfectHash virtual-table join
- full permutation matrix mode:
  `rawdog-jit` + `llvm-jit` x curated hash functions x vector widths
  `1/2/4/8/16`
- repeated build-run analysis (`--build-runs`) for creation-time distributions

## Integration Shape

- sqlite is vendored as a pinned amalgamation snapshot under `sqlite/`.
- The module `perfecthash` is registered in-process.
- `CREATE VIRTUAL TABLE ... USING perfecthash(...)` materializes a PerfectHash
  lookup index from a source table (`dim`) and key/value columns.
- Join queries probe the virtual table through `xBestIndex` equality constraints.

## Vendored sqlite Snapshot

- Version: `3.51.2` (`3510200`)
- Source: `https://www.sqlite.org/2026/sqlite-amalgamation-3510200.zip`
- Files: `sqlite3.c`, `sqlite3.h`, `sqlite3ext.h`

## Build

### Linux/macOS

```bash
cmake -S examples/sqlite-online-jit \
      -B build/examples/sqlite-online-jit \
      -DPERFECTHASH_ROOT=/path/to/perfecthash

cmake --build build/examples/sqlite-online-jit -j
```

### Windows (Visual Studio)

```powershell
cmake -S examples/sqlite-online-jit `
      -B build\examples\sqlite-online-jit `
      -G "Visual Studio 17 2022" -A x64 `
      -DPERFECTHASH_ROOT=C:\path\to\perfecthash

cmake --build build\examples\sqlite-online-jit --config Release
```

## Run

### Full Permutation Matrix (Default)

Running with no backend/hash/vector overrides executes the full comparison
matrix with strict vector width enabled:

```bash
./build/examples/sqlite-online-jit/sqlite-online-jit
```

### Comprehensive Multi-Run Matrix (Recommended)

Use the helper script to generate notebook-ready CSV outputs under
`examples/sqlite-online-jit/results/latest/`:

```bash
examples/sqlite-online-jit/scripts/run_matrix_benchmark.sh
```

Environment overrides for larger/smaller runs:

```bash
BUILD_RUNS=50 DIM_SIZE=50000 FACT_SIZE=200000 ITERATIONS=1 \
  examples/sqlite-online-jit/scripts/run_matrix_benchmark.sh
```

### Single Configuration

```bash
./build/examples/sqlite-online-jit/sqlite-online-jit \
  --single \
  --backend rawdog-jit \
  --hash mulshrolate2rx \
  --vector-width 16 \
  --strict-vector-width 1 \
  --build-runs 20
```

Optional flags:

- `--matrix` or `--single`
- `--backend <rawdog-jit|llvm-jit|auto>`
- `--hash <name>`
- `--vector-width <0|1|2|4|8|16>`
- `--strict-vector-width <0|1>` (single mode)
- `--build-runs <count>`
- `--output-detailed-csv <path>`
- `--output-summary-csv <path>`
- `--dim-size <count>`
- `--fact-size <count>`
- `--iterations <count>`
- `--seed <value>`

## Output Model

Two CSV output levels are supported:

- Detailed CSV (`--output-detailed-csv`): one row per build-run
  permutation sample.
- Summary CSV (`--output-summary-csv`): one row per permutation with aggregate
  stats.

Timing fields now separate build phases:

- `create_*`: external full CREATE VIRTUAL TABLE wall time
- `table_create_*`: `PhOnlineJitCreateTable32()` wall time
  (captures the probabilistic random-search build phase)
- `compile_*`: JIT compile wall time
- `end_to_end_*`: `create + query` wall time
- `query_speedup`: baseline/query
- `end_to_end_speedup`: baseline/(create+query)
- `break_even_queries`: estimated number of queries to amortize build cost

The run fails if query results diverge between baseline and PerfectHash paths.

## Notebook Visualization

Open the included notebook:

```bash
python -m pip install -r examples/sqlite-online-jit/notebooks/requirements.txt
jupyter notebook examples/sqlite-online-jit/notebooks/sqlite_online_jit_matrix_analysis.ipynb
```

Notebook expectations:

- `examples/sqlite-online-jit/results/latest/summary.csv`
- `examples/sqlite-online-jit/results/latest/detailed.csv`

The notebook provides:

- heatmaps for speedup and build cost across hash/vector/backend combinations
- repeated-run table-creation distribution plots
- query-speedup vs creation-cost scatter plots
- ranked top configurations for query-only and end-to-end scenarios
