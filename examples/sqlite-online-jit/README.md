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
      -B build\\examples\\sqlite-online-jit `
      -G \"Visual Studio 17 2022\" -A x64 `
      -DPERFECTHASH_ROOT=C:\\path\\to\\perfecthash

cmake --build build\\examples\\sqlite-online-jit --config Release
```

## Run

```bash
./build/examples/sqlite-online-jit/sqlite-online-jit \
  --backend rawdog-jit \
  --dim-size 50000 \
  --fact-size 1000000 \
  --iterations 5 \
  --vector-width 16
```

Optional flags:

- `--backend <rawdog-jit|llvm-jit|auto>`
- `--hash <name>`
- `--vector-width <0|1|2|4|8|16>`
- `--dim-size <count>`
- `--fact-size <count>`
- `--iterations <count>`
- `--seed <value>`

## Output

The executable prints:

- `EXPLAIN QUERY PLAN` for baseline and PerfectHash queries
- average runtime for both modes
- relative speedup (`baseline/perfecthash`)

The run fails if query results diverge between baseline and PerfectHash paths.
