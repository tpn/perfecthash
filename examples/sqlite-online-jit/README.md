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

Default mode pulls PerfectHash from GitHub using FetchContent.

### Linux/macOS

```bash
cmake -S examples/sqlite-online-jit \
      -B build/examples/sqlite-online-jit \
      -DPERFECTHASH_BUILD_PROFILE=online-rawdog-and-llvm-jit

cmake --build build/examples/sqlite-online-jit -j
```

### Windows (Visual Studio)

```powershell
cmake -S examples/sqlite-online-jit `
      -B build\\examples\\sqlite-online-jit `
      -G \"Visual Studio 17 2022\" -A x64 `
      -DPERFECTHASH_BUILD_PROFILE=online-rawdog-and-llvm-jit

cmake --build build\\examples\\sqlite-online-jit --config Release
```

### Local Tree Fallback (No FetchContent)

```bash
cmake -S examples/sqlite-online-jit \
      -B build/examples/sqlite-online-jit \
      -DPH_ONLINE_JIT_USE_FETCHCONTENT=OFF \
      -DPERFECTHASH_ROOT=/path/to/perfecthash
```

Optional knobs:

- `-DPERFECTHASH_GIT_REPOSITORY=<repo-url-or-path>` (default: `https://github.com/tpn/perfecthash.git`)
- `-DPERFECTHASH_GIT_TAG=<tag-or-branch>` (default: `main`)

## Run

### Full Permutation Matrix (Default)

Running with no backend/hash/vector overrides executes the full comparison
matrix:

```bash
./build/examples/sqlite-online-jit/sqlite-online-jit
```

This runs:

- backends: `rawdog-jit`, `llvm-jit`
- hash functions:
  `multiplyshiftr`, `multiplyshiftrx`, `mulshrolate1rx`,
  `mulshrolate2rx`, `mulshrolate3rx`, `mulshrolate4rx`
- vector widths: `1`, `2`, `4`, `8`, `16`

For matrix mode, strict vector-width behavior is enabled so AVX-512 (`16`) and
other widths are measured as requested per permutation.

### Single Configuration

Use explicit backend/hash/vector options to run one configuration:

```bash
./build/examples/sqlite-online-jit/sqlite-online-jit \
  --single \
  --backend rawdog-jit \
  --hash mulshrolate2rx \
  --vector-width 16
```

Optional flags:

- `--matrix` or `--single`
- `--backend <rawdog-jit|llvm-jit|auto>`
- `--hash <name>`
- `--vector-width <0|1|2|4|8|16>`
- `--strict-vector-width <0|1>` (single mode)
- `--dim-size <count>`
- `--fact-size <count>`
- `--iterations <count>`
- `--seed <value>`

## Output

The executable prints:

- `EXPLAIN QUERY PLAN` for baseline and PerfectHash queries
- baseline runtime summary
- single mode:
  requested/effective backend + vector width, compile HRESULT, speedup
- matrix mode:
  one CSV row per permutation with
  backend/hash/requested+effective vector width/JIT status/compile HRESULT/runtime/speedup

The run fails if query results diverge between baseline and PerfectHash paths.
