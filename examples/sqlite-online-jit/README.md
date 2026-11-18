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

Older Windows builds may have copied `PerfectHashLLVM.dll` beside the executable before CMake tracked staged files. Staging stops if it finds an untracked plugin there and the selected SDK supplies none. Move the previous SDK's DLL out of the executable directory and rebuild. To use a compatible plugin you supplied, keep its matching `PerfectHash.dll` beside it and reconfigure with `-DPERFECTHASH_ONLINE_JIT_LLVM_RUNTIME_DLL="C:/path/to/PerfectHashLLVM.dll"`.

When reusing a build directory with a cached `ON` or `OFF`, pass `-DPH_ONLINE_JIT_USE_FETCHCONTENT=AUTO` to select from the current `PERFECTHASH_ROOT`.

To rediscover SDK paths cached by an older finder, add `-U "PERFECTHASH_ONLINE_JIT_*" -U PERFECTHASH_ASM_LIBRARY` to the configure command and repeat any intentional path overrides after those options.

If a Windows repository contains builds for several target architectures, set `PERFECTHASH_ROOT` to the intended build or install directory. For a build tree without headers, also pass `-DPERFECTHASH_ONLINE_JIT_INCLUDE_DIR=<repository>/include`.

For a Windows SDK with a static archive outside a `static` directory, pass `-DPERFECTHASH_ONLINE_JIT_STATIC_LIBRARY=C:/sdk/lib/Debug/PerfectHashOnlineCore.lib` with the path to the archive.

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

Matrix mode strictly requests each width when compiling the JIT. SQLite builds
its index map and probes it one key at a time through `PhOnlineJitIndex32`.
Callers of the 16-key batch API can use scalar batches with RawDog on ARM64
or vector instructions with LLVM.

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
