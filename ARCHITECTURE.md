# Architecture

## Purpose
PerfectHash generates perfect hash tables from 32-bit key sets using an acyclic
random 2-part hypergraph (CHM-style) algorithm. The primary workflow is offline
creation: a command-line tool builds a small compiled library that exposes an
`Index()` routine for O(1) lookups with minimal memory access.

## High-Level Components

### Core Library: `PerfectHash`
Location: `src/PerfectHash/` with public headers in `include/`.

Responsibilities:
- Parse input key sets and creation parameters.
- Build and solve graphs (CHM01/CHM02 implementations).
- Materialize tables and generate output files for compiled tables.
- Provide API surface for contexts, tables, and error handling.

Key submodules (representative files):
- Graph solving and CHM variants: `Graph.c`, `GraphImpl*.c`, `Chm01.c`,
  `Chm02Shared.c`.
- Table lifecycle: `PerfectHashTable*.c` (create, load, hash, lookup, compile).
- Key handling: `PerfectHashKeys*.c`.
- Platform/utility layer: `Rtl*.c`, `PerfectHashCompat.c` (non-Windows).
- ETW/event metadata: `PerfectHashEvents.*`.

The core library uses COM-style interfaces (vtables) for contexts and tables
(`PerfectHashContext`, `PerfectHashTable`) and is designed to be callable from
the CLI tools and optional Python wrappers.

### Runtime Table Implementation: `CompiledPerfectHashTable`
Location: `src/CompiledPerfectHashTable/` and generated C headers in
`src/PerfectHash/CompiledPerfectHash*`.

Responsibilities:
- Provide the generated `Index()` routines and supporting code for compiled
  perfect hash tables.
- Host multiple index/hash variants (e.g., multiply-shift, CRC32, Jenkins)
  used by the generator to emit final code.

### CLI Tools
Locations:
- `src/PerfectHashCreateExe/` (`PerfectHashCreateExe`)
- `src/PerfectHashBulkCreateExe/` (`PerfectHashBulkCreateExe`)

Responsibilities:
- Bootstrap the `PerfectHash` library.
- Create a context and invoke table creation with command-line parameters.
- Emit output directories containing generated table sources, build files, and
  benchmark/test harnesses.

### CUDA Acceleration (Optional)
Location: `src/PerfectHashCuda/`.

Responsibilities:
- GPU-accelerated graph solving and related kernels.
- Optional build via `USE_CUDA=ON` and appropriate CUDA toolkit settings.

### Assembly and Instrumentation (Optional)
- `src/PerfectHashAsm/`: hand-tuned assembly routines.
- `src/FunctionHook/`: optional PENTER-based instrumentation support.

### Python Tooling
Location: `python/perfecthash/`.

Responsibilities:
- CLI wrappers, analysis utilities, and helpers for inspecting output.
- Optional DLL interop stubs in `python/perfecthash/dll/`.

## Data Flow (Table Creation)
1. CLI tool reads arguments and bootstraps `PerfectHash`.
2. `PerfectHashContext` loads keys and validates parameters.
3. Graphs are constructed and solved (CPU or GPU depending on build flags).
4. On success, table data and index routines are emitted into an output
   directory (C sources, headers, build files, and optional benchmarks/tests).
5. Optional compilation produces a ready-to-link DLL/SO containing `Index()`.

## Build and Configuration
- Build system: CMake (`CMakeLists.txt`, `CMakePresets.json`, `src/cmake/`).
- Public headers install from `include/`.
- Configuration defaults live in `conf/perfecthash.conf`.
- Windows-centric tooling is present but Linux builds are supported via CMake.

## Generated Artifacts
Output directories contain:
- `CompiledPerfectHash.h`, `CompiledPerfectHashMacroGlue.h`, and helpers.
- Per-table C sources and build files for tests and benchmarks.
- Compiled binaries when `--Compile` or equivalent flags are used.

## Repository Layout (Quick Map)
- `include/`: public API headers.
- `src/PerfectHash/`: core generator library.
- `src/CompiledPerfectHashTable/`: runtime/index implementations.
- `src/PerfectHashCreateExe/`, `src/PerfectHashBulkCreateExe/`: CLI tools.
- `src/PerfectHashCuda/`: optional CUDA acceleration.
- `python/`: Python tooling and wrappers.
- `keys/`, `data/`, `notebooks/`: sample data and analysis assets.
