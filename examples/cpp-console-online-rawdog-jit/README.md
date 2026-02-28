# cpp-console-online-rawdog-jit

Minimal C++ console example that creates a runtime perfect hash table for
32-bit keys using:

- PerfectHash online mode
- RawDog JIT backend
- the slim public header `PerfectHashOnlineRawdog.h`

This example is intentionally simple so humans and LLMs can both ingest it
quickly.

## What It Finds

The CMake finder module prefers the smallest runtime-first targets:

- `PerfectHashOnlineCore` (preferred)
- `PerfectHashOnline` (fallback)

On Windows, static detection is handled so `PERFECT_HASH_ONLINE_RAWDOG_STATIC`
is applied automatically when needed.

## Build Requirements

- CMake 3.20+
- C++17 compiler
- Internet access for default FetchContent mode

Optional fallback:

- A built (or installed) PerfectHash tree with online/RawDog-JIT enabled targets

## Configure and Build

### Linux/macOS

```bash
cmake -S examples/cpp-console-online-rawdog-jit \
      -B build/examples/cpp-console-online-rawdog-jit \
      -DPERFECTHASH_BUILD_PROFILE=online-rawdog-jit

cmake --build build/examples/cpp-console-online-rawdog-jit -j
```

### Windows (Visual Studio)

```powershell
cmake -S examples/cpp-console-online-rawdog-jit `
      -B build\examples\cpp-console-online-rawdog-jit `
      -G "Visual Studio 17 2022" -A x64 `
      -DPERFECTHASH_BUILD_PROFILE=online-rawdog-jit

cmake --build build\examples\cpp-console-online-rawdog-jit --config Release
```

### Local Tree Fallback (No FetchContent)

```bash
cmake -S examples/cpp-console-online-rawdog-jit \
      -B build/examples/cpp-console-online-rawdog-jit \
      -DPH_ONLINE_RAWDOG_USE_FETCHCONTENT=OFF \
      -DPERFECTHASH_ROOT=/path/to/perfecthash
```

Optional knobs:

- `-DPERFECTHASH_GIT_REPOSITORY=<repo-url-or-path>` (default: `https://github.com/tpn/perfecthash.git`)
- `-DPERFECTHASH_GIT_TAG=<tag-or-branch>` (default: `main`)

## Run

```bash
./build/examples/cpp-console-online-rawdog-jit/cpp-console-online-rawdog-jit
```

Optional arguments:

- `--hash <name>`
- `--vector-width <0|1|2|4|8|16>`

Example:

```bash
./cpp-console-online-rawdog-jit --hash mulshrolate2rx --vector-width 16
```

## Notes

- The app validates that every key maps to a unique index.
- The slim `PhOnlineRawdogCompileTable()` wrapper retries smaller vector widths
  when a requested RawDog JIT vector kernel is unavailable on the current
  host/hash combination.
- If RawDog JIT still returns `PH_E_NOT_IMPLEMENTED` after retries (for example,
  unsupported hash scalar kernel on that platform), the sample continues with
  the non-JIT index path and prints a clear fallback mode.
- `x86_64` and `arm64` RawDog routine selection stays architecture-specific via
  upstream library build flags/macros.
