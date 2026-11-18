# cpp-console-online-jit

Minimal C++ console example that creates a runtime perfect hash table for
32-bit keys using PerfectHash online mode with either:

- RawDog JIT backend (`rawdog-jit`)
- LLVM JIT backend (`llvm-jit`)

The example uses the slim public header `PerfectHashOnlineJit.h`.

## What It Finds

The CMake finder module resolves:

- `PerfectHashOnline` runtime (`PerfectHash::OnlineJit` imported target)
- `PerfectHashLLVM` runtime (required for `llvm-jit`)

On Windows, required DLLs are copied next to the executable automatically.
On Linux/macOS, build-time rpaths are set for local runs.

## Build Requirements

- CMake 3.20+
- C++17 compiler
- Internet access for default FetchContent mode

Optional fallback:

- A built (or installed) PerfectHash tree with online and LLVM support enabled

## Configure and Build

### Linux/macOS

```bash
cmake -S examples/cpp-console-online-jit \
      -B build/examples/cpp-console-online-jit \
      -DPERFECTHASH_BUILD_PROFILE=online-rawdog-and-llvm-jit

cmake --build build/examples/cpp-console-online-jit -j
```

### Windows (Visual Studio)

```powershell
cmake -S examples/cpp-console-online-jit `
      -B build\examples\cpp-console-online-jit `
      -G "Visual Studio 17 2022" -A x64 `
      -DPERFECTHASH_BUILD_PROFILE=online-rawdog-and-llvm-jit

cmake --build build\examples\cpp-console-online-jit --config Release
```

### Local Tree Fallback (No FetchContent)

Older Windows builds may have copied `PerfectHashLLVM.dll` beside the executable before CMake tracked staged files. Staging stops if it finds an untracked plugin there and the selected SDK supplies none. Move the previous SDK's DLL out of the executable directory and rebuild. To use a compatible plugin you supplied, reconfigure with `-DPERFECTHASH_ONLINE_JIT_LLVM_RUNTIME_DLL="C:/path/to/PerfectHashLLVM.dll"`.

When reusing a build directory with a cached `ON` or `OFF`, pass `-DPH_ONLINE_JIT_USE_FETCHCONTENT=AUTO` to select local discovery whenever `PerfectHashOnlineJit_ROOT` or `PERFECTHASH_ROOT` is set as a CMake or environment variable.

To rediscover SDK paths cached by an older finder, add `-U "PERFECTHASH_ONLINE_JIT_*" -U PERFECTHASH_ASM_LIBRARY` to the configure command and repeat any intentional path overrides after those options.

`PerfectHashOnlineJit_ROOT` accepts an ordered list of SDKs for the target architecture. Discovery uses the first complete SDK. If a Windows repository contains builds for several target architectures, set `PERFECTHASH_ROOT` to the intended build or install directory. A headerless build can use the named checkout when the build is inside it or its CMake cache identifies that source directory. For a headerless SDK from another location, also pass `-DPERFECTHASH_ONLINE_JIT_INCLUDE_DIR=<repository>/include`.

For a Windows SDK with a static archive outside a `static` directory, pass `-DPERFECTHASH_ONLINE_JIT_STATIC_LIBRARY=C:/sdk/lib/Debug/PerfectHashOnlineCore.lib` with the path to the archive.

```bash
cmake -S examples/cpp-console-online-jit \
      -B build/examples/cpp-console-online-jit \
      -DPH_ONLINE_JIT_USE_FETCHCONTENT=OFF \
      -DPERFECTHASH_ROOT=/path/to/perfecthash
```

Optional knobs:

- `-DPERFECTHASH_GIT_REPOSITORY=<repo-url-or-path>` (default: `https://github.com/tpn/perfecthash.git`)
- `-DPERFECTHASH_GIT_TAG=<tag-or-branch>` (default: `main`)

## Run

```bash
./build/examples/cpp-console-online-jit/cpp-console-online-jit --backend rawdog-jit
./build/examples/cpp-console-online-jit/cpp-console-online-jit --backend llvm-jit --vector-width 1
```

Optional arguments:

- `--backend <rawdog-jit|llvm-jit|auto>`
- `--hash <name>`
- `--vector-width <0|1|2|4|8|16>`
- `--dump-cuda-source`
- `--omit-kernels`
- `--source-out <path>`

### Dump Generated CUDA Source

```bash
./build/examples/cpp-console-online-jit/cpp-console-online-jit \
  --backend rawdog-jit \
  --hash mulshrolate3rx \
  --source-out /tmp/online_jit_table.cu \
  --omit-kernels
```

`--omit-kernels` emits only the inline lookup fragment (`index_from_key()`,
constants, table data, and helpers) and leaves kernel shape to the downstream
consumer. This is the mode intended for NVRTC/LTO-IR pipelines that want
call-site specialization such as `ITEMS_PER_THREAD` unrolling.

## Notes

- The app validates that each input key maps to a unique output index.
- For `rawdog-jit`, the wrapper retries smaller vector widths when a requested
  vector kernel is unavailable.
- If the selected JIT backend is unavailable on the current host/table
  combination, the sample falls back to the non-JIT index path.
