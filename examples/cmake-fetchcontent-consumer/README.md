# PerfectHash FetchContent Consumer Example

This example demonstrates a CMake consumer that can either:

- Build PerfectHash directly from GitHub with `FetchContent`.
- Use an installed package via `find_package(PerfectHash CONFIG REQUIRED)`.

## FetchContent Mode (default)

```bash
cmake -S . -B build \
  -DPERFECTHASH_CONSUMER_USE_FETCHCONTENT=ON \
  -DPERFECTHASH_GIT_REPOSITORY=https://github.com/tpn/perfecthash.git \
  -DPERFECTHASH_GIT_TAG=main \
  -DPERFECTHASH_BUILD_PROFILE=online-rawdog-jit
cmake --build build --parallel
```

## Installed Package Mode

```bash
cmake -S . -B build \
  -DPERFECTHASH_CONSUMER_USE_FETCHCONTENT=OFF \
  -DCMAKE_PREFIX_PATH=/path/to/perfecthash/install
cmake --build build --parallel
```

## Switching Build Profiles

- `online-rawdog-jit`: online/rawdog libraries without LLVM and without CLI exes.
- `online-rawdog-and-llvm-jit`: online/rawdog plus LLVM support.
- `online-llvm-jit`: online LLVM JIT without RawDog generation.
- `full`: full build including CLI executables.

The default linked target is `PerfectHash::PerfectHashOnlineCore`; override with
`-DPERFECTHASH_CONSUMER_TARGET=<target>` if needed.
