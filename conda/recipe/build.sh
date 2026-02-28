#!/usr/bin/env bash
set -euo pipefail

profile="${PERFECTHASH_BUILD_PROFILE:-online-rawdog-jit}"

cmake -S . -B build-conda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DPERFECTHASH_BUILD_PROFILE="$profile" \
  -DPERFECTHASH_ENABLE_TESTS=OFF \
  -DBUILD_TESTING=OFF

cmake --build build-conda --parallel
cmake --install build-conda
