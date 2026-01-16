#!/usr/bin/env bash
set -euo pipefail

root_dir="$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)"
build_dir="${BUILD_DIR:-${root_dir}/build}"
install_dir="${INSTALL_DIR:-${root_dir}/install}"

generator="${CMAKE_GENERATOR:-}"
cache_generator=""

if [ -f "$build_dir/CMakeCache.txt" ]; then
  cache_generator="$(sed -n \
    's/^CMAKE_GENERATOR:INTERNAL=//p' \
    "$build_dir/CMakeCache.txt")"
fi

cmake_help="$(cmake --help 2>/dev/null || true)"

if [ -z "$generator" ]; then
  if printf '%s\n' "$cmake_help" | grep -q "Ninja Multi-Config"; then
    generator="Ninja Multi-Config"
  elif printf '%s\n' "$cmake_help" | grep -q "^  Ninja$"; then
    generator="Ninja"
  fi
fi

if [ -n "$cache_generator" ] && [ -n "$generator" ] \
  && [ "$cache_generator" != "$generator" ]; then
  printf 'Existing CMake generator: %s\n' "$cache_generator"
  printf 'Requested generator: %s\n' "$generator"
  printf 'Delete prior CMake cache files first? (y/N): '
  read -r reply
  case "$reply" in
    y|Y)
      rm -f "$build_dir/CMakeCache.txt"
      rm -rf "$build_dir/CMakeFiles"
      cache_generator=""
      ;;
    *)
      generator="$cache_generator"
      ;;
  esac
fi

cmake_args=(
  -S "$root_dir"
  -B "$build_dir"
  "-DCMAKE_INSTALL_PREFIX=$install_dir"
)

if [ -n "$generator" ]; then
  cmake_args=(-G "$generator" "${cmake_args[@]}")
fi

cmake "${cmake_args[@]}" "$@"
