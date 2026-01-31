#!/usr/bin/env bash
set -euo pipefail

sudo_cmd=""

if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
  sudo_cmd="sudo"
fi

run() {
  if [ -n "$sudo_cmd" ]; then
    "$sudo_cmd" "$@"
  else
    "$@"
  fi
}

packages=(
  build-essential
  cmake
  ninja-build
)

if [ "${WITH_RUST:-0}" = "1" ]; then
  packages+=(
    rustc
    cargo
  )
fi

run apt-get update
run apt-get install -y "${packages[@]}"

if [ "${WITH_CUDA:-0}" = "1" ]; then
  cat <<'CUDA_MSG'
CUDA toolkit install is optional and distro-specific.
On Ubuntu, see NVIDIA's CUDA repo instructions:
https://developer.nvidia.com/cuda-downloads
CUDA_MSG
fi