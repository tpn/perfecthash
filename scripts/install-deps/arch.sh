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
  base-devel
  cmake
  ninja
)

run pacman -Sy --needed --noconfirm "${packages[@]}"

if [ "${WITH_CUDA:-0}" = "1" ]; then
  cat <<'CUDA_MSG'
CUDA toolkit install is optional and distro-specific.
On Arch, CUDA is available in the official repo:
https://archlinux.org/packages/extra/x86_64/cuda/
CUDA_MSG
fi
