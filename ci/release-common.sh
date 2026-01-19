#!/usr/bin/env bash
set -euo pipefail

PH_SCRIPT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)"
PH_ROOT_DIR="$(
  cd "${PH_SCRIPT_DIR}/.."
  pwd
)"

ph_info() {
  printf '%s\n' "$*"
}

ph_die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

ph_require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    ph_die "missing command: $cmd"
  fi
}

ph_run() {
  if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

ph_rm_rf() {
  local path="$1"
  if [ -z "$path" ] || [ "$path" = "/" ]; then
    ph_die "refuse to remove '$path'"
  fi
  ph_run rm -rf "$path"
}

ph_guess_version() {
  local version=""
  if [ -n "${RELEASE_VERSION:-}" ]; then
    echo "${RELEASE_VERSION#v}"
    return 0
  fi
  version="$(awk 'match($0, /^[[:space:]]*VERSION[[:space:]]+[0-9]/) { print $2; exit }' \
    "$PH_ROOT_DIR/CMakeLists.txt" 2>/dev/null || true)"
  if [ -z "$version" ]; then
    if command -v git >/dev/null 2>&1; then
      version="$(git -C "$PH_ROOT_DIR" describe --tags --abbrev=0 2>/dev/null || true)"
    fi
  fi
  if [ -z "$version" ]; then
    ph_die "unable to determine release version (set RELEASE_VERSION)"
  fi
  echo "${version#v}"
}

ph_platform_label() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"
  case "$os" in
    Linux) os="linux" ;;
    Darwin) os="macos" ;;
    *) os="$(printf '%s' "$os" | tr '[:upper:]' '[:lower:]')" ;;
  esac
  case "$arch" in
    x86_64|amd64) arch="x86_64" ;;
    aarch64|arm64) arch="arm64" ;;
    *) arch="$arch" ;;
  esac
  printf '%s-%s\n' "$os" "$arch"
}

ph_default_generator() {
  local cmake_help
  cmake_help="$(cmake --help 2>/dev/null || true)"
  if printf '%s\n' "$cmake_help" | grep -q "Ninja Multi-Config"; then
    printf '%s\n' "Ninja Multi-Config"
    return 0
  fi
  if printf '%s\n' "$cmake_help" | grep -q "^  Ninja$"; then
    printf '%s\n' "Ninja"
    return 0
  fi
  return 1
}

ph_copy_docs() {
  local dest_dir="$1"
  local doc
  for doc in "$PH_ROOT_DIR/README.md" \
             "$PH_ROOT_DIR/LICENSE" \
             "$PH_ROOT_DIR/USAGE.txt"; do
    if [ -f "$doc" ]; then
      ph_run cp "$doc" "$dest_dir/"
    fi
  done
}

ph_write_sha256() {
  local file="$1"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    ph_info "dry-run: would write checksum for $file"
    return 0
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" > "${file}.sha256"
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" > "${file}.sha256"
  else
    ph_die "missing sha256sum or shasum for checksum generation"
  fi
}
