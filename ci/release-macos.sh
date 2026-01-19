#!/usr/bin/env bash
set -euo pipefail

script_dir="$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)"
source "${script_dir}/release-common.sh"

usage() {
  cat <<'EOF'
Usage: ci/release-macos.sh [options]

Options:
  --version <ver>       Release version (defaults to RELEASE_VERSION/git/CMakeLists)
  --config <cfg>        Build config (default: Release)
  --arch <arch>         macOS arch (arm64 or x86_64; default: host arch)
  --build-dir <dir>     Build directory
  --install-dir <dir>   Install directory
  --stage-dir <dir>     Staging directory
  --dist-dir <dir>      Artifact output directory
  --skip-tests          Skip ctest
  --skip-install        Skip cmake --install
  --skip-package        Skip packaging
  --clean               Remove the default output base dir before building
  --dry-run             Print commands without executing them
  -h, --help            Show this help

Environment:
  RELEASE_VERSION, CONFIG, CMAKE_GENERATOR, CMAKE_OSX_ARCHITECTURES
  PERFECTHASH_ENABLE_NATIVE_ARCH, SKIP_TESTS, SKIP_INSTALL, SKIP_PACKAGE
  DRY_RUN
EOF
}

release_version=""
config="${CONFIG:-Release}"
arch="${CMAKE_OSX_ARCHITECTURES:-}"
build_dir="${BUILD_DIR:-}"
install_dir="${INSTALL_DIR:-}"
stage_dir="${STAGE_DIR:-}"
dist_dir="${DIST_DIR:-}"
skip_tests="${SKIP_TESTS:-0}"
skip_install="${SKIP_INSTALL:-0}"
skip_package="${SKIP_PACKAGE:-0}"
clean="${CLEAN:-0}"
DRY_RUN="${DRY_RUN:-0}"

while [ $# -gt 0 ]; do
  case "$1" in
    --version)
      release_version="$2"
      shift 2
      ;;
    --config)
      config="$2"
      shift 2
      ;;
    --arch)
      arch="$2"
      shift 2
      ;;
    --build-dir)
      build_dir="$2"
      shift 2
      ;;
    --install-dir)
      install_dir="$2"
      shift 2
      ;;
    --stage-dir)
      stage_dir="$2"
      shift 2
      ;;
    --dist-dir)
      dist_dir="$2"
      shift 2
      ;;
    --skip-tests)
      skip_tests=1
      shift
      ;;
    --skip-install)
      skip_install=1
      shift
      ;;
    --skip-package)
      skip_package=1
      shift
      ;;
    --clean)
      clean=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      ph_die "unknown argument: $1"
      ;;
  esac
done

if [ -z "$release_version" ]; then
  release_version="$(ph_guess_version)"
fi

if [ -z "$arch" ]; then
  arch="$(uname -m)"
fi
case "$arch" in
  arm64|aarch64) arch="arm64" ;;
  x86_64|amd64) arch="x86_64" ;;
  *) ph_die "unsupported arch: $arch" ;;
esac

platform="macos-${arch}"
base_dir="${PH_ROOT_DIR}/out/release/${release_version}/${platform}"

if [ -z "$build_dir" ]; then
  build_dir="${base_dir}/build"
fi
if [ -z "$install_dir" ]; then
  install_dir="${base_dir}/install"
fi
if [ -z "$stage_dir" ]; then
  stage_dir="${base_dir}/stage/perfecthash-${release_version}-${platform}"
fi
if [ -z "$dist_dir" ]; then
  dist_dir="${base_dir}/dist"
fi

if [ "$clean" = "1" ]; then
  ph_rm_rf "$base_dir"
fi

ph_require_cmd cmake
ph_require_cmd ctest
ph_require_cmd tar

generator="${CMAKE_GENERATOR:-}"
if [ -z "$generator" ]; then
  generator="$(ph_default_generator || true)"
fi
if [ -n "$generator" ] && printf '%s' "$generator" | grep -qi "ninja"; then
  ph_require_cmd ninja
fi

native_arch="${PERFECTHASH_ENABLE_NATIVE_ARCH:-ON}"

ph_info "release version: ${release_version}"
ph_info "platform: ${platform}"
ph_info "build dir: ${build_dir}"
ph_info "install dir: ${install_dir}"
ph_info "dist dir: ${dist_dir}"

cmake_args=(
  -S "$PH_ROOT_DIR"
  -B "$build_dir"
  "-DCMAKE_INSTALL_PREFIX=$install_dir"
  "-DCMAKE_BUILD_TYPE=$config"
  "-DCMAKE_OSX_ARCHITECTURES=$arch"
  "-DPERFECTHASH_ENABLE_TESTS=ON"
  "-DBUILD_TESTING=ON"
  "-DPERFECTHASH_ENABLE_NATIVE_ARCH=${native_arch}"
)

if [ -n "$generator" ]; then
  cmake_args=(-G "$generator" "${cmake_args[@]}")
fi

ph_run cmake "${cmake_args[@]}"
ph_run cmake --build "$build_dir" --config "$config" --parallel

if [ "$skip_tests" != "1" ]; then
  if [ ! -f "${PH_ROOT_DIR}/keys/HologramWorld-31016.keys" ]; then
    ph_die "missing keys/HologramWorld-31016.keys for tests"
  fi
  ph_run ctest --test-dir "$build_dir" --output-on-failure -C "$config" --timeout 300
fi

if [ "$skip_install" != "1" ]; then
  ph_run cmake --install "$build_dir" --config "$config"
fi

if [ "$skip_package" != "1" ]; then
  ph_rm_rf "$(dirname "$stage_dir")"
  ph_run mkdir -p "$stage_dir"
  ph_run cp -a "$install_dir"/. "$stage_dir"/
  ph_copy_docs "$stage_dir"
  ph_run mkdir -p "$dist_dir"
  package_name="perfecthash-${release_version}-${platform}"
  tarball="${dist_dir}/${package_name}.tar.gz"
  COPYFILE_DISABLE=1 ph_run tar -C "$(dirname "$stage_dir")" -czf "$tarball" "$(basename "$stage_dir")"
  ph_write_sha256 "$tarball"
fi
