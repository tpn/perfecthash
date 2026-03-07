#!/usr/bin/env bash
set -euo pipefail

script_dir="$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)"
root_dir="$(
  cd "${script_dir}/../../.."
  pwd
)"

usage() {
  cat <<'EOF'
Usage: conda-cycle.sh [options]

Build, smoke-test, and optionally upload PerfectHash conda packages.

Options:
  --version <ver>       Package version override (default: latest git tag or 0.0.0)
  --output-folder <p>   Conda output folder (default: conda/out)
  --skip-build          Skip conda build steps
  --skip-local-test     Skip local file:// channel smoke test
  --upload              Upload built artifacts to anaconda.org
  --remote-test         Smoke test install from remote org channel
  --user <name>         Upload/channel user (default: $ANACONDA_UPLOAD_USER or perfecthash)
  --dry-run             Print commands without executing them
  -h, --help            Show this help
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

run() {
  local previous=""
  local rendered=""
  local arg=""
  printf '+'
  for arg in "$@"; do
    rendered="${arg}"
    if [ "${previous}" = "-t" ] || [ "${previous}" = "--token" ]; then
      rendered="***REDACTED***"
    elif [ -n "${ANACONDA_API_TOKEN:-}" ] && [ "${arg}" = "${ANACONDA_API_TOKEN}" ]; then
      rendered="***REDACTED***"
    fi
    printf ' %q' "${rendered}"
    previous="${arg}"
  done
  printf '\n'
  if [ "${dry_run}" = "1" ]; then
    return 0
  fi
  "$@"
}

cleanup_prefix() {
  local prefix="$1"
  if [ -z "${prefix}" ] || [ ! -d "${prefix}" ]; then
    return 0
  fi
  conda env remove -y -p "${prefix}" >/dev/null 2>&1 || rm -rf "${prefix}"
}

make_temp_prefix() {
  local label="$1"
  if [ "${dry_run}" = "1" ]; then
    printf '/tmp/%s\n' "${label}"
  else
    mktemp -d "${TMPDIR:-/tmp}/perfecthash-${label}-XXXXXX"
  fi
}

version="${PERFECTHASH_CONDA_VERSION:-}"
output_folder="conda/out"
skip_build=0
skip_local_test=0
upload=0
remote_test=0
dry_run=0
upload_user="${ANACONDA_UPLOAD_USER:-perfecthash}"
local_prefix=""
remote_prefix=""

while [ $# -gt 0 ]; do
  case "$1" in
    --version)
      version="$2"
      shift 2
      ;;
    --output-folder)
      output_folder="$2"
      shift 2
      ;;
    --skip-build)
      skip_build=1
      shift
      ;;
    --skip-local-test)
      skip_local_test=1
      shift
      ;;
    --upload)
      upload=1
      shift
      ;;
    --remote-test)
      remote_test=1
      shift
      ;;
    --user)
      upload_user="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [ -z "${version}" ]; then
  tag="$(git -C "${root_dir}" describe --tags --abbrev=0 --match 'v[0-9]*' 2>/dev/null || true)"
  if [ -n "${tag}" ]; then
    version="${tag#v}"
  else
    version="0.0.0"
  fi
fi

case "${version}" in
  v*) version="${version#v}" ;;
esac

case "${output_folder}" in
  /*) abs_output_folder="${output_folder}" ;;
  *) abs_output_folder="${root_dir}/${output_folder}" ;;
esac

if [ "${upload}" = "1" ] && [ "${dry_run}" != "1" ] && [ -z "${ANACONDA_API_TOKEN:-}" ]; then
  die "ANACONDA_API_TOKEN is required for --upload"
fi

if command -v mamba >/dev/null 2>&1; then
  solver="mamba"
else
  solver="conda"
fi

cleanup() {
  if [ "${dry_run}" != "1" ]; then
    cleanup_prefix "${local_prefix}"
    cleanup_prefix "${remote_prefix}"
  fi
}

trap cleanup EXIT

if [ "${skip_build}" != "1" ]; then
  run env PERFECTHASH_CONDA_VERSION="${version}" \
    conda build "${root_dir}/conda/recipe" --output
  run env PERFECTHASH_CONDA_VERSION="${version}" \
    conda build "${root_dir}/conda/recipe" \
      --output-folder "${abs_output_folder}" \
      --no-anaconda-upload
fi

if [ "${skip_local_test}" != "1" ]; then
  local_prefix="$(make_temp_prefix "conda-local-test")"
  local_channel="file://${abs_output_folder}"
  run conda index "${abs_output_folder}"
  run "${solver}" create -y -p "${local_prefix}" -c "${local_channel}" -c conda-forge perfecthash
  run conda run -p "${local_prefix}" bash -lc \
    'test -f "$CONDA_PREFIX/include/PerfectHash/PerfectHash.h" && test -f "$CONDA_PREFIX/lib/cmake/PerfectHash/PerfectHashConfig.cmake"'
fi

if [ "${upload}" = "1" ]; then
  if [ "${dry_run}" = "1" ]; then
    artifacts=(
      "${abs_output_folder}/linux-64/perfecthash-0.0.0-0.conda"
      "${abs_output_folder}/linux-64/perfecthash-full-0.0.0-0.conda"
    )
  else
    mapfile -t artifacts < <(find "${abs_output_folder}" -type f \( -name '*.conda' -o -name '*.tar.bz2' \) | sort)
    if [ "${#artifacts[@]}" -eq 0 ]; then
      die "no conda artifacts found under ${abs_output_folder}"
    fi
  fi
  run anaconda -t "${ANACONDA_API_TOKEN:-}" upload \
    --user "${upload_user}" \
    --label main \
    --skip-existing \
    "${artifacts[@]}"
fi

if [ "${remote_test}" = "1" ]; then
  remote_prefix="$(make_temp_prefix "conda-remote-test")"
  run "${solver}" create -y -p "${remote_prefix}" -c "${upload_user}" -c conda-forge perfecthash
  run conda run -p "${remote_prefix}" bash -lc \
    'test -f "$CONDA_PREFIX/include/PerfectHash/PerfectHash.h" && test -f "$CONDA_PREFIX/lib/cmake/PerfectHash/PerfectHashConfig.cmake"'
fi
