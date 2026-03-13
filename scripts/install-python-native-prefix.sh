#!/usr/bin/env bash

set -euo pipefail

root_dir="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"

prefix="${PERFECTHASH_PREFIX:-${root_dir}/.perfecthash-prefix}"
configure_preset="${PERFECTHASH_CONFIGURE_PRESET:-ninja-multi-full}"
build_preset="${PERFECTHASH_BUILD_PRESET:-ninja-full-release}"
build_dir="${PERFECTHASH_BUILD_DIR:-${root_dir}/build-full}"
install_config="${PERFECTHASH_INSTALL_CONFIG:-Release}"

printf 'PerfectHash Python native prefix install\n'
printf '  root: %s\n' "${root_dir}"
printf '  prefix: %s\n' "${prefix}"
printf '  configure preset: %s\n' "${configure_preset}"
printf '  build preset: %s\n' "${build_preset}"
printf '  build dir: %s\n' "${build_dir}"
printf '  install config: %s\n' "${install_config}"

cmake --preset "${configure_preset}"
cmake --build --preset "${build_preset}"
cmake --install "${build_dir}" --config "${install_config}" --prefix "${prefix}"

cat <<EOF

Native artifacts installed to:
  ${prefix}

Recommended next steps:
  export PERFECTHASH_PREFIX="${prefix}"
  env -u PYTHONPATH uv sync

Optional smoke checks:
  env -u PYTHONPATH uv run python -m perfecthash create --help
  env -u PYTHONPATH uv run python - <<'PY'
from perfecthash import build_table
with build_table([1, 3, 5, 7], hash_function="MultiplyShiftR") as table:
    print(table.index(3))
PY
EOF
