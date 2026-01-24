#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
CONFIG="${CONFIG:-Debug}"
KEYS_DIR="${KEYS_DIR:-../perfecthash-keys/sys32}"
OUTPUT_DIR="${OUTPUT_DIR:-${BUILD_DIR}/stress-sys32}"
ALGORITHM="${ALGORITHM:-Chm01}"
HASH_FUNCTION="${HASH_FUNCTION:-Mulshrolate4RX}"
MASK_FUNCTION="${MASK_FUNCTION:-And}"

if command -v nproc >/dev/null 2>&1; then
    MAX_CONCURRENCY="${MAX_CONCURRENCY:-$(nproc)}"
else
    MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
fi

EXE="${BUILD_DIR}/bin/${CONFIG}/PerfectHashBulkCreate"
if [[ ! -x "${EXE}" ]]; then
    EXE="${BUILD_DIR}/bin/PerfectHashBulkCreate"
fi

if [[ ! -x "${EXE}" ]]; then
    echo "Bulk create exe not found: ${EXE}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

"${EXE}" \
    "${KEYS_DIR}" \
    "${OUTPUT_DIR}" \
    "${ALGORITHM}" \
    "${HASH_FUNCTION}" \
    "${MASK_FUNCTION}" \
    "${MAX_CONCURRENCY}" \
    "$@"
