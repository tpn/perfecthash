#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." >/dev/null 2>&1 && pwd)"

DIM_SIZE="${DIM_SIZE:-50000}"
FACT_SIZE="${FACT_SIZE:-200000}"
ITERATIONS="${ITERATIONS:-1}"
BUILD_RUNS="${BUILD_RUNS:-20}"

RESULTS_DIR="${1:-${EXAMPLE_DIR}/results/latest}"
mkdir -p "${RESULTS_DIR}"

BIN_CANDIDATES=(
  "${REPO_ROOT}/build/examples/sqlite-online-jit/sqlite-online-jit"
  "${REPO_ROOT}/build/examples/sqlite-online-jit/Release/sqlite-online-jit"
)

BIN_PATH=""
for candidate in "${BIN_CANDIDATES[@]}"; do
  if [[ -x "${candidate}" ]]; then
    BIN_PATH="${candidate}"
    break
  fi
done

if [[ -z "${BIN_PATH}" ]]; then
  echo "Unable to find sqlite-online-jit binary. Build the example first:" >&2
  echo "  cmake -S examples/sqlite-online-jit -B build/examples/sqlite-online-jit -DPERFECTHASH_ROOT=${REPO_ROOT}" >&2
  echo "  cmake --build build/examples/sqlite-online-jit --parallel" >&2
  exit 1
fi

SUMMARY_CSV="${RESULTS_DIR}/summary.csv"
DETAILED_CSV="${RESULTS_DIR}/detailed.csv"
LOG_PATH="${RESULTS_DIR}/run.log"

{
  echo "[sqlite-online-jit] binary: ${BIN_PATH}"
  echo "[sqlite-online-jit] results: ${RESULTS_DIR}"
  echo "[sqlite-online-jit] dim-size=${DIM_SIZE} fact-size=${FACT_SIZE} iterations=${ITERATIONS} build-runs=${BUILD_RUNS}"

  "${BIN_PATH}" \
    --matrix \
    --dim-size "${DIM_SIZE}" \
    --fact-size "${FACT_SIZE}" \
    --iterations "${ITERATIONS}" \
    --build-runs "${BUILD_RUNS}" \
    --output-detailed-csv "${DETAILED_CSV}" \
    --output-summary-csv "${SUMMARY_CSV}"
} | tee "${LOG_PATH}"

cat <<DONE

Generated:
  - ${SUMMARY_CSV}
  - ${DETAILED_CSV}
  - ${LOG_PATH}

Open notebook:
  jupyter notebook ${EXAMPLE_DIR}/notebooks/sqlite_online_jit_matrix_analysis.ipynb
DONE
