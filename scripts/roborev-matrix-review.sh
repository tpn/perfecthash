#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

ref="${1:-}"
if [[ -z "${ref}" ]]; then
  default_branch="$(git symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null || true)"
  default_branch="${default_branch#origin/}"

  if [[ -n "${default_branch}" ]] && git rev-parse --verify --quiet "origin/${default_branch}" >/dev/null; then
    base_ref="origin/${default_branch}"
  elif git rev-parse --verify --quiet origin/main >/dev/null; then
    base_ref="origin/main"
  elif git rev-parse --verify --quiet main >/dev/null; then
    base_ref="main"
  else
    printf '%s\n' "Could not resolve a default branch ref. Pass an explicit range like HEAD^..HEAD." >&2
    exit 1
  fi

  merge_base="$(git merge-base "${base_ref}" HEAD)"
  ref="${merge_base}..HEAD"
fi

agents="${ROBOREV_AGENTS:-codex,claude-code}"
review_types="${ROBOREV_REVIEW_TYPES:-default}"
reasoning="${ROBOREV_REASONING:-thorough}"
min_severity="${ROBOREV_MIN_SEVERITY:-medium}"
synthesis_agent="${ROBOREV_SYNTHESIS_AGENT:-claude-code}"
output_path="${ROBOREV_OUTPUT_PATH:-${repo_root}/.roborev/last-review.md}"
gate_timeout="${ROBOREV_GATE_TIMEOUT:-}"

mkdir -p "$(dirname "${output_path}")"

printf '%s\n' "Running roborev matrix review"
printf '  %s\n' "ref: ${ref}"
printf '  %s\n' "agents: ${agents}"
printf '  %s\n' "review types: ${review_types}"
printf '  %s\n' "reasoning: ${reasoning}"
printf '  %s\n' "min severity: ${min_severity}"
printf '  %s\n' "synthesis agent: ${synthesis_agent}"
printf '  %s\n' "output: ${output_path}"
if [[ -n "${gate_timeout}" ]]; then
  printf '  %s\n' "gate timeout: ${gate_timeout}"
fi

cmd=(
  roborev ci review
  --ref "${ref}"
  --agent "${agents}"
  --review-types "${review_types}"
  --synthesis-agent "${synthesis_agent}"
  --reasoning "${reasoning}"
  --min-severity "${min_severity}"
)

if [[ -n "${gate_timeout}" ]] && command -v timeout >/dev/null 2>&1; then
  timeout --foreground "${gate_timeout}" "${cmd[@]}" | tee "${output_path}"
else
  "${cmd[@]}" | tee "${output_path}"
fi
