#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

check_timeout="${ROBOREV_CHECK_TIMEOUT:-30}"
required_agents=(codex claude-code)

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

require_cmd git
require_cmd roborev

printf '%s\n' "Configuring worktree-local Git hooks..."
git config extensions.worktreeConfig true
git config --worktree core.hooksPath .githooks

printf '%s\n' "Installing or updating Roborev skills..."
roborev skills install

printf '%s\n' "Ensuring Roborev daemon is running..."
roborev daemon start >/dev/null 2>&1 || true

status=0
for agent in "${required_agents[@]}"; do
  printf 'Checking agent: %s\n' "${agent}"
  if ! roborev check-agents --agent "${agent}" --timeout "${check_timeout}"; then
    status=1
  fi
done

printf '\n%s\n' "Current Roborev status:"
roborev status || true

printf '\n%s\n' "Git hook resolution:"
git config --show-origin --get core.hooksPath || true
git rev-parse --git-path hooks/pre-commit
git rev-parse --git-path hooks/post-commit
git rev-parse --git-path hooks/post-rewrite

printf '\n%s\n' "Suggested next commands:"
printf '  %s\n' "roborev show HEAD"
printf '  %s\n' "scripts/roborev-matrix-review.sh HEAD^..HEAD"

if [[ "${status}" -ne 0 ]]; then
  printf '\n%s\n' "One or more required agents are missing or unhealthy. Install/authenticate them, then rerun this script." >&2
  exit "${status}"
fi
