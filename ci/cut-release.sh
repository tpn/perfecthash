#!/usr/bin/env bash
set -euo pipefail

script_dir="$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)"
root_dir="$(
  cd "${script_dir}/.."
  pwd
)"

usage() {
  cat <<'EOF'
Usage: ci/cut-release.sh --version X.Y[.Z[.W]] [options]

Options:
  --version <ver>      Release version (required, with or without leading "v")
  --push               Push tag to remote after creating it
  --remote <name>      Remote name for push/check (default: origin)
  --allow-dirty        Allow a dirty working tree
  --notes-output <p>   Release notes output path
  --dry-run            Print actions without modifying git state
  -h, --help           Show this help
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

run_cmd() {
  if [ "$dry_run" = "1" ]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

version=""
push_tag=0
remote_name="origin"
allow_dirty=0
dry_run=0
notes_output=""

while [ $# -gt 0 ]; do
  case "$1" in
    --version)
      version="$2"
      shift 2
      ;;
    --push)
      push_tag=1
      shift
      ;;
    --remote)
      remote_name="$2"
      shift 2
      ;;
    --allow-dirty)
      allow_dirty=1
      shift
      ;;
    --notes-output)
      notes_output="$2"
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

[ -n "$version" ] || die "--version is required"

if ! command -v git >/dev/null 2>&1; then
  die "git is required"
fi

version="${version#v}"
if ! printf '%s' "$version" | grep -Eq '^[0-9]+(\.[0-9]+){1,3}$'; then
  die "invalid version '$version' (expected X.Y, X.Y.Z, or X.Y.Z.W)"
fi
tag="v${version}"

if [ "$allow_dirty" != "1" ]; then
  if [ -n "$(git -C "$root_dir" status --porcelain)" ]; then
    die "git working tree is dirty (use --allow-dirty to override)"
  fi
fi

if git -C "$root_dir" rev-parse --verify --quiet "refs/tags/${tag}" >/dev/null; then
  die "tag already exists locally: ${tag}"
fi

if [ "$push_tag" = "1" ]; then
  if ! git -C "$root_dir" remote get-url "$remote_name" >/dev/null 2>&1; then
    die "remote '${remote_name}' does not exist"
  fi
  if git -C "$root_dir" ls-remote --exit-code --tags "$remote_name" "refs/tags/${tag}" >/dev/null 2>&1; then
    die "tag already exists on remote '${remote_name}': ${tag}"
  fi
fi

if [ -z "$notes_output" ]; then
  notes_output="${root_dir}/out/release/${version}/release-notes-${tag}.md"
fi
notes_dir="$(dirname "$notes_output")"
if [ "$dry_run" = "1" ]; then
  printf '+ mkdir -p %q\n' "$notes_dir"
else
  mkdir -p "$notes_dir"
fi

if [ -x "${root_dir}/ci/generate-release-notes.sh" ]; then
  run_cmd "${root_dir}/ci/generate-release-notes.sh" --version "$version" --output "$notes_output"
else
  printf 'warning: ci/generate-release-notes.sh not found/executable; skipping notes generation\n' >&2
fi

run_cmd git -C "$root_dir" tag -a "$tag" -m "$tag"

if [ "$push_tag" = "1" ]; then
  run_cmd git -C "$root_dir" push "$remote_name" "refs/tags/${tag}"
  printf 'release tag pushed: %s (%s)\n' "$tag" "$remote_name"
else
  printf 'release tag created locally: %s\n' "$tag"
  printf 'push with: git -C %q push %q refs/tags/%s\n' "$root_dir" "$remote_name" "$tag"
fi

printf 'release notes: %s\n' "$notes_output"
