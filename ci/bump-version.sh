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
Usage: ci/bump-version.sh --version X.Y.Z [options]

Options:
  --version <ver>   New version number (required)
  --commit          Create a git commit after updating
  --tag             Create an annotated tag "v<ver>" after updating
  --allow-dirty     Allow running with a dirty git status
  --dry-run         Print actions without modifying files
  -h, --help        Show this help
EOF
}

new_version=""
do_commit=0
do_tag=0
allow_dirty=0
dry_run=0

while [ $# -gt 0 ]; do
  case "$1" in
    --version)
      new_version="$2"
      shift 2
      ;;
    --commit)
      do_commit=1
      shift
      ;;
    --tag)
      do_tag=1
      shift
      ;;
    --allow-dirty)
      allow_dirty=1
      shift
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
      printf 'error: unknown argument: %s\n' "$1" >&2
      exit 1
      ;;
  esac
done

if [ -z "$new_version" ]; then
  usage
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  printf 'error: python3 is required for version updates\n' >&2
  exit 1
fi

if command -v git >/dev/null 2>&1; then
  if [ "$allow_dirty" -ne 1 ]; then
    if [ -n "$(git -C "$root_dir" status --porcelain)" ]; then
      printf 'error: git working tree is dirty (use --allow-dirty to override)\n' >&2
      exit 1
    fi
  fi
fi

if [ "$dry_run" -eq 1 ]; then
  printf 'dry-run: would update CMakeLists.txt to VERSION %s\n' "$new_version"
else
  python3 - "$root_dir/CMakeLists.txt" "$new_version" <<'PY'
import io
import re
import sys

path = sys.argv[1]
version = sys.argv[2]

with open(path, "r", encoding="utf-8") as fh:
    data = fh.read()

pattern = re.compile(r"^(\s*VERSION\s+)([0-9][0-9.\-]*)(\s*)$",
                     re.MULTILINE)
if not pattern.search(data):
    raise SystemExit("error: VERSION line not found in CMakeLists.txt")

data = pattern.sub(rf"\g<1>{version}\g<3>", data, count=1)

with open(path, "w", encoding="utf-8", newline="") as fh:
    fh.write(data)
PY
fi

if [ "$do_commit" -eq 1 ]; then
  if ! command -v git >/dev/null 2>&1; then
    printf 'error: git is required for --commit\n' >&2
    exit 1
  fi
  if [ "$dry_run" -eq 1 ]; then
    printf 'dry-run: would commit version bump\n'
  else
    git -C "$root_dir" add CMakeLists.txt
    git -C "$root_dir" commit -m "Bump version to ${new_version}"
  fi
fi

if [ "$do_tag" -eq 1 ]; then
  if ! command -v git >/dev/null 2>&1; then
    printf 'error: git is required for --tag\n' >&2
    exit 1
  fi
  if [ "$dry_run" -eq 1 ]; then
    printf 'dry-run: would create tag v%s\n' "$new_version"
  else
    git -C "$root_dir" tag -a "v${new_version}" -m "v${new_version}"
  fi
fi
