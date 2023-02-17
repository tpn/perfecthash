#/bin/zsh

# Usage: git rebase -x git-rebase-update-copyrights.zsh $(git merge-base HEAD origin/main)
# Requires this bin directory to be on your path.
_hash=$(git rev-parse HEAD)
_files=$(git diff --name-only $_hash^1 2>&1)
for _f in $_files; do
    ph update-copyright -p $_f
done
git add -u
git commit --amend --no-edit
