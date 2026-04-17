# Local Roborev Workflow For PerfectHash

Status for this machine as of 2026-04-16:

- `roborev v0.46.1` is installed and the daemon is healthy.
- `claude-code` and `codex` pass `roborev check-agents`.
- automatic post-commit review is configured for this repo via `.githooks`.
- the normal queued review path prefers `claude-code` with `codex` as backup.
- `codex` is the default fixer/refiner agent.

## Important Local Limitation

The automatic post-commit Roborev path is a queued review, not the full
multi-agent matrix. Use it for automatic per-commit coverage, then use the
explicit matrix gate when you want a stronger synchronous check.

## Standard Commands

Automatic review loop:

```bash
roborev wait
roborev show HEAD
```

Explicit commit-level matrix gate:

```bash
scripts/roborev-matrix-review.sh HEAD^..HEAD
```

Explicit branch-level gate against the merge-base with the default branch:

```bash
scripts/roborev-matrix-review.sh
```

Stable synthesized review artifact:

```bash
.roborev/last-review.md
```

Useful supporting commands:

```bash
roborev status
roborev list
roborev fix --open
roborev refine --since HEAD~1
```
