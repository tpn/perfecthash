# Roborev Playbook For PerfectHash

This is the recommended review loop for Codex-driven work in this repo.

## Recommendation

Use both layers:

- automatic post-commit queued review for every commit,
- explicit two-agent matrix review before push or whenever a contained change
  needs a synchronous gate.

The queued review gives cheap automatic coverage immediately after commit. The
matrix script is the stronger local gate.

## Formal Loop

For each contained piece of work:

1. Read `ROBOREV-CONTEXT.md` and `AGENTS.md`.
2. Implement the contained change.
3. Run targeted validation for the touched scope.
4. Commit locally.
5. Wait for the automatic review to finish:

```bash
roborev wait
```

6. Inspect the latest review:

```bash
roborev show HEAD
```

7. If the change is substantial, user-facing, or about to be pushed, run the
   explicit matrix gate:

```bash
scripts/roborev-matrix-review.sh HEAD^..HEAD
```

8. Read the synthesized summary in `.roborev/last-review.md`.
9. Fix actionable findings, rerun targeted validation, and commit again.
10. Repeat until the relevant review comes back clean.

## Final Pre-Push Gate

Before push or PR update, run the branch-level matrix gate:

```bash
scripts/roborev-matrix-review.sh
```

That reviews the full branch diff from the merge-base with the default branch
and refreshes `.roborev/last-review.md`.

## Matrix Defaults

The matrix script defaults to:

- reviewers: `codex`, `claude-code`
- synthesis agent: `claude-code`
- review type: `default`
- reasoning: `thorough`
- minimum severity: `medium`

You can override these for a run with environment variables such as
`ROBOREV_AGENTS`, `ROBOREV_SYNTHESIS_AGENT`, or `ROBOREV_GATE_TIMEOUT`.
