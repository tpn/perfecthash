# Roborev Bootstrap For PerfectHash

Use this file for new machines, fresh clones, or agent-driven environment
setup.

## Strategy

Split the setup into two layers:

1. Machine-global prerequisites:
   - install `roborev`,
   - install the agent CLIs you want Roborev to call,
   - authenticate those CLIs,
   - ensure `roborev check-agents` passes.
2. Repo-local activation:
   - point Git hooks at this repo's tracked `.githooks`,
   - install/update Roborev skills for Codex and Claude,
   - verify daemon and agent health from this clone.

This repo already tracks:

- `.roborev.toml`
- `.githooks/pre-commit`
- `.githooks/post-commit`
- `.githooks/post-rewrite`
- `scripts/roborev-matrix-review.sh`
- `ROBOREV-CONTEXT.md`
- `ROBOREV-LOCAL.md`
- `ROBOREV-PLAYBOOK.md`

Because of that, prefer the local bootstrap script below over `roborev init`.
`roborev init` is useful for greenfield repos, but here it would generate
parallel hook/config state we do not need.

## Fast Path

After `roborev`, `codex`, and `claude` are installed and authenticated on the
machine, run:

```bash
scripts/bootstrap-roborev-local.sh
```

That script:

- enables worktree-local Git config,
- sets `core.hooksPath=.githooks`,
- installs or updates Roborev skills,
- starts the Roborev daemon if needed,
- checks `codex` and `claude-code`,
- prints the next verification commands.

## Verification

From the repo root:

```bash
roborev status
roborev check-agents --agent codex --timeout 30
roborev check-agents --agent claude-code --timeout 30
scripts/roborev-matrix-review.sh HEAD^..HEAD
```

## Agent-Oriented Prompt

If you want another agent to perform setup, point it at this file and ask it
to:

1. install machine prerequisites if missing,
2. run `scripts/bootstrap-roborev-local.sh`,
3. confirm `core.hooksPath` resolves to `.githooks`,
4. confirm `roborev check-agents` passes for `codex` and `claude-code`,
5. run a smoke-test review and report the result.
