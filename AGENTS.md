# Agent rules / constitution

Rules any AI coding agent (Claude Code, Cursor, Cline, etc.) must follow when working in this repository. If your tool expects a different filename, copy this content into `.clinerules` or `.cursorrules` as well — the content matters more than the filename.

## Scope
Applies to all AI-assisted changes in this repo, whether run interactively or autonomously.

## Non-negotiable rules
1. Never commit secrets, API keys, or `.env` files. `.env.example` is for documentation only.
2. Every change that touches application logic must include or update a test.
3. No force-push to `main`. Feature branches only, merged via PR.
4. Run the linter and test suite locally before committing. A red pipeline blocks merge.
5. Commit messages follow Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:`, `test:`.

## Workflow the agent should follow
1. Read `SPEC.md` and `TASKS.md` before starting a task.
2. Implement the smallest working change for the task.
3. Run tests and linter.
4. Update documentation if behavior changed.
5. Commit with a clear, scoped message.

## Things the agent must NOT do without explicit human approval
- Delete or rewrite existing tests just to make them pass.
- Change the CI/CD workflow files.
- Introduce a new external dependency.

## Escalation
If a task is ambiguous or underspecified, the agent should stop and ask rather than guessing.
