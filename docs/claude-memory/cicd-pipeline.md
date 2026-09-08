---
name: cicd-pipeline
description: "How the GitHub CI/CD pipeline is wired (ruleset, REVIEWER switch, auth) — config that lives in GitHub settings, not the repo"
metadata: 
  node_type: memory
  type: project
  originSessionId: 8418c869-b00e-4acc-b7ee-04d461835e5c
  modified: 2026-08-30T19:19:56.086Z
---

CI/CD for `Axzyo/Llm-Sandbox` was set up 2026-08-29. Much of this lives in
GitHub *settings*, not the repo, so it isn't discoverable from the code.

**Repo went PUBLIC** (2026-08-29) — required to enforce branch protection on the
free plan (private repos need Pro). Secret scan was clean before flipping (local
Ollama only, no API keys in code).

**Gate on `main`** — GitHub ruleset "main protection", id **21782551**, active:
- PR required (no direct push), no force-push, no branch deletion
- Required status checks: **`smoke-test`** + **`claude-review`** (strict/up-to-date)
- **0 required approvals** -> merges auto once both checks are green, no bypass
  needed. (History: started at 1, but on a solo repo 1-approval forced admin
  bypass every merge, and bypass skips the whole ruleset incl. checks -> soft
  gate. 0 approvals makes the status checks a true hard gate.)
- Bypass actor: Repository admin role (`actor_id 5`, always) — escape hatch.

**Reviewer switch** = repo variable **`REVIEWER`** (claude|copilot|both|off),
read by `.github/workflows/claude-review.yml`:
- `claude` (current) — Claude review runs, gates the merge.
- `copilot`/`off` — the claude-review job runs but the review step is skipped via
  `if:`, so the job reports green as a no-op (required check never blocks).
- Switch with `gh variable set REVIEWER --body <value>` (no code change).

**Auth = API key.** Secret **`ANTHROPIC_API_KEY`** (pay-per-use Console billing,
~cents/PR). The workflow passes it as the `anthropic_api_key` input to
`anthropics/claude-code-action@v1`.

**Model pinned to Sonnet 5** via `claude_args: --model claude-sonnet-5` (2026-08-30,
cost cut). Was briefly Opus 4.8 (user finds Opus 5 inconsistent; action default is
Opus 5). Sonnet 5 is ~60% cheaper and still catches cross-file issues.
**Cost controls (2026-08-30):** trigger is `types: [opened, reopened]` (NOT
`synchronize`) so pushing fixes does NOT re-review — the 3-4 re-reviews/merge were
the main spend (~$2/merge, ~$5/day on Opus). Close+reopen to force a fresh review.
Plus `concurrency: cancel-in-progress` cancels a superseded run.
NOTE: the action SKIPS (green no-op, ~8s) on any PR that edits `claude-review.yml`
itself (anti-tampering: workflow must match the default branch) — such changes only
take effect once merged. Console usage is per-model; Opus cost looked low (~$4 for
~2M tokens) because most input is cache reads (0.1x) from re-reading repo each run. The **Claude GitHub App**
(github.com/apps/claude) is installed on the repo — required for the action's
GitHub-side identity; without it the run fails "Claude Code is not installed".
Gotcha hit twice: `gh secret set` via the interactive paste-prompt saved the
value EMPTY (action failed "ANTHROPIC_API_KEY ... is required"); the reliable fix
was `gh secret set ANTHROPIC_API_KEY --body "<key>"`.
- Copilot review was tried (`copilot_code_review` ruleset rule) but never fired —
  the account has no Copilot code-review access; rule was removed. OAuth-token
  auth (`claude setup-token`) was abandoned: no standalone Claude Code CLI on
  PATH (only the desktop app), so API key was chosen instead.

**Human process** is documented in `.github/CONTRIBUTING.md`; standards enforced
by the review are in `CLAUDE.md` (see [[code-quality-rules]]).

**Why:** the pipeline's source of truth is split between repo files and GitHub
settings; this captures the settings half + the traps hit.
**How to apply:** to change the gate, edit ruleset 21782551 (PUT via `gh api`);
to change reviewer, set the `REVIEWER` variable; to re-auth, use `gh secret set
... --body`. Verify changes with a throwaway PR.
