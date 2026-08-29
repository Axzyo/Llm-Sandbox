# CI/CD pipeline

How code reaches `main` in this repo, and how the automated checks are wired.
This documents the setup as of 2026-08-29. The human-facing workflow summary
lives in [`.github/CONTRIBUTING.md`](../.github/CONTRIBUTING.md); the coding
standards the reviewer enforces live in [`CLAUDE.md`](../CLAUDE.md).

## Overview

Every change reaches `main` through a pull request that must pass two automated
checks. There are no direct pushes to `main`.

```
feature branch  ->  PR into main  ->  [ smoke-test ] + [ claude-review ]  ->  merge on green
```

Enforcement is a GitHub **repository ruleset** ("main protection"), not the older
branch-protection UI. The repo is **public**, which is what makes ruleset
enforcement available on the free plan.

## The gate (ruleset "main protection")

Applied to the default branch (`main`):

| Rule | Effect |
| ---- | ------ |
| Require a pull request | No direct pushes; all commits arrive via PR |
| Require status checks | **`smoke-test`** and **`claude-review`** must pass (strict: branch must be up to date) |
| Required approvals | **0** — merges automatically once checks are green |
| Block force pushes | No history rewrites on `main` |
| Restrict deletions | `main` cannot be deleted |
| Bypass | Repository admins may bypass in an emergency |

Because required approvals is 0, the **status checks are the real gate**: a PR
merges as soon as both are green, with no manual approval and no admin bypass
needed. Nothing lands on `main` without passing tests *and* being reviewed by the
agent.

> Note: `claude-review` going green means Claude successfully ran and posted its
> review, not that it "approved." Its findings are advisory comments you read on
> the PR; a PR simply cannot merge without that review having run.

## The two checks

### `smoke-test` (`.github/workflows/ci.yml`)

Runs on every PR to `main` (and on pushes to non-`main` branches). Installs
`requirements.txt` and runs `python smoke_test.py`. Must print `SMOKE OK`.

### `claude-review` (`.github/workflows/claude-review.yml`)

Runs `anthropics/claude-code-action@v1` on every PR. Claude reads the full diff
and reviews it against `CLAUDE.md` and `DESIGN.md`, posting a summary comment
(and inline comments on specific lines). It focuses on the project's rules:
no hardcoded behavioral prose, reuse of existing systems, no dead code, no
premature complexity, strict validation, and test coverage for changed behavior.

## Switching the reviewer

The active reviewer is controlled by the repository **variable `REVIEWER`**:

| `REVIEWER` | `claude-review` check | Copilot |
| ---------- | --------------------- | ------- |
| `claude` (current) | Claude reviews, gates the merge | off |
| `both`     | Claude reviews, gates the merge | on (if enabled in settings) |
| `copilot`  | stands down (job passes as a no-op) | on |
| `off`      | stands down (job passes as a no-op) | off |

Change it with no code edit:

```bash
gh variable set REVIEWER --body claude   # or copilot / both / off
```

When Claude stands down, the `claude-review` job still runs but skips the review
step and reports green, so the required check never blocks a merge.

> Copilot code review is a separate GitHub setting and requires Copilot access on
> the account. It was configured but does not run here because the account has no
> Copilot code-review access, so `claude` is the active reviewer.

## Authentication

`claude-review` authenticates to Anthropic with the repository secret
**`ANTHROPIC_API_KEY`** (pay-per-use, billed from Anthropic Console credits,
roughly cents per PR). The **Claude GitHub App**
(https://github.com/apps/claude) must be installed on the repo — it provides the
action's GitHub-side identity for posting comments.

To (re)set the key, use the `--body` form — the interactive paste prompt has been
observed to save an empty value:

```bash
gh secret set ANTHROPIC_API_KEY -R Axzyo/Llm-Sandbox --body "sk-ant-..."
```

## Day-to-day workflow

```bash
git switch -c feat/thing          # one branch per feature; never commit to main
# ...work, commit...
git push -u origin feat/thing
gh pr create --base main --fill   # smoke-test + Claude review run automatically
# wait for both checks to go green, then:
gh pr merge --squash --delete-branch
```

## Changing the setup

- **Gate rules** (checks, approvals, protections): edit the ruleset via the API —
  `gh api --method PUT repos/Axzyo/Llm-Sandbox/rulesets/<id>`.
- **Reviewer**: set the `REVIEWER` variable (above).
- **Credentials**: `gh secret set ... --body`.

Verify any change with a throwaway PR before relying on it.
