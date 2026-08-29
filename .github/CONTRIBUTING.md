# Contributing / Branch workflow

`main` is protected. All work follows this flow:

1. **Every feature gets its own branch** off `main`
   (e.g. `feat/inventory-effects`, `fix/spatial-decay`). Never commit to `main`.
2. Push the branch and open a **pull request into `main`**.
3. On every PR, two required checks run automatically:
   - **`smoke-test`** — runs `python smoke_test.py` (must pass).
   - **`claude-review`** — an AI agent reviews the diff against `CLAUDE.md` /
     `DESIGN.md` and posts inline + summary comments.
4. A PR can only merge once checks pass and the review has been addressed and
   approved. **Every merge is reviewed by the agent before acceptance.**

## Choosing the reviewer (Claude vs Copilot)

The active reviewer is controlled by the repo variable **`REVIEWER`**:

| `REVIEWER` value | Claude (`claude-review` check) | Copilot |
| ---------------- | ------------------------------ | ------- |
| `claude` (or unset) | runs, gates the merge        | off     |
| `both`           | runs, gates the merge          | add on (see below) |
| `copilot`        | stands down (check passes no-op) | on    |
| `off`            | stands down (check passes no-op) | off     |

Switch it any time (no code change, takes effect on the next PR event):

```
gh variable set REVIEWER --body copilot     # or claude / both / off
```

When Claude stands down, the `claude-review` job still runs but does nothing and
reports green, so the required check never blocks a merge.

**Copilot side** is a separate GitHub setting (it posts advisory review comments,
it is not a status check): enable *Settings -> Code review -> "Automatically
request Copilot review"* (needs Copilot access on the account). Turn it on for
`both`/`copilot`, off otherwise.

## One-time setup (repo owner)

- Install the **Claude GitHub App** on this repo (https://github.com/apps/claude)
  — the action uses it to post review comments. Without it the job fails with
  "Claude Code is not installed on this repository".
- Add repo secret **`ANTHROPIC_API_KEY`** (Settings → Secrets and variables →
  Actions) so the review job can call the model. Without it, `claude-review` fails.
- Branch protection on `main` will be set to require the checks above plus one
  approving review before merging. See this PR's discussion for the `gh api`
  call used to apply it.
