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

## One-time setup (repo owner)

- Add repo secret **`ANTHROPIC_API_KEY`** (Settings → Secrets and variables →
  Actions) so the review job can run. Without it, `claude-review` fails.
- Branch protection on `main` is configured to require the checks above plus one
  approving review. See `git` history of this PR for the `gh api` call used.
