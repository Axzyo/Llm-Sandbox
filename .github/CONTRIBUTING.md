# Contributing / Branch workflow

`main` is protected by a repository ruleset. All work follows this flow:

1. **Every feature gets its own branch** off `main`
   (e.g. `feat/inventory-effects`, `fix/spatial-decay`). Never commit to `main`.
2. Push the branch and open a **pull request into `main`**.
3. On every PR, two required checks run automatically:
   - **`smoke-test`** — runs `python smoke_test.py` (must pass).
   - **`claude-review`** — an AI agent reviews the diff against `CLAUDE.md` /
     `DESIGN.md` and posts inline + summary comments.
4. **No approving review is required** — a PR merges once both required checks are
   green. While `REVIEWER=claude`, `claude-review` is a required check, so every
   merge has been through the agent (its comments are advisory; read them before
   merging).

For everything else — the full ruleset, the `REVIEWER` reviewer switch,
authentication, one-time setup, and how to change the pipeline — see
[`docs/ci-cd-pipeline.md`](../docs/ci-cd-pipeline.md).
