---
name: code-quality-rules
description: The six code-quality rules enforced on every PR (live in CLAUDE.md + PR review agent)
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 71c8cf19-1539-45aa-8243-33201181c3bb
  modified: 2026-08-30T10:18:27.223Z
---

The project's code-quality standards, authored 2026-08-28 into `CLAUDE.md` and
enforced on every PR into `main` by both humans and the `claude-review` GitHub
Action. Derived from decisions across the project history (see
[[memory-system-design]], [[decide-training-teacher]], [[goal-schema]]).

1. **No hardcoded reasoning or behavioral prose in logic** — provide facts/data;
   let the decider (LLM or code) reason. No coaching sentences or scripted
   interpretations in per-turn context. Behavior emerges from data.
2. **Reuse existing systems over parallel one-offs** — flow new behavior through
   existing pipelines (interact/inventory/memory/perception); data-driven
   payloads over bespoke functions and kind-special-casing.
3. **Delete dead code; don't let it linger** — rebuild fresh when a real need arises.
4. **No premature complexity** — simplest structure that fits current scale.
5. **Strict validation, no silent coercion** — validators reject on any contract
   violation (return None / route to bad), never guess or auto-fix.
6. **Keep tests in sync** — update and re-run `smoke_test.py` (-> `SMOKE OK`) on
   every schema/behavior change.
7. **No indirection tables** (added 2026-08-30) — a dict/table must carry data you
   can't get by accessing the real field directly. Bans alias tables, identity
   entries (`key == value`), and keys-never-read tables (use a list); fix the name
   at the source instead. Legit data payloads + name->function dispatch (key used)
   are fine. Motivated by a `FELT_STATS = {"health":"hp",...}` alias table.

**Why:** these are the user's confirmed quality bar; the PR agent cites them, so
knowing them lets me pre-empt review findings and write code that passes first try.

**How to apply:** check any change against all seven before proposing it. Note the
built-in tension between #2 (reuse) and #4 (no premature complexity): tolerate
some duplication over a forced/wrong abstraction — reuse only when the shared
mechanism genuinely fits. #1 is the domain-specific one (LLM-agent design); the
rest are general best practice (DRY, YAGNI, KISS, fail-fast).
