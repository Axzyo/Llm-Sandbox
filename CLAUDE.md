# Coding standards

Project rules for this repo. Read alongside `DESIGN.md` (the architecture and
philosophy). Both humans and the automated PR reviewer enforce these on every
merge into `main`.

## Core rules

1. **No hardcoded reasoning or behavioral prose in logic.**
   Provide facts and data; let the decider (LLM or code) reason from them. Do not
   bake authored conclusions, coaching sentences, or scripted interpretations into
   per-turn context or runtime paths. Personality and behavior must emerge from
   data (memories, drive weights, sensations, familiarity flags), never from a
   hardcoded string.

2. **Reuse existing systems over parallel one-offs.**
   New behavior should flow through the mechanisms that already exist
   (interact / inventory / memory / perception pipelines). "Everything already
   exists, effects are properties" — prefer data-driven payloads over bespoke
   functions and kind-special-casing.

3. **Delete dead code; don't let it linger.**
   Remove dormant/unused code rather than keeping it "just in case." Rebuild fresh
   when a concrete need actually arises.

4. **No premature complexity.**
   Choose the simplest structure that fits the current scale. Don't add compact
   representations, extra tunable constants, or abstractions before they're
   needed. Don't build training/optimization on top of a world that doesn't yet
   have the stakes to justify it.

5. **Strict validation, no silent coercion.**
   Validators reject on any contract violation (return `None` / route to `bad`)
   rather than guessing or auto-fixing malformed input. Keep the LLM I/O contract
   (goal-sets) strict everywhere.

6. **Keep tests in sync with every schema/behavior change.**
   Update `smoke_test.py` and re-run it (`python smoke_test.py` → `SMOKE OK`)
   whenever behavior or the I/O contract changes. New or changed behavior should
   have coverage.

7. **No indirection tables.**
   A dict/table earns its place only when its values are data you can't get by
   accessing the real field directly. Delete a table whose values are just other
   names or attributes (aliases), whose entries are identity (`key == value`), or
   whose keys are never read (that's a list). If a table exists only to translate
   one name into another, fix the name at the source instead — e.g. don't map
   `{"health": "hp"}` when the field could just be called `health`. Data tables
   (real payloads, coordinates, effect specs, name->function dispatch whose key is
   used) are fine and encouraged (see rule 2).

## Working notes

- The NPC's only source of what things do is outcome memories, not names or
  prompt text — preserve that causal-learning property.
- Geometry/spatial knowledge lives in the spatial-memory layer (`sim/spatial.py`),
  not the episodic `MemoryStore`.
- Use ASCII (e.g. `->`, not `→`) in prompt/console/pygame-facing strings to avoid
  Windows cp1252 issues.
