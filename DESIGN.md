# LLM NPC Sandbox — Design

Sandbox for testing LLM-powered NPCs in a real-time grid world of stacked levels. Goal: human-like behavior emerging from a minimal substrate — perception, memory, and drives — with **zero behavioral hardcoding**.

## Core principles

1. **No behavioral hardcoding.** No scripted personalities, no "if threatened then flee" rules. Code provides physics and motor control only. All *decisions* come from the LLM reasoning over perceptions + memories + drives.
2. **Intent vs. motor control.** The LLM emits intent-level actions (`move to (12,8)`, `interact npc_2`). The simulation handles execution (pathfinding, timing, collision). Humans don't consciously compute leg movements; agents don't emit per-step instructions.
3. **Drives produce sensations, not actions.** A drive never directly triggers behavior. It generates internal state that the agent perceives (hunger pangs, fear from low HP, restlessness) and the LLM decides what, if anything, to do about it. This is the main defense against hardcode creep.
4. **Model-agnostic.** All LLM calls go through one provider interface. Any model — API or local checkpoint — slots in without touching game code.

## World

- A stack of levels, each a grid of cells (`sim/world.py`). A cell at `(x, y, level)` has two layers:
  - **floor layer** — a surface at height `level`, named by material (`grass`, `stone`), or none;
  - **connector layer** — one object in the slab between `level` and `level + 1` (dirt, a wall, a table, a stair step), filling a vertical span `[bottom, top)` of that slab in slab units. A full block is `0..1`, a table `0..0.5`, the upper step of a staircase `0.5..1`.
- Heights are continuous. An entity stands at height `z` on a *surface* (a floor, or the top of a connector); its level is `floor(z)`. The top of a full block on level L is a surface at L + 1, so the ground is a level of dirt blocks with grass floor tiles on top: remove the grass and the dirt top still holds you up; remove the dirt too and the column drops a level.
- Movement: stepping into a column lands you on the highest surface there that is at most your `climb` above you and has `height` of clear space over it. Dropping any distance is allowed. Sight: a horizontal line at mid-body height, blocked by any connector span covering that height (you see over a table, not through a wall). No code knows what a "table" or a "stair" is; only spans and heights.
- Real-time loop. One entity per column per level.
- A connector has a **type**, and a type is data (`sim/terrain.py::CONNECTORS`): a default span plus a list of **tags**. Tags are the only place a type's properties live. Each tag is one record `{on, do, ...payload}`: `on` is the trigger (`interact`, `use`, `time`), `do` the effect (`restore` a stat, `give` an item, `become` another type). A bush is `give berry` + `become empty_bush` on interact; an `empty_bush` is `become bush` after a timed wait; a well is `restore thirst`. Items are tagged the same way. No code special-cases a type name; `apply_tags` reads the records.
- Placed objects (a bush, a well) are *named* connectors (they carry an `id`). The world exposes entities and named objects alike as "things" (`World.things()` / `thing(id)`), so perception and `interact` treat a bush like any other body, and a thing changing type (a bush picked bare) is perceived like a move. Bulk connectors (dirt, walls) stay anonymous and only show on the map.

## Entities & attributes

Every entity (player and NPCs) is an instance of the same structure. Perceptual/motor tunables live **on the entity**, not as global constants:

| Attribute         | Default      | Notes                              |
|-------------------|--------------|------------------------------------|
| `pos`             | —            | `(x, y, z)`: column + standing height |
| `hp`              | 100          |                                    |
| `vision_radius`   | 8            | LOS-based                          |
| `interact_range`  | 1 (adjacent) | chebyshev distance                 |
| `height`          | 1.0 slab     | body height; must fit under overheads |
| `climb`           | 0.5 slab     | tallest step up in one move        |
| `move_interval`   | 150 ms/tile  | motor execution speed              |
| `think_interval`  | 3 s          | idle decision cadence (NPCs)       |
| `inventory`       | []           | items usable via `use`             |
| `drives`          | see below    | personality = drive weights        |
| `memory`          | []           | private, per-entity                |

## Perception

- Each tick, compute what each NPC can see: entities/objects within `vision_radius`, unobstructed at eye height (Bresenham LOS over connector spans).
- **Diffed:** only changes generate events ("npc_2 entered view", "npc_2 moved", "object X appeared"). Prevents prompt noise and meaningless memories.
- Perception events feed two consumers: immediate interrupts (see LLM loop) and memory writes.

## Actions

Three verbs, available to every entity:

| Action     | Target                  | Semantics                                        |
|------------|-------------------------|--------------------------------------------------|
| `move`     | tile coordinate         | pathfind + walk there over time                  |
| `use`      | own inventory item      | internal (consume, wield, etc.)                  |
| `interact` | external entity/object  | external (talk, manipulate, engage)              |

Rule of thumb encoded nowhere but enforced by validation: **`use` = internal, `interact` = external.**

Validation: range check (`interact_range`), LOS check, target existence. Failures return feedback to the actor and get logged — failed attempts are themselves memorable events.

## Memory

Each memory: `{id, t, type, content, entities, salience, location, last_accessed, access_count}`

Types: `observation`, `action_result`, `reflection` (later).

- **Write time:** salient perception/dialogue/outcome events become memories. The forming call tags salience (1–10); location = where the rememberer stood. No numeric rules in code.
- **Read time:** retrieval score = `0.30·recency + 0.35·salience + 0.20·lexical + 0.15·spatial`. Recency anchors on the later of created/last-accessed; each actual retrieval updates `last_accessed`/`access_count` (rehearsal keeps useful memories vivid). Spatial term = `8/(8+chebyshev(query_pos, mem_loc))`. Weights are constants in `sim/memory.py`, tunable.
- **Conjunction emerges at read time.** Co-located/co-timed memories co-retrieve; the decision-time LLM connects them. We never compute that combination ourselves.
- **Reflections (later):** periodic synthesis calls cluster related memories into conclusions stored as high-salience memories.
- **Embedding column: deferred** — lexical+spatial covers current scale; paraphrase variance will eventually justify hybrid semantic scoring.
- Memories are private per entity. NPC-A's actions become NPC-B's memories only through B's perception.

## Drives & personality

Drives are a weighted set of internal needs. Personality = a **weight profile** over universal drives — nothing else:

```
drives: { survival: 1.0, curiosity: 0.0 }   // start: survival-only
// later e.g. { survival: 0.5, curiosity: 0.5 }  -> knowledge-valuing NPC
// later e.g. { survival: 0.3, power: 0.7 }      -> risk-taking NPC
```

**Weights are shares of one whole and must sum to 1** (`sim/reward.py::validate_drives`).
The same weights scale reward during training, so unequal totals would pay some
NPCs more for identical behavior; normalizing makes returns comparable across
profiles. Personality is how the budget is split, not how big it is.

Mechanism: each drive has a mechanical state (HP/hunger for survival, novelty-deficit for curiosity). Drive state is reported to the agent as **internal sensation** in its prompt ("you feel hungry", "your surroundings feel stale"). The LLM weighs sensations against memories and decides. An NPC valuing power over safety is just a different weighting of the same machinery — no new code.

Survival substrate (physics, not behavior): HP, hunger draining HP slowly, damage sources in the world, death → respawn + log.

Curiosity: deferred until survival-only proves too passive (it will — safe agents have no reason to act).

## LLM loop (NPCs)

```
sense (LOS diff) ──► interrupt? ──► think now
                     │no
                     ▼
              think_interval elapsed ──► think
think:
  observations + internal drive sensations + retrieved memories
    → prompt → structured JSON response { action, params }
  malformed → retry once → fallback no-op (logged)
execution:
  sim executes intent over real time (pathfinding, timers)
  outcomes → memory writes → next cycle
```

- Interrupts bypass cadence: damage taken, interacted with, novel perception.
- **Dialogue sessions:** `interact` between two conversational entities opens a session — alternating turns, both sides are LLMs (NPC↔NPC) or human-typed (player↔NPC). Session ends when either moves away or signals end. Every exchange becomes a memory for both parties.
- Player↔NPC: player types text; NPC responds within the session.

## Controls (player)

- Movement: `e/s/d/f` (up/left/down/right).
- Target: whatever entity is under the mouse cursor (highlighted; HUD shows distance/LOS to it).
- Interact: **right-click** the targeted entity — in range+LOS it acts (talk), otherwise explicit failure feedback.
- Talk: right-click an NPC opens conversation; type immediately, `ENTER` sends, `ESC` or **left-click** leaves. Movement locks while chatting.
- Quit: `ESC` when not in conversation.

Tunables in `config.json`: `interact_range` (applied to all entities at spawn, default 4 tiles), `max_dialogue_turns`, `memory_*`, model settings.

## Targeting

Every entity has a `target`: the thing it is "looking at". Interactions always flow through the target, regardless of who acts.

- Player: target tracks the mouse hover each frame.
- NPCs: attention rule v0 — most recently perceived entity (`entity_entered`/`entity_moved` sets it, `entity_left` clears if it was the target). Will be replaced by LLM-driven attention later.
- NPC autonomous behavior is ENABLED (`ENABLE_NPC_THINKS = True`, now in `sim/engine.py`): NPCs think when a novel memory forms AND on an idle cadence (`think_interval`, per entity, default 3 s) — internal pressure like hunger is never a perception event, so the cadence is what lets an agent reconsider as its meters fall.

## Logging

Append-only **JSONL**, one event per line, per run:

```json
{"t": 1234.5, "run": "r7", "actor": "npc_1", "type": "prompt", "payload": {...}}
```

Event types: `perception`, `prompt`, `response`, `action_start`, `action_complete`, `action_failed`, `memory_write`, `memory_retrieve`, `dialogue_msg`, `death`, `respawn`.

Dual purpose: debugging/readability now, and **training data later** — logged observation→response pairs are exactly the corpus needed for fine-tuning a small model on any given NPC's behavior.

## Tech stack

- Windowed desktop app (not web). Test 1: Python 3 + pygame for window/render/input.
- Sim layer is pure Python and headless-testable; the renderer is thin and swappable.
- LLM calls later: provider interface in-process or as a local sidecar; API keys via env/config, never in code.

### Running

```
pip install -r requirements.txt
python main.py            # play
python smoke_test.py      # headless sim assertions
```

## Test 1 — acceptance criteria (PASSED)

Small grid, some walls, player + 2 brainless NPCs.

- [x] esdf movement works, wall collision works
- [x] render shows grid, walls, all three entities
- [x] NPCs exist as bodies: correct collision, visible, idle
- [x] `interact` on NPC in range+LOS → explicit feedback: `target: npc_2 | range: ok | los: clear`
- [x] `interact` out of range or blocked LOS → explicit failure feedback
- [x] JSONL logging captures every input/action event from minute one

## Implementation status

| Step | Status | Notes |
|---|---|---|
| Test 1: window + bodies + interact validation | done | pygame app, headless-testable sim |
| LLM brains (perceive → think → act) | done | gemma4 via Ollama (`think:false` required); worker thread; A* intent execution; interrupts on interact/entity_entered |
| Memory system | done | batched LLM tagging per think-cycle → first-person salience-tagged memories; retrieval = 0.35·recency(halflife 300s) + 0.35·salience + 0.30·lexical overlap; top-k into prompt |
| NPC↔NPC dialogue sessions | done | unified `say`/`end_dialogue` actions in one schema; sessions close on distance/left/ended/max_turns(12); transcript flushed to both parties' pending events at close → remembered |
| Player↔NPC dialogue | done, **verified via `--autotest talk`** | right-click opens (no NPC opener line); type immediately, ENTER sends, ESC/left-click leaves; movement locked while chatting; responses echo to console + on-screen; harness injects real input events |
| NPC↔NPC dialogue | done, **verified via `--autotest npc-talk`** | same interact code path as player (injected intent → apply_intent → open_session → reply alternation). Live run: 12-turn coherent survival negotiation, closed by max_turns cap |
| Curiosity drive / personalities | next | |
| Felt memories (interoception) | done | internal stat changes (health/hunger/thirst — anything in `sim/engine.py::FELT_STATS`) flow through the same event->memory pipeline as sense `felt` (`did` = external act, `felt` = internal state), emitted when the rounded value the agent perceives changes. Direction lives in subject.type, so a same-direction run consolidates into one spanning record ("I felt my thirst fell from 99 to 0") while a reversal or another stat starts its own; a full episode of drain (300 ticks) folds to 3 records. Consolidation gap 2s->3s so the slowest stat cadence can't fragment runs; `recall` accepts sense `felt` |
| Headless engine + reward training scaffold | done | per-tick sim extracted to `sim/engine.py` (main.py and the headless runner drive the same Engine; headless thinks synchronously for reproducibility). `sim/reward.py`: drive-weighted, world-computed reward (survival = min meter; curiosity = types newly familiar). Expert iteration: `train/run_episodes.py` (rollouts + per-NPC returns), reward-filtered `extract_dataset.py` (top quantile per drive profile), `train_lora.py` (Qwen2.5-3B LoRA, completion-only loss), `train/expert_iter.py` orchestrator; round-1 rollout with gemma4 is the baseline yardstick |
| Reflections + `use` objects | — | |
| Vector embeddings for retrieval | deferred | lexical overlap sufficient at current scale; revisit when memory corpora grow |

Testing philosophy going forward: every interaction gets an isolated, controlled `--autotest` mode that injects events through the real pygame pipeline (not by calling handlers directly).

Note: with `ENABLE_NPC_THINKS = False`, dialogue transcripts flush into pending events at session close but are not yet converted to memories (that happens on the next think cycle, currently frozen). They are fully preserved in the JSONL journal.

Known warts (accepted for now):
- ~~Cross-think-cycle duplicate memories accumulate~~ **Resolved:** write-time consolidation (`sim/consolidation.py`, `MemoryStore.record`). One rule, no per-type policy: two memories fold into one when they share `sense`, at most **one** subject field (kind/ref/type/pos/info) differs, and they are within `gap_s`. Time always spans (`t..t_end`, `count`); a folded record keeps `origin` (its first subject) alongside `subject` (its latest) — first+last, which is all render and retrieval use (the common case is `pos`-only-differs, i.e. a movement `origin->latest`). Intermediate waypoints are intentionally dropped: under this rule every snapshot shares identity, so they add nothing. `observer_loc`/`direction` don't affect the decision.
- Memories are per-run only (no persistence across app restarts).
- Interact→memory path unit-tested via fake provider; live verification pending human-at-keyboard run.
- **REVISIT: `landing` / `climb` semantics.** The step rule (`World.landing`: up by
  at most `climb`, drop any distance, body must fit) shipped with the layered map
  but its level placement isn't fully trusted yet — verify entities land on the
  intended surfaces once vertical content (dig/build, multi-level areas) exists.
  Not important at current single-level scale.
- **REVISIT with the same work: sight and targeting are z-blind beyond the eye ray.**
  Sight is a horizontal line at eye height (floors never block it) and range checks
  are 2-D, so a second inhabited level would let agents see/interact through solid
  ground. Move targets are columns, not cells — `(x, y)` reaches any level of that
  column. All fine while exactly one level is inhabited; fix alongside dig/build.

Verified live (90 s sim): 34/34 LLM calls returned schema-valid actions; cadence ~2.6 s/think warm. Baseline behavior as predicted: survival-only NPCs wander locally, passive until threatened.

## Roadmap after test 1

1. Give one NPC an LLM brain: perceive → think → wander with purpose (survival-only).
2. Memory system live: does it remember being interacted with?
3. NPC↔NPC dialogue sessions.
4. Curiosity drive; drive-weight personalities.
5. Reflections; objects with `use` mechanics.
6. Fine-tuning experiments on logged corpora.
