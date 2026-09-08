---
name: input-spatial-memory
description: LLM input contract — geometry is memory-based via a separate spatial layer; strict response filter
metadata: 
  node_type: memory
  type: project
  originSessionId: a8dcbd6f-f51d-4e85-b4e6-c423df7a339c
  modified: 2026-08-27T21:58:18.748Z
---

After finalizing the goal-set OUTPUT ([[goal-schema]]), work moved to the LLM INPUT contract (2026-08-27).

**Response filter (done):** `sim/brain.py` `filter_response(raw)` is the single strict entry point → routes every reply to `goals` / `wait` / `recall` / `bad`. `validate_goals` is now strict (returns None on ANY contract violation — no silent coercion): must be `{"goals":[{"actions":[≥1 world action], "importance":<num>, "reason":?}, …≥1]}`; importance required+numeric (clamped 0–10); recall/wait inside a plan, empty goals, bare list/action, unknown/malformed action all → bad. `decide()` journals `response_rejected{reason}`, does one corrective retry, else empty agenda.

**Geometry = memory-based (user decision).** User wants everything experiential. Tiles are remembered, not hand-fed live. Chosen design: a **separate spatial-memory layer** (`sim/spatial.py` `SpatialMemory`) — a per-entity remembered occupancy grid ({(x,y)->floor|wall}), NOT in the episodic `MemoryStore` (one glance = ~289 tiles would evict the 300-cap / swamp k=5 recall). 
- `sim/perception.py` `visible_tiles(world, viewer)` + `_sightline_clear_to` (a wall target IS visible — you see its face — unlike `has_los`).
- `Brain.perceive_tiles(seen, t)` folds tiles in; new geometry sets `pending_think`. `main.py` calls it in the perception tick (only when ENABLE_NPC_MEMORY).
- `SpatialMemory.render_local(center, radius)` → coordinate-framed ASCII map (`@`=you `#`=wall `.`=floor ` `=unseen; header gives x-range, rows labelled by y so the LLM can derive absolute coords for move goals). `build_user_prompt` appends it. Verified: renders correctly, LOS-masked, reaches the prompt (~750 chars).
- Used ASCII `->` (not `→`) in goal summaries + map header to avoid Windows cp1252 console / pygame-font issues.
- **Storage:** in-RAM sparse dict `{(x,y):{type,last_seen}}`, NOT persisted (per-run). Scales linearly with unique tiles SEEN (~430 B/tile RAM, ~20 B/tile JSON); coordinate magnitude is irrelevant (int is 28 B whether 5 or 10^7). Teleporting without perceiving adds nothing. Prompt stays bounded (render_local is a local window).
- **Forgetting infra (2026-08-27):** `forget(coord)`, `forget_where(pred)`, `forget_older_than(t)`, `forget_beyond(center,r)`, `clear()`, optional `max_tiles` cap with LRU `_evict_to_cap()`.
- **Forgetting = memorability model (2026-08-27, replaced the radius idea — user rejected distance-based).** Plan settled on **plain dict + forgetting** (NOT compact reps: at 4000 NPCs a bounded working set is ~1 GB, and the dict carries the per-tile metadata fine-grained forgetting needs). Each tile stores `memorability` (replaced `last_seen`). Mechanism = **decay + cap backstop**; inputs = **reinforcement + proximity to goal locations** (user's choice). In `sim/spatial.py` (tunables are module constants): `observe` adds `SIGHT_BOOST`(1.0) capped at `MEMORABILITY_CAP`(4.0); `age(goal_locations)` multiplies all by `DECAY`(0.9), floors tiles within `GOAL_RANGE`(8) of a goal to **(that goal's `importance` × proximity)**, forgets ≤ `FORGET_THRESHOLD`(0.15), then evicts least-memorable past `MAX_TILES`(8000). No separate GOAL_STRENGTH constant — user: "Goal.importance replaces GOAL_STRENGTH, no point adding new values." `GoalList.locations()` returns `((x,y), importance)` pairs (move-targets + the goal's importance); so geometry near an urgent goal is held stronger and fades slower when the goal ends. Verified: importance-8 goal floors its tile to 8.0, importance-1 to 1.0. Brain: `perceive_tiles(seen)` reinforces, `maintain_spatial(goal_locs)` runs `age`; main.py calls both each perception tick with `npc.goals.locations()`. Verified: near-goal tile floors at 2.667 and holds; far tile decays and is forgotten. Compact reps (packed-int set ~74 B/tile, roaring ~1–2 B) remain the fallback only if NPCs must RETAIN huge maps without forgetting — deferred; interface isolates the backend.
- Scale targets (user): infinite world + LOD sphere of chunks around player; peak battle ~2000v2000 over ≥1000×1000 tiles (sparse: ~1 NPC in an 8-radius view avg). Real ceilings at that scale are perception CPU (needs a spatial hash, not O(N²)) and LLM inference (LOD-gated thinking), NOT tile storage.

**Map access = push + pull (2026-08-27, user spec).** The LLM references its map two ways: (1) PUSH — `build_user_prompt._map_blocks` auto-renders a self window (vision radius) PLUS a `MAP_ANCHOR_RADIUS`(4) window around each retrieved memory's location (`subject.pos` or `observer_loc`), deduped so overlapping terrain isn't drawn twice. So located memories drag in the remembered terrain around them ("saw Dave at [40,20]" → map there). (2) PULL — a new `look` action `{"action":"look","params":{"x","y"}}` (thinking-layer, like recall) renders a remembered-terrain window at a coord even where no episodic memory exists; folded into the `decide` loop via `extra_anchors`, bounded `MAX_LOOKS`(2), agent continues. `recall` still works and its located memories also bring windows. `render_local(center, radius, marker=self_pos)` — `@` marks only the NPC's real tile, so anchor windows centered elsewhere show no false `@`. `filter_response`/`validate_intent` route `look`; decide journals `map_look`. Verified + tested (`test_look_action_loop`).

**OPEN decision — per-entity perception enrichment:** whether visible_entities should carry derived `distance`/`in_interact_range`/`bearing` or stay raw `{id,type,pos}`. User leaned "probably nothing, need more context." Fits the memory-based philosophy: the agent learns range through *failure* memories ("tried to interact … out_of_range") rather than hand-fed computation. Not yet decided.

All smoke_test pass (added `test_spatial_memory`). ENABLE_NPC_THINKS still False (autonomous thinking off during isolation).

**Why:** captures the input contract half of the LLM interface + the deliberate memory-based geometry choice.
**How to apply:** keep geometry/spatial knowledge in the spatial layer, not episodic; resolve the perception-enrichment question before relying on interact-range reasoning.
