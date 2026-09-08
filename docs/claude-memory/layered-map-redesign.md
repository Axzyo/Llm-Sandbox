---
name: layered-map-redesign
description: "Map redesign (Sept 2026) - stacked levels of cells with floor + connector span; slice 1 (cell record, heights, climb, sight) shipped; support/collapse, dig/build, falling/breaking, looking-down still to do"
metadata: 
  node_type: memory
  type: project
  originSessionId: 626fda4f-404d-487c-9eec-7e44922c6348
  modified: 2026-09-05T00:47:24.135Z
---

On 2026-09-04 the user redesigned the map: "floor layer" (tiles = material + position) and
"connector layer" (objects between floors: dirt, walls, tables, stairs) with heights.
Slice 1 shipped on branch ci/setup-pipeline: `sim/world.py` cells keyed (x, y, level) with
`floor` material and a `Connector(material, bottom, top)` span; entities have float `z`
(standing height, level = floor(z)) plus `height` / `climb` properties; movement uses
`World.landing`, sight uses mid-body eye height over connector spans. Ground = level 0
dirt blocks + stone floor, level 1 grass; walls are stone connectors on level 1.

User said "go with your defaults" on the four open decisions:
- support span for floors over rooms: NOT built yet, was to be a flood-fill from full-height
  columns within N tiles (N unpicked); collapse drops floor + objects, breaks by toughness.
- standable: any surface with clearance for the body (done, no per-object flag).
- break rule: one toughness number per object/entity vs fall distance (not built).
- looking down through holes: per-level sight for now; 'v' glyph marks a drop (done).

**Why:** the user wants buildable multi-level worlds (dig holes, reshape dirt into ramps,
climb out, build floors that need support). Stairs were the reason connectors got spans
(bottom/top) rather than a single height: you can walk under the upper step.

**How to apply:** next slices in order: dig/build/reshape as external world actions with
data payloads (no kind special-casing), then support + collapse, then falling/breaking.
Keep `move` params as x,y only (the LLM names a column). See [[code-quality-rules]].

REVISIT (owner-flagged 2026-09-05): `World.landing`/`climb` semantics not fully trusted (level placement of landings unverified); sight/targeting are z-blind beyond the horizontal eye ray (floors never block sight, range is 2-D, move targets are columns not cells). All harmless while one level is inhabited; fix together when dig/build/multi-level content lands. Standing on wells (top 0.5 = climb 0.5) is intended.
