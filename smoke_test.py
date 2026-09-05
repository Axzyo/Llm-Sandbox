import json
import os
import tempfile

from sim.actions import attempt_move, evaluate_interact
from sim.brain import Brain, validate_intent, validate_goals, filter_response, memories_from_event, render_memory, bearing, MAX_RECALLS
from sim.entities import Entity
from sim.goals import Goal, GoalList, goal_from_intent, DEFAULT_IMPORTANCE
from sim.journal import Journal
from sim.terrain import CONNECTORS, GROUND, build_test_map, make, place_resources, tick_connectors, interact_with, use_item
from sim.consolidation import Consolidator, subject_diff
from sim.memory import MemoryStore, tokenize
from sim.pathing import next_step
from sim.perception import PerceptionTracker, visible_tiles
from sim.spatial import SpatialMemory
from sim.engine import Engine, death_cause
from sim.needs import tick_needs, HUNGER_DRAIN_PER_S, THIRST_DRAIN_PER_S, HEALTH_REGEN_PER_S
from sim.reward import curiosity_reward, note_novelty, reward, survival_reward, validate_drives
from sim.provider import OllamaProvider, _text_value_so_far
from sim.world import Connector, World, has_los, level_of


class FakeProvider:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat_json(self, system, user):
        self.calls.append((system, user))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply



def build_world():
    world, spawns = build_test_map()
    player = Entity("player", "you", "player", *spawns["player"])
    npc1 = Entity("npc_1", "npc_1", "npc", *spawns["npc_1"])
    npc2 = Entity("npc_2", "npc_2", "npc", *spawns["npc_2"])
    for e in (player, npc1, npc2):
        world.entities[e.id] = e
    return world, player, npc1, npc2


def test_movement_and_interact():
    world, player, npc1, npc2 = build_world()

    r = attempt_move(world, player, 1, 0)
    assert r["ok"] and player.pos == (4, 3, 1.0), r

    player.x, player.y = 7, 7
    r = attempt_move(world, player, 0, 1)
    assert not r["ok"] and r["reason"] == "blocked", r

    player.x, player.y = 19, 5
    r = attempt_move(world, player, -1, 0)
    assert not r["ok"] and r["reason"] == "occupied" and r.get("by") == "npc_2", r

    res = evaluate_interact(world, player)
    assert (
        res["ok"]
        and res["target"] == "npc_2"
        and res["distance"] == 1
        and res["range_ok"]
        and res["los_ok"]
    ), res

    player.x, player.y = 17, 4
    assert not has_los(world, 17, 4, 18, 5, 1.5), "corner must block LOS"
    res = evaluate_interact(world, player)
    assert not res["ok"] and res["range_ok"] and not res["los_ok"], res

    player.x, player.y = 3, 3
    res = evaluate_interact(world, player)
    assert not res["ok"] and not res["range_ok"], res


def test_layers():
    """The cell record: floors and connector spans decide where a body can stand,
    what it can step to, and what it sees -- nothing knows a 'table' or 'stair'."""
    w = World(6, 3)
    for x in range(6):
        w.put(x, 1, 0, floor="stone", connector=Connector("dirt"))   # ground: dirt slab, top at 1
        w.put(x, 1, 1, floor="grass")                                # grass floor laid on it
    assert w.surfaces(0, 1) == [0.0, 1.0] and w.tile_type(0, 1, 1) == "floor"
    assert w.tile_type(0, 1, 0) == "wall", "from inside the ground slab the dirt is solid"

    # a 1x1 hole: grass and dirt gone at x=1 -> the column drops to the stone at 0
    w.put(1, 1, 0, floor="stone")
    w.put(1, 1, 1)
    assert w.surfaces(1, 1) == [0.0] and w.tile_type(1, 1, 1) == "drop"
    tall = Entity("t", "t", "npc", 0, 1, 1.0)
    r = attempt_move(w, tall, 1, 0)
    assert r["ok"] and tall.z == 0.0 and tall.level == 0, r          # any drop is allowed
    assert attempt_move(w, tall, -1, 0)["reason"] == "blocked", "a full dirt block is too tall to climb"
    assert w.tile_type(2, 1, 0) == "wall", "from the hole the dirt next door is solid"

    # reshape the neighbouring dirt into a half-height step -> climbable, then out
    w.put(2, 1, 0, floor="stone", connector=Connector("dirt", 0.0, 0.5))
    w.put(2, 1, 1)
    assert w.tile_type(2, 1, 0) == "step"
    assert attempt_move(w, tall, 1, 0)["ok"] and tall.z == 0.5, "onto the step"
    assert attempt_move(w, tall, 1, 0)["ok"] and tall.z == 1.0, "and up onto the grass"

    # a table (0..0.5) under a hole in the floor above: a faller lands on the table,
    # never touching the floor beneath it
    w.put(3, 1, 0, floor="stone", connector=Connector("wood", 0.0, 0.5))
    w.put(3, 1, 1)
    assert w.landing(3, 1, 1.0, 0.5, 1.0) == 0.5 and w.tile_type(3, 1, 1) == "drop"
    # ... but close the floor over that table and, approached from the ground slab,
    # a full-height body no longer fits on it (nor under it): only a short one does
    w.put(3, 1, 1, floor="grass")
    assert w.landing(3, 1, 0.0, 0.5, 1.0) is None
    assert w.landing(3, 1, 0.0, 0.5, 0.5) == 0.5, "a short body still fits on the table"
    w.put(3, 1, 1)

    # under the stairs: the upper step spans 0.5..1 of its slab, so the space
    # beneath it is a half-slab gap -- a short body walks under, a tall one cannot
    w.put(4, 1, 1, floor="grass", connector=Connector("wood", 0.5, 1.0))
    assert w.tile_type(4, 1, 1) == "step"
    assert w.landing(4, 1, 1.0, 0.5, 1.0) is None, "no room under the step for a full body"
    assert w.landing(4, 1, 1.0, 0.5, 0.4) == 1.0, "a short body ducks under"
    assert w.landing(4, 1, 1.5, 0.5, 1.0) == 2.0, "from half a slab up, onto the step's top: level 2"
    assert level_of(2.0) == 2 and level_of(1.5) == 1

    # sight: mid-body height. A half wall is seen over, a full wall is not
    w.put(5, 1, 1, floor="grass", connector=Connector("stone", 0.0, 0.5))
    assert not w.opaque(5, 1, 1.5) and w.opaque(5, 1, 1.25), "sight passes over a half wall, a crouch does not"
    assert w.opaque(0, 1, 0.5) and not w.opaque(0, 1, 1.5), "dirt blocks sight inside its slab only"
    assert has_los(w, 3, 1, 5, 1, 1.25), "a sight line under the upper stair step is clear"
    assert not has_los(w, 3, 1, 5, 1, 1.5), "one through the step is not"
    assert not has_los(w, 1, 1, 3, 1, 0.25), "down in the hole, the half-height dirt step blocks a low sight line"

    # the same-column occupancy rule is per level
    low = Entity("l", "l", "npc", 4, 1, 1.0)
    high = Entity("h", "h", "npc", 4, 1, 2.0)
    w.entities = {e.id: e for e in (low, high)}
    assert w.entity_at(4, 1, 1) is low and w.entity_at(4, 1, 2) is high and w.entity_at(4, 1) is not None


def test_pathing():
    world, player, _, _ = build_world()
    player.x, player.y = 5, 5
    goal = (19, 9)
    for _ in range(200):
        step = next_step(world, player, goal)
        assert step is not None, "path should exist"
        assert step[2] == world.landing(step[0], step[1], player.z, 0.5, 1.0), "step must be a legal landing"
        player.x, player.y, player.z = step
        if (player.x, player.y) == goal:
            break
    assert (player.x, player.y) == goal and player.z == 1.0
    assert next_step(world, player, goal) is None
    assert next_step(world, player, (0, 0)) is None, "goal inside wall unreachable"


def test_perception():
    world, player, npc1, npc2 = build_world()
    tracker = PerceptionTracker("npc_1")

    events = tracker.update(world, npc1)
    kinds = {(e["kind"], e["id"]) for e in events}
    assert ("entity_entered", "npc_2") in kinds, events
    assert ("entity_entered", "player") not in kinds, "player spawns outside vision radius"

    events = tracker.update(world, npc1)
    assert events == [], events

    player.x, player.y = 20, 6
    events = tracker.update(world, npc1)
    assert any(e["kind"] == "entity_entered" and e["id"] == "player" for e in events), events

    player.x, player.y = 21, 6
    events = tracker.update(world, npc1)
    assert any(e["kind"] == "entity_moved" and e["id"] == "player" for e in events), events

    player.x, player.y = 3, 14
    events = tracker.update(world, npc1)
    entered = [(e["kind"], e["id"]) for e in events]
    assert ("entity_left", "player") in entered, events


def test_spatial_memory():
    world, player, npc1, npc2 = build_world()   # npc_1 at (19,9) on level 1, vision 8
    seen = dict(visible_tiles(world, npc1))
    # own tile is seen floor; the right border wall (23,9) is in view and remembered as wall
    assert seen[(19, 9, 1)] == "floor", seen.get((19, 9, 1))
    assert seen[(23, 9, 1)] == "wall", "the border wall in view is seen"
    # a tile occluded by the x=15 wall column is NOT seen (LOS only)
    assert (14, 9, 1) not in seen, "tile behind a wall must be unseen"
    assert all(k[2] == 1 for k in seen), "an entity sees its own level"

    sm = SpatialMemory("npc_1")
    new = sm.observe_many(seen.items())
    assert new == len(seen) and len(sm) == len(seen)
    assert sm.observe_many(seen.items()) == 0, "re-seeing discovers nothing new"
    assert sm.get((23, 9, 1)) == "wall" and sm.known((19, 9, 1)) and not sm.known((14, 9, 1))

    m = sm.render_local((19, 9, 1), radius=4)
    assert "@" in m and "#" in m and "y=  9" in m and "level 1" in m, m
    assert sm.render_local((19, 9, 0), radius=4) is None, "another level is unknown"
    assert SpatialMemory("x").render_local((0, 0, 1), 3) is None, "nothing remembered -> no map"

    # memorability: seeing a tile reinforces it, up to the cap
    from sim.spatial import SIGHT_BOOST, MEMORABILITY_CAP
    f = SpatialMemory("f", max_tiles=None)
    f.observe((0, 0, 1), "floor")
    assert f.memorability((0, 0, 1)) == SIGHT_BOOST
    for _ in range(20):
        f.observe((0, 0, 1), "floor")
    assert f.memorability((0, 0, 1)) == MEMORABILITY_CAP, "reinforcement caps out"

    # decay + threshold: an un-refreshed tile fades and is forgotten; a nearby goal
    # floors memorability by proximity so goal-relevant geometry survives
    a = SpatialMemory("a", max_tiles=None)
    a.observe_many([((0, 0, 1), "floor"), ((10, 0, 1), "floor")])   # both start at SIGHT_BOOST
    passes = 0
    while a.known((0, 0, 1)) and passes < 100:
        a.age(goal_locations=[((10, 0), 3.0)])                     # importance-3 goal on (10,0)
        passes += 1
    assert not a.known((0, 0, 1)), "the tile far from any goal decays away"
    assert a.known((10, 0, 1)) and a.memorability((10, 0, 1)) >= 3.0 * 0.9, "goal tile floored to importance"
    # a MORE important goal floors its tile higher
    a.observe((20, 0, 1), "floor")
    a.age(goal_locations=[((20, 0), 9.0)])
    assert a.memorability((20, 0, 1)) > a.memorability((10, 0, 1)), "higher-importance goal = stickier geometry"

    # cap backstop evicts the LEAST memorable first (not the oldest)
    capped = SpatialMemory("c", max_tiles=3)
    capped.observe_many([((i, 0, 1), "floor") for i in range(5)])   # all equal memorability
    capped.observe((4, 0, 1), "floor")                               # bump (4,0) above the rest
    capped.observe((3, 0, 1), "floor")
    capped.observe_many([((9, 0, 1), "floor")])                      # force a 6th -> over cap
    assert len(capped) == 3 and capped.known((4, 0, 1)) and capped.known((3, 0, 1)), "kept most memorable"

    # Brain wiring: perceive reinforces, maintain_spatial decays + protects goals
    b = Brain("b", object())
    b.perceive_tiles([((5, 5, 1), "floor"), ((40, 40, 1), "floor")])
    for _ in range(60):
        b.maintain_spatial(goal_locations=[((5, 5), 4.0)])        # (5,5) protected, (40,40) not
    assert b.spatial.known((5, 5, 1)) and not b.spatial.known((40, 40, 1)), "goal-far geometry fades"


def test_death():
    import main
    from sim.world import World
    world = World(6, 6)
    alive = Entity("npc_1", "npc_1", "npc", 1, 1, 1.0)
    doomed = Entity("npc_2", "npc_2", "npc", 2, 2, 1.0)
    doomed.stats["health"], doomed.stats["thirst"] = 0.0, 0.0   # dehydrated to death
    for e in (alive, doomed):
        world.entities[e.id] = e
    npcs = [alive, doomed]
    by_id = {e.id: e for e in npcs}
    path = os.path.join(tempfile.mkdtemp(), "death.jsonl")
    j = Journal(path, "death")
    dead = main.reap_dead(npcs, by_id, world, j, sim_t=9.0)
    j.close()
    assert [d.id for d in dead] == ["npc_2"], dead
    assert "npc_2" not in world.entities and doomed not in npcs and "npc_2" not in by_id
    assert alive in npcs and "npc_1" in world.entities, "the living are untouched"
    rec = [json.loads(l) for l in open(path, encoding="utf-8")]
    assert any(r["type"] == "death" and r["payload"]["cause"] == "dehydration" for r in rec), rec


def test_terrain_resources():
    import random as _random
    world, spawns = build_test_map()
    spawn_tiles = {(x, y) for x, y, _z in spawns.values()}
    placed = place_resources(world, _random.Random(1))
    kinds = sorted(o.kind for o in placed)
    assert kinds == ["bush", "bush", "bush", "well"], kinds
    for o in placed:
        # a resource is the cell's connector object, set on the ground level
        assert world.cell(o.x, o.y, GROUND).connector.id == o.id and o.z == GROUND
        assert world.tile_type(o.x, o.y, GROUND) == "step", "a part-height object on the map"
        assert (o.x, o.y) not in spawn_tiles, "resource on a spawn tile"
        assert world.thing(o.id) == o and world.thing_at(o.x, o.y, GROUND) == o
    assert len({(o.x, o.y) for o in placed}) == 4, "resources overlap"
    assert len(world.objects()) == 4 and len(world.things()) == 4, "named objects only; dirt and walls are anonymous"

    water = next(o for o in placed if o.kind == "well")
    bush = next(o for o in placed if o.kind == "bush")
    # spans are the objects' only physics: a bush is too tall to climb and blocks a
    # sight line; a well's rim can be stood on and seen over
    walker = Entity("w", "w", "npc", bush.x, bush.y, 1.0)
    assert world.landing(bush.x, bush.y, 1.0, walker.properties["climb"], 1.0) is None
    assert world.opaque(bush.x, bush.y, 1.5) and not world.opaque(water.x, water.y, 1.5)
    assert world.landing(water.x, water.y, 1.0, 0.5, 1.0) == 1.5

    a = Entity("a", "a", "npc", 0, 0, 1.0)
    a.stats["hunger"], a.stats["thirst"] = 20.0, 10.0

    # tags are the whole of a type's behavior: the well's interact tag restores
    # thirst on the spot and the well is unchanged (nothing says otherwise)
    out = interact_with(water, a, now=5.0)
    assert out == {"ok": True, "effects": [{"stat": "thirst", "gained": 90.0}]}, out
    assert a.stats["thirst"] == 100.0 and a.inventory == [] and water.connector.type == "well"
    assert interact_with(a, a, now=5.0) is None, "an entity has no tags"

    # the bush's interact tags: give a berry, then become an empty_bush (same cell,
    # same id, the empty type's span and its regrow timer). A watcher sees the change.
    watcher = Entity("w2", "w2", "npc", bush.x, bush.y - 1 if bush.y > 1 else bush.y + 1, 1.0)
    world.entities[watcher.id] = watcher
    tracker = PerceptionTracker(watcher.id)
    assert any(e["id"] == bush.id and e["etype"] == "bush" for e in tracker.update(world, watcher))
    k = bush.connector
    out = interact_with(bush, a, now=5.0, rng=_random.Random(2))
    assert out == {"ok": True, "effects": [{"item": "berry"}, {"became": "empty_bush"}]}, out
    changed = [e for e in tracker.update(world, watcher) if e["id"] == bush.id]
    assert changed == [{"kind": "entity_changed", "id": bush.id, "pos": [bush.x, bush.y, 1.0], "etype": "empty_bush"}], changed
    del world.entities[watcher.id]
    assert a.inventory == ["berry"] and a.stats["hunger"] == 20.0, "picking doesn't feed you - eating does"
    assert k.type == "empty_bush" and k.id == bush.id and (k.bottom, k.top) == CONNECTORS["empty_bush"]["span"]
    assert world.thing(bush.id).kind == "empty_bush", "others now perceive an empty bush"
    lo, hi = CONNECTORS["empty_bush"]["tags"][0]["after"]
    assert k.timer is not None and 5.0 + lo <= k.timer <= 5.0 + hi
    # an empty bush has no interact tags: touching it does nothing, and says so
    assert interact_with(world.thing(bush.id), a, now=6.0) == {"ok": True, "effects": []}

    # eat the berry (inventory use): its use tag raises hunger, the berry is consumed
    out = use_item(a, "berry")
    assert out == {"ok": True, "did": "use", "item": "berry", "effects": [{"stat": "hunger", "gained": 40.0}]}, out
    assert a.stats["hunger"] == 60.0 and a.inventory == []
    assert use_item(a, "berry") is None, "no berry left to eat"

    # the empty bush's timed tag: it becomes a bush again when its timer comes due
    tick_connectors(world, now=k.timer - 1)
    assert k.type == "empty_bush"
    tick_connectors(world, now=k.timer)
    assert k.type == "bush" and k.timer is None and k.top == 0.75, "regrew into a bush"
    assert all(c.timer is None for c in world.connectors() if c.type in ("dirt", "stone")), "bulk has no timers"


def test_needs():
    e = Entity("npc_1", "npc_1", "npc", 5, 5, 1.0)
    s = e.stats
    assert s["health"] == 100 and s["hunger"] == 100.0 and s["thirst"] == 100.0, "start full"

    tick_needs(e, dt=10.0)                          # 10 seconds pass
    assert s["hunger"] == 100.0 - HUNGER_DRAIN_PER_S * 10, s["hunger"]
    assert s["thirst"] == 100.0 - THIRST_DRAIN_PER_S * 10, s["thirst"]
    assert s["health"] == 100.0, "well-fed + hydrated: health regenerates but caps at full"

    # health regenerates while both needs are above threshold
    s["health"], s["hunger"], s["thirst"] = 50.0, 90.0, 90.0
    tick_needs(e, dt=10.0)
    assert s["health"] == 50.0 + HEALTH_REGEN_PER_S * 10, s["health"]

    # a need at 0 starves health down; needs never go below empty
    s["hunger"] = 1.0
    tick_needs(e, dt=100.0)                          # hunger clamps at 0, then health drains
    assert s["hunger"] == 0.0, "needs never go below empty"
    assert s["health"] < 60.0, "empty hunger drains health"

    # in-between (a need below threshold but not empty) holds health steady
    s["health"], s["hunger"], s["thirst"] = 40.0, 50.0, 90.0
    tick_needs(e, dt=10.0)
    assert s["health"] == 40.0, "one need mid-range: health neither drains nor regens"


def test_reward():
    from sim.world import World
    world = World(4, 4)

    # topped-off survivalist: reward ~ 1 * survival signal
    e = Entity("a", "a", "npc", 1, 1, 1.0)
    e.drives = {"survival": 1.0, "curiosity": 0.0}
    assert survival_reward(e, world) == 1.0 and reward(e, world) == 1.0

    # only as safe as the worst meter; starving ~ 0; dead = 0
    e.stats["thirst"] = 30.0
    assert survival_reward(e, world) == 0.3
    e.stats["hunger"] = 0.0
    assert survival_reward(e, world) == 0.0, "an empty meter zeroes the signal"
    e.stats["hunger"], e.stats["health"] = 100.0, 0.0
    assert survival_reward(e, world) == 0.0, "death ends reward accrual"

    # curiosity pays out when a type newly becomes familiar, once
    c = Entity("c", "c", "npc", 1, 1, 1.0)
    c.drives = {"survival": 0.5, "curiosity": 0.5}
    brain = Brain("c", object())
    seen = {}
    assert note_novelty(c, brain, seen) == 0 and reward(c, world) == 0.5   # survival share only
    brain.record_events([{"kind": "did_interact", "target": "bush_1", "target_type": "berry_bush",
                          "target_pos": [2, 1], "outcome": "ok", "effect": "picked a berry"}],
                        now_t=1.0, location=[1, 1])
    assert note_novelty(c, brain, seen) == 1 and curiosity_reward(c, world) == 1.0
    assert reward(c, world) == 1.0, "novelty weighted by the curiosity share"
    assert note_novelty(c, brain, seen) == 0, "a discovery pays only once"

    # drive weights are shares of one whole: sum != 1 is rejected, never renormalized
    validate_drives({"survival": 0.75, "curiosity": 0.25})
    try:
        validate_drives({"survival": 1.0, "curiosity": 0.5})
        assert False, "sum 1.5 must be rejected"
    except ValueError:
        pass

    # a survival-only profile's reward ignores novelty entirely
    c.drives = {"survival": 1.0, "curiosity": 0.0}
    c.novelty_gained = 3
    assert reward(c, world) == survival_reward(c, world)


def test_engine_headless():
    from sim.world import World
    world = World(8, 8)
    npc = Entity("npc_1", "npc_1", "npc", 1, 1, 1.0)
    npc.stats["thirst"] = 40.0
    world.entities[npc.id] = npc
    world.put(2, 1, 1, floor="grass", connector=make("well", "water_1"))

    # synchronous think: perceiving the (novel) water triggers a decide inline,
    # and the returned plan executes through the same advance_goals as the game
    prov = FakeProvider([
        {"goals": [{"actions": [{"action": "interact", "params": {"target": "water_1"}}],
                    "importance": 8, "reason": "drink"}]},
    ])
    path = os.path.join(tempfile.mkdtemp(), "engine.jsonl")
    j = Journal(path, "engine")
    engine = Engine(world, [npc], {"npc_1": Brain("npc_1", prov)}, j)
    engine.step(0.3)
    assert prov.calls, "novel perception must trigger a synchronous think"
    assert npc.stats["thirst"] > 95.0, f"the planned drink executed (thirst={npc.stats['thirst']})"
    fam = engine.brains["npc_1"]._familiar_types()
    assert "well" in fam, "the outcome memory makes the well familiar"
    # and the reward pipeline sees the discovery
    assert note_novelty(npc, engine.brains["npc_1"], {}) == 1

    # death mid-run: the engine reaps, the survivor keeps stepping
    doomed = Entity("npc_2", "npc_2", "npc", 5, 5, 1.0)
    doomed.stats["health"], doomed.stats["hunger"] = 0.5, 0.0
    world.entities[doomed.id] = doomed
    engine.npcs.append(doomed)
    engine.npcs_by_id[doomed.id] = doomed
    engine.trackers[doomed.id] = PerceptionTracker(doomed.id)
    engine.pending_obs[doomed.id] = []
    engine.thinking[doomed.id] = False
    died = engine.step(1.0)
    assert [d.id for d in died] == ["npc_2"] and death_cause(doomed) == "starvation"
    assert "npc_2" not in world.entities and engine.npcs == [npc]

    # idle cadence: internal pressure is never a perception event, so with nothing
    # novel left the agent still rethinks once think_interval elapses
    ncalls = len(prov.calls)
    engine.step(2.5)                     # sim_t 3.8 > next_think_at
    assert len(prov.calls) > ncalls, "the idle think cadence must fire"

    # wait semantics: a wait action holds indefinitely (no clock) and only the
    # next decision's goals end it
    wait_goal = Goal(actions=[{"action": "wait"}], importance=5.0)
    npc.goals.add(wait_goal)
    for _ in range(4):
        engine.step(1.0)
    assert npc.goals.current() is wait_goal and wait_goal.status == "active", \
        "waiting never completes on its own"
    engine.post_goals("npc_1", [Goal(actions=[{"action": "say", "params": {"text": "up"}}],
                                     importance=1.0)])
    assert wait_goal.status == "done", "a new decision ends the wait"
    assert npc.goals.current().actions[0]["action"] == "say", "even a lower-importance one"
    j.close()
    rec = [json.loads(l) for l in open(path, encoding="utf-8")]
    types = {r["type"] for r in rec}
    assert {"goals_added", "effects", "death"} <= types, types


def test_episode_runner():
    import random as _random
    from train.run_episodes import profile_key, run_episode, sample_drives, summarize
    from train.extract_dataset import load_keep_set

    rng = _random.Random(7)
    for _ in range(20):
        d = sample_drives(rng)
        assert d["survival"] in (0.0, 0.25, 0.5, 0.75, 1.0), d
        assert abs(sum(d.values()) - 1.0) < 1e-9, f"drives must sum to 1: {d}"
    assert profile_key({"survival": 0.75, "curiosity": 0.25}) == "curiosity=0.25,survival=0.75"

    # a micro-episode end to end: engine + reward accrual + score rows.
    # the exhausted FakeProvider makes every think fail closed -> empty agendas,
    # so NPCs idle and accrue near-full survival reward for the whole budget.
    cfg = {"interact_range": 4, "memory_k": 5, "memory_halflife_s": 300.0}
    tmp = tempfile.mkdtemp()
    rows = run_episode("ep_smoke", os.path.join(tmp, "ep_smoke.jsonl"),
                       FakeProvider([]), cfg, _random.Random(3), n_npcs=2, budget=4.0, dt=0.5)
    assert len(rows) == 2
    for r in rows:
        assert r["cause"] == "alive" and r["survival_s"] == 4.0, r
        # return ~= survival_weight * min-meter(≈1) * budget
        assert abs(r["return"] - r["drives"]["survival"] * 4.0) < 0.2, r
    summarize(rows)

    # reward filtering keeps the top quantile PER PROFILE, not globally
    scores = os.path.join(tmp, "scores.jsonl")
    with open(scores, "w", encoding="utf-8") as f:
        for i, (prof, ret) in enumerate([("s=1", 9.0), ("s=1", 1.0),
                                         ("c=1", 0.4), ("c=1", 0.1)]):
            f.write(json.dumps({"file": f"ep_{i}.jsonl", "npc": f"npc_{i}",
                                "profile": prof, "return": ret}) + "\n")
    keep = load_keep_set(scores, top=0.5)
    assert keep == {("ep_0.jsonl", "npc_0"), ("ep_2.jsonl", "npc_2")}, \
        "low-return explorers must not lose to high-return survivalists' raw numbers"
    # max_keep caps each bucket independently (bounds an ever-growing elite pool)
    keep = load_keep_set(scores, top=1.0, max_keep=1)
    assert keep == {("ep_0.jsonl", "npc_0"), ("ep_2.jsonl", "npc_2")}, \
        "cap keeps each bucket's best, never evicting one profile for another"


def test_think_debounce():
    # near-simultaneous novel memories share ONE think: the novelty flag becomes a
    # short due-window instead of firing a call per memory
    from sim.world import World
    world = World(6, 6)
    npc = Entity("npc_1", "npc_1", "npc", 1, 1, 1.0)
    world.entities[npc.id] = npc
    goalset = {"goals": [{"actions": [{"action": "wait"}], "importance": 1}]}
    prov = FakeProvider([goalset] * 5)
    j = Journal(os.path.join(tempfile.mkdtemp(), "debounce.jsonl"), "debounce")
    eng = Engine(world, [npc], {"npc_1": Brain("npc_1", prov)}, j)
    eng.step(0.1)                        # spawn think fires via cadence
    base = len(prov.calls)
    brain = eng.brains["npc_1"]
    brain.record_events([{"kind": "heard_say", "speaker": "a", "speaker_type": "npc",
                          "speaker_pos": [2, 2], "text": "one"}], eng.sim_t, [1, 1])
    eng.step(0.1)                        # inside the window: held, not dispatched
    brain.record_events([{"kind": "heard_say", "speaker": "b", "speaker_type": "npc",
                          "speaker_pos": [3, 3], "text": "two"}], eng.sim_t, [1, 1])
    eng.step(0.1)                        # second novelty joins the same window
    assert len(prov.calls) == base, "the window holds near-simultaneous novelties"
    eng.step(0.3)                        # window closed -> exactly one think for both
    assert len(prov.calls) == base + 1, "one call covers the batch"
    assert eng.think_due_at["npc_1"] is None
    j.close()


def test_felt_memories():
    # event -> structured 'felt' memory (did = external act, felt = internal state)
    mems = memories_from_event({"kind": "felt_stat", "stat": "thirst", "value": 82,
                                "direction": "falling"}, [5, 5], 10.0, "npc_1")
    assert len(mems) == 1 and mems[0]["sense"] == "felt"
    s = mems[0]["subject"]
    assert s["kind"] == "stat" and s["ref"] == "thirst" and s["type"] == "falling"
    assert s["info"] == {"stat": "thirst", "value": 82}

    def stat_subject(name, val, direction):
        return {"kind": "stat", "ref": name, "type": direction, "pos": None,
                "info": {"stat": name, "value": val}}

    # a ticking stat folds into ONE spanning run (only info differs each step)
    st = MemoryStore("n", consolidator=Consolidator())
    run, _ = st.record("felt", stat_subject("hunger", 99, "falling"), [1, 1], 2.0)
    for val, t in ((98, 4.0), (97, 6.0), (96, 8.0)):     # 2s cadence = the real drain rate
        _, merged = st.record("felt", stat_subject("hunger", val, "falling"), [1, 1], t)
        assert merged is True, f"tick at t={t} must fold into the run"
    assert len(st) == 1 and run["count"] == 4
    assert run["origin"]["info"]["value"] == 99 and run["subject"]["info"]["value"] == 96
    assert render_memory(run) == "I felt my hunger fell from 99 to 96", render_memory(run)

    # another stat at the SAME value must not fold in (ref + info differ)
    _, merged = st.record("felt", stat_subject("thirst", 96, "falling"), [1, 1], 8.5)
    assert merged is False and len(st) == 2

    # a direction reversal starts a new record (type + info differ)
    _, merged = st.record("felt", stat_subject("hunger", 97, "rising"), [1, 1], 9.0)
    assert merged is False and len(st) == 3

    # recall can filter on the new sense, and felt runs are retrievable
    rec = validate_intent({"action": "recall", "params": {"query": "hunger", "sense": "felt"}})
    assert rec["params"]["sense"] == "felt"
    got = st.retrieve(tokenize("thirst"), now_t=9.5, query_sense="felt", k=1)
    assert got and got[0]["subject"]["ref"] == "thirst"

    # the engine wires interoception through the same pipeline: draining needs
    # produce merged felt runs, not a flood of records
    from sim.world import World
    world = World(6, 6)
    npc = Entity("npc_1", "npc_1", "npc", 1, 1, 1.0)
    world.entities[npc.id] = npc
    j = Journal(os.path.join(tempfile.mkdtemp(), "felt.jsonl"), "felt")
    eng = Engine(world, [npc], {"npc_1": Brain("npc_1", FakeProvider([]))}, j)
    for _ in range(16):
        eng.step(0.5)                    # 8 sim-seconds of hunger/thirst drain
    j.close()
    felt = [m for m in eng.brains["npc_1"].store.memories if m["sense"] == "felt"]
    by_stat = {m["subject"]["ref"] for m in felt}
    assert by_stat == {"hunger", "thirst"}, by_stat   # health held full -> nothing felt
    assert all(m["count"] > 1 for m in felt), "ticking stats must consolidate into runs"
    assert all(m["subject"]["type"] == "falling" for m in felt)


def test_curiosity():
    b = Brain("npc_1", object())
    snap = {"t": 1.0, "self_id": "npc_1", "self_pos": [5, 5, 1.0],
            "visible_entities": [{"id": "bush_1", "type": "berry_bush", "pos": [6, 5]}],
            "drives": {"survival": 0.75, "curiosity": 0.25}, "recent_perceptions": []}
    # familiarity is a plain fact on each visible thing, derived from memory
    ann = b._annotate_familiarity(snap)
    assert ann["visible_entities"][0]["familiar"] is False, "never interacted -> unfamiliar"
    assert "familiar" not in snap["visible_entities"][0], "the input snapshot is not mutated"
    # once it has INTERACTED with a berry_bush, its kind is familiar
    b.record_events([{"kind": "did_interact", "target": "bush_1", "target_type": "berry_bush",
                      "target_pos": [6, 5], "outcome": "ok", "effect": "picked a berry"}],
                    now_t=2.0, location=[5, 5])
    assert "berry_bush" in b._familiar_types()
    assert b._annotate_familiarity(snap)["visible_entities"][0]["familiar"] is True


def test_brain_validation():
    good = {"action": "move", "params": {"x": 5, "y": 6}}
    assert validate_intent(good) == good
    assert validate_intent({"action": "wait"}) == {"action": "wait"}
    assert validate_intent({"action": "none"}) is None, "none was replaced by wait"
    assert validate_intent({"action": "idle"}) is None, "idle is no longer an action"
    assert validate_intent({"action": "fly"}) is None
    assert validate_intent({"action": "move", "params": {"x": "a", "y": 2}}) is None
    assert validate_intent({"action": "interact", "params": {}}) is None
    assert validate_intent("nope") is None
    inv = {"action": "inventory", "params": {"op": "arrange", "item": "torch"}}
    assert validate_intent(inv) == inv
    assert validate_intent({"action": "inventory", "params": {"op": "equip", "item": "torch"}}) is None, "equip/stow folded into arrange"
    assert validate_intent({"action": "inventory", "params": {"op": "bogus", "item": "torch"}}) is None
    assert validate_intent({"action": "inventory", "params": {"op": "use"}}) is None
    assert validate_intent({"action": "use", "params": {"item": "torch"}}) is None, "use is now an inventory op"
    say = {"action": "say", "params": {"text": "hello there"}}
    assert validate_intent(say) == say
    assert validate_intent({"action": "say", "params": {"text": "   "}}) is None
    assert validate_intent({"action": "say", "params": {}}) is None
    assert validate_intent({"action": "end_dialogue"}) is None, "dialogue sessions are gone"
    rec = validate_intent({"action": "recall", "params": {"query": "bears", "sense": "saw"}})
    assert rec == {"action": "recall", "params": {"query": "bears", "sense": "saw"}}
    assert validate_intent({"action": "recall", "params": {"query": "x", "sense": "bogus"}}) == {"action": "recall", "params": {"query": "x"}}
    assert validate_intent({"action": "recall", "params": {"query": "  "}}) is None
    look = {"action": "look", "params": {"x": 3, "y": 4}}
    assert validate_intent(look) == look
    assert validate_intent({"action": "look", "params": {"x": 3}}) is None, "look needs both coords"
    assert validate_intent({"action": "look", "params": {"x": 1.5, "y": 2}}) is None, "look coords are ints"
    # an optional "reason" passes through, capped; absence is fine
    assert validate_intent({"action": "wait", "reason": "safe, nothing to do"}) == {"action": "wait", "reason": "safe, nothing to do"}
    assert "reason" not in validate_intent({"action": "wait"})


def _subject(kind, ref, typ, pos, info=None):
    return {"kind": kind, "ref": ref, "type": typ, "pos": pos, "info": info or {}}


def test_memory_store():
    s = MemoryStore("a")
    assert len(s) == 0
    s.add("saw", _subject("entity", "player", "player", [5, 5], {"text": "lava is dangerous"}), [5, 5], 10)
    s.add("saw", _subject("entity", "rock_1", "rock", [20, 20]), [20, 20], 500)
    assert len(s) == 2
    assert "salience" not in s.memories[0], "salience is gone"
    assert s.memories[0]["observer_loc"] == [5, 5]
    assert s.memories[0]["last_accessed"] == 10 and s.memories[0]["access_count"] == 0

    got = s.retrieve(tokenize("lava"), now_t=520, k=2, query_loc=[6, 6])
    assert got[0]["subject"]["ref"] == "player", got[0]
    assert s.memories[0]["access_count"] == 1 and s.memories[0]["last_accessed"] == 520

    got = s.retrieve(tokenize("rock"), now_t=520, k=2)
    assert got[0]["subject"]["type"] == "rock"

    # spatial anchors on the SUBJECT's position, not the observer's stance
    near = s.add("saw", _subject("entity", "a", "thing", [10, 10]), [0, 0], 500)
    far = s.add("saw", _subject("entity", "b", "thing", [40, 40]), [0, 0], 500)
    got = s.retrieve(tokenize("thing"), now_t=501, k=4, query_loc=[11, 11])
    ids = [m["id"] for m in got]
    assert ids.index(near["id"]) < ids.index(far["id"]), "subject proximity must outrank distance"


def test_memory_consolidation():
    """One rule: same sense + <=1 differing subject field + close in time -> merge."""
    # the diff primitive itself
    a = _subject("entity", "npc_2", "npc", [3, 3])
    assert subject_diff(a, a) == 0
    assert subject_diff(a, _subject("entity", "npc_2", "npc", [4, 3])) == 1          # pos only (movement)
    assert subject_diff(a, _subject("entity", "npc_3", "npc", [4, 3])) == 2          # ref + pos

    s = MemoryStore("npc_1", consolidator=Consolidator())
    m1, merged = s.record("saw", _subject("entity", "npc_2", "npc", [3, 3]), [0, 0], 1.0, direction="E")
    assert merged is False and len(s) == 1 and m1["count"] == 1

    # same entity, only pos changes (1 diff), small gap -> folds into the same record
    m2, merged = s.record("saw", _subject("entity", "npc_2", "npc", [4, 3]), [1, 1], 1.2, direction="NE")
    assert merged is True and m2 is m1 and len(s) == 1
    m3, merged = s.record("saw", _subject("entity", "npc_2", "npc", [5, 3]), [2, 2], 1.4, direction="E")
    assert merged is True and len(s) == 1

    # spans both the path (origin -> latest subject) and the time (t -> t_end)
    assert m1["t"] == 1.0 and m1["t_end"] == 1.4 and m1["count"] == 3
    assert m1["origin"]["pos"] == [3, 3] and m1["subject"]["pos"] == [5, 3]
    assert m1["direction"] == "E" and m1["observer_loc"] == [2, 2]   # endpoint values carried
    assert render_memory(m1) == "I saw npc (npc_2) move from [3, 3] to [5, 3] to the E", render_memory(m1)

    # the run stays findable and still carries its origin after the latest moved on
    assert s.retrieve(tokenize("npc_2"), now_t=1.5, k=1)[0]["origin"]["pos"] == [3, 3]

    # a different entity differs in 2 fields (ref + pos) -> NOT merged, its own record
    other, merged = s.record("saw", _subject("entity", "npc_3", "npc", [8, 8]), [0, 0], 1.5)
    assert merged is False and len(s) == 2, "two differing fields must not merge"

    # a long gap breaks the run even for the same entity
    late, merged = s.record("saw", _subject("entity", "npc_2", "npc", [6, 3]), [0, 0], 10.0)
    assert merged is False and len(s) == 3

    # 'left view' differs from the run in pos AND info (2 fields) -> its own record
    left, merged = s.record("saw", _subject("entity", "npc_2", "npc", None, {"event": "left view"}), [0, 0], 10.1)
    assert merged is False and render_memory(left) == "npc (npc_2) left my view"

    # the separate npc_3 record is findable by its own tokens
    got = s.retrieve(tokenize("npc_3"), now_t=1.6, k=1)
    assert got[0]["subject"]["ref"] == "npc_3"

    # recency anchors on t_end: a run last seen at 2.0 scores as just-happened
    fresh = MemoryStore("npc_2", consolidator=Consolidator())
    run, _ = fresh.record("saw", _subject("entity", "x", "thing", [1, 1]), [0, 0], 1.0)
    fresh.record("saw", _subject("entity", "x", "thing", [2, 1]), [0, 0], 2.0)
    assert run["t_end"] == 2.0
    assert fresh.retrieve(tokenize("thing"), now_t=2.0, k=1)[0]["_score"] > 0.99
    print("memory consolidation ok")


def test_bears_example():
    """The worked example: type-match + subject-position spatial surfaces the bear."""
    s = MemoryStore("npc_1")
    s.add("saw", _subject("entity", "bear_1", "bear", [12, 5]), [19, 9], 8.0, direction="NW")
    s.add("saw", _subject("entity", "npc_2", "npc", [3, 3]), [19, 9], 9.0)
    # player at [13,6] asks about bears; "bears" must stem to the "bear" type
    got = s.retrieve(tokenize("have you seen any bears around here lately"), now_t=20.0, k=2, query_loc=[13, 6])
    assert got[0]["subject"]["ref"] == "bear_1", got
    assert got[0]["_score"] > got[1]["_score"], "bear should outrank the far npc"


def test_recall_action_loop():
    # LLM recalls first, then answers; the recalled memory must reach the second prompt.
    prov = FakeProvider([
        {"action": "recall", "params": {"query": "bear"}},
        {"goals": [{"actions": [{"action": "say", "params": {"text": "yes, a bear to the north"}}],
                    "importance": 2, "reason": "answer the question"}]},
    ])
    brain = Brain("npc_1", prov, memory_k=3)
    brain.record_events([{"kind": "entity_entered", "id": "bear_1", "pos": [12, 5], "etype": "bear"}], 5.0, [10, 5])
    snap = {"t": 8.0, "self_id": "npc_1", "self_pos": [10, 5, 1.0], "hp": 100,
            "visible_entities": [], "recent_perceptions": []}
    goals = brain.decide(snap)
    assert goals[0].actions[0]["action"] == "say", goals
    assert len(prov.replies) == 0, "both the recall and the say were consumed"
    second_prompt = prov.calls[-1][1]
    assert "bear" in second_prompt, "recalled bear memory must be in the second prompt"


def test_recall_cap():
    prov = FakeProvider([{"action": "recall", "params": {"query": "x"}}] * (MAX_RECALLS + 3))
    brain = Brain("npc_1", prov, memory_k=2)
    snap = {"t": 1.0, "self_pos": [0, 0, 1.0], "hp": 100, "visible_entities": [], "recent_perceptions": []}
    goals = brain.decide(snap)
    assert goals == [], goals                      # exhausted recall -> empty agenda
    assert len(prov.calls) == MAX_RECALLS + 2, "recall loop is bounded"


def test_look_action_loop():
    # LLM looks at a remembered area far from itself, then plans; the looked-at map
    # must reach the second prompt so the move is grounded in remembered terrain.
    prov = FakeProvider([
        {"action": "look", "params": {"x": 40, "y": 40}},
        {"goals": [{"actions": [{"action": "move", "params": {"x": 41, "y": 40}}],
                    "importance": 2, "reason": "head to the remembered clearing"}]},
    ])
    brain = Brain("npc_1", prov)
    brain.spatial.observe_many([((x, y, 1), "floor") for x in range(39, 42) for y in range(39, 42)])
    snap = {"t": 1.0, "self_id": "npc_1", "self_pos": [1, 1, 1.0], "hp": 100,
            "vision_radius": 8, "visible_entities": [], "recent_perceptions": []}
    goals = brain.decide(snap)
    assert goals and goals[0].actions[0]["params"] == {"x": 41, "y": 40}, goals
    first_prompt, second_prompt = prov.calls[0][1], prov.calls[-1][1]
    assert "x=36..44" not in first_prompt, "the far area isn't shown until looked at"
    assert "x=36..44" in second_prompt, "look renders the remembered map around (40,40)"
    assert len(prov.replies) == 0, "look then goals both consumed"


def test_sense_and_direction_axes():
    s = MemoryStore("npc")
    seen = s.add("saw", _subject("entity", "bear_1", "bear", [12, 5]), [10, 5], 5.0, direction="E")
    told = s.add("heard", _subject("entity", "player", "player", [9, 5], {"text": "bears live east"}), [9, 5], 6.0)

    # sense axis: "have you SEEN bears" should prefer the saw-memory over the heard one
    got = s.retrieve(tokenize("bear"), now_t=7.0, query_sense="saw", k=2)
    assert got[0]["id"] == seen["id"], got

    # direction axis: asking about the east favors the E-bearing memory (told has no direction)
    got = s.retrieve(set(), now_t=7.0, query_direction="E", k=2)
    assert got[0]["id"] == seen["id"], got

    # an unspecified axis must not dilute: a pure-subject probe still ranks by subject+time
    got = s.retrieve(tokenize("bear"), now_t=7.0, k=2)
    assert {m["id"] for m in got} == {seen["id"], told["id"]}


def test_memory_write_is_code_only():
    class NoLLM:
        def __init__(self):
            self.calls = 0

        def chat_json(self, system, user):
            self.calls += 1
            raise AssertionError("memory writing must never call the LLM")

    prov = NoLLM()
    brain = Brain("npc_x", prov, memory_k=3)
    events = [
        {"kind": "entity_entered", "id": "player", "pos": [1, 1], "etype": "player"},
        {"kind": "entity_moved", "id": "player", "pos": [2, 1], "etype": "player"},
    ]
    saved = brain.record_events(events, now_t=100.0, location=[2, 3])
    assert prov.calls == 0, "no LLM call during memory writing"
    # the two sightings of the same entity consolidate into one spanning record
    assert len(saved) == 2 and len(brain.store) == 1, "same-entity run folds into one memory"
    m = brain.store.memories[0]
    assert m["observer_loc"] == [2, 3], "memory records where the rememberer stood"
    assert m["subject"]["ref"] == "player"
    assert m["origin"]["pos"] == [1, 1] and m["subject"]["pos"] == [2, 1]
    assert m["count"] == 2 and m["t"] == 100.0 and m["t_end"] == 100.0
    assert m["sense"] == "saw" and "salience" not in m


def test_speech_memories():
    brain = Brain("npc_1", FakeProvider([]))
    # hearing someone speak -> a `heard` memory anchored on the speaker, text searchable
    heard = brain.record_events([{"kind": "heard_say", "speaker": "player", "speaker_type": "player",
                                  "speaker_pos": [4, 4], "text": "have you seen bears?"}], now_t=50.0, location=[5, 5])
    assert len(heard) == 1 and heard[0]["sense"] == "heard"
    assert heard[0]["subject"]["ref"] == "player" and heard[0]["subject"]["pos"] == [4, 4]
    got = brain.store.retrieve(tokenize("bears"), now_t=51.0, k=1, query_loc=[4, 4])
    assert "bears" in got[0]["subject"]["info"]["text"]

    # speaking myself -> a `did` memory (speech is an action I enacted)
    said = brain.record_events([{"kind": "did_say", "text": "no, only wolves"}], now_t=52.0, location=[5, 5])
    assert said[0]["sense"] == "did" and said[0]["subject"]["kind"] == "self"
    assert "wolves" in said[0]["subject"]["info"]["text"]


def test_action_memories():
    brain = Brain("npc_1", FakeProvider([]))
    # a completed move -> a 'did' memory anchored on the destination tile
    saved = brain.record_events([{"kind": "did_move", "pos": [8, 3], "outcome": "arrived"}], now_t=10.0, location=[8, 3])
    m = saved[0]
    assert m["sense"] == "did" and m["subject"]["kind"] == "tile" and m["subject"]["pos"] == [8, 3]
    assert "moved to [8, 3]" in render_memory(m)

    # a failed move renders the outcome
    saved = brain.record_events([{"kind": "did_move", "pos": [40, 40], "outcome": "unreachable"}], now_t=11.0, location=[5, 5])
    assert "unreachable" in render_memory(saved[0])

    # a failed interact -> 'did' memory about the target
    saved = brain.record_events([{"kind": "did_interact", "target": "door_1", "target_type": "door",
                                  "target_pos": [6, 6], "outcome": "out_of_range"}], now_t=12.0, location=[3, 3])
    assert saved[0]["sense"] == "did" and saved[0]["subject"]["ref"] == "door_1"
    assert "out_of_range" in render_memory(saved[0])
    # and it's searchable by the thing I acted on
    got = brain.store.retrieve(tokenize("door"), now_t=13.0, k=1)
    assert got[0]["subject"]["ref"] == "door_1"


def test_render_and_bearing():
    assert bearing([5, 5], [5, 1]) == "N"
    assert bearing([5, 5], [9, 9]) == "SE"
    assert bearing([5, 5], [5, 5]) is None
    m = {"sense": "saw", "direction": "NW", "subject": _subject("entity", "bear_1", "bear", [12, 5])}
    text = render_memory(m)
    assert "bear" in text and "NW" in text and "[12, 5]" in text


def test_recall_surfaces_memories():
    prov = FakeProvider([{"goals": [{"actions": [{"action": "wait"}], "importance": 1}]}])
    brain = Brain("npc_x", prov, memory_k=3)
    brain.record_events(
        [{"kind": "entity_entered", "id": "player", "pos": [1, 1], "etype": "player"}],
        now_t=100.0,
        location=[2, 3],
    )
    snap = {
        "t": 110.0,
        "self_id": "npc_x",
        "self_pos": [2, 3, 1.0],
        "hp": 100,
        "visible_entities": [{"id": "player", "pos": [1, 1]}],
        "recent_perceptions": [{"kind": "entity_entered", "id": "player"}],
    }
    goals = brain.decide(snap, events=[])
    assert len(goals) == 1 and goals[0].actions == [{"action": "wait"}], "waiting is a decision"
    assert len(brain.store) == 1, "decide must not create memories, only recall"
    # internal state joins the probe: stat names surface the outcome memories that
    # mention them, so what relieved a need in the past reaches the prompt unasked
    probe = brain.query_from({"stats": {"thirst": 12, "hunger": 80}, "visible_entities": []})
    assert {"thirst", "hunger"} <= probe, probe
    last_user = prov.calls[-1][1]
    assert "player" in last_user and "saw" in last_user, last_user


def test_stream_text_extraction():
    # key not seen yet
    assert _text_value_so_far('{"action":"say","par') is None
    # value streaming, still open
    assert _text_value_so_far('{"action":"say","params":{"text":"Hel') == ("Hel", False)
    # value closed
    assert _text_value_so_far('{"action":"say","params":{"text":"Hello"}}') == ("Hello", True)
    # escaped quote inside the text
    assert _text_value_so_far('{"text":"she said \\"hi\\""}') == ('she said "hi"', True)
    # newline escape decoded (this frame closes the value)
    assert _text_value_so_far('{"text":"a\\nb"') == ("a\nb", True)
    # same content still open (no closing quote yet)
    assert _text_value_so_far('{"text":"a\\nb') == ("a\nb", False)
    # escape split across a chunk boundary: stop before the lone backslash, no broken char
    assert _text_value_so_far('{"text":"a\\') == ("a", False)

    # simulate incremental streaming: deltas must be monotonic and concatenate to the whole
    frames = ['{"action":"say","params":{"text":"',
              '{"action":"say","params":{"text":"Hi ',
              '{"action":"say","params":{"text":"Hi the',
              '{"action":"say","params":{"text":"Hi there"}}']
    emitted, out = 0, ""
    for raw in frames:
        res = _text_value_so_far(raw)
        if res is None:
            continue
        decoded, _ = res
        assert decoded.startswith(out), "text must only grow"
        out = decoded
    assert out == "Hi there"


def test_provider_body_shape():
    # localhost is normalized to IPv4 to dodge the Windows ::1 connect stall
    p = OllamaProvider("http://localhost:11434/", "gemma4")
    assert p.url == "http://127.0.0.1:11434", p.url
    assert OllamaProvider("http://192.168.1.5:11434", "m").url == "http://192.168.1.5:11434"
    assert p.keep_alive == "10m"
    body = {
        "model": p.model,
        "stream": False,
        "format": "json",
        "think": False,
    }
    assert body["think"] is False


def test_journal():
    path = os.path.join(tempfile.mkdtemp(), "smoke.jsonl")
    j = Journal(path, "smoke")
    j.log("system", "spawn", entities={"player": [3, 3]})
    j.clock = lambda: 42.5                # the Engine points this at its sim clock
    j.log("player", "dialogue_msg", partner="npc_1", text="hi")
    j.close()
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    assert all({"t", "run", "actor", "type", "payload"} <= set(rec) for rec in records)
    assert records[0]["t"] == 0.0 and records[1]["t"] == 42.5, "t is sim time: 0 before an engine, its clock after"
    print(f"journal records: {len(records)}")


def test_goal_infrastructure():
    # importance orders the list; ties fall back to insertion order (FIFO)
    def goal(action, params, importance):
        return Goal(actions=[{"action": action, "params": params}], importance=importance)

    gl = GoalList()
    gl.add(goal("move", {"x": 1, "y": 1}, 2.0))
    gl.add(goal("say", {"text": "first low"}, 1.0))
    gl.add(goal("say", {"text": "second low"}, 1.0))
    gl.add(goal("interact", {"target": "npc_2"}, 5.0))
    order = [g.summary() for g in gl]
    assert order[0] == "interact(npc_2)", order          # highest importance first
    assert order[1] == "move(1,1)", order
    assert order[2] == 'say("first low")', order          # tie -> earlier added first
    assert order[3] == 'say("second low")', order
    assert gl.current().summary() == "interact(npc_2)"

    # completing the top surfaces the next; a later high-importance goal preempts
    top = gl.current()
    gl.complete(top)
    assert top.status == "done" and len(gl) == 3
    assert gl.current().summary() == "move(1,1)"
    gl.add(goal("say", {"text": "RUN"}, 99.0))
    assert gl.current().summary() == 'say("RUN")', "high importance must jump the queue"

    # locations() feeds spatial memory: move-targets paired with their goal importance
    assert ((1, 1), 2.0) in gl.locations(), gl.locations()      # the move goal added earlier

    # a multi-action plan runs its actions in order; step tracks progress
    plan = Goal(actions=[{"action": "move", "params": {"x": 2, "y": 2}},
                         {"action": "say", "params": {"text": "arrived"}}], importance=3.0)
    assert plan.current_action["action"] == "move"
    assert plan.advance() is True and plan.current_action["action"] == "say"
    assert plan.summary() == 'move(2,2)->say("arrived") [1/2]'
    assert plan.advance() is False, "no actions left -> plan complete"

    # validate_goals is STRICT: a full match returns Goals; ANY contract
    # violation returns None (a bad response), never a coerced partial.
    parsed = validate_goals({"goals": [
        {"actions": [{"action": "move", "params": {"x": 3, "y": 4}},
                     {"action": "say", "params": {"text": "hi"}}],
         "importance": 7, "reason": "reposition"},
        {"actions": [{"action": "interact", "params": {"target": "npc_2"}}],
         "importance": 15},                                    # out of range -> clamped
    ]})
    assert [[a["action"] for a in g.actions] for g in parsed] == [["move", "say"], ["interact"]], parsed
    assert parsed[0].importance == 7.0 and parsed[0].reason == "reposition"
    assert parsed[1].importance == 10.0, "importance clamps to [0,10]"

    assert validate_goals({"goals": []}) is None, "an empty goals list is not a decision"
    assert validate_goals({"goals": [{"actions": [{"action": "move", "params": {"x": 1, "y": 1}},
                                                  {"action": "recall", "params": {"query": "x"}}],
                                      "importance": 3}]}) is None, "recall inside a plan is invalid"
    assert validate_goals({"goals": [{"actions": [{"action": "fly"}], "importance": 3}]}) is None, "unknown action"
    waitset = validate_goals({"goals": [{"actions": [{"action": "wait"}], "importance": 3}]})
    assert waitset is not None and waitset[0].actions == [{"action": "wait"}], \
        "wait is a world action inside a plan (a trainable decision)"
    assert validate_goals({"goals": [{"actions": [{"action": "say", "params": {"text": "hi"}}]}]}) is None, "importance is required"
    assert validate_goals({"goals": [{"actions": [{"action": "say", "params": {"text": "hi"}}],
                                      "importance": True}]}) is None, "boolean is not importance"
    assert validate_goals({"action": "move", "params": {"x": 1, "y": 1}}) is None, "bare action is not a goal-set"
    assert validate_goals([{"actions": [{"action": "say", "params": {"text": "hi"}}], "importance": 1}]) is None, "bare list rejected"
    assert validate_goals("nope") is None
    assert validate_goals({"goals": "bad"}) is None

    # filter_response routes the whole reply space: goals / recall / look / bad
    assert filter_response({"action": "wait", "reason": "safe"})["kind"] == "bad", \
        "standalone wait left the contract; wait lives inside goal-sets now"
    assert filter_response({"action": "recall", "params": {"query": "bear"}})["kind"] == "recall"
    assert filter_response({"goals": [{"actions": [{"action": "say", "params": {"text": "hi"}}],
                                       "importance": 2}]})["kind"] == "goals"
    assert filter_response({"action": "look", "params": {"x": 4, "y": 5}})["kind"] == "look"
    assert filter_response({"nonsense": 1})["kind"] == "bad"
    assert filter_response({"action": "recall"})["kind"] == "bad", "recall needs a query"
    assert filter_response({"action": "look", "params": {"x": 4}})["kind"] == "bad", "look needs both coords"
    assert filter_response({"goals": [{"actions": [{"action": "fly"}], "importance": 1}]})["kind"] == "bad"

    # goal_from_intent bridges a single validated intent; recall/look/garbage -> None
    g = goal_from_intent({"action": "say", "params": {"text": "hey"}}, importance=3.0)
    assert g is not None and g.actions[0]["action"] == "say" and g.importance == 3.0
    assert goal_from_intent({"action": "wait"}) is not None, "wait is a world action now"
    assert goal_from_intent({"action": "recall", "params": {"query": "x"}}) is None
    assert goal_from_intent({"action": "look", "params": {"x": 1, "y": 2}}) is None
    assert goal_from_intent("nope") is None
    print("goal infrastructure ok")


def main() -> None:
    test_movement_and_interact()
    test_layers()
    test_pathing()
    test_perception()
    test_spatial_memory()
    test_terrain_resources()
    test_needs()
    test_death()
    test_reward()
    test_engine_headless()
    test_episode_runner()
    test_think_debounce()
    test_felt_memories()
    test_curiosity()
    test_brain_validation()
    test_goal_infrastructure()
    test_memory_store()
    test_memory_consolidation()
    test_bears_example()
    test_recall_action_loop()
    test_recall_cap()
    test_look_action_loop()
    test_sense_and_direction_axes()
    test_memory_write_is_code_only()
    test_speech_memories()
    test_action_memories()
    test_render_and_bearing()
    test_recall_surfaces_memories()
    test_stream_text_extraction()
    test_provider_body_shape()
    test_journal()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
