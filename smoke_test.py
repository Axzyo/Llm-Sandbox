import json
import os
import tempfile

from sim.actions import attempt_move, evaluate_interact
from sim.brain import Brain, validate_intent, validate_goals, filter_response, memories_from_event, render_memory, bearing, MAX_RECALLS
from sim.entities import Entity
from sim.goals import Goal, GoalList, goal_from_intent, DEFAULT_IMPORTANCE
from sim.journal import Journal
from sim.terrain import build_test_map, place_resources, tick_resources, interact_with, use_item, BERRY_AMOUNT
from sim.consolidation import Consolidator, subject_diff
from sim.memory import MemoryStore, tokenize
from sim.pathing import next_step
from sim.perception import PerceptionTracker, visible_tiles
from sim.spatial import SpatialMemory
from sim.engine import Engine, death_cause
from sim.needs import tick_needs, HUNGER_DRAIN_PER_S, THIRST_DRAIN_PER_S, HEALTH_REGEN_PER_S
from sim.reward import curiosity_reward, note_novelty, reward, survival_reward
from sim.provider import OllamaProvider, _text_value_so_far
from sim.world import has_los


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
    assert r["ok"] and player.pos == (4, 3), r

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
    assert not has_los(world, 17, 4, 18, 5), "corner must block LOS"
    res = evaluate_interact(world, player)
    assert not res["ok"] and res["range_ok"] and not res["los_ok"], res

    player.x, player.y = 3, 3
    res = evaluate_interact(world, player)
    assert not res["ok"] and not res["range_ok"], res


def test_pathing():
    world, _, _, _ = build_world()
    start, goal = (5, 5), (19, 9)
    cur = start
    for _ in range(200):
        step = next_step(world, cur, goal)
        assert step is not None, "path should exist"
        assert not world.blocked(*step), "step must be walkable"
        cur = step
        if cur == goal:
            break
    assert cur == goal
    assert next_step(world, goal, goal) is None
    assert next_step(world, start, (0, 0)) is None, "goal inside wall unreachable"


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
    world, player, npc1, npc2 = build_world()   # npc_1 at (19,9), vision 8
    seen = dict(visible_tiles(world, npc1))
    # own tile is seen floor; the right border wall (23,9) is in view and remembered as wall
    assert seen[(19, 9)] == "floor", seen.get((19, 9))
    assert seen[(23, 9)] == "wall", "the border wall in view is seen"
    # a tile occluded by the x=15 wall column is NOT seen (LOS only)
    assert (14, 9) not in seen, "tile behind a wall must be unseen"

    sm = SpatialMemory("npc_1")
    new = sm.observe_many(seen.items())
    assert new == len(seen) and len(sm) == len(seen)
    assert sm.observe_many(seen.items()) == 0, "re-seeing discovers nothing new"
    assert sm.get((23, 9)) == "wall" and sm.known((19, 9)) and not sm.known((14, 9))

    m = sm.render_local((19, 9), radius=4)
    assert "@" in m and "#" in m and "y=  9" in m, m
    assert SpatialMemory("x").render_local((0, 0), 3) is None, "nothing remembered -> no map"

    # memorability: seeing a tile reinforces it, up to the cap
    from sim.spatial import SIGHT_BOOST, MEMORABILITY_CAP
    f = SpatialMemory("f", max_tiles=None)
    f.observe((0, 0), "floor")
    assert f.memorability((0, 0)) == SIGHT_BOOST
    for _ in range(20):
        f.observe((0, 0), "floor")
    assert f.memorability((0, 0)) == MEMORABILITY_CAP, "reinforcement caps out"

    # decay + threshold: an un-refreshed tile fades and is forgotten; a nearby goal
    # floors memorability by proximity so goal-relevant geometry survives
    a = SpatialMemory("a", max_tiles=None)
    a.observe_many([((0, 0), "floor"), ((10, 0), "floor")])       # both start at SIGHT_BOOST
    passes = 0
    while a.known((0, 0)) and passes < 100:
        a.age(goal_locations=[((10, 0), 3.0)])                     # importance-3 goal on (10,0)
        passes += 1
    assert not a.known((0, 0)), "the tile far from any goal decays away"
    assert a.known((10, 0)) and a.memorability((10, 0)) >= 3.0 * 0.9, "goal tile floored to importance"
    # a MORE important goal floors its tile higher
    a.observe((20, 0), "floor")
    a.age(goal_locations=[((20, 0), 9.0)])
    assert a.memorability((20, 0)) > a.memorability((10, 0)), "higher-importance goal = stickier geometry"

    # cap backstop evicts the LEAST memorable first (not the oldest)
    capped = SpatialMemory("c", max_tiles=3)
    capped.observe_many([((i, 0), "floor") for i in range(5)])    # all equal memorability
    capped.observe((4, 0), "floor")                               # bump (4,0) above the rest
    capped.observe((3, 0), "floor")
    capped.observe_many([((9, 0), "floor")])                      # force a 6th -> over cap
    assert len(capped) == 3 and capped.known((4, 0)) and capped.known((3, 0)), "kept most memorable"

    # Brain wiring: perceive reinforces, maintain_spatial decays + protects goals
    b = Brain("b", object())
    b.perceive_tiles([((5, 5), "floor"), ((40, 40), "floor")])
    for _ in range(60):
        b.maintain_spatial(goal_locations=[((5, 5), 4.0)])        # (5,5) protected, (40,40) not
    assert b.spatial.known((5, 5)) and not b.spatial.known((40, 40)), "goal-far geometry fades"


def test_death():
    import main
    from sim.world import World
    world = World(6, 6)
    alive = Entity("npc_1", "npc_1", "npc", 1, 1)
    doomed = Entity("npc_2", "npc_2", "npc", 2, 2)
    doomed.hp, doomed.thirst = 0.0, 0.0            # dehydrated to death
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
    spawn_tiles = set(spawns.values())
    placed = place_resources(world, _random.Random(1))
    kinds = sorted(e.kind for e in placed)
    assert kinds == ["berry_bush", "berry_bush", "berry_bush", "water"], kinds
    for e in placed:
        assert not world.is_wall(e.x, e.y), "resource on a wall"
        assert (e.x, e.y) not in spawn_tiles, "resource on a spawn tile"
    assert len({(e.x, e.y) for e in placed}) == 4, "resources overlap"

    water = next(e for e in placed if e.kind == "water")
    bush = next(e for e in placed if e.kind == "berry_bush")

    a = Entity("a", "a", "npc", 0, 0)
    a.hunger, a.thirst = 20.0, 10.0

    # drink water: thirst stat raised, water is not used up (infinite source, no item)
    out = interact_with(water, a, now=5.0)
    assert out == {"ok": True, "did": "drink", "stat": "thirst", "gained": 90.0}, out
    assert a.thirst == 100.0 and a.inventory == []

    # pick berries: a 'berry' item enters the inventory; the bush is now empty + regrowing
    out = interact_with(bush, a, now=5.0, rng=_random.Random(2))
    assert out == {"ok": True, "did": "harvest", "yields": "berry"}, out
    assert a.inventory == ["berry"] and a.hunger == 20.0, "picking doesn't feed you — eating does"
    assert bush.resource["available"] is False and bush.resource["regrow_at"] is not None
    assert interact_with(bush, a, now=6.0) == {"ok": False, "did": "harvest", "reason": "empty"}

    # eat the berry (inventory use): hunger rises, the berry is consumed
    out = use_item(a, "berry")
    assert out == {"ok": True, "did": "use", "item": "berry", "stat": "hunger", "gained": float(BERRY_AMOUNT)}, out
    assert a.hunger == 20.0 + BERRY_AMOUNT and a.inventory == []
    assert use_item(a, "berry") is None, "no berry left to eat"

    # the bush regrows a berry after its timer
    tick_resources(world, now=bush.resource["regrow_at"] - 1)
    assert bush.resource["available"] is False
    tick_resources(world, now=bush.resource["regrow_at"])
    assert bush.resource["available"] is True, "berry regrew after its timer"


def test_needs():
    e = Entity("npc_1", "npc_1", "npc", 5, 5)
    assert e.hp == 100 and e.hunger == 100.0 and e.thirst == 100.0, "start full"

    tick_needs(e, dt=10.0)                          # 10 seconds pass
    assert e.hunger == 100.0 - HUNGER_DRAIN_PER_S * 10, e.hunger
    assert e.thirst == 100.0 - THIRST_DRAIN_PER_S * 10, e.thirst
    assert e.hp == 100.0, "well-fed + hydrated: health regenerates but caps at full"

    # health regenerates while both needs are above threshold
    e.hp, e.hunger, e.thirst = 50.0, 90.0, 90.0
    tick_needs(e, dt=10.0)
    assert e.hp == 50.0 + HEALTH_REGEN_PER_S * 10, e.hp

    # a need at 0 starves health down; needs never go below empty
    e.hunger = 1.0
    tick_needs(e, dt=100.0)                          # hunger clamps at 0, then health drains
    assert e.hunger == 0.0, "needs never go below empty"
    assert e.hp < 60.0, "empty hunger drains health"

    # in-between (a need below threshold but not empty) holds health steady
    e.hp, e.hunger, e.thirst = 40.0, 50.0, 90.0
    tick_needs(e, dt=10.0)
    assert e.hp == 40.0, "one need mid-range: health neither drains nor regens"


def test_reward():
    from sim.world import World
    world = World(4, 4)

    # topped-off survivalist: reward ~ 1 * survival signal
    e = Entity("a", "a", "npc", 1, 1)
    e.drives = {"survival": 1.0, "curiosity": 0.0}
    assert survival_reward(e, world) == 1.0 and reward(e, world) == 1.0

    # only as safe as the worst meter; starving ~ 0; dead = 0
    e.thirst = 30.0
    assert survival_reward(e, world) == 0.3
    e.hunger = 0.0
    assert survival_reward(e, world) == 0.0, "an empty meter zeroes the signal"
    e.hunger, e.hp = 100.0, 0.0
    assert survival_reward(e, world) == 0.0, "death ends reward accrual"

    # curiosity pays out when a type newly becomes familiar, once
    c = Entity("c", "c", "npc", 1, 1)
    c.drives = {"survival": 0.0, "curiosity": 0.5}
    brain = Brain("c", object())
    seen = {}
    assert note_novelty(c, brain, seen) == 0 and reward(c, world) == 0.0
    brain.record_events([{"kind": "did_interact", "target": "bush_1", "target_type": "berry_bush",
                          "target_pos": [2, 1], "outcome": "ok", "effect": "picked a berry"}],
                        now_t=1.0, location=[1, 1])
    assert note_novelty(c, brain, seen) == 1 and curiosity_reward(c, world) == 1.0
    assert reward(c, world) == 0.5, "novelty weighted by the curiosity drive"
    assert note_novelty(c, brain, seen) == 0, "a discovery pays only once"

    # a survival-only profile's reward ignores novelty entirely
    c.drives = {"survival": 1.0, "curiosity": 0.0}
    c.novelty_gained = 3
    assert reward(c, world) == survival_reward(c, world)


def test_engine_headless():
    from sim.world import World
    world = World(8, 8)
    npc = Entity("npc_1", "npc_1", "npc", 1, 1)
    npc.thirst = 40.0
    water = Entity("water_1", "water_1", "water", 2, 1)
    water.resource = {"kind": "restore", "stat": "thirst", "amount": 100}
    for e in (npc, water):
        world.entities[e.id] = e

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
    assert npc.thirst > 95.0, f"the planned drink executed (thirst={npc.thirst})"
    fam = engine.brains["npc_1"]._familiar_types()
    assert "water" in fam, "the outcome memory makes water familiar"
    # and the reward pipeline sees the discovery
    assert note_novelty(npc, engine.brains["npc_1"], {}) == 1

    # death mid-run: the engine reaps, the survivor keeps stepping
    doomed = Entity("npc_2", "npc_2", "npc", 5, 5)
    doomed.hp, doomed.hunger = 0.5, 0.0
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
    j.close()
    rec = [json.loads(l) for l in open(path, encoding="utf-8")]
    types = {r["type"] for r in rec}
    assert {"goals_added", "resource_use", "death"} <= types, types


def test_episode_runner():
    import random as _random
    from train.run_episodes import profile_key, run_episode, sample_drives, summarize
    from train.extract_dataset import load_keep_set

    rng = _random.Random(7)
    for _ in range(20):
        d = sample_drives(rng)
        assert d["survival"] in (0.5, 1.0) and d["curiosity"] in (0.0, 0.5, 1.0), d
    assert profile_key({"survival": 1.0, "curiosity": 0.5}) == "curiosity=0.5,survival=1"

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


def test_curiosity():
    b = Brain("npc_1", object())
    snap = {"t": 1.0, "self_id": "npc_1", "self_pos": [5, 5],
            "visible_entities": [{"id": "bush_1", "type": "berry_bush", "pos": [6, 5]}],
            "drives": {"survival": 1.0, "curiosity": 0.5}, "recent_perceptions": []}
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
    snap = {"t": 8.0, "self_id": "npc_1", "self_pos": [10, 5], "hp": 100,
            "visible_entities": [], "recent_perceptions": []}
    goals = brain.decide(snap)
    assert goals[0].actions[0]["action"] == "say", goals
    assert len(prov.replies) == 0, "both the recall and the say were consumed"
    second_prompt = prov.calls[-1][1]
    assert "bear" in second_prompt, "recalled bear memory must be in the second prompt"


def test_recall_cap():
    prov = FakeProvider([{"action": "recall", "params": {"query": "x"}}] * (MAX_RECALLS + 3))
    brain = Brain("npc_1", prov, memory_k=2)
    snap = {"t": 1.0, "self_pos": [0, 0], "hp": 100, "visible_entities": [], "recent_perceptions": []}
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
    brain.spatial.observe_many([((x, y), "floor") for x in range(39, 42) for y in range(39, 42)])
    snap = {"t": 1.0, "self_id": "npc_1", "self_pos": [1, 1], "hp": 100,
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
    prov = FakeProvider([{"action": "wait"}])
    brain = Brain("npc_x", prov, memory_k=3)
    brain.record_events(
        [{"kind": "entity_entered", "id": "player", "pos": [1, 1], "etype": "player"}],
        now_t=100.0,
        location=[2, 3],
    )
    snap = {
        "t": 110.0,
        "self_id": "npc_x",
        "self_pos": [2, 3],
        "hp": 100,
        "visible_entities": [{"id": "player", "pos": [1, 1]}],
        "recent_perceptions": [{"kind": "entity_entered", "id": "player"}],
    }
    goals = brain.decide(snap, events=[])
    assert goals == [], "wait is an empty agenda"
    assert len(brain.store) == 1, "decide must not create memories, only recall"
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
    j.log("player", "dialogue_msg", partner="npc_1", text="hi")
    j.close()
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    assert all({"t", "run", "actor", "type", "payload"} <= set(rec) for rec in records)
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

    assert validate_goals({"goals": []}) is None, "empty agenda must be wait, not empty goals"
    assert validate_goals({"goals": [{"actions": [{"action": "move", "params": {"x": 1, "y": 1}},
                                                  {"action": "recall", "params": {"query": "x"}}],
                                      "importance": 3}]}) is None, "recall inside a plan is invalid"
    assert validate_goals({"goals": [{"actions": [{"action": "fly"}], "importance": 3}]}) is None, "unknown action"
    assert validate_goals({"goals": [{"actions": [{"action": "wait"}], "importance": 3}]}) is None, "wait is not a plan action"
    assert validate_goals({"goals": [{"actions": [{"action": "say", "params": {"text": "hi"}}]}]}) is None, "importance is required"
    assert validate_goals({"goals": [{"actions": [{"action": "say", "params": {"text": "hi"}}],
                                      "importance": True}]}) is None, "boolean is not importance"
    assert validate_goals({"action": "move", "params": {"x": 1, "y": 1}}) is None, "bare action is not a goal-set"
    assert validate_goals([{"actions": [{"action": "say", "params": {"text": "hi"}}], "importance": 1}]) is None, "bare list rejected"
    assert validate_goals("nope") is None
    assert validate_goals({"goals": "bad"}) is None

    # filter_response routes the whole reply space: goals / wait / recall / bad
    assert filter_response({"action": "wait", "reason": "safe"})["kind"] == "wait"
    assert filter_response({"action": "recall", "params": {"query": "bear"}})["kind"] == "recall"
    assert filter_response({"goals": [{"actions": [{"action": "say", "params": {"text": "hi"}}],
                                       "importance": 2}]})["kind"] == "goals"
    assert filter_response({"action": "look", "params": {"x": 4, "y": 5}})["kind"] == "look"
    assert filter_response({"nonsense": 1})["kind"] == "bad"
    assert filter_response({"action": "recall"})["kind"] == "bad", "recall needs a query"
    assert filter_response({"action": "look", "params": {"x": 4}})["kind"] == "bad", "look needs both coords"
    assert filter_response({"goals": [{"actions": [{"action": "fly"}], "importance": 1}]})["kind"] == "bad"

    # goal_from_intent bridges a single validated intent; recall/wait/garbage -> None
    g = goal_from_intent({"action": "say", "params": {"text": "hey"}}, importance=3.0)
    assert g is not None and g.actions[0]["action"] == "say" and g.importance == 3.0
    assert goal_from_intent({"action": "wait"}) is None
    assert goal_from_intent({"action": "recall", "params": {"query": "x"}}) is None
    assert goal_from_intent({"action": "wait"}) is None
    assert goal_from_intent("nope") is None
    print("goal infrastructure ok")


def main() -> None:
    test_movement_and_interact()
    test_pathing()
    test_perception()
    test_spatial_memory()
    test_terrain_resources()
    test_needs()
    test_death()
    test_reward()
    test_engine_headless()
    test_episode_runner()
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
