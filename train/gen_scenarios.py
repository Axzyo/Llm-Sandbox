"""Generate a schema-current, situation-diverse DECIDE dataset by distillation.

Instead of replaying the game, we synthesize a broad spread of *situations*
(safe/idle, distant agent, approaching threat, low HP, adjacency, memory-
relevant) and run each one through the REAL `Brain.decide()` against a teacher
model. Because the decision flows through the live brain, every logged pair uses
the current system prompt + user-prompt builder and is gated by the real
validator — so the corpus can never drift from what the game actually feeds.

Output is an ordinary run log (runs/synth_*.jsonl). Turn it into an SFT file with
the existing extractor:

    python train/gen_scenarios.py --n 400            # -> runs/synth_<ts>.jsonl
    python train/extract_dataset.py --out train/data/sft.jsonl

Notes on what the situation space can (and cannot) provoke, given the CURRENT
snapshot schema:
  * `use` is effectively un-promptable: build_snapshot() exposes no inventory,
    so the agent never knows what it carries. We do not fake one in.
  * Under a survival-only drive, a faithful teacher chooses `move`/`none`/
    `recall` most of the time; `say`/`interact` stay rare by design. We balance
    the SITUATIONS, not the action labels, and let the teacher's policy show.
"""
import argparse
import json
import os
import random
import sys
import time
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.brain import Brain, memories_from_event, validate_intent  # noqa: E402
from sim.journal import Journal  # noqa: E402
from sim.terrain import build_test_map  # noqa: E402
from sim.world import chebyshev, has_los  # noqa: E402
from train.teachers import make_teacher  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cfg() -> dict:
    path = os.path.join(ROOT, "config.json")
    cfg = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    return cfg


def floor_tiles(world) -> list:
    return [(x, y) for y in range(world.height) for x in range(world.width)
            if not world.is_wall(x, y)]


def visible_from(world, origin, tiles, vision, rng, need_los=True):
    """A random floor tile within vision (and LOS) of origin, excluding origin."""
    ox, oy = origin
    cand = [(x, y) for (x, y) in tiles
            if (x, y) != origin and chebyshev(ox, oy, x, y) <= vision
            and (not need_los or has_los(world, ox, oy, x, y))]
    return rng.choice(cand) if cand else None


def at_range(world, origin, tiles, dist, vision, rng):
    """A tile at exactly chebyshev `dist` from origin, in vision + LOS."""
    ox, oy = origin
    cand = [(x, y) for (x, y) in tiles
            if chebyshev(ox, oy, x, y) == dist and chebyshev(ox, oy, x, y) <= vision
            and has_los(world, ox, oy, x, y)]
    return rng.choice(cand) if cand else None


# ---- situation families -----------------------------------------------------
# Each returns (snapshot, seed_memories) where seed_memories is a list of
# synthetic perception events to write into the brain's store before deciding.

FAMILIES = []


def family(weight):
    def deco(fn):
        FAMILIES.append((weight, fn))
        return fn
    return deco


def _base_snapshot(self_id, pos, hp, irange, t):
    return {
        "t": round(t, 2),
        "self_id": self_id,
        "self_pos": [pos[0], pos[1]],
        "hp": hp,
        "vision_radius": 8,
        "hearing_radius": 12,
        "interact_range": irange,
        "visible_entities": [],
        "recent_perceptions": [],
    }


@family(0.24)
def safe_idle(world, tiles, self_id, irange, rng, t):
    """Full HP, nothing in view. The canonical 'none' situation."""
    pos = rng.choice(tiles)
    return _base_snapshot(self_id, pos, 100, irange, t), []


@family(0.20)
def distant_agent(world, tiles, self_id, irange, rng, t):
    """Healthy; another agent visible but far and static. Seeing != acting."""
    pos = rng.choice(tiles)
    other = visible_from(world, pos, tiles, 8, rng)
    snap = _base_snapshot(self_id, pos, rng.choice([90, 100, 100]), irange, t)
    if other and chebyshev(pos[0], pos[1], other[0], other[1]) > irange:
        oid = rng.choice(["npc_2", "player", "npc_3"])
        snap["visible_entities"] = [{"id": oid, "type": "npc", "pos": list(other)}]
        snap["recent_perceptions"] = [
            {"kind": "entity_entered", "id": oid, "pos": list(other), "etype": "npc"}
        ]
    return snap, []


@family(0.16)
def approaching(world, tiles, self_id, irange, rng, t):
    """An agent that has been moving toward you across recent perceptions."""
    pos = rng.choice(tiles)
    far = at_range(world, pos, tiles, min(7, 8), 8, rng) or visible_from(world, pos, tiles, 8, rng)
    near = at_range(world, pos, tiles, max(irange + 1, 3), 8, rng)
    snap = _base_snapshot(self_id, pos, rng.choice([70, 85, 100]), irange, t)
    oid = rng.choice(["npc_2", "player"])
    if far and near:
        snap["visible_entities"] = [{"id": oid, "type": "npc", "pos": list(near)}]
        snap["recent_perceptions"] = [
            {"kind": "entity_entered", "id": oid, "pos": list(far), "etype": "npc"},
            {"kind": "entity_moved", "id": oid, "pos": list(near), "etype": "npc"},
        ]
    return snap, []


@family(0.10)
def low_hp_alone(world, tiles, self_id, irange, rng, t):
    """Wounded, nothing in view. Flee-to-safety or hold?"""
    pos = rng.choice(tiles)
    return _base_snapshot(self_id, pos, rng.choice([12, 20, 30, 45]), irange, t), []


@family(0.10)
def low_hp_with_agent(world, tiles, self_id, irange, rng, t):
    """Wounded with another agent nearby — the pressure case."""
    pos = rng.choice(tiles)
    other = visible_from(world, pos, tiles, 6, rng)
    snap = _base_snapshot(self_id, pos, rng.choice([10, 18, 25]), irange, t)
    oid = rng.choice(["npc_2", "player"])
    if other:
        snap["visible_entities"] = [{"id": oid, "type": "npc", "pos": list(other)}]
        snap["recent_perceptions"] = [
            {"kind": "entity_moved", "id": oid, "pos": list(other), "etype": "npc"}
        ]
    return snap, []


@family(0.10)
def adjacent_agent(world, tiles, self_id, irange, rng, t):
    """Another agent within interact range + LOS. Talk? Ignore? Move off?"""
    pos = rng.choice(tiles)
    near = at_range(world, pos, tiles, rng.randint(1, max(1, min(irange, 2))), 8, rng)
    snap = _base_snapshot(self_id, pos, rng.choice([80, 100]), irange, t)
    oid = rng.choice(["npc_2", "player"])
    if near:
        snap["visible_entities"] = [{"id": oid, "type": "npc", "pos": list(near)}]
        snap["recent_perceptions"] = [
            {"kind": "entity_entered", "id": oid, "pos": list(near), "etype": "npc"}
        ]
    return snap, []


@family(0.10)
def memory_relevant(world, tiles, self_id, irange, rng, t):
    """Ambiguous present, but memories carry a prior threat/observation that a
    recall could surface. Tests whether the policy consults memory."""
    pos = rng.choice(tiles)
    snap = _base_snapshot(self_id, pos, rng.choice([55, 70, 100]), irange, t)
    # seed a couple of older memories about a threat seen elsewhere
    threat_pos = rng.choice(tiles)
    seeds = [
        {"kind": "entity_left", "id": "npc_2", "etype": "npc"},
        {"kind": "heard_say", "speaker": "player", "speaker_pos": list(threat_pos),
         "speaker_type": "player", "text": rng.choice(
             ["stay away from the east room", "something attacked me over there",
              "it's not safe near the wall"])},
    ]
    # maybe a faint current perception to give the recall a hook
    other = visible_from(world, pos, tiles, 8, rng)
    if other and rng.random() < 0.5:
        snap["visible_entities"] = [{"id": "npc_2", "type": "npc", "pos": list(other)}]
        snap["recent_perceptions"] = [
            {"kind": "entity_entered", "id": "npc_2", "pos": list(other), "etype": "npc"}
        ]
    return snap, seeds


def pick_family(rng):
    r = rng.random()
    acc = 0.0
    for w, fn in FAMILIES:
        acc += w
        if r <= acc:
            return fn
    return FAMILIES[-1][1]


def seed_memories(brain, seeds, loc, now_t):
    """Write synthetic perception events as memories a bit in the past."""
    for i, ev in enumerate(seeds):
        past = now_t - (len(seeds) - i) * 8.0
        for kw in memories_from_event(ev, loc, past, brain.entity_id):
            brain.store.add(kw["sense"], kw["subject"], kw["observer_loc"],
                            kw["now"], direction=kw.get("direction"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="scenarios to generate")
    ap.add_argument("--teacher", default="ollama",
                    choices=["ollama", "claude", "kimi", "openai_compat"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None, help="run log path (default runs/synth_<ts>.jsonl)")
    ap.add_argument("--self-id", default="npc_1")
    args = ap.parse_args()

    cfg = load_cfg()
    rng = random.Random(args.seed)
    world, _spawns = build_test_map()
    tiles = floor_tiles(world)
    irange = cfg.get("interact_range", 4)

    teacher = make_teacher(args.teacher, cfg)
    if hasattr(teacher, "warm"):
        teacher.warm()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = args.out or os.path.join(ROOT, "runs", f"synth_{ts}.jsonl")
    journal = Journal(out, f"synth_{ts}")

    counts = collections.Counter()
    fam_counts = collections.Counter()
    start = time.time()
    for i in range(args.n):
        fn = pick_family(rng)
        # virtual clock so memory recency behaves; spread scenarios over time
        now_t = 100.0 + i * 5.0
        snap, seeds = fn(world, tiles, args.self_id, irange, rng, now_t)
        brain = Brain(
            args.self_id, teacher, journal,
            memory_k=cfg.get("memory_k", 5),
            memory_halflife_s=cfg.get("memory_halflife_s", 300.0),
        )
        if seeds:
            seed_memories(brain, seeds, snap["self_pos"], now_t)
        try:
            intent = brain.decide(snap)
        except Exception as exc:  # network/teacher hiccup: skip, keep going
            print(f"[{i}] {fn.__name__}: ERROR {exc}", file=sys.stderr)
            continue
        counts[intent.get("action", "?")] += 1
        fam_counts[fn.__name__] += 1
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = max(time.time() - start, 1e-9)
            rate = (i + 1) / elapsed
            print(f"[{i+1}/{args.n}] {rate:.2f}/s  actions={dict(counts)}", flush=True)

    journal.close()
    dur = time.time() - start
    print(f"\ndone: {args.n} scenarios in {dur:.0f}s -> {out}")
    print("by situation :", dict(fam_counts.most_common()))
    print("by action    :", dict(counts.most_common()))
    print("\nnext: python train/extract_dataset.py --out train/data/sft.jsonl")


if __name__ == "__main__":
    main()
