"""Terrain: connector types and their tags, the test map, and the resources
scattered on it. All map/terrain code lives here.

A connector's TYPE is data: its default span and a list of TAGS. Tags are the
only place a type's properties live - what happens when something interacts with
it, when an item made from it is used, or when time passes. Each tag is one
record: `on` says when it fires, `do` names the effect, the rest is that effect's
payload. Nothing here is special-cased on a type name; `apply_tags` reads the
records and applies them.

    {"on": "interact", "do": "restore", "stat": "thirst", "amount": 100}   drink from it
    {"on": "interact", "do": "give", "item": "berry"}                      it hands you an item
    {"on": "interact", "do": "become", "type": "empty_bush"}               it turns into another type
    {"on": "time", "after": (lo, hi), "do": "become", "type": "bush"}      ... on its own, after a while

ITEMS are tagged the same way (`on: "use"`). Eating a picked berry is what raises
the hunger stat - so the NPC learns "ate a berry -> hunger rose" from that outcome
memory, never from being told a bush is food.

The ground is level 0: a stone floor with a full slab of dirt on top, so its top
is a surface at height 1. Level 1 is grass floor tiles laid on that dirt; walls
are full-slab stone connectors on level 1. Entities live on level 1 (z = 1.0).
"""
import random

from .world import Connector, World

WIDTH = 24
HEIGHT = 16
GROUND = 1                       # the level entities spawn on: grass over a slab of dirt
SPAWNS = {"player": (3, 3, 1.0), "npc_1": (19, 9, 1.0), "npc_2": (18, 5, 1.0)}

CONNECTORS = {                   # type -> default span within its slab + tags
    "dirt":       {"span": (0.0, 1.0), "tags": []},
    "stone":      {"span": (0.0, 1.0), "tags": []},
    "well":       {"span": (0.0, 0.5), "tags": [
        {"on": "interact", "do": "restore", "stat": "thirst", "amount": 100}]},
    "bush":       {"span": (0.0, 0.75), "tags": [
        {"on": "interact", "do": "give", "item": "berry"},
        {"on": "interact", "do": "become", "type": "empty_bush"}]},
    "empty_bush": {"span": (0.0, 0.75), "tags": [
        {"on": "time", "after": (30.0, 60.0), "do": "become", "type": "bush"}]},
}

ITEMS = {                        # item name -> tags; a 'use' fires the `on: "use"` ones and consumes it
    "berry": [{"on": "use", "do": "restore", "stat": "hunger", "amount": 40}],
}


def make(type: str, id: str | None = None, now: float = 0.0, rng=None) -> Connector:
    """A connector of `type` at its default span, its timer set if the type has a
    timed tag."""
    k = Connector(type, *CONNECTORS[type]["span"], id=id)
    set_type(k, type, now, rng)
    return k


def set_type(k: Connector, type: str, now: float, rng=None) -> None:
    """Turn connector `k` into `type` in place (same cell, same id): its span
    becomes the type's default, and its timer is set from the type's timed tag
    (or cleared)."""
    k.type = type
    k.bottom, k.top = CONNECTORS[type]["span"]
    timed = [t for t in CONNECTORS[type]["tags"] if t["on"] == "time"]
    k.timer = now + (rng or random.Random()).uniform(*timed[0]["after"]) if timed else None


# --- effects: `do` -> handler(tag, target connector or None, actor, now, rng) -> fact ---

def _raise_stat(actor, stat: str, amount: float) -> float:
    """Raise one of the actor's stats toward 100; returns how much it actually rose."""
    before = actor.stats[stat]
    actor.stats[stat] = min(100.0, before + amount)
    return round(actor.stats[stat] - before, 1)


def _restore(tag, target, actor, now, rng) -> dict:
    return {"stat": tag["stat"], "gained": _raise_stat(actor, tag["stat"], tag["amount"])}


def _give(tag, target, actor, now, rng) -> dict:
    actor.inventory.append(tag["item"])
    return {"item": tag["item"]}


def _become(tag, target, actor, now, rng) -> dict:
    set_type(target, tag["type"], now, rng)
    return {"became": tag["type"]}


EFFECTS = {"restore": _restore, "give": _give, "become": _become}


def apply_tags(tags: list, on: str, target, actor, now: float, rng=None) -> list:
    """Fire every tag of `tags` whose trigger is `on`. Returns one fact dict per
    effect applied, in order - what actually happened, for outcome memories."""
    return [EFFECTS[t["do"]](t, target, actor, now, rng) for t in tags if t["on"] == on]


def describe_effects(facts: list) -> str | None:
    """Facts -> a short outcome phrase for a memory ("thirst +90; got a berry;
    it became empty_bush"), or None if nothing happened."""
    parts = []
    for f in facts:
        if "stat" in f:
            parts.append(f"{f['stat']} +{f['gained']:g}")
        elif "item" in f:
            parts.append(f"got a {f['item']}")
        elif "became" in f:
            parts.append(f"it became {f['became']}")
    return "; ".join(parts) or None


def interact_with(thing, actor, now: float, rng=None) -> dict | None:
    """Fire `thing`'s interact tags on `actor`. Returns {"ok", "effects"} (effects
    may be empty: touching a thing with nothing to give), or None if `thing` is not
    a connector object (an entity - dialogue and the like are handled elsewhere)."""
    k = getattr(thing, "connector", None)
    if k is None:
        return None
    return {"ok": True, "effects": apply_tags(CONNECTORS[k.type]["tags"], "interact", k, actor, now, rng)}


def use_item(actor, item: str) -> dict | None:
    """Fire an inventory item's use tags on `actor` and consume one of it.
    Returns an outcome dict, or None if the item is unknown / isn't carried."""
    tags = ITEMS.get(item)
    if not tags or item not in actor.inventory:
        return None
    facts = apply_tags(tags, "use", None, actor, 0.0)
    actor.inventory.remove(item)
    return {"ok": True, "did": "use", "item": item, "effects": facts}


def tick_connectors(world, now: float, rng=None) -> None:
    """Fire the timed tags of every connector whose timer has come due."""
    for k in world.connectors():
        if k.timer is not None and now >= k.timer:
            k.timer = None
            apply_tags(CONNECTORS[k.type]["tags"], "time", k, None, now, rng)


# --- the map ---------------------------------------------------------------------

def wall(world, x, y) -> None:
    world.put(x, y, GROUND, floor="grass", connector=make("stone"))


def build_test_map() -> tuple[World, dict]:
    w = World(WIDTH, HEIGHT)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            w.put(x, y, 0, floor="stone", connector=make("dirt"))
            w.put(x, y, GROUND, floor="grass")
    for x in range(WIDTH):
        wall(w, x, 0)
        wall(w, x, HEIGHT - 1)
    for y in range(HEIGHT):
        wall(w, 0, y)
        wall(w, WIDTH - 1, y)
    for x in range(4, 13):
        if x != 8:
            wall(w, x, 8)
    for y in range(3, 12):
        if y != 7:
            wall(w, 15, y)
    wall(w, 17, 5)
    wall(w, 18, 4)
    return w, SPAWNS


def free_standing_tiles(world, taken) -> list:
    """Columns on the ground level an entity could be placed on, as (x, y, z):
    a surface exactly at the ground height, nothing on it, not in `taken`."""
    return [(x, y, float(GROUND)) for y in range(world.height) for x in range(world.width)
            if world.tile_type(x, y, GROUND) == "floor" and (x, y) not in taken]


def place_resources(world, rng=None) -> list:
    """Set 1 well and 3 bushes into random free ground cells - a fresh layout each
    call - as named connector objects (perceived and interactable like any body).
    Returns them as Placed records. Avoids walls, existing entities (spawns), and
    doubling up."""
    rng = rng or random.Random()
    taken = {(e.x, e.y) for e in world.entities.values()}
    tiles = free_standing_tiles(world, taken)
    rng.shuffle(tiles)

    ids = []
    for type, oid in [("well", "well_1"), ("bush", "bush_1"), ("bush", "bush_2"), ("bush", "bush_3")]:
        if not tiles:
            break
        x, y, _z = tiles.pop()
        world.cell(x, y, GROUND).connector = make(type, oid, rng=rng)
        ids.append(oid)
    return [o for o in world.objects() if o.id in ids]
