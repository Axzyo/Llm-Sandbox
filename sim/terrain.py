"""Terrain generation: the world's tiles (walls) and the resources scattered on it.
All map/terrain code lives here.

Resources are placed objects — entities of kind 'water' / 'berry_bush' sitting on
floor tiles — each carrying a data-driven `resource` payload that says what
interacting with it does. No behavior is special-cased on kind; `interact_with`
just reads the payload and applies it. An effect raises one of the actor's STATS
(hunger, thirst, health, ...) by an amount — nothing here knows about "needs".
Two interaction kinds today:
    {"kind": "restore", "stat": <stat>, "amount": <n>}       # water: drink -> raise a stat
    {"kind": "harvest", "yields": <item>, "available", "regrow_at"}  # bush: pick -> item

Items are just names in an entity's inventory; ITEMS says what using one does
(also data). Eating a picked berry (an inventory 'use') is what actually raises
the hunger stat — so the NPC learns "ate a berry -> hunger rose" from that outcome
memory, never from being told a bush is food.
"""
import random

from .entities import Entity
from .world import World

WIDTH = 24
HEIGHT = 16
SPAWNS = {"player": (3, 3), "npc_1": (19, 9), "npc_2": (18, 5)}

BERRY_REGROW_S = (30.0, 60.0)   # a picked bush grows a new berry after this many seconds
BERRY_AMOUNT = 40               # hunger restored by eating one berry

ITEMS = {                        # item name -> what a 'use' (eat/drink/etc.) does
    "berry": {"stat": "hunger", "amount": BERRY_AMOUNT},
}


def build_test_map() -> tuple[World, dict]:
    w = World(WIDTH, HEIGHT)
    for x in range(WIDTH):
        w.add_wall(x, 0)
        w.add_wall(x, HEIGHT - 1)
    for y in range(HEIGHT):
        w.add_wall(0, y)
        w.add_wall(WIDTH - 1, y)
    for x in range(4, 13):
        if x != 8:
            w.add_wall(x, 8)
    for y in range(3, 12):
        if y != 7:
            w.add_wall(15, y)
    w.add_wall(17, 5)
    w.add_wall(18, 4)
    return w, SPAWNS


def _free_floor_tiles(world, taken) -> list:
    return [(x, y) for y in range(world.height) for x in range(world.width)
            if not world.is_wall(x, y) and (x, y) not in taken]


def place_resources(world, rng=None) -> list:
    """Scatter 1 infinite water source and 3 one-time-use berry bushes on random
    free floor tiles — a fresh layout each call. Adds them to world.entities (so
    they are perceived and interactable like any other body) and returns them.
    Avoids walls, existing entities (spawns), and doubling up."""
    rng = rng or random.Random()
    taken = {(e.x, e.y) for e in world.entities.values()}
    tiles = _free_floor_tiles(world, taken)
    rng.shuffle(tiles)

    specs = [("water_1", "water",
              {"kind": "restore", "stat": "thirst", "amount": 100})]
    for i in range(1, 4):
        specs.append((f"bush_{i}", "berry_bush",
                      {"kind": "harvest", "yields": "berry", "available": True, "regrow_at": None}))

    placed = []
    for eid, kind, resource in specs:
        if not tiles:
            break
        x, y = tiles.pop()
        e = Entity(eid, eid, kind, x, y)
        e.resource = resource
        world.entities[eid] = e
        placed.append(e)
    return placed


def tick_resources(world, now: float) -> None:
    """Regrow any harvestable resource whose regrow timer has elapsed."""
    for e in world.entities.values():
        r = getattr(e, "resource", None)
        if not r or r.get("kind") != "harvest":
            continue
        if not r.get("available") and r.get("regrow_at") is not None and now >= r["regrow_at"]:
            r["available"] = True
            r["regrow_at"] = None


def _raise_stat(actor, stat: str, amount: float) -> float:
    """Raise one of the actor's stats toward 100; returns how much it actually rose."""
    before = getattr(actor, stat)
    setattr(actor, stat, min(100.0, before + amount))
    return round(getattr(actor, stat) - before, 1)


def interact_with(entity, actor, now: float, rng=None) -> dict | None:
    """Apply `entity`'s interaction to `actor`, per the entity's `resource` data.
    Returns an outcome dict, or None if `entity` isn't an interactable resource.

    'restore' (water): drink — raise a stat on the spot; the source is not used up.
    'harvest' (bush): pick — put the yielded item in the actor's inventory, if a
    berry is ripe; the bush is then empty and scheduled to regrow.
    """
    r = getattr(entity, "resource", None)
    if not r:
        return None
    kind = r.get("kind")
    if kind == "restore":
        gained = _raise_stat(actor, r["stat"], r["amount"])
        return {"ok": True, "did": "drink", "stat": r["stat"], "gained": gained}
    if kind == "harvest":
        if not r.get("available"):
            return {"ok": False, "did": "harvest", "reason": "empty"}
        actor.inventory.append(r["yields"])
        r["available"] = False
        r["regrow_at"] = now + (rng or random.Random()).uniform(*BERRY_REGROW_S)
        return {"ok": True, "did": "harvest", "yields": r["yields"]}
    return None


def use_item(actor, item: str) -> dict | None:
    """Apply an inventory item's `use` effect (from ITEMS) to `actor`, consuming one.
    Returns an outcome dict, or None if the item has no effect / isn't carried."""
    eff = ITEMS.get(item)
    if not eff or item not in actor.inventory:
        return None
    gained = _raise_stat(actor, eff["stat"], eff["amount"])
    actor.inventory.remove(item)
    return {"ok": True, "did": "use", "item": item, "stat": eff["stat"], "gained": gained}
