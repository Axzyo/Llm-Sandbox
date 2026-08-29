from .world import World, chebyshev, has_los

DIRS = {
    "up": (0, -1),
    "left": (-1, 0),
    "down": (0, 1),
    "right": (1, 0),
}


def attempt_move(world: World, entity, dx: int, dy: int) -> dict:
    nx, ny = entity.x + dx, entity.y + dy
    if world.blocked(nx, ny):
        return {"ok": False, "reason": "blocked", "to": [nx, ny]}
    occupant = world.entity_at(nx, ny)
    if occupant is not None:
        return {"ok": False, "reason": "occupied", "by": occupant.id, "to": [nx, ny]}
    fx, fy = entity.x, entity.y
    entity.x, entity.y = nx, ny
    return {"ok": True, "from": [fx, fy], "to": [nx, ny]}


def evaluate_interact(world: World, actor, target_id: str | None = None) -> dict:
    others = [e for e in world.entities.values() if e.id != actor.id]
    if not others:
        return {"ok": False, "reason": "no_targets"}
    if target_id is not None and target_id in world.entities and target_id != actor.id:
        target = world.entities[target_id]
    else:
        target = min(others, key=lambda e: chebyshev(actor.x, actor.y, e.x, e.y))
    d = chebyshev(actor.x, actor.y, target.x, target.y)
    range_ok = d <= actor.interact_range
    los_ok = has_los(world, actor.x, actor.y, target.x, target.y)
    return {
        "ok": range_ok and los_ok,
        "target": target.id,
        "distance": d,
        "range_ok": range_ok,
        "los_ok": los_ok,
    }
