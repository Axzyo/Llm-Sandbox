from .world import World, chebyshev, eye_height, has_los, level_of

DIRS = {
    "up": (0, -1),
    "left": (-1, 0),
    "down": (0, 1),
    "right": (1, 0),
}


def attempt_move(world: World, entity, dx: int, dy: int) -> dict:
    """Step one column over. The entity lands on the highest surface there it can
    reach (world.landing): up by at most its climb, down any distance."""
    nx, ny = entity.x + dx, entity.y + dy
    nz = world.landing(nx, ny, entity.z, entity.properties["climb"], entity.properties["height"])
    if nz is None:
        return {"ok": False, "reason": "blocked", "to": [nx, ny]}
    occupant = world.entity_at(nx, ny, level_of(nz))
    if occupant is not None:
        return {"ok": False, "reason": "occupied", "by": occupant.id, "to": [nx, ny]}
    frm = [entity.x, entity.y, entity.z]
    entity.x, entity.y, entity.z = nx, ny, nz
    return {"ok": True, "from": frm, "to": [nx, ny, nz]}


def evaluate_interact(world: World, actor, target_id: str | None = None) -> dict:
    others = [t for t in world.things() if t.id != actor.id]
    if not others:
        return {"ok": False, "reason": "no_targets"}
    target = world.thing(target_id) if target_id is not None and target_id != actor.id else None
    if target is None:
        target = min(others, key=lambda t: chebyshev(actor.x, actor.y, t.x, t.y))
    d = chebyshev(actor.x, actor.y, target.x, target.y)
    range_ok = d <= actor.properties["interact_range"]
    los_ok = has_los(world, actor.x, actor.y, target.x, target.y, eye_height(actor))
    return {
        "ok": range_ok and los_ok,
        "target": target.id,
        "distance": d,
        "range_ok": range_ok,
        "los_ok": los_ok,
    }
