from .world import chebyshev, has_los


def _sightline_clear_to(world, x0, y0, x1, y1) -> bool:
    """Bresenham ray from (x0,y0) to (x1,y1): clear iff no wall lies STRICTLY
    before the target. Unlike has_los, the target itself may be a wall — you see
    the face of a wall in front of you, you just can't see past it."""
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx - dy
    x, y = x0, y0
    while (x, y) != (x1, y1):
        if (x, y) != (x0, y0) and world.is_wall(x, y):
            return False                       # a wall before the target blocks it
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return True


def visible_tiles(world, viewer) -> list:
    """Tiles within vision + line of sight of the viewer, as ((x, y), type) where
    type is 'wall' or 'floor'. Feeds the spatial-memory layer (a remembered map),
    not episodic memory."""
    r = viewer.properties["vision_radius"]
    out = []
    for y in range(viewer.y - r, viewer.y + r + 1):
        for x in range(viewer.x - r, viewer.x + r + 1):
            if not world.in_bounds(x, y):
                continue
            if chebyshev(viewer.x, viewer.y, x, y) > r:
                continue
            if not _sightline_clear_to(world, viewer.x, viewer.y, x, y):
                continue
            out.append(((x, y), "wall" if world.is_wall(x, y) else "floor"))
    return out


def visible_entities(world, viewer) -> list:
    out = []
    for e in world.entities.values():
        if e.id == viewer.id:
            continue
        if (
            chebyshev(viewer.x, viewer.y, e.x, e.y) <= viewer.properties["vision_radius"]
            and has_los(world, viewer.x, viewer.y, e.x, e.y)
        ):
            out.append(e)
    return out


class PerceptionTracker:
    def __init__(self, viewer_id: str):
        self.viewer_id = viewer_id
        self.last: dict = {}

    def update(self, world, viewer) -> list:
        seen = {}
        for e in world.entities.values():
            if e.id == viewer.id:
                continue
            if (
                chebyshev(viewer.x, viewer.y, e.x, e.y) <= viewer.properties["vision_radius"]
                and has_los(world, viewer.x, viewer.y, e.x, e.y)
            ):
                seen[e.id] = (e.x, e.y)

        def etype(eid):
            e = world.entities.get(eid)
            return e.kind if e is not None else "entity"

        events = []
        for eid, pos in seen.items():
            if eid not in self.last:
                events.append({"kind": "entity_entered", "id": eid, "pos": list(pos), "etype": etype(eid)})
            elif self.last[eid] != pos:
                events.append({"kind": "entity_moved", "id": eid, "pos": list(pos), "etype": etype(eid)})
        for eid in self.last:
            if eid not in seen:
                events.append({"kind": "entity_left", "id": eid, "etype": etype(eid)})
        self.last = seen
        return events
