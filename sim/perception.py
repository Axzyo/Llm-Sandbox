from .world import chebyshev, eye_height, has_los


def _sightline_clear_to(world, x0, y0, x1, y1, z) -> bool:
    """Bresenham ray at height z from (x0,y0) to (x1,y1): clear iff nothing opaque
    lies STRICTLY before the target. Unlike has_los, the target itself may be
    opaque: you see the face of a wall in front of you, you just can't see past it."""
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx - dy
    x, y = x0, y0
    while (x, y) != (x1, y1):
        if (x, y) != (x0, y0) and world.opaque(x, y, z):
            return False                       # something before the target blocks it
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return True


def visible_tiles(world, viewer) -> list:
    """Columns within vision + line of sight of the viewer, as ((x, y, level), type)
    on the viewer's own level, type per world.tile_type ('wall' / 'step' / 'floor'
    / 'drop'). Feeds the spatial-memory layer (a remembered map), not episodic
    memory."""
    r = viewer.properties["vision_radius"]
    eye = eye_height(viewer)
    level = viewer.level
    out = []
    for y in range(viewer.y - r, viewer.y + r + 1):
        for x in range(viewer.x - r, viewer.x + r + 1):
            if not world.in_bounds(x, y):
                continue
            if chebyshev(viewer.x, viewer.y, x, y) > r:
                continue
            if not _sightline_clear_to(world, viewer.x, viewer.y, x, y, eye):
                continue
            out.append(((x, y, level), world.tile_type(x, y, level)))
    return out


def _sees(world, viewer, e) -> bool:
    return (chebyshev(viewer.x, viewer.y, e.x, e.y) <= viewer.properties["vision_radius"]
            and has_los(world, viewer.x, viewer.y, e.x, e.y, eye_height(viewer)))


def visible_things(world, viewer) -> list:
    """Entities and named objects the viewer can see (never itself)."""
    return [t for t in world.things() if t.id != viewer.id and _sees(world, viewer, t)]


class PerceptionTracker:
    def __init__(self, viewer_id: str):
        self.viewer_id = viewer_id
        self.last: dict = {}

    def update(self, world, viewer) -> list:
        seen = {t.id: ((t.x, t.y, t.z), t.kind) for t in visible_things(world, viewer)}

        events = []
        for eid, (pos, kind) in seen.items():
            if eid not in self.last:
                events.append({"kind": "entity_entered", "id": eid, "pos": list(pos), "etype": kind})
            elif self.last[eid][0] != pos:
                events.append({"kind": "entity_moved", "id": eid, "pos": list(pos), "etype": kind})
            elif self.last[eid][1] != kind:       # a bush picked bare, a thing transformed
                events.append({"kind": "entity_changed", "id": eid, "pos": list(pos), "etype": kind})
        for eid, (_pos, kind) in self.last.items():
            if eid not in seen:
                events.append({"kind": "entity_left", "id": eid, "etype": kind})
        self.last = seen
        return events
