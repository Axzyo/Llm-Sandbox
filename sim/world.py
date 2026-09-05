"""The world: a stack of levels, each a grid of cells.

A cell at (x, y, level) has two parts, matching the two layers of the map:
  * `floor`     -- the floor layer: a surface at height `level`, named by its
                   material ("grass", "stone", ...), or None where there is no floor.
  * `connector` -- the connector layer: one object standing in the slab between
                   heights `level` and `level + 1` (dirt, a wall, a table, a stair
                   step ...). It occupies the vertical span [bottom, top) of that
                   slab, in slab units, so a full block is 0..1, a table 0..0.5 and
                   the upper step of a staircase 0.5..1 (you can walk under it).

Heights are continuous: an entity stands at some height z on a *surface* -- a floor,
or the top of a connector -- and its level is floor(z). The top of a full block at
level L is a surface at L + 1, so ground is a level of dirt blocks with grass floor
tiles on top: remove the grass and the dirt top is still there to stand on. Only
the connectors' spans and the floors' heights matter to movement and sight; no
code here knows what a "table" or a "wall" is.
"""
import math
from dataclasses import dataclass


@dataclass
class Connector:
    """One object on the connector layer of a cell, of some `type` (dirt, stone,
    bush, well ...), filling the vertical span [bottom, top) of its slab (0 = the
    slab's floor, 1 = the floor above). What a type does is its tags, data in
    sim/terrain.py; `timer` is when its timed tag comes due, if it has one. Bulk
    (dirt, a wall) is anonymous. A named one (`id`: a bush, a well) is a thing
    others perceive and can interact with."""
    type: str
    bottom: float = 0.0
    top: float = 1.0
    id: str | None = None
    timer: float | None = None


@dataclass
class Cell:
    floor: str | None = None          # floor-layer material, or no floor
    connector: Connector | None = None


@dataclass(frozen=True)
class Placed:
    """A named connector object as the rest of the sim sees it: the same id / kind /
    position surface an Entity has, so perception and interaction treat both alike.
    `connector` is the cell's own object, so effects applied through it persist."""
    id: str
    kind: str
    x: int
    y: int
    z: float                          # the height its base rests at
    connector: Connector

    @property
    def pos(self) -> tuple:
        return (self.x, self.y, self.z)


def level_of(z: float) -> int:
    """The slab an entity standing at height z is in."""
    return math.floor(z)


def chebyshev(x0: int, y0: int, x1: int, y1: int) -> int:
    return max(abs(x1 - x0), abs(y1 - y0))


class World:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.columns: dict = {}       # (x, y) -> {level: Cell}; absent = empty air
        self.entities: dict = {}

    # --- cells ---------------------------------------------------------------

    def cell(self, x: int, y: int, level: int) -> Cell | None:
        return self.columns.get((x, y), {}).get(level)

    def put(self, x: int, y: int, level: int, floor: str | None = None,
            connector: Connector | None = None) -> Cell:
        c = Cell(floor, connector)
        self.columns.setdefault((x, y), {})[level] = c
        return c

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    # --- geometry queries ----------------------------------------------------

    def surfaces(self, x: int, y: int) -> list:
        """Every height in column (x, y) something could stand on: each floor, and
        the top of each connector. Ascending, deduplicated."""
        out = set()
        for level, c in self.columns.get((x, y), {}).items():
            if c.floor is not None:
                out.add(float(level))
            if c.connector is not None:
                out.add(level + c.connector.top)
        return sorted(out)

    def column_free(self, x: int, y: int, lo: float, hi: float) -> bool:
        """Is the vertical span [lo, hi) of column (x, y) empty? A floor plane at
        exactly `lo` is what the span rests on, so it does not obstruct; a floor
        strictly inside the span, or any connector overlapping it, does."""
        for level, c in self.columns.get((x, y), {}).items():
            if c.floor is not None and lo < level < hi:
                return False
            k = c.connector
            if k is not None and level + k.bottom < hi and lo < level + k.top:
                return False
        return True

    def landing(self, x: int, y: int, from_z: float, climb: float, height: float) -> float | None:
        """Where an entity of `height` standing at `from_z` ends up if it steps into
        column (x, y): the highest surface no more than `climb` above it whose body
        space is free. Dropping any distance is allowed. None = cannot enter."""
        if not self.in_bounds(x, y):
            return None
        for s in reversed(self.surfaces(x, y)):
            if s > from_z + climb:
                continue
            if self.column_free(x, y, s, s + height):
                return s
        return None

    def opaque(self, x: int, y: int, z: float) -> bool:
        """Does column (x, y) block a horizontal sight line at height z? Only
        connectors block sight; floors are horizontal planes."""
        if not self.in_bounds(x, y):
            return True
        for level, c in self.columns.get((x, y), {}).items():
            k = c.connector
            if k is not None and level + k.bottom <= z < level + k.top:
                return True
        return False

    def tile_type(self, x: int, y: int, level: int) -> str:
        """What an entity on `level` perceives at column (x, y), as a shape class:
        'wall' a connector filling the whole slab, 'step' a partial connector,
        'floor' a surface at exactly this level's height, 'drop' nothing to stand
        on at this level (the column falls away)."""
        c = self.cell(x, y, level)
        k = c.connector if c is not None else None
        if k is not None:
            return "wall" if k.bottom <= 0.0 and k.top >= 1.0 else "step"
        return "floor" if float(level) in self.surfaces(x, y) else "drop"

    # --- things: entities and named connector objects --------------------------

    def connectors(self) -> list:
        """Every connector object in the world, named or not."""
        return [c.connector for column in self.columns.values()
                for c in column.values() if c.connector is not None]

    def objects(self) -> list:
        """Every named connector object, as Placed records."""
        return [Placed(c.connector.id, c.connector.type, x, y, level + c.connector.bottom, c.connector)
                for (x, y), column in self.columns.items()
                for level, c in column.items()
                if c.connector is not None and c.connector.id is not None]

    def things(self) -> list:
        """Everything perceivable and interactable: entities, then named objects."""
        return list(self.entities.values()) + self.objects()

    def thing(self, id: str):
        """The entity or named object with this id, or None."""
        e = self.entities.get(id)
        if e is not None:
            return e
        return next((o for o in self.objects() if o.id == id), None)

    def entity_at(self, x: int, y: int, level: int | None = None):
        """The entity in column (x, y), on `level` if given (else any level)."""
        for e in self.entities.values():
            if e.x == x and e.y == y and (level is None or level_of(e.z) == level):
                return e
        return None

    def thing_at(self, x: int, y: int, level: int):
        """The entity on `level` of column (x, y), else the named object in that cell."""
        e = self.entity_at(x, y, level)
        if e is not None:
            return e
        return next((o for o in self.objects() if (o.x, o.y, level_of(o.z)) == (x, y, level)), None)


def has_los(world: World, x0: int, y0: int, x1: int, y1: int, z: float) -> bool:
    """Bresenham sight line at height z from (x0, y0) to (x1, y1): clear iff no
    opaque column lies between them (the endpoints themselves never block)."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        if (x, y) == (x1, y1):
            return True
        if (x, y) != (x0, y0) and world.opaque(x, y, z):
            return False
        e2 = 2 * err
        step_x = e2 > -dy
        step_y = e2 < dx
        if step_x and step_y and world.opaque(x + sx, y, z) and world.opaque(x, y + sy, z):
            return False
        if step_x:
            err -= dy
            x += sx
        if step_y:
            err += dx
            y += sy


def eye_height(entity) -> float:
    """Height of an entity's sight line: mid-body. Sight passes over anything
    lower than that and is blocked by anything covering it."""
    return entity.z + entity.properties["height"] / 2.0
