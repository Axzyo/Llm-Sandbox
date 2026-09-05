from dataclasses import dataclass, field

from .goals import GoalList
from .world import level_of


def _default_stats() -> dict:
    # Depletable/replenishable meters, 0 (empty) .. 100 (full). Interoceived as
    # `felt` memories when they change. Later: stamina, mana, ...
    return {"health": 100.0, "hunger": 100.0, "thirst": 100.0}


def _default_properties() -> dict:
    # Static capabilities that other systems read (perception ranges, movement
    # cadence). Later: damage, defence, attack/casting speed, ...
    return {
        "vision_radius": 8,
        "hearing_radius": 12,
        "interact_range": 1,
        "height": 1.0,              # body height in slab units; must fit under whatever is overhead
        "climb": 0.5,               # tallest step up (slab units) it can take in one move
        "move_interval": 0.15,
        "think_interval": 3.0,      # idle decision cadence (s); novel events think sooner
    }


@dataclass
class Entity:
    # Identity and position: every entity has these, so they stay top-level.
    # (x, y) is the column; z is the height it stands at (a surface: a floor, or the
    # top of a connector), so its level is floor(z) -- see sim/world.py.
    id: str
    name: str
    kind: str
    x: int
    y: int
    z: float
    # Components grouped by what reads them (pass a whole section into a memory,
    # or one field: entity.stats["hunger"]).
    stats: dict = field(default_factory=_default_stats)
    properties: dict = field(default_factory=_default_properties)
    inventory: list = field(default_factory=list)
    drives: dict = field(default_factory=lambda: {"survival": 0.75, "curiosity": 0.25})  # personality = weights, sum to 1
    goals: GoalList = field(default_factory=GoalList)
    # Per-tick runtime state (not a component; transient scheduling/targeting).
    next_move_at: float = 0.0
    target: str | None = None

    @property
    def pos(self) -> tuple:
        return (self.x, self.y, self.z)

    @property
    def level(self) -> int:
        return level_of(self.z)
