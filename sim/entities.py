from dataclasses import dataclass, field

from .goals import GoalList


@dataclass
class Entity:
    id: str
    name: str
    kind: str
    x: int
    y: int
    hp: float = 100.0                 # health, 0..100 (higher is better); ticks with needs
    hunger: float = 100.0            # 100 = sated, 0 = starving; drains over time
    thirst: float = 100.0            # 100 = hydrated, 0 = parched; drains over time
    vision_radius: int = 8
    hearing_radius: int = 12
    interact_range: int = 1
    move_interval: float = 0.15
    think_interval: float = 3.0       # idle decision cadence (s); novel events think sooner
    inventory: list = field(default_factory=list)
    drives: dict = field(default_factory=lambda: {"survival": 1.0, "curiosity": 0.5})  # personality = weights
    next_move_at: float = 0.0
    move_target: tuple | None = None      # legacy; NPC movement is goal-driven now
    target: str | None = None
    goals: GoalList = field(default_factory=GoalList)
    resource: dict | None = None      # set on terrain resources (water / berry bush); see sim/terrain.py

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)
