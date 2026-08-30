"""Survival needs: health, hunger, thirst — the substrate that gives an agent a
reason to act. Each ranges 0 (empty) to 100 (full); higher is better.

Hunger and thirst drain over time on their own — that drain IS the survival
pressure. Health is now coupled to them: while hunger OR thirst sits at 0 the body
starves/dehydrates and health ticks DOWN; while both are well-supplied (above
REGEN_THRESHOLD) health slowly regenerates. In between, health holds. At 0 hp the
entity dies (handled in main.py's reap_dead). No external damage sources yet.

Rates are module constants (placeholders, like sim/memory.py's weights).
"""

NEED_MAX = 100.0
HUNGER_DRAIN_PER_S = 0.5   # ~200s from full to empty
THIRST_DRAIN_PER_S = 0.8   # thirst outpaces hunger, ~125s from full to empty
STARVE_DAMAGE_PER_S = 1.0  # health lost per second while a need is at 0
HEALTH_REGEN_PER_S = 0.5   # health gained per second while well-fed AND hydrated
REGEN_THRESHOLD = 75.0     # both hunger and thirst must exceed this to regenerate


def tick_needs(entity, dt: float) -> None:
    """Advance needs by elapsed real time `dt` (seconds). Hunger and thirst drain
    (clamped at 0). Then health responds: it drops while either need is empty, and
    regenerates while both are above REGEN_THRESHOLD; otherwise it holds. Health is
    clamped to [0, 100]."""
    entity.hunger = max(0.0, entity.hunger - HUNGER_DRAIN_PER_S * dt)
    entity.thirst = max(0.0, entity.thirst - THIRST_DRAIN_PER_S * dt)
    if entity.hunger <= 0.0 or entity.thirst <= 0.0:
        entity.hp = max(0.0, entity.hp - STARVE_DAMAGE_PER_S * dt)
    elif entity.hunger > REGEN_THRESHOLD and entity.thirst > REGEN_THRESHOLD:
        entity.hp = min(100.0, entity.hp + HEALTH_REGEN_PER_S * dt)
