"""Drive-weighted reward, computed by code from measurable world outcomes.

Each drive owns one component function `(npc, world) -> float` that MEASURES a
fact about the entity's state — never a judgment of what an action "meant". An
action credits whatever drives its outcomes served; there is no action
classification. The total reward reuses the entity's own `drives` weights, so
the same profile that conditions the policy's decisions also scores them:

    reward = sum_d drives[d] * signal_d

Constants are minimal placeholders, like sim/needs.py's rates.
"""

DISCOUNT_PER_S = 1.0   # per-second return discount; 1.0 = undiscounted (episodes are bounded)


def survival_reward(npc, world) -> float:
    """You are only as safe as your worst meter: min(hp, hunger, thirst)/100
    while alive, 0 once dead. Accrued over time, so living longer while
    topped-off is worth more than scraping by — and death ends accrual."""
    if npc.hp <= 0.0:
        return 0.0
    return min(npc.hp, npc.hunger, npc.thirst) / 100.0


def curiosity_reward(npc, world) -> float:
    """Novelty gained this step: how many entity types newly became familiar
    (the NPC's first interaction outcome with that kind of thing). A discrete
    discovery signal measured from memory — NOT tiles stepped on, which is
    gameable. The delta is a memory-set change only the episode runner can see
    (it owns the brains), so the runner records it each step via
    note_novelty(); an entity with nothing recorded scores 0."""
    return float(getattr(npc, "novelty_gained", 0))


REWARD_COMPONENTS = {
    "survival": survival_reward,
    "curiosity": curiosity_reward,
    # "power": power_reward,   # seam: possessions/strength — not built yet
}


def reward(npc, world) -> float:
    """Drive-weighted sum of the per-drive outcome signals."""
    return sum(npc.drives.get(d, 0.0) * fn(npc, world) for d, fn in REWARD_COMPONENTS.items())


def note_novelty(npc, brain, seen: dict) -> int:
    """Measure the NPC's familiar-type delta since the last call and record it
    on the entity for curiosity_reward. `seen` is a caller-owned dict
    npc.id -> set of types already counted. Returns the types gained."""
    now = brain._familiar_types()
    gained = len(now - seen.get(npc.id, set()))
    seen[npc.id] = now
    npc.novelty_gained = gained
    return gained
