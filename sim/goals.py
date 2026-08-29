"""Goal infrastructure: an NPC's agenda as importance-ranked plans.

A **Goal** is one plan the LLM authored in a single think: an *ordered list of
actions* (the same per-action schema the world already validates) that share a
single `importance` and `reason`. The actions run in listed order — action `step`
tracks progress within the plan. A **GoalList** keeps goals ordered by importance
(desc), ties broken by insertion order, so an earlier plan runs to completion
while a later, more-important plan preempts whatever is mid-flight and resumes it
afterward. Execution (in main.py) always works the top goal's current action.

Pure data structures — no LLM, no world, no imports from the rest of sim.
"""
from dataclasses import dataclass, field
from itertools import count

DEFAULT_IMPORTANCE = 1.0
_seq = count()  # monotonic; stable tie-break so equal-importance goals keep listed order


def _action_label(action: dict) -> str:
    """Compact one-token label for a single action within a plan."""
    a = action.get("action")
    p = action.get("params") or {}
    if a == "move":
        return f"move({p.get('x')},{p.get('y')})"
    if a == "interact":
        return f"interact({p.get('target')})"
    if a == "inventory":
        return f"{p.get('op')}({p.get('item')})"
    if a == "say":
        text = str(p.get("text", ""))
        return f'say("{text[:20]}{"…" if len(text) > 20 else ""}")'
    return str(a)


@dataclass
class Goal:
    actions: list = field(default_factory=list)   # ordered [{action, params}, ...]
    importance: float = DEFAULT_IMPORTANCE
    reason: str = ""
    seq: int = field(default_factory=lambda: next(_seq))
    status: str = "pending"                        # pending -> active -> done | failed
    step: int = 0                                  # index of the current action in `actions`
    started_step: int = -1                         # last step for which action_start was logged

    @property
    def current_action(self) -> dict | None:
        """The action to work right now, or None once the plan is exhausted."""
        return self.actions[self.step] if 0 <= self.step < len(self.actions) else None

    def advance(self) -> bool:
        """Mark the current action done and move to the next. Returns True if the
        plan has more actions to run, False if it is now complete."""
        self.step += 1
        return self.step < len(self.actions)

    def summary(self) -> str:
        """Compact one-line label for HUD / journal: the plan's actions in order.
        A single-action plan reads as just that action; a multi-step plan shows
        `k/n` progress when it is under way."""
        if not self.actions:
            return "(empty)"
        plan = "->".join(_action_label(a) for a in self.actions)
        if len(self.actions) > 1 and self.step > 0:
            return f"{plan} [{self.step}/{len(self.actions)}]"
        return plan


class GoalList:
    """An NPC's goals, always sorted by (importance desc, insertion order asc)."""

    def __init__(self):
        self._goals: list[Goal] = []

    def add(self, goal: Goal) -> None:
        self._goals.append(goal)
        self._sort()

    def add_many(self, goals) -> None:
        self._goals.extend(goals)
        self._sort()

    def _sort(self) -> None:
        self._goals.sort(key=lambda g: (-g.importance, g.seq))

    def current(self) -> Goal | None:
        """Highest-importance goal still to do (pending or mid-flight)."""
        for g in self._goals:
            if g.status in ("pending", "active"):
                return g
        return None

    def locations(self) -> list:
        """The current plans' move-target tiles paired with each goal's importance,
        as ((x,y), importance) — spatial memory floors nearby tiles' memorability to
        this, so geometry near an important goal is forgotten more slowly."""
        out = []
        for g in self._goals:
            for a in g.actions:
                if a.get("action") == "move":
                    p = a.get("params") or {}
                    x, y = p.get("x"), p.get("y")
                    if isinstance(x, int) and isinstance(y, int):
                        out.append(((x, y), g.importance))
        return out

    def complete(self, goal: Goal) -> None:
        goal.status = "done"
        self._remove(goal)

    def fail(self, goal: Goal) -> None:
        goal.status = "failed"
        self._remove(goal)

    def _remove(self, goal: Goal) -> None:
        self._goals = [g for g in self._goals if g is not goal]

    def clear(self) -> None:
        self._goals.clear()

    def __len__(self) -> int:
        return len(self._goals)

    def __iter__(self):
        return iter(self._goals)


def goal_from_intent(intent, importance: float = DEFAULT_IMPORTANCE) -> Goal | None:
    """Wrap a single validated intent dict as a one-action Goal (bridge for
    injected/scripted actions). Returns None for recall/wait (thinking-layer
    choices, not world goals) or malformed input."""
    if not isinstance(intent, dict):
        return None
    action = intent.get("action")
    if action is None or action in ("recall", "wait"):
        return None
    return Goal(
        actions=[{"action": action, "params": dict(intent.get("params") or {})}],
        importance=importance,
        reason=(intent.get("reason") or ""),
    )
