"""The per-tick simulation, independent of any renderer: needs, resources,
death, perception, thinking, and goal execution. main.py (pygame) and the
headless episode runner (train/run_episodes.py) both drive the same Engine —
one sim, two front ends, no fork.

Thinking is synchronous by default (brain.decide runs inline, so a headless
rollout is reproducible given a seed and a deterministic provider). main.py
passes dispatch_think to route think jobs to its worker thread instead and
feeds the results back through post_goals().
"""
from .actions import attempt_move, evaluate_interact
from .journal import Journal
from .needs import tick_needs
from .pathing import next_step
from .perception import PerceptionTracker, visible_entities, visible_tiles
from .terrain import tick_resources, interact_with, use_item
from .world import chebyshev

PERCEPTION_INTERVAL = 0.2

ENABLE_NPC_MEMORY = True
ENABLE_NPC_THINKS = True

def build_snapshot(npc, world, pending_events: list, now_t: float) -> dict:
    return {
        "t": round(now_t, 2),
        "self_id": npc.id,
        "self_pos": [npc.x, npc.y],
        "health": round(npc.stats["health"]),    # three survival needs, 0 (empty) .. 100 (full)
        "hunger": round(npc.stats["hunger"]),
        "thirst": round(npc.stats["thirst"]),
        "drives": npc.drives,                    # personality weights (survival, curiosity, ...)
        "vision_radius": npc.properties["vision_radius"],
        "hearing_radius": npc.properties["hearing_radius"],
        "interact_range": npc.properties["interact_range"],
        "visible_entities": [{"id": e.id, "type": e.kind, "pos": [e.x, e.y]} for e in visible_entities(world, npc)],
        "recent_perceptions": pending_events[-12:],
    }


def remember_action(brains: dict, npc, event: dict, sim_t: float) -> None:
    """Write a 'did' memory of a resolved action outcome, through the same pipeline."""
    if ENABLE_NPC_MEMORY and npc.id in brains:
        brains[npc.id].record_events([event], sim_t, location=[npc.x, npc.y])


def broadcast(world, speaker, text: str, brains: dict, pending_obs: dict, sim_t: float,
              hear_log: list, journal: Journal) -> None:
    """Speech is a broadcast: the speaker remembers saying it, and every entity within
    its own hearing_radius forms a `heard` memory (walls don't block sound). The player,
    if in earshot, just sees it in the log."""
    journal.log(speaker.id, "say", text=text)
    print(f"[{sim_t:7.1f}] {'you' if speaker.id == 'player' else speaker.id}: {text}", flush=True)
    if ENABLE_NPC_MEMORY and speaker.id in brains:
        remember_action(brains, speaker, {"kind": "did_say", "text": text}, sim_t)
    for e in world.entities.values():
        if e.id == speaker.id:
            continue
        if chebyshev(speaker.x, speaker.y, e.x, e.y) > e.properties["hearing_radius"]:
            continue
        event = {"kind": "heard_say", "speaker": speaker.id, "speaker_type": speaker.kind,
                 "speaker_pos": [speaker.x, speaker.y], "text": text}
        if ENABLE_NPC_MEMORY and e.id in brains:
            brains[e.id].record_events([event], sim_t, location=[e.x, e.y])
            pending_obs[e.id].append(event)
        elif e.id == "player":
            hear_log.append(f"{speaker.id}: {text}")


def enact_instant(world, npc, action_obj, journal: Journal, sim_t: float, brains: dict,
                  pending_obs: dict, hear_log: list) -> str:
    """Enact a one-tick action (interact / inventory / say). Returns 'done' or 'failed'."""
    action = action_obj.get("action")
    params = action_obj.get("params") or {}
    if action == "interact":
        res = evaluate_interact(world, npc, params.get("target"))
        target = world.entities.get(res.get("target"))
        tpos = [target.x, target.y] if target is not None else None
        ttype = target.kind if target is not None else "entity"
        if res["ok"]:
            journal.log(npc.id, "action_complete", action="interact", **res)
            # a resource target's data decides what interacting does (drink / pick)
            eff = interact_with(target, npc, sim_t) if target is not None else None
            effect = None
            if eff is not None:
                journal.log(npc.id, "resource_use", target=target.id, **eff)
                if eff["did"] == "drink":
                    effect = f"{eff['stat']} +{eff['gained']:g}"
                elif eff["did"] == "harvest":
                    effect = f"picked a {eff['yields']}" if eff["ok"] else "nothing to pick"
            remember_action(brains, npc, {"kind": "did_interact", "target": res.get("target"),
                                          "target_type": ttype, "target_pos": tpos,
                                          "outcome": "ok", "effect": effect}, sim_t)
            return "done"
        journal.log(npc.id, "action_failed", action="interact", **res)
        outcome = "out_of_range" if not res.get("range_ok") else ("no_los" if not res.get("los_ok") else "failed")
        remember_action(brains, npc, {"kind": "did_interact", "target": params.get("target"),
                                      "target_type": ttype, "target_pos": tpos, "outcome": outcome}, sim_t)
        return "failed"
    if action == "inventory":
        op, item = params.get("op"), params.get("item")
        if item in npc.inventory:
            # 'use' applies the item's own effect (eat a berry -> hunger); the item
            # is consumed. drop/arrange have no effect yet.
            used = use_item(npc, item) if op == "use" else None
            journal.log(npc.id, "action_complete", action="inventory", op=op, item=item,
                        **(used or {}))
            effect = f"{used['stat']} +{used['gained']:g}" if used else None
            remember_action(brains, npc, {"kind": "did_inventory", "op": op, "item": item,
                                          "outcome": "ok", "effect": effect}, sim_t)
            return "done"
        journal.log(npc.id, "action_failed", action="inventory", op=op, item=item, reason="no_such_item")
        remember_action(brains, npc, {"kind": "did_inventory", "op": op, "item": item, "outcome": "no_such_item"}, sim_t)
        return "failed"
    if action == "say":
        broadcast(world, npc, params.get("text", ""), brains, pending_obs, sim_t, hear_log, journal)
        return "done"
    # wait/recall are thinking-layer choices and never become goals, so they don't reach here
    journal.log(npc.id, "action_failed", action=str(action), reason="unknown_action")
    return "failed"


def progress_move(world, npc, goal, action_obj, journal: Journal, sim_t: float, brains: dict) -> str:
    """Advance a durative move action one step. Returns 'active' (more to go),
    'done' (arrived) or 'failed' (unreachable)."""
    params = action_obj.get("params") or {}
    tx, ty = params.get("x"), params.get("y")
    if (npc.x, npc.y) == (tx, ty):
        remember_action(brains, npc, {"kind": "did_move", "pos": [tx, ty], "outcome": "arrived"}, sim_t)
        return "done"
    if goal.started_step != goal.step:
        goal.started_step = goal.step
        journal.log(npc.id, "action_start", action="move", to=[tx, ty], source="goal")
    if sim_t < npc.next_move_at:
        return "active"
    step = next_step(world, (npc.x, npc.y), (tx, ty))
    if step is None:
        journal.log(npc.id, "action_failed", action="move", reason="unreachable", to=[tx, ty])
        remember_action(brains, npc, {"kind": "did_move", "pos": [tx, ty], "outcome": "unreachable"}, sim_t)
        return "failed"
    result = attempt_move(world, npc, step[0] - npc.x, step[1] - npc.y)
    npc.next_move_at = sim_t + npc.properties["move_interval"]
    if result["ok"]:
        journal.log(npc.id, "action_complete", action="move", **result)
    # blocked/occupied: stay active and retry next tick (pathing reroutes)
    return "active"


def advance_goals(world, npc, journal: Journal, sim_t: float, brains: dict,
                  pending_obs: dict, hear_log: list) -> None:
    """Work the NPC's top goal for this tick. Durative goals stay 'active' and
    resume next tick unless a higher-importance goal has since preempted them."""
    goal = npc.goals.current()
    if goal is None:
        return
    action_obj = goal.current_action
    if action_obj is None:                 # plan exhausted -> the goal is done
        npc.goals.complete(goal)
        return
    goal.status = "active"
    if action_obj.get("action") == "move":
        outcome = progress_move(world, npc, goal, action_obj, journal, sim_t, brains)
    else:
        outcome = enact_instant(world, npc, action_obj, journal, sim_t, brains, pending_obs, hear_log)
    if outcome == "done":
        if not goal.advance():             # no actions left -> whole plan complete
            npc.goals.complete(goal)
    elif outcome == "failed":
        npc.goals.fail(goal)               # a failed step abandons the whole plan
    # "active": leave it in place for next tick


def death_cause(npc) -> str:
    """What killed a 0-hp entity, read from its meters."""
    return "dehydration" if npc.stats["thirst"] <= 0.0 else ("starvation" if npc.stats["hunger"] <= 0.0 else "unknown")


def reap_dead(npcs: list, npcs_by_id: dict, world, journal: Journal, sim_t: float) -> list:
    """Remove NPCs whose health has hit 0 — death is permanent, there is no respawn.
    A dead NPC leaves the world (others perceive it disappear), stops thinking and
    perceiving, and its death is logged. Returns the ones that died this tick."""
    dead = [n for n in npcs if n.stats["health"] <= 0.0]
    for npc in dead:
        cause = death_cause(npc)
        journal.log(npc.id, "death", pos=[npc.x, npc.y], cause=cause)
        print(f"[{sim_t:7.1f}] {npc.id} died of {cause}", flush=True)
        world.entities.pop(npc.id, None)
        npcs_by_id.pop(npc.id, None)
        npcs.remove(npc)
    return dead


class Engine:
    """One sim tick, front-end agnostic. step(dt) runs: needs -> resources ->
    death -> perception (on its own cadence) -> thinking -> goal execution."""

    def __init__(self, world, npcs: list, brains: dict, journal: Journal, dispatch_think=None):
        self.world = world
        self.npcs = npcs                                  # live list; death mutates it
        self.npcs_by_id = {n.id: n for n in npcs}
        self.brains = brains
        self.journal = journal
        self.dispatch_think = dispatch_think              # None = think inline (headless)
        self.trackers = {n.id: PerceptionTracker(n.id) for n in npcs}
        self.pending_obs = {n.id: [] for n in npcs}
        self.thinking = {n.id: False for n in npcs}
        self.next_think_at = {n.id: 0.0 for n in npcs}
        self._felt_last: dict = {}                        # npc id -> {stat: last rounded value}
        self.hear_log: list = []
        self.sim_t = 0.0
        self.next_perceive_at = 0.0

    def track(self, entity) -> None:
        """Register a brainless body (the player) so broadcast/pending bookkeeping
        has a slot for it; it never perceives or thinks."""
        self.pending_obs.setdefault(entity.id, [])

    def step(self, dt: float) -> list:
        """Advance the whole sim by dt seconds. Returns the entities that died."""
        self.sim_t += dt
        # survival needs drain in real time -> the pressure NPCs perceive and weigh
        for npc in self.npcs:
            tick_needs(npc, dt)
        tick_resources(self.world, self.sim_t)   # regrow berries whose timer elapsed
        died = reap_dead(self.npcs, self.npcs_by_id, self.world, self.journal, self.sim_t)
        if self.sim_t >= self.next_perceive_at:
            self.next_perceive_at = self.sim_t + PERCEPTION_INTERVAL
            self._perceive()
        self._think()
        for npc in self.npcs:
            advance_goals(self.world, npc, self.journal, self.sim_t, self.brains,
                          self.pending_obs, self.hear_log)
        return died

    def _perceive(self) -> None:
        for npc in self.npcs:
            brain = self.brains.get(npc.id)
            if ENABLE_NPC_MEMORY and brain is not None:
                # geometry perception -> spatial memory (a remembered map), separate
                # from episodic memory so ~289 seen tiles never swamp recall. Then a
                # maintenance pass decays memorability and forgets faded tiles, keeping
                # geometry near current goal locations.
                brain.perceive_tiles(visible_tiles(self.world, npc))
                brain.maintain_spatial(npc.goals.locations())
                # interoception: stat shifts become `felt` memories through the same
                # pipeline. They are not perception events (the snapshot already
                # carries the current numbers); memory is what gives them a history.
                felt = self._sense_stats(npc)
                if felt:
                    brain.record_events(felt, self.sim_t, location=[npc.x, npc.y])
            events = self.trackers[npc.id].update(self.world, npc)
            if events:
                self.journal.log(npc.id, "perception", events=events)
                if ENABLE_NPC_MEMORY and brain is not None:
                    brain.record_events(events, self.sim_t, location=[npc.x, npc.y])
                self.pending_obs[npc.id].extend(events)
                for ev in events:
                    if ev["kind"] in ("entity_entered", "entity_moved"):
                        npc.target = ev["id"]
                    elif ev["kind"] == "entity_left" and npc.target == ev["id"]:
                        npc.target = None

    def _sense_stats(self, npc) -> list:
        """One felt event per stat whose value — rounded, exactly as the agent
        perceives it in its snapshot — changed since the last pass. Interoception
        covers every entry in npc.stats (health, hunger, thirst, later mana, ...);
        the internal counterpart of a `did`. The first pass only sets the baseline
        (being spawned is not a change)."""
        current = {name: round(val) for name, val in npc.stats.items()}
        last = self._felt_last.get(npc.id)
        self._felt_last[npc.id] = current
        if last is None:
            return []
        return [{"kind": "felt_stat", "stat": name, "value": cur,
                 "direction": "rising" if cur > last[name] else "falling"}
                for name, cur in current.items() if cur != last[name]]

    def _think(self) -> None:
        for npc in list(self.npcs):
            if not ENABLE_NPC_THINKS or self.thinking[npc.id]:
                continue
            brain = self.brains.get(npc.id)
            if brain is None:
                continue
            # think when something novel happened, or on the idle cadence — internal
            # pressure (hunger, thirst) is never a perception event, so without the
            # cadence an agent would starve without ever reconsidering (DESIGN: LLM loop)
            if not brain.pending_think and self.sim_t < self.next_think_at[npc.id]:
                continue
            brain.pending_think = False
            self.next_think_at[npc.id] = self.sim_t + npc.properties["think_interval"]
            events = list(self.pending_obs[npc.id])
            self.pending_obs[npc.id] = []
            snapshot = build_snapshot(npc, self.world, events, self.sim_t)
            self.journal.log(npc.id, "action_start", action="think")
            if self.dispatch_think is not None:
                self.thinking[npc.id] = True
                self.dispatch_think(npc.id, snapshot, events)
            else:
                self.post_goals(npc.id, brain.decide(snapshot, events))

    def post_goals(self, eid: str, goals: list) -> None:
        """Deliver a finished think's goal set (sync or from a worker thread's
        result queue). The entity may have died while thinking."""
        self.thinking[eid] = False
        if goals and eid in self.npcs_by_id:
            self.npcs_by_id[eid].goals.add_many(goals)     # merged + re-sorted by importance
            self.journal.log(eid, "goals_added", goals=[g.summary() for g in goals],
                             importances=[g.importance for g in goals])
