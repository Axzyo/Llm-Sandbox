import json

from .consolidation import Consolidator
from .goals import DEFAULT_IMPORTANCE, Goal
from .memory import MemoryStore, index_tokens, tokenize
from .spatial import SpatialMemory

AGENT_MARKER = "autonomous agent"
VALID_MEM_TYPES = ("observation", "action_result")
MAX_MEM_CONTENT = 240
MAX_RECALLS = 2  # deliberate memory searches allowed before the agent must act
MAX_LOOKS = 2    # deliberate map look-ups (remembered terrain at a coord) before acting
MAP_ANCHOR_RADIUS = 4  # window size drawn around a memory location / an explicit look
NOVELTY_THRESHOLD = 0.5  # a new memory scoring below this vs the store is "novel" -> think now


def _merge_memories(base: list, more: list) -> list:
    seen = {m["id"] for m in base}
    return base + [m for m in more if m["id"] not in seen]

SYSTEM_TEMPLATE = """You are __NAME__, an autonomous agent in a 2D grid world.
Your only directive: survive.
You accumulate experiences; they shape how you act, but they never force you.

You have three survival needs — health, hunger, and thirst — each from 0 (empty) to 100 (full). Hunger and thirst fall on their own over time. If either reaches 0 your health drains; keep both well up and your health slowly recovers. Keeping your needs high is what surviving means. Your state reports their values.

Each turn you receive your current state as JSON. You reply with EXACTLY one JSON object and nothing else.

A GOAL is one plan: an ordered list of actions you intend to carry out, with a single "importance" (0–10, higher = more urgent) and a short "reason". Your reply is your set of goals right now:

{"goals":[
  {"actions":[<action>, <action>, ...], "importance":<0-10>, "reason":"<why this plan>"}
]}

Most of the time one goal with one or two actions is enough; author several only when you truly hold several separate intentions, and let importance rank them.

The actions a plan may contain:
- move to a tile: {"action":"move","params":{"x":<int>,"y":<int>}}
- interact with an entity or pick up an item from the ground: {"action":"interact","params":{"target":"<entity id>"}}
- say something aloud — anyone nearby hears it: {"action":"say","params":{"text":"<what you say>"}}
- manage an item you already carry: {"action":"inventory","params":{"op":"use|drop|arrange","item":"<item>"}}

Three special replies stand alone (NOT inside a goal). After a recall or look you will be shown the result and get to choose again:
- do nothing this turn, just observe: {"action":"wait","reason":"<why>"}
- search your memory before deciding: {"action":"recall","params":{"query":"<what to remember>","sense":"saw|heard|did|felt (optional)"},"reason":"<why>"}
- look at the terrain you remember around a tile: {"action":"look","params":{"x":<int>,"y":<int>},"reason":"<why>"}

Rules:
- Act only when you have a reason to. Movement, contact, speech, and effort all carry risk; when you feel safe and nothing needs doing, reply wait. Waiting is a valid and often correct choice — an empty agenda is fine.
- Your state lists your `drives` (your motivations, such as survival and curiosity) and marks each visible thing `familiar` (its kind is one you have interacted with before) or not. They are yours to weigh.
- A goal's actions run in the order you list them; importance decides which goal runs first and lets an urgent new goal preempt one in progress.
- Speech is a broadcast: everyone nearby hears whatever you say, and something you heard may or may not have been meant for you.
- inventory manages items you already have (use, drop, arrange); taking an item off the ground is instead an interact with it.
- Coordinates are tile positions you could stand on.
- Interacting requires the target within your interact range and line of sight.
- You perceive through line of sight only; unseen things do not exist for you yet.
- You are shown a remembered map around yourself and around the locations your recalled memories refer to. Coordinates you have never seen are blank/unknown; do not move to unknown tiles.
- recall returns matching memories and then you choose again; use it when your current memories are not enough. look returns the remembered terrain around a coordinate (even where nothing happened) and then you choose again; use it to check a route or a place you recall."""


def validate_intent(obj) -> dict | None:
    intent = _validate_action(obj)
    if intent is None:
        return None
    reason = obj.get("reason") if isinstance(obj, dict) else None
    if isinstance(reason, str) and reason.strip():
        intent["reason"] = reason.strip()[:160]
    return intent


def _validate_action(obj) -> dict | None:
    if not isinstance(obj, dict):
        return None
    action = obj.get("action")
    params = obj.get("params") or {}
    if action == "move":
        x, y = params.get("x"), params.get("y")
        if not isinstance(x, int) or not isinstance(y, int):
            return None
        return {"action": "move", "params": {"x": x, "y": y}}
    if action == "inventory":
        op = params.get("op")
        item = params.get("item")
        if op not in ("use", "drop", "arrange"):
            return None
        if not isinstance(item, str) or not item.strip():
            return None
        return {"action": "inventory", "params": {"op": op, "item": item.strip()}}
    if action == "interact":
        target = params.get("target")
        if not isinstance(target, str) or not target:
            return None
        return {"action": "interact", "params": {"target": target}}
    if action == "say":
        text = params.get("text") if isinstance(params, dict) else None
        if not isinstance(text, str) or not text.strip():
            return None
        return {"action": "say", "params": {"text": text.strip()[:280]}}
    if action == "wait":
        return {"action": "wait"}
    if action == "recall":
        query = params.get("query")
        if not isinstance(query, str) or not query.strip():
            return None
        out = {"action": "recall", "params": {"query": query.strip()[:120]}}
        sense = params.get("sense")
        if sense in ("saw", "heard", "did", "felt"):
            out["params"]["sense"] = sense
        return out
    if action == "look":
        x, y = params.get("x"), params.get("y")
        if not isinstance(x, int) or not isinstance(y, int):
            return None
        return {"action": "look", "params": {"x": x, "y": y}}
    return None


GOAL_ACTIONS = ("move", "interact", "say", "inventory")   # verbs allowed inside a plan
IMPORTANCE_MIN, IMPORTANCE_MAX = 0.0, 10.0


def validate_goals(obj) -> list[Goal] | None:
    """Strictly parse a goal-set reply. Returns the Goal list on a full match, or
    None if the reply violates the contract (the caller treats None as a bad
    response). There is no silent coercion: the reply MUST be

        {"goals":[ {"actions":[<world action>, ...], "importance":<num>, "reason":?}, ... ]}

    Rejection (→ None) if: not that object shape, an empty goals list, a goal that
    is not a dict, a missing/empty/non-list `actions`, any action that fails the
    world schema or is a thinking-layer verb (recall/wait), or a missing /
    non-numeric `importance`. `importance` is clamped to [0, 10]; `reason` is
    optional. Ordering by importance happens later in GoalList.
    """
    if not isinstance(obj, dict):
        return None
    items = obj.get("goals")
    if not isinstance(items, list) or not items:
        return None
    goals: list[Goal] = []
    for gd in items:
        if not isinstance(gd, dict):
            return None
        raw_actions = gd.get("actions")
        if not isinstance(raw_actions, list) or not raw_actions:
            return None
        actions = []
        for a in raw_actions:
            va = _validate_action(a)
            if va is None or va["action"] not in GOAL_ACTIONS:
                return None
            actions.append(va)
        imp = gd.get("importance")
        if isinstance(imp, bool) or not isinstance(imp, (int, float)):
            return None                                    # importance is required + numeric
        importance = float(min(IMPORTANCE_MAX, max(IMPORTANCE_MIN, imp)))
        reason = gd.get("reason")
        reason = reason.strip()[:160] if isinstance(reason, str) and reason.strip() else ""
        goals.append(Goal(actions=actions, importance=importance, reason=reason))
    return goals


def goals_to_obj(goals) -> dict:
    """Serialize validated Goals back to the canonical wire object the game accepts:
    {"goals":[{"actions":[...],"importance":<num>,"reason":?}]}. Whole importances
    render as ints (8.0 -> 8) so training targets read naturally; empty reasons are
    omitted. Inverse of validate_goals for the fields that survive validation."""
    out = []
    for g in goals:
        imp = int(g.importance) if float(g.importance).is_integer() else g.importance
        goal = {"actions": g.actions, "importance": imp}
        if g.reason:
            goal["reason"] = g.reason
        out.append(goal)
    return {"goals": out}


def filter_response(raw):
    """The response filter: classify a raw LLM reply against the strict contract.

    Returns a small verdict dict — exactly one of:
        {"kind": "goals",  "goals": [Goal, ...]}   a valid goal-set
        {"kind": "wait"}                            valid do-nothing reply
        {"kind": "recall", "params": {...}}         valid memory-search reply
        {"kind": "look",   "params": {"x","y"}}     valid remembered-terrain look-up
        {"kind": "bad",    "reason": "<why>"}       does not match the contract
    """
    if isinstance(raw, dict) and raw.get("action") in ("wait", "recall", "look"):
        intent = validate_intent(raw)
        if intent is None:
            return {"kind": "bad", "reason": f"malformed {raw.get('action')}"}
        act = intent["action"]
        if act == "wait":
            return {"kind": "wait"}
        return {"kind": act, "params": intent.get("params", {})}
    goals = validate_goals(raw)
    if goals is None:
        return {"kind": "bad", "reason": "not a valid goal-set"}
    return {"kind": "goals", "goals": goals}


_DIRS8 = [
    (0, -1, "N"), (1, -1, "NE"), (1, 0, "E"), (1, 1, "SE"),
    (0, 1, "S"), (-1, 1, "SW"), (-1, 0, "W"), (-1, -1, "NW"),
]


def bearing(observer, target) -> str | None:
    """Compass direction from observer tile to target tile, or None if same/unknown."""
    if not observer or not target:
        return None
    dx, dy = target[0] - observer[0], target[1] - observer[1]
    if dx == 0 and dy == 0:
        return None
    best, bestdot = None, -2.0
    norm = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / norm, dy / norm
    for cx, cy, name in _DIRS8:
        cn = (cx * cx + cy * cy) ** 0.5
        dot = (ux * cx + uy * cy) / cn
        if dot > bestdot:
            bestdot, best = dot, name
    return best


def memories_from_event(ev, observer_loc, now: float, self_id: str) -> list:
    """Turn one perception/dialogue event into a list of structured memory kwargs.

    Pure code, no LLM. Dialogue expands to one memory per line. Fields are the
    source of truth; the readable sentence is derived later by render_memory.
    Unknown kinds are recorded generically, never dropped.
    """
    if not isinstance(ev, dict):
        return []
    kind = ev.get("kind")
    if kind in ("entity_entered", "entity_moved"):
        pos = ev.get("pos")
        subj = {"kind": "entity", "ref": ev.get("id"), "type": ev.get("etype", "entity"),
                "pos": list(pos) if pos else None, "info": {}}
        return [dict(sense="saw", subject=subj, observer_loc=observer_loc, now=now,
                     direction=bearing(observer_loc, pos))]
    if kind == "entity_left":
        subj = {"kind": "entity", "ref": ev.get("id"), "type": ev.get("etype", "entity"),
                "pos": None, "info": {"event": "left view"}}
        return [dict(sense="saw", subject=subj, observer_loc=observer_loc, now=now, direction=None)]
    if kind == "did_move":
        pos = ev.get("pos")
        subj = {"kind": "tile", "ref": None, "type": "tile",
                "pos": list(pos) if pos else None, "info": {"outcome": ev.get("outcome", "arrived")}}
        return [dict(sense="did", subject=subj, observer_loc=observer_loc, now=now,
                     direction=bearing(observer_loc, pos))]
    if kind == "did_interact":
        tpos = ev.get("target_pos")
        info = {"outcome": ev.get("outcome", "ok")}
        if ev.get("effect"):
            info["effect"] = ev["effect"]     # what interacting yielded (e.g. "thirst +100", "picked a berry")
        subj = {"kind": "entity", "ref": ev.get("target"), "type": ev.get("target_type", "entity"),
                "pos": list(tpos) if tpos else None, "info": info}
        return [dict(sense="did", subject=subj, observer_loc=observer_loc, now=now,
                     direction=bearing(observer_loc, tpos))]
    if kind == "did_inventory":
        info = {"op": ev.get("op", "use"), "outcome": ev.get("outcome", "ok")}
        if ev.get("effect"):
            info["effect"] = ev["effect"]     # what using the item did (e.g. "hunger +40")
        subj = {"kind": "item", "ref": ev.get("item"), "type": ev.get("item_type", "item"),
                "pos": None, "info": info}
        return [dict(sense="did", subject=subj, observer_loc=observer_loc, now=now, direction=None)]
    if kind == "did_say":
        # speaking is a 'did' (an action you enacted); the words live in info.text
        subj = {"kind": "self", "ref": self_id, "type": "self",
                "pos": list(observer_loc) if observer_loc else None, "info": {"text": ev.get("text", "")}}
        return [dict(sense="did", subject=subj, observer_loc=observer_loc, now=now, direction=None)]
    if kind == "felt_stat":
        # interoception: an internal stat shifted (`did` = external act, `felt` =
        # internal state). Direction lives in subject.type so a run of same-direction
        # changes consolidates (only info differs) while a reversal, or another stat,
        # differs in two fields and starts its own record.
        stat = ev.get("stat")
        subj = {"kind": "stat", "ref": stat, "type": ev.get("direction", "changed"),
                "pos": None, "info": {"stat": stat, "value": ev.get("value")}}
        return [dict(sense="felt", subject=subj, observer_loc=observer_loc, now=now, direction=None)]
    if kind == "heard_say":
        spk = ev.get("speaker")
        spos = ev.get("speaker_pos")
        subj = {"kind": "entity", "ref": spk, "type": ev.get("speaker_type", "agent"),
                "pos": list(spos) if spos else None, "info": {"text": ev.get("text", "")}}
        return [dict(sense="heard", subject=subj, observer_loc=observer_loc, now=now,
                     direction=bearing(observer_loc, spos))]
    subj = {"kind": "event", "ref": None, "type": "event", "pos": None,
            "info": {"raw": json.dumps(ev)[:MAX_MEM_CONTENT]}}
    return [dict(sense="felt", subject=subj, observer_loc=observer_loc, now=now, direction=None)]


def render_memory(m: dict) -> str:
    """Structured memory -> a first-person sentence for the LLM prompt (derived, not stored)."""
    s = m.get("subject") or {}
    sense = m.get("sense")
    ref, typ, pos = s.get("ref"), s.get("type"), s.get("pos")
    who = f"{typ} ({ref})" if ref and ref != typ else (typ or "something")
    origin = (m.get("origin") or {}).get("pos")   # first pos of a consolidated run, if any
    if sense == "saw":
        info = s.get("info") or {}
        if info.get("event") == "left view":
            return f"{who} left my view"
        dirn = f" to the {m['direction']}" if m.get("direction") else ""
        if origin and pos and origin != pos:    # a consolidated sighting run
            return f"I saw {who} move from {origin} to {pos}{dirn}"
        where = f" at {pos}" if pos else ""
        return f"I saw {who}{dirn}{where}"
    if sense == "heard":
        return f'{who} said: "{(s.get("info") or {}).get("text", "")}"'
    if sense == "did":
        info = s.get("info") or {}
        if "text" in info:                       # speaking is a 'did'
            return f'I said: "{info["text"]}"'
        outcome = info.get("outcome")
        if s.get("kind") == "tile":              # a move
            if origin and pos and origin != pos:  # a consolidated movement run
                return f"I moved from {origin} to {pos}"
            base = f"I moved to {pos}" if pos else "I moved"
            return base if outcome in (None, "arrived") else f"I tried to move to {pos} ({outcome})"
        effect = info.get("effect")
        if s.get("kind") == "item":              # an inventory op
            op = info.get("op", "use")
            verb = {"use": "used", "drop": "dropped", "arrange": "arranged"}.get(op, op)
            if outcome in (None, "ok"):
                return f"I {verb} {who} and {effect}" if effect else f"I {verb} {who}"
            return f"I tried to {op} {who} ({outcome})"
        if outcome in (None, "ok"):              # interact
            return f"I interacted with {who} and {effect}" if effect else f"I interacted with {who}"
        return f"I tried to interact with {who} ({outcome})"
    if sense == "felt" and s.get("kind") == "stat":
        val = (s.get("info") or {}).get("value")
        first = ((m.get("origin") or {}).get("info") or {}).get("value")
        if first is not None and first != val:   # a consolidated run spans first -> latest
            verb = {"rising": "rose", "falling": "fell"}.get(s.get("type"), "changed")
            return f"I felt my {ref} {verb} from {first} to {val}"
        word = {"rising": "rise", "falling": "fall"}.get(s.get("type"), "change")
        return f"I felt my {ref} {word} to {val}"
    return f"something happened ({(s.get('info') or {}).get('raw', '')})"


class Brain:
    def __init__(
        self,
        entity_id: str,
        provider,
        journal=None,
        memory_k: int = 5,
        memory_halflife_s: float = 300.0,
    ):
        self.entity_id = entity_id
        self.provider = provider
        self.journal = journal
        self.memory_k = memory_k
        self.memory_halflife_s = memory_halflife_s
        self.store = MemoryStore(entity_id, consolidator=Consolidator())
        self.spatial = SpatialMemory(entity_id)   # remembered map, separate from episodic memory
        self.system = SYSTEM_TEMPLATE.replace("__NAME__", entity_id)
        self.pending_think = False   # set when a novel memory forms; consumed by the think loop

    def perceive_tiles(self, seen) -> int:
        """Fold seen ((x,y), type) tiles into spatial memory (reinforcing them).
        Newly discovered geometry is novel -> think now, like a novel episodic memory."""
        new = self.spatial.observe_many(seen)
        if new:
            self.pending_think = True
        return new

    def maintain_spatial(self, goal_locations=()) -> int:
        """One spatial-memory maintenance pass: decay memorability, protect tiles
        near the NPC's current goal locations, forget what faded, cap the rest.
        Called on the perception cadence; returns tiles forgotten."""
        return self.spatial.age(goal_locations)

    def _retrieve_for(self, query_tokens: set, now_t: float, query_loc=None) -> list:
        memories = self.store.retrieve(
            query_tokens=query_tokens,
            now_t=now_t,
            k=self.memory_k,
            halflife_s=self.memory_halflife_s,
            query_loc=query_loc,
        )
        if memories and self.journal:
            self.journal.log(
                self.entity_id,
                "memory_retrieve",
                ids=[m["id"] for m in memories],
                scores=[m["_score"] for m in memories],
            )
        return memories

    def _chat_raw(self, system: str, user: str, on_delta=None) -> dict | None:
        """One provider call. Streams when on_delta is given and supported (so a
        `say` types out live). Journals the prompt + raw response (or the error).
        Returns the raw parsed dict, or None on transport/parse failure."""
        streamed = on_delta is not None and hasattr(self.provider, "chat_json_stream")
        try:
            if streamed:
                raw = self.provider.chat_json_stream(system, user, on_delta)
            else:
                raw = self.provider.chat_json(system, user)
        except Exception as exc:
            if self.journal:
                self.journal.log(self.entity_id, "response", error=str(exc), streamed=streamed)
            return None
        if self.journal:
            self.journal.log(self.entity_id, "prompt", system=system, user=user)
            self.journal.log(self.entity_id, "response", raw=raw, streamed=streamed)
        return raw

    def _run_recall(self, params: dict, memories: list, now_t: float, query_loc) -> list:
        """Execute a `recall` request and fold the results into the recalled set."""
        found = self.store.retrieve(
            tokenize(params.get("query", "")),
            now_t,
            k=self.memory_k,
            halflife_s=self.memory_halflife_s,
            query_loc=query_loc,
            query_sense=params.get("sense"),
        )
        if self.journal:
            self.journal.log(self.entity_id, "memory_recall", query=params.get("query"),
                             sense=params.get("sense"), ids=[m["id"] for m in found])
        return _merge_memories(memories, found)

    def query_from(self, snapshot: dict) -> set:
        tokens = set()
        for e in snapshot.get("visible_entities", []):
            tokens |= tokenize(e.get("id", ""))
        for ev in snapshot.get("recent_perceptions", []):
            tokens |= tokenize(ev.get("kind", "")) | tokenize(ev.get("id", ""))
        return tokens

    def _is_novel(self, subject: dict, observer_loc, now_t: float) -> bool:
        """Is this about to be a memory unlike anything already stored? Compared by the
        same similarity scorer used for recall (checked BEFORE the memory is added)."""
        probe = index_tokens({"subject": subject})
        loc = subject.get("pos") or observer_loc
        found = self.store.retrieve(probe, now_t, k=1, query_loc=loc)
        return (not found) or (found[0]["_score"] < NOVELTY_THRESHOLD)

    def record_events(self, events: list, now_t: float, location=None) -> list:
        """Write structured memories straight from events. Code-only, instant, no LLM.
        A novel memory (unlike anything stored) raises pending_think -> the agent thinks now."""
        saved = []
        for ev in events or []:
            for kw in memories_from_event(ev, location, now_t, self.entity_id):
                # novelty is judged before the write; a memory that consolidates into
                # an existing run is by definition not novel, so it never wakes a think.
                novel = self._is_novel(kw["subject"], kw["observer_loc"], now_t)
                mem, merged = self.store.record(
                    kw["sense"], kw["subject"], kw["observer_loc"], kw["now"],
                    direction=kw.get("direction"),
                )
                if merged:
                    novel = False
                if novel:
                    self.pending_think = True
                saved.append(mem)
                if self.journal:
                    self.journal.log(self.entity_id, "memory_write", memory=mem, novel=novel, merged=merged)
        return saved

    def _map_blocks(self, snapshot: dict, memories: list, extra_anchors=()) -> list:
        """Rendered remembered-map windows: one around self (at vision radius) and
        one around each distinct location the recalled memories refer to or the
        agent explicitly looked at. Anchors already covered by a rendered window are
        skipped so the same terrain isn't drawn twice."""
        self_pos = snapshot.get("self_pos")
        if self_pos is None:
            return []
        me = (self_pos[0], self_pos[1])
        vision = snapshot.get("vision_radius", 8)
        blocks = []
        rendered = []  # (center, radius) already drawn

        def covered(a):
            return any(max(abs(a[0] - cx), abs(a[1] - cy)) <= rad for (cx, cy), rad in rendered)

        m = self.spatial.render_local(me, vision, marker=me)
        if m:
            blocks.append(m)
        rendered.append((me, vision))

        anchors = []
        for mem in memories:
            loc = (mem.get("subject") or {}).get("pos") or mem.get("observer_loc")
            if loc:
                anchors.append((loc[0], loc[1]))
        anchors += [(a[0], a[1]) for a in extra_anchors]
        for a in anchors:
            if covered(a):
                continue
            block = self.spatial.render_local(a, MAP_ANCHOR_RADIUS, marker=me)
            if block:
                blocks.append(block)
            rendered.append((a, MAP_ANCHOR_RADIUS))
        return blocks

    def _familiar_types(self) -> set:
        """Types this NPC has actually INTERACTED with (a `did` memory about them) —
        i.e. kinds of thing it has learned something about, not merely seen."""
        out = set()
        for m in self.store.memories:
            subj = m.get("subject") or {}
            if m.get("sense") == "did" and subj.get("kind") == "entity" and subj.get("type"):
                out.add(subj["type"])
        return out

    def _annotate_familiarity(self, snapshot: dict) -> dict:
        """Tag each visible thing `familiar` (its kind is one this NPC has interacted
        with) or not — a plain fact, derived from memory. The LLM weighs it against
        its own `curiosity` drive; nothing here tells it what to do. Returns a copy;
        the input snapshot is not mutated."""
        vis = snapshot.get("visible_entities")
        if not vis:
            return snapshot
        familiar = self._familiar_types()
        annotated = [{**e, "familiar": e.get("type") in familiar} for e in vis]
        return {**snapshot, "visible_entities": annotated}

    def build_context(self, snapshot: dict, memories: list, corrective: bool = False,
                      extra_anchors=()) -> str:
        parts = ["state: " + json.dumps(self._annotate_familiarity(snapshot))]
        parts += self._map_blocks(snapshot, memories, extra_anchors)
        if memories:
            lines = [f"- {render_memory(m)}" for m in memories]
            parts.append("memories you recall:\n" + "\n".join(lines))
        parts.append("Choose your goals as JSON, or reply wait / recall / look.")
        msg = "\n\n".join(parts)
        if corrective:
            msg += (
                "\nIMPORTANT: your previous output was not valid. Reply with exactly one"
                ' JSON object: {"goals":[{"actions":[...],"importance":<0-10>,"reason":"..."}]},'
                ' or {"action":"wait","reason":"..."}, or a recall/look object.'
            )
        return msg

    def decide(self, snapshot: dict, events: list | None = None, on_delta=None) -> list[Goal]:
        """Return the NPC's goal set for this think (possibly empty = do nothing).

        Memory is written at perception time now, not here; decide only recalls +
        plans. The LLM may first `recall` (bounded by MAX_RECALLS) or `wait`;
        otherwise its output is parsed into importance-ranked Goal plans.
        """
        now_t = float(snapshot.get("t", 0.0))
        self_pos = snapshot.get("self_pos")
        memories = self._retrieve_for(self.query_from(snapshot), now_t, query_loc=self_pos)
        anchors = []                            # extra map windows the agent looked at
        corrective = False
        recalls = looks = 0
        for _ in range(MAX_RECALLS + MAX_LOOKS + 2):
            context = self.build_context(snapshot, memories, corrective=corrective,
                                         extra_anchors=anchors)
            raw = self._chat_raw(self.system, context, on_delta=on_delta)
            if raw is None:
                if corrective:
                    return []
                corrective = True
                continue
            res = filter_response(raw)
            kind = res["kind"]
            if kind == "recall" and recalls < MAX_RECALLS:
                recalls += 1
                memories = self._run_recall(res["params"], memories, now_t, self_pos)
                corrective = False
                continue
            if kind == "look" and looks < MAX_LOOKS:
                looks += 1
                p = res["params"]
                anchors.append((p["x"], p["y"]))    # next prompt renders the map there
                if self.journal:
                    self.journal.log(self.entity_id, "map_look", x=p["x"], y=p["y"])
                corrective = False
                continue
            if kind == "wait":
                return []                        # explicit do-nothing: empty agenda
            if kind == "goals":
                return res["goals"]
            # bad response (or a pull past its cap): log it, nudge once, then give up
            if self.journal:
                reason = res.get("reason") if kind == "bad" else f"{kind}_exhausted"
                self.journal.log(self.entity_id, "response_rejected", reason=reason)
            if corrective:
                return []
            corrective = True
        return []
