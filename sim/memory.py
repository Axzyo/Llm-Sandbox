import re
import threading

# Retrieval scores a probe against each memory on five axes:
#   time (recency) · location (subject.pos) · direction (bearing) · subject (what/who) · sense (saw/heard/did)
# No salience, no embeddings, no entities list: a memory is about exactly one subject;
# multi-entity events are implicit (they become several memories).
#
# WEIGHTS ARE PLACEHOLDERS — all axes deemed important, real balance still TBD.
WEIGHT_TIME = 1.0
WEIGHT_LOCATION = 1.0
WEIGHT_DIRECTION = 1.0
WEIGHT_SUBJECT = 1.0
WEIGHT_SENSE = 1.0

_COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9_]+", str(text).lower()))


def _singular(tok: str) -> str:
    # crude morphology so "bears" (query) matches the "bear" type; not semantics.
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _norm(tokens) -> set:
    return {_singular(t) for t in tokens}


def chebyshev(a, b) -> int:
    if a is None or b is None:
        return 0
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _dir_match(a, b) -> float:
    """1.0 same compass point, fading to 0.0 at opposite; 0 if either is unknown."""
    if a not in _COMPASS or b not in _COMPASS:
        return 0.0
    ia, ib = _COMPASS.index(a), _COMPASS.index(b)
    steps = min((ia - ib) % 8, (ib - ia) % 8)  # 0..4
    return 1.0 - steps / 4.0


def _subject_tokens(subj: dict) -> set:
    toks = set()
    if subj.get("type"):
        toks |= tokenize(subj["type"])
    if subj.get("ref"):
        toks.add(str(subj["ref"]).lower())
    for v in (subj.get("info") or {}).values():
        toks |= tokenize(str(v))
    return toks


def index_tokens(m: dict) -> set:
    """Searchable tokens of a memory, from its subject fields. Makes `bear` (a type)
    findable even though the id is `bear_1`. A consolidated memory also indexes its
    `origin` subject, so the run stays findable by where/what it started as."""
    toks = _subject_tokens(m.get("subject") or {})
    if m.get("origin"):
        toks |= _subject_tokens(m["origin"])
    return toks


class MemoryStore:
    def __init__(self, owner_id: str, max_memories: int = 300, consolidator=None):
        self.owner_id = owner_id
        self.max_memories = max_memories
        self.memories: list = []
        self._next_id = 1
        self._lock = threading.Lock()
        # optional write-time consolidation (duck-typed: candidate() + absorb())
        self._consolidator = consolidator

    def __len__(self) -> int:
        return len(self.memories)

    def _new_record(self, sense: str, subject: dict, observer_loc, t: float, direction) -> dict:
        return {
            "id": None,                  # assigned on append
            "t": t,                      # first occurrence of this (possibly consolidated) memory
            "t_end": t,                  # last occurrence; == t until it consolidates
            "count": 1,                  # raw events folded into this record
            "sense": sense,
            "observer_loc": list(observer_loc) if observer_loc else None,
            "direction": direction,
            "subject": subject,          # {kind, ref, type, pos, info}
            "last_accessed": t,
            "access_count": 0,
        }

    def _append_locked(self, mem: dict) -> None:
        mem["id"] = f"{self.owner_id}_m{self._next_id}"
        self._next_id += 1
        self.memories.append(mem)
        if len(self.memories) > self.max_memories:
            self.memories.pop(0)

    def add(self, sense: str, subject: dict, observer_loc, t: float, direction=None) -> dict:
        """Low-level append: always a fresh record, no consolidation."""
        mem = self._new_record(sense, subject, observer_loc, t, direction)
        with self._lock:
            self._append_locked(mem)
        return mem

    def record(self, sense: str, subject: dict, observer_loc, t: float, direction=None) -> tuple:
        """Consolidation-aware write. If a recent memory is compatible (same sense,
        at most one differing subject field, close in time), fold this into it;
        otherwise append a new record. Returns (record, merged)."""
        with self._lock:
            if self._consolidator is not None:
                prev = self._consolidator.candidate(self.memories, sense, subject, t)
                if prev is not None:
                    self._consolidator.absorb(prev, sense, subject, observer_loc, t, direction)
                    return prev, True
            mem = self._new_record(sense, subject, observer_loc, t, direction)
            self._append_locked(mem)
        return mem, False

    def retrieve(self, query_tokens: set, now_t: float, k: int = 5, halflife_s: float = 300.0,
                 query_loc=None, query_direction=None, query_sense=None) -> list:
        """Score every memory against a probe. A probe only engages the axes it
        specifies; time is always engaged. The denominator is the sum of engaged
        weights so an unspecified axis never dilutes the score."""
        q = _norm(query_tokens) if query_tokens else set()
        use_subject = bool(q)
        use_loc = query_loc is not None
        use_dir = query_direction is not None
        use_sense = query_sense is not None

        with self._lock:
            snapshot = list(self.memories)
        if not snapshot:
            return []

        denom = (
            WEIGHT_TIME
            + (WEIGHT_SUBJECT if use_subject else 0.0)
            + (WEIGHT_LOCATION if use_loc else 0.0)
            + (WEIGHT_DIRECTION if use_dir else 0.0)
            + (WEIGHT_SENSE if use_sense else 0.0)
        ) or 1.0

        scored = []
        for m in snapshot:
            anchor = max(m.get("t_end", m["t"]), m["last_accessed"])
            num = WEIGHT_TIME * (0.5 ** (max(0.0, now_t - anchor) / halflife_s))
            if use_subject:
                idx = _norm(index_tokens(m))
                num += WEIGHT_SUBJECT * (len(q & idx) / len(q))
            if use_loc:
                pos = (m.get("subject") or {}).get("pos")
                num += WEIGHT_LOCATION * (8.0 / (8.0 + chebyshev(query_loc, pos)) if pos is not None else 0.0)
            if use_dir:
                num += WEIGHT_DIRECTION * _dir_match(query_direction, m.get("direction"))
            if use_sense:
                num += WEIGHT_SENSE * (1.0 if m.get("sense") == query_sense else 0.0)
            scored.append((num / denom, m))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        out = []
        for score, m in scored[:k]:
            m["last_accessed"] = now_t
            m["access_count"] += 1
            copy = dict(m)
            copy["_score"] = round(score, 3)
            out.append(copy)
        return out
