"""Write-time memory consolidation — one simple rule.

Two memories fold into one when they are the *same kind of observation of nearly
the same thing, close in time*:

  - same `sense`, and
  - at most ONE of the subject fields (kind, ref, type, pos, info) differs, and
  - the time gap is within `gap_s`.

`observer_loc`, `direction`, and the bookkeeping fields play no part in the
decision. Time always spans (`t`..`t_end`, `count`), and a folded record keeps
`origin` — the first subject it absorbed — alongside `subject`, the latest. That
first+last pair is all any consumer needs (render shows "from origin to latest";
retrieval indexes both); intermediate waypoints carry no extra information under
this rule, since every snapshot in a run shares the same identity.

This is deliberately generic: it is not movement-specific. Movement is simply the
common case where the one differing field is `pos`.
"""
SUBJECT_FIELDS = ("kind", "ref", "type", "pos", "info")
# Gap sits comfortably above the slowest felt-stat cadence (1 unit / 2s at the
# current hunger drain rate), so a continuously ticking stat folds into one run
# instead of fragmenting on timing jitter.
DEFAULT_GAP_S = 3.0
DEFAULT_MAX_DIFF = 1


def subject_diff(a: dict, b: dict) -> int:
    """How many subject fields differ between two subjects."""
    a, b = a or {}, b or {}
    return sum(1 for f in SUBJECT_FIELDS if a.get(f) != b.get(f))


class Consolidator:
    def __init__(self, gap_s: float = DEFAULT_GAP_S, max_diff: int = DEFAULT_MAX_DIFF):
        self.gap_s = gap_s
        self.max_diff = max_diff

    def candidate(self, memories: list, sense: str, subject: dict, t: float) -> dict | None:
        """The most recent memory this one may fold into, or None. Scans for a
        same-sense memory within the time window whose subject differs in at most
        `max_diff` fields; ties broken toward the latest."""
        best = None
        for m in memories:
            if m["sense"] != sense:
                continue
            gap = t - m["t_end"]
            if gap < 0 or gap > self.gap_s:
                continue
            if subject_diff(m["subject"], subject) > self.max_diff:
                continue
            if best is None or m["t_end"] > best["t_end"]:
                best = m
        return best

    def absorb(self, prev: dict, sense: str, subject: dict, observer_loc, t: float, direction) -> None:
        """Fold a new observation into `prev` in place: pin `origin` to the run's
        first subject the first time it folds, advance the endpoint (latest
        subject/direction/where), extend the time span, and bump the count."""
        if "origin" not in prev:
            prev["origin"] = dict(prev["subject"])   # the run's first subject, kept for good
        prev["subject"] = subject                    # latest subject
        if direction is not None:
            prev["direction"] = direction
        if observer_loc:
            prev["observer_loc"] = list(observer_loc)
        prev["t_end"] = t
        prev["count"] += 1
