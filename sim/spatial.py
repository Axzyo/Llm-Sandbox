"""Spatial memory: an NPC's remembered map, kept separate from episodic memory.

Geometry is learned through perception, but a remembered occupancy grid would
swamp the small episodic store (one glance reveals ~289 tiles; recall returns
only k). So tiles live in their own per-entity layer: what floor/wall the NPC has
seen, at absolute coordinates. Unseen tiles are simply unknown. Rendered into the
think prompt as a local map so the agent picks move destinations from geometry it
remembers, never from tiles it has never seen.

Forgetting is driven by a per-tile **memorability**, not by distance:
  * seeing a tile reinforces it (recency + frequency), capped;
  * every maintenance pass memorability decays;
  * tiles near the NPC's current goal locations are floored to (that goal's
    importance x proximity), so geometry near an urgent goal resists fading harder
    than geometry near a trivial one, and irrelevant terrain drops out;
  * a tile is forgotten once memorability falls below a threshold, and a hard
    tile cap is a backstop that evicts the least-memorable tiles.

Tunables are module constants (placeholders, like sim/memory.py's weights).
Pure data + string rendering — no world, no LLM.
"""

SIGHT_BOOST = 1.0          # memorability added each time a tile is seen
MEMORABILITY_CAP = 4.0     # ceiling so oft-seen tiles don't grow without bound
DECAY = 0.9                # memorability multiplier per maintenance pass
FORGET_THRESHOLD = 0.15    # forget a tile once memorability drops to/below this
MAX_TILES = 8000           # hard backstop; evict least-memorable beyond this
GOAL_RANGE = 8             # tiles within this chebyshev distance of a goal are protected
# The memorability floor AT a goal location is that goal's own `importance` (0-10),
# fading to 0 at GOAL_RANGE — no separate strength constant; important goals stick harder.


class SpatialMemory:
    def __init__(self, owner_id: str, max_tiles: int | None = MAX_TILES):
        self.owner_id = owner_id
        self.tiles: dict = {}          # (x, y) -> {"type": "floor"|"wall", "memorability": float}
        self.max_tiles = max_tiles     # cap backstop; None disables it

    def __len__(self) -> int:
        return len(self.tiles)

    def observe(self, coord, tile_type: str) -> None:
        cell = self.tiles.get(coord)
        if cell is None:
            self.tiles[coord] = {"type": tile_type, "memorability": SIGHT_BOOST}
        else:
            cell["type"] = tile_type   # tiles are static today; last write wins
            cell["memorability"] = min(MEMORABILITY_CAP, cell["memorability"] + SIGHT_BOOST)
        self._evict_to_cap()

    def observe_many(self, seen) -> int:
        """Record a batch of ((x,y), type). Returns how many were newly discovered."""
        new = 0
        for coord, ttype in seen:
            cell = self.tiles.get(coord)
            if cell is None:
                new += 1
                self.tiles[coord] = {"type": ttype, "memorability": SIGHT_BOOST}
            else:
                cell["type"] = ttype
                cell["memorability"] = min(MEMORABILITY_CAP, cell["memorability"] + SIGHT_BOOST)
        self._evict_to_cap()
        return new

    def get(self, coord):
        cell = self.tiles.get(coord)
        return cell["type"] if cell else None

    def known(self, coord) -> bool:
        return coord in self.tiles

    def memorability(self, coord) -> float:
        cell = self.tiles.get(coord)
        return cell["memorability"] if cell else 0.0

    # --- forgetting: decay + goal-proximity + threshold + cap backstop ---------

    def age(self, goal_locations=()) -> int:
        """One maintenance pass: decay every tile, floor tiles near goal locations
        to (goal importance x proximity) so geometry near an urgent goal resists
        fading harder, then forget whatever fell below threshold and evict past the
        cap. Returns tiles forgotten.

        `goal_locations` is an iterable of ((x,y), importance) — the NPC's current
        goal targets paired with each goal's own importance value."""
        for cell in self.tiles.values():
            cell["memorability"] *= DECAY
        goals = list(goal_locations)
        if goals:
            for coord, cell in self.tiles.items():
                best = 0.0
                for (gx, gy), importance in goals:
                    d = max(abs(coord[0] - gx), abs(coord[1] - gy))
                    if d <= GOAL_RANGE:
                        val = importance * (1.0 - d / (GOAL_RANGE + 1))
                        if val > best:
                            best = val
                if best > cell["memorability"]:
                    cell["memorability"] = best
        forgotten = self.forget_where(lambda _c, cell: cell["memorability"] <= FORGET_THRESHOLD)
        return forgotten + self._evict_to_cap()

    def forget_where(self, predicate) -> int:
        """Forget every tile for which predicate(coord, cell) is True. Returns count."""
        doomed = [c for c, cell in self.tiles.items() if predicate(c, cell)]
        for c in doomed:
            del self.tiles[c]
        return len(doomed)

    def _evict_to_cap(self) -> int:
        """If a max_tiles cap is set and exceeded, drop the least-memorable tiles
        down to the cap. No-op while max_tiles is None."""
        if self.max_tiles is None or len(self.tiles) <= self.max_tiles:
            return 0
        excess = len(self.tiles) - self.max_tiles
        weakest = sorted(self.tiles.items(), key=lambda kv: kv[1]["memorability"])[:excess]
        for c, _cell in weakest:
            del self.tiles[c]
        return excess

    def render_local(self, center, radius: int, marker=None) -> str | None:
        """A coordinate-framed local map centered on `center`, or None if nothing
        nearby is remembered yet. `marker` is the tile to draw as '@' (the NPC's
        real position); it defaults to the center for a self-centered view, and for
        a window centered elsewhere (a memory/look location) '@' only appears if the
        NPC actually stands inside it. Columns run x low->high left->right; each row
        is labelled by its y. Glyphs: '@' you, '#' wall, '.' floor, ' ' unseen."""
        cx, cy = center
        mark = tuple(marker) if marker is not None else (cx, cy)
        x0, x1 = cx - radius, cx + radius
        y0, y1 = cy - radius, cy + radius
        if not any((x, y) in self.tiles
                   for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)):
            return None
        header = (f"remembered map - columns are x={x0}..{x1} (left to right), each row "
                  f"labelled by y; '@'=you '#'=wall '.'=floor ' '=unseen:")
        lines = [header]
        for y in range(y0, y1 + 1):
            row = []
            for x in range(x0, x1 + 1):
                if (x, y) == mark:
                    row.append("@")
                elif (x, y) in self.tiles:
                    row.append("#" if self.tiles[(x, y)]["type"] == "wall" else ".")
                else:
                    row.append(" ")
            lines.append(f"y={y:>3}: " + "".join(row))
        return "\n".join(lines)
