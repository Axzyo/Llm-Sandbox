import heapq

DIR_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def next_step(world, entity, goal: tuple[int, int], max_nodes: int = 4096) -> tuple | None:
    """First step of a path from the entity's column to the `goal` column, as
    (x, y, z): the column to step into and the height it lands at. Steps follow
    the same rule as a real move (world.landing with the entity's climb and
    height), so the path never promises a step the body cannot take. None if
    already there or unreachable."""
    start = (entity.x, entity.y, entity.z)
    climb, height = entity.properties["climb"], entity.properties["height"]
    if (entity.x, entity.y) == tuple(goal):
        return None
    open_heap = [(0, start)]
    came: dict = {}
    g_score = {start: 0}
    expanded = 0
    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if (cur[0], cur[1]) == tuple(goal):
            path = [cur]
            while path[-1] != start:
                path.append(came[path[-1]])
            path.reverse()
            return path[1]
        expanded += 1
        if expanded > max_nodes:
            return None
        cx, cy, cz = cur
        for dx, dy in DIR_STEPS:
            nx, ny = cx + dx, cy + dy
            nz = world.landing(nx, ny, cz, climb, height)
            if nz is None:
                continue
            nxt = (nx, ny, nz)
            tentative = g_score[cur] + 1
            if tentative < g_score.get(nxt, 1 << 30):
                g_score[nxt] = tentative
                came[nxt] = cur
                h = abs(nx - goal[0]) + abs(ny - goal[1])
                heapq.heappush(open_heap, (tentative + h, nxt))
    return None
