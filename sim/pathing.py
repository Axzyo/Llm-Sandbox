import heapq

DIR_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def next_step(world, start: tuple[int, int], goal: tuple[int, int], max_nodes: int = 4096) -> tuple[int, int] | None:
    if start == goal:
        return None
    if world.blocked(*goal):
        return None
    open_heap = [(0, start)]
    came: dict = {}
    g_score = {start: 0}
    expanded = 0
    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while path[-1] != start:
                path.append(came[path[-1]])
            path.reverse()
            return path[1]
        expanded += 1
        if expanded > max_nodes:
            return None
        cx, cy = cur
        for dx, dy in DIR_STEPS:
            nxt = (cx + dx, cy + dy)
            if world.blocked(*nxt):
                continue
            tentative = g_score[cur] + 1
            if tentative < g_score.get(nxt, 1 << 30):
                g_score[nxt] = tentative
                came[nxt] = cur
                h = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                heapq.heappush(open_heap, (tentative + h, nxt))
    return None
