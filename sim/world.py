TILE_FLOOR = "."
TILE_WALL = "#"


def chebyshev(x0: int, y0: int, x1: int, y1: int) -> int:
    return max(abs(x1 - x0), abs(y1 - y0))


class World:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.tiles = [[TILE_FLOOR] * width for _ in range(height)]
        self.entities: dict = {}

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_wall(self, x: int, y: int) -> bool:
        return self.tiles[y][x] == TILE_WALL

    def blocked(self, x: int, y: int) -> bool:
        return not self.in_bounds(x, y) or self.is_wall(x, y)

    def entity_at(self, x: int, y: int):
        for e in self.entities.values():
            if e.x == x and e.y == y:
                return e
        return None

    def add_wall(self, x: int, y: int) -> None:
        self.tiles[y][x] = TILE_WALL


def has_los(world: World, x0: int, y0: int, x1: int, y1: int) -> bool:
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        if (x, y) != (x0, y0) and world.is_wall(x, y):
            return False
        if (x, y) == (x1, y1):
            return True
        e2 = 2 * err
        step_x = e2 > -dy
        step_y = e2 < dx
        if step_x and step_y and world.is_wall(x + sx, y) and world.is_wall(x, y + sy):
            return False
        if step_x:
            err -= dy
            x += sx
        if step_y:
            err += dx
            y += sy
