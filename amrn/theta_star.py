
import heapq
import numpy as np

SQRT2 = np.sqrt(2.0)
NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
             (0, 1), (1, -1), (1, 0), (1, 1)]


def line_of_sight(grid, a, b):
    x0, y0 = a
    x1, y1 = b
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        if grid[x, y]:
            return False
        if (x, y) == (x1, y1):
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def theta_star(grid, start, goal):
    n, m = grid.shape
    if grid[start] or grid[goal]:
        return None

    def h(c):
        return np.hypot(c[0] - goal[0], c[1] - goal[1])

    g = {start: 0.0}
    parent = {start: start}
    open_heap = [(h(start), start)]
    closed = set()

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while parent[path[-1]] != path[-1]:
                path.append(parent[path[-1]])
            return path[::-1]
        if cur in closed:
            continue
        closed.add(cur)

        for dx, dy in NEIGHBORS:
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < n and 0 <= nb[1] < m):
                continue
            if grid[nb] or nb in closed:
                continue
            if dx and dy and grid[cur[0] + dx, cur[1]] and grid[cur[0], cur[1] + dy]:
                continue

            pa = parent[cur]
            if line_of_sight(grid, pa, nb):
                cand_g = g[pa] + np.hypot(nb[0] - pa[0], nb[1] - pa[1])
                cand_parent = pa
            else:
                cand_g = g[cur] + (SQRT2 if dx and dy else 1.0)
                cand_parent = cur

            if cand_g < g.get(nb, np.inf):
                g[nb] = cand_g
                parent[nb] = cand_parent
                heapq.heappush(open_heap, (cand_g + h(nb), nb))
    return None


def plan_on_grid(grid, resolution, start, goal):
    """Plan on a boolean occupancy grid. Returns (N,2) waypoints or None."""

    def to_cell(p):
        c = (int(p[0] / resolution), int(p[1] / resolution))
        return (min(max(c[0], 0), grid.shape[0] - 1),
                min(max(c[1], 0), grid.shape[1] - 1))

    def nearest_free(c):
        if not grid[c]:
            return c
        for radius in range(1, 15):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nb = (c[0] + dx, c[1] + dy)
                    if (0 <= nb[0] < grid.shape[0] and
                            0 <= nb[1] < grid.shape[1] and not grid[nb]):
                        return nb
        return c

    cells = theta_star(grid, nearest_free(to_cell(start)),
                       nearest_free(to_cell(goal)))
    if cells is None:
        return None
    pts = [(np.array(c) + 0.5) * resolution for c in cells]
    pts[0] = np.asarray(start, dtype=float)
    pts[-1] = np.asarray(goal, dtype=float)
    return np.array(pts)


def plan_path(world, start, goal, resolution=0.1, inflate=0.3):
    """Plan on the world's known-obstacle grid. (N,2) waypoints or None."""
    grid = world.occupancy_grid(resolution, inflate=inflate)
    return plan_on_grid(grid, resolution, start, goal)
