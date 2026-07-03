
import numpy as np

L_OCC = 0.9
L_FREE = -0.4
L_CLAMP = 6.0


def _dilate(mask, radius):
    out = mask.copy()
    n, m = mask.shape
    r2 = radius * radius
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx * dx + dy * dy > r2 or (dx == 0 and dy == 0):
                continue
            src = mask[max(0, -dx):min(n, n - dx), max(0, -dy):min(m, m - dy)]
            out[max(0, dx):min(n, n + dx), max(0, dy):min(m, m + dy)] |= src
    return out


class OccupancyMap:
    def __init__(self, size=10.0, resolution=0.1):
        self.size = size
        self.res = resolution
        self.n = int(round(size / resolution))
        self.logodds = np.zeros((self.n, self.n))

    def _cell(self, p):
        i = int(p[0] / self.res)
        j = int(p[1] / self.res)
        return min(max(i, 0), self.n - 1), min(max(j, 0), self.n - 1)

    def _bresenham(self, a, b):
        x0, y0 = a
        x1, y1 = b
        cells = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            cells.append((x, y))
            if (x, y) == (x1, y1):
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return cells

    def update(self, pose, angles, ranges, hits, max_range):
        x, y, th = pose
        origin = self._cell((x, y))
        for a, r, hit in zip(angles, ranges, hits):
            end = (x + r * np.cos(th + a), y + r * np.sin(th + a))
            cells = self._bresenham(origin, self._cell(end))
            for c in cells[:-1]:
                self.logodds[c] = max(self.logodds[c] + L_FREE, -L_CLAMP)
            c = cells[-1]
            if hit and r < max_range - 1e-6:
                self.logodds[c] = min(self.logodds[c] + L_OCC, L_CLAMP)
            else:
                self.logodds[c] = max(self.logodds[c] + L_FREE, -L_CLAMP)

    def probability(self):
        """Occupancy probability grid in [0,1]; 0.5 = unknown."""
        return 1.0 - 1.0 / (1.0 + np.exp(self.logodds))

    def explored_fraction(self):
        """Fraction of cells observed at least once."""
        return float((np.abs(self.logodds) > 0.1).mean())

    def planning_grid(self, p_occ=0.65, inflate_cells=3):
        """Occupied grid from the live map, dilated for planning."""
        occ = self.probability() > p_occ
        return _dilate(occ, inflate_cells) if inflate_cells > 0 else occ

    def image(self):
        """Grayscale image for imshow: white free, gray unknown, black occupied."""
        return (1.0 - self.probability()).T
