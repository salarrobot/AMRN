
import numpy as np
import matplotlib.patches as mpatches


class Obstacle:
    name = ""

    def distance(self, p):
        p = np.asarray(p, dtype=float)
        return float(self.distance_grid(p[0], p[1]))

    def distance_grid(self, X, Y):
        raise NotImplementedError

    def contains(self, p):
        return self.distance(p) <= 0.0

    @property
    def center(self):
        raise NotImplementedError

    def patch(self, offset=(0.0, 0.0), **kw):
        raise NotImplementedError


class Circle(Obstacle):
    def __init__(self, cx, cy, r, name=""):
        self.c = np.array([cx, cy], dtype=float)
        self.r = float(r)
        self.name = name

    def distance_grid(self, X, Y):
        return np.hypot(X - self.c[0], Y - self.c[1]) - self.r

    @property
    def center(self):
        return self.c

    def patch(self, offset=(0.0, 0.0), **kw):
        return mpatches.Circle(self.c + np.asarray(offset), self.r, **kw)


class Rect(Obstacle):
    def __init__(self, cx, cy, w, h, name=""):
        self.c = np.array([cx, cy], dtype=float)
        self.hw = float(w) / 2.0
        self.hh = float(h) / 2.0
        self.name = name

    def distance_grid(self, X, Y):
        dx = np.abs(X - self.c[0]) - self.hw
        dy = np.abs(Y - self.c[1]) - self.hh
        outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
        inside = np.minimum(np.maximum(dx, dy), 0.0)
        return outside + inside

    @property
    def center(self):
        return self.c

    def patch(self, offset=(0.0, 0.0), **kw):
        return mpatches.Rectangle(self.c - (self.hw, self.hh)
                                  + np.asarray(offset),
                                  2 * self.hw, 2 * self.hh, **kw)


class Square(Rect):
    def __init__(self, cx, cy, side, name=""):
        super().__init__(cx, cy, side, side, name)


class ConvexPolygon(Obstacle):
    def __init__(self, *vertices, name=""):
        self.v = np.array(vertices, dtype=float)
        self.name = name
        area = 0.0
        for i in range(len(self.v)):
            a, b = self.v[i], self.v[(i + 1) % len(self.v)]
            area += a[0] * b[1] - b[0] * a[1]
        if area < 0:
            self.v = self.v[::-1]

    @staticmethod
    def regular(cx, cy, r, n, rot=0.0, name=""):
        ang = rot + np.arange(n) * 2.0 * np.pi / n
        verts = np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)])
        return ConvexPolygon(*verts, name=name)

    def distance_grid(self, X, Y):
        d2_min = np.full(np.shape(X), np.inf)
        inside = np.ones(np.shape(X), dtype=bool)
        n = len(self.v)
        for i in range(n):
            a, b = self.v[i], self.v[(i + 1) % n]
            ex, ey = b - a
            wx, wy = X - a[0], Y - a[1]
            t = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
            dx, dy = wx - t * ex, wy - t * ey
            d2_min = np.minimum(d2_min, dx * dx + dy * dy)
            inside &= (ex * wy - ey * wx) >= 0
        d = np.sqrt(d2_min)
        return np.where(inside, -d, d)

    @property
    def center(self):
        return self.v.mean(axis=0)

    def patch(self, offset=(0.0, 0.0), **kw):
        return mpatches.Polygon(self.v + np.asarray(offset), closed=True, **kw)


class Triangle(ConvexPolygon):
    def __init__(self, v0, v1, v2, name=""):
        super().__init__(v0, v1, v2, name=name)


class World:
    SDF_RES = 0.05

    def __init__(self, size=10.0):
        self.size = float(size)
        self.obstacles = [
            Rect(2.6, 6.6, 1.4, 0.8, name="LIBRARY"),
            Rect(2.2, 7.1, 0.6, 1.0),
            Rect(7.2, 3.0, 1.4, 0.6, name="DEPOT"),
            Rect(7.2, 2.3, 0.6, 0.8),
            Rect(5.0, 5.0, 1.2, 1.0, name="CITY HALL"),
            Circle(2.7, 2.8, 0.75, name="ROTUNDA"),
            Circle(1.4, 4.7, 0.5, name="SILO"),
            ConvexPolygon.regular(7.6, 7.2, 0.78, 6, rot=np.pi / 6,
                                  name="ARENA"),
            ConvexPolygon.regular(5.5, 8.2, 0.70, 5, rot=np.pi / 2,
                                  name="MUSEUM"),
            Triangle((5.0, 0.9), (6.4, 2.1), (4.2, 2.3), name="PAVILION"),
            ConvexPolygon((7.85, 4.60), (8.95, 5.05), (8.90, 5.55),
                          (7.80, 5.10), name="GALLERY"),
        ]
        self.landmarks = [
            Circle(0.85, 3.30, 0.28, name="Fountain"),
            Circle(8.70, 8.85, 0.20, name="Statue"),
        ]
        self.obstacles += self.landmarks
        self.surprises = []
        self._rebuild_sdf()

    def _rebuild_sdf(self):
        self._sdf = self.distance_field(self.SDF_RES, include_surprises=True)

    def add_surprise(self, obstacle):
        self.surprises.append(obstacle)
        self._rebuild_sdf()

    def all_obstacles(self):
        return self.obstacles + self.surprises

    def in_bounds(self, p):
        return 0.0 <= p[0] <= self.size and 0.0 <= p[1] <= self.size

    def min_distance(self, p):
        d = min(o.distance(p) for o in self.all_obstacles())
        wall = min(p[0], p[1], self.size - p[0], self.size - p[1])
        return min(d, wall)

    def is_free(self, p, margin=0.0):
        return self.in_bounds(p) and self.min_distance(p) > margin

    def distance_field(self, resolution, include_surprises=False):
        obs = self.all_obstacles() if include_surprises else self.obstacles
        n = int(round(self.size / resolution))
        xs = (np.arange(n) + 0.5) * resolution
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        d = np.minimum.reduce([o.distance_grid(X, Y) for o in obs])
        wall = np.minimum.reduce([X, Y, self.size - X, self.size - Y])
        return np.minimum(d, wall)

    def occupancy_grid(self, resolution, inflate=0.0):
        return self.distance_field(resolution) <= inflate

    def _sdf_at(self, x, y):
        n = self._sdf.shape[0]
        i = int(x / self.SDF_RES)
        j = int(y / self.SDF_RES)
        if i < 0 or j < 0 or i >= n or j >= n:
            return 0.0
        return self._sdf[i, j]

    def raycast(self, origin, angle, max_range, step=0.02):
        ox, oy = origin
        c, s = np.cos(angle), np.sin(angle)
        r = step
        while r < max_range:
            d = self._sdf_at(ox + r * c, oy + r * s)
            if d <= 0.0:
                return r, True
            r += max(step, d * 0.9)
        return max_range, False
