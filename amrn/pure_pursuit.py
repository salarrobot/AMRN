
import numpy as np


def resample_path(path, ds=0.1):
    """Densify a sparse waypoint polyline to ~ds spacing."""
    pts = [np.asarray(path[0], dtype=float)]
    for a, b in zip(path[:-1], path[1:]):
        seg = np.asarray(b, dtype=float) - a
        n = max(int(np.ceil(np.linalg.norm(seg) / ds)), 1)
        for k in range(1, n + 1):
            pts.append(a + seg * (k / n))
    return np.array(pts)


class PurePursuit:
    MAX_STEER = 0.85

    def __init__(self, wheelbase=0.26, lookahead=0.55, k_speed=0.35,
                 lookahead_min=0.4, lookahead_max=1.1):
        self.L = wheelbase
        self.ld0 = lookahead
        self.k_speed = k_speed
        self.ld_min = lookahead_min
        self.ld_max = lookahead_max
        self._idx = 0

    def reset(self):
        self._idx = 0

    @property
    def idx(self):
        return self._idx

    def deviation(self, path, pos):
        """Cross-track distance from the robot to the local path window."""
        lo = max(self._idx - 5, 0)
        w = path[lo:self._idx + 30]
        return float(np.min(np.linalg.norm(w - np.asarray(pos, dtype=float),
                                           axis=1)))

    def lookahead_dist(self, v):
        return float(np.clip(self.ld0 + self.k_speed * v,
                             self.ld_min, self.ld_max))

    def target_point(self, path, pos, v):
        """Lookahead point on the path with monotonic progress tracking."""
        pos = np.asarray(pos, dtype=float)
        ld = self.lookahead_dist(v)
        window = path[self._idx:self._idx + 60]
        d = np.linalg.norm(window - pos, axis=1)
        self._idx += int(np.argmin(d))
        for i in range(self._idx, len(path)):
            if np.linalg.norm(path[i] - pos) >= ld:
                return path[i], i
        return path[-1], len(path) - 1

    def steering(self, pose, target, v):
        """Steering angle and bearing error toward target from pose."""
        x, y, th = pose
        dx, dy = target[0] - x, target[1] - y
        alpha = np.arctan2(dy, dx) - th
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        ld = max(self.lookahead_dist(v), np.hypot(dx, dy), 1e-6)
        delta = np.arctan2(2.0 * self.L * np.sin(alpha), ld)
        return float(np.clip(delta, -self.MAX_STEER, self.MAX_STEER)), alpha
