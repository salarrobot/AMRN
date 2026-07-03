
import numpy as np


class Lidar:
    def __init__(self, n_beams=72, max_range=3.5, noise_std=0.01, seed=0):
        self.n_beams = n_beams
        self.max_range = max_range
        self.noise_std = noise_std
        self.angles = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
        self.rng = np.random.default_rng(seed)

    def scan(self, world, pose):
        """Scan from pose (x, y, theta) -> (ranges, hits)."""
        x, y, th = pose
        ranges = np.empty(self.n_beams)
        hits = np.empty(self.n_beams, dtype=bool)
        for i, a in enumerate(self.angles):
            r, hit = world.raycast((x, y), th + a, self.max_range)
            if hit:
                r = min(max(r + self.rng.normal(0.0, self.noise_std), 0.05),
                        self.max_range)
            ranges[i] = r
            hits[i] = hit
        return ranges, hits

    def points(self, pose, ranges, hits=None):
        """Convert a scan to world-frame (x, y) hit points."""
        x, y, th = pose
        mask = hits if hits is not None else np.ones(self.n_beams, bool)
        a = self.angles[mask] + th
        r = ranges[mask]
        return np.column_stack([x + r * np.cos(a), y + r * np.sin(a)])
