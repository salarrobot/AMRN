
import numpy as np


class IntelligentAPF:
    def __init__(self, k_att=1.0, k_rep=0.5, k_tan=0.4,
                 influence=0.9, goal_relax=0.9, stuck_window=40,
                 rep_cap=2.2):
        self.k_att = k_att
        self.k_rep = k_rep
        self.k_tan = k_tan
        self.influence = influence
        self.goal_relax = goal_relax
        self.stuck_window = stuck_window
        self.rep_cap = rep_cap
        self._history = []
        self._escape_boost = 0.0

    def _update_stuck(self, pos):
        self._history.append(np.array(pos))
        if len(self._history) > self.stuck_window:
            self._history.pop(0)
            travelled = np.linalg.norm(self._history[-1] - self._history[0])
            if travelled < 0.15:
                self._escape_boost = 1.0
        self._escape_boost *= 0.97

    def force(self, pos, target, goal, obstacle_points):
        """Total APF force at pos toward target, repelled from scan points."""
        pos = np.asarray(pos, dtype=float)
        self._update_stuck(pos)

        to_t = np.asarray(target, dtype=float) - pos
        d_t = np.linalg.norm(to_t)
        f_att = self.k_att * to_t / d_t if d_t > 1e-9 else np.zeros(2)

        f_rep = np.zeros(2)
        f_tan = np.zeros(2)
        n_pts = 0
        d_goal = np.linalg.norm(np.asarray(goal, dtype=float) - pos)
        gnron = min(1.0, d_goal ** self.goal_relax)

        for q in obstacle_points:
            diff = pos - q
            d = np.linalg.norm(diff)
            if d < 1e-9 or d > self.influence:
                continue
            n = diff / d
            mag = self.k_rep * (1.0 / d - 1.0 / self.influence) / d ** 2
            f_rep += mag * n
            t = np.array([-n[1], n[0]])
            if np.dot(t, f_att) < 0:
                t = -t
            f_tan += (self.k_tan + self._escape_boost) * mag * t
            n_pts += 1

        if n_pts:
            rep = gnron * (f_rep + f_tan) / n_pts
            m = np.linalg.norm(rep)
            if m > self.rep_cap:
                rep *= self.rep_cap / m
        else:
            rep = np.zeros(2)

        total = f_att + rep
        norm = np.linalg.norm(total)
        if norm < 0.05 and d_t > 0.2:
            t = np.array([-to_t[1], to_t[0]]) / d_t
            total = 0.5 * t + 0.2 * f_att
        return total

    def danger_level(self, obstacle_points, pos):
        """0 (clear) .. 1 (imminent) from the nearest sensed obstacle."""
        if len(obstacle_points) == 0:
            return 0.0
        d = np.min(np.linalg.norm(obstacle_points - np.asarray(pos), axis=1))
        return float(np.clip(1.0 - d / self.influence, 0.0, 1.0))
