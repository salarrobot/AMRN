
import argparse
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from amrn.world import World, Circle, Rect, ConvexPolygon
from amrn.theta_star import plan_path, plan_on_grid
from amrn.lidar import Lidar
from amrn.slam import OccupancyMap
from amrn.apf import IntelligentAPF
from amrn.pure_pursuit import PurePursuit, resample_path
from amrn.robot import CarRobot

SCENARIOS = {
    1: {"start": (0.6, 0.6), "goal": (9.4, 9), "name": "bottom-left -> top-right"},
    2: {"start": (0.6, 9.4), "goal": (9.4, 0.6), "name": "top-left -> bottom-right"},
    3: {"start": (0.6, 6.5), "goal": (9.4, 3.2), "name": "mid-left -> mid-right"},
    4: {"start": (8.0, 0.5), "goal": (2.0, 9.4), "name": "bottom-right -> top-left"},
}

DT = 0.05
STEPS_PER_FRAME = 2
GOAL_TOL = 0.3
MAX_TIME = 120.0
FRONT_SECTOR = np.pi / 5
REVERSE_STEPS = 20
PLAN_RES = 0.1
PLAN_INFLATE = 0.3
REPLAN_COOLDOWN = 2.0
HIST_SECTORS = 36


class Simulation:
    def __init__(self, scenario, surprise=False):
        sc = SCENARIOS[scenario]
        self.scenario = scenario
        self.world = World(size=10.0)
        self.start = np.array(sc["start"])
        self.goal = np.array(sc["goal"])
        self.name = sc["name"]

        print(f"[AMRN] scenario {scenario}: {self.name}"
              + (" (+surprise)" if surprise else ""))
        print("[AMRN] planning global path with Theta* ...")
        self.path = plan_path(self.world, self.start, self.goal,
                              PLAN_RES, PLAN_INFLATE)
        if self.path is None:
            raise RuntimeError("Theta* found no path — goal unreachable")
        seg = np.diff(self.path, axis=0)
        self.path_len = float(np.sum(np.linalg.norm(seg, axis=1)))
        print(f"[AMRN] path: {len(self.path)} waypoints, "
              f"length {self.path_len:.2f} m")
        self.track = resample_path(self.path, ds=0.1)

        heading = np.arctan2(*(self.path[1] - self.start)[::-1])
        self.robot = CarRobot(*self.start, heading)
        self.lidar = Lidar(n_beams=72, max_range=3.5)
        self.slam = OccupancyMap(size=10.0, resolution=PLAN_RES)
        self.apf = IntelligentAPF()
        self.pp = PurePursuit(wheelbase=self.robot.L)

        self.t = 0.0
        self.done = False
        self.reached = False
        self.trail = [self.robot.pos.copy()]
        self.min_clearance = np.inf
        self.collisions = 0
        self.reverse_ctr = 0
        self.replans = 0
        self._recoveries = 0
        self._last_replan_t = -REPLAN_COOLDOWN
        self._path_blocked = False
        self._steps = 0
        self.scan_pts = np.empty((0, 2))
        self.ranges = np.full(self.lidar.n_beams, self.lidar.max_range)
        self.force = np.zeros(2)
        self.target = self.goal.copy()

        if surprise:
            self._place_surprise()

    def _place_surprise(self):
        n = len(self.track)
        for frac in (0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65):
            c = self.track[int(frac * n)]
            if (min(o.distance(c) for o in self.world.obstacles) > 0.9
                    and np.linalg.norm(c - self.start) > 1.3
                    and np.linalg.norm(c - self.goal) > 1.3
                    and min(c.min(), self.world.size - c.max()) > 0.6):
                self.world.add_surprise(Circle(c[0], c[1], 0.35))
                print(f"[AMRN] surprise obstacle at ({c[0]:.1f}, {c[1]:.1f})"
                      " — unknown to the planner, LiDAR must find it")
                return
        print("[AMRN] no safe spot for a surprise obstacle; skipping")

    def _check_path_blocked(self):
        sensed = self.slam.planning_grid(p_occ=0.65, inflate_cells=1)
        for p in self.track[self.pp.idx:self.pp.idx + 30:3]:
            i, j = int(p[0] / PLAN_RES), int(p[1] / PLAN_RES)
            if (0 <= i < sensed.shape[0] and 0 <= j < sensed.shape[1]
                    and sensed[i, j]):
                self._path_blocked = True
                return

    def replan(self):
        self._last_replan_t = self.t
        self._path_blocked = False
        self._recoveries = 0
        known = self.world.occupancy_grid(PLAN_RES, inflate=PLAN_INFLATE)
        sensed = self.slam.planning_grid(
            p_occ=0.65, inflate_cells=int(round(PLAN_INFLATE / self.slam.res)))
        new = plan_on_grid(known | sensed, PLAN_RES, self.robot.pos, self.goal)
        if new is None:
            print(f"[AMRN] t={self.t:.1f}s replan failed — keeping old path")
            return
        self.replans += 1
        self.path = new
        self.track = resample_path(new, ds=0.1)
        self.pp.reset()
        print(f"[AMRN] t={self.t:5.1f}s replanned (#{self.replans}): "
              f"{len(new)} waypoints")

    def step(self):
        if self.done:
            return
        robot = self.robot
        self._steps += 1

        self.ranges, hits = self.lidar.scan(self.world, robot.pose)
        self.slam.update(robot.pose, self.lidar.angles, self.ranges, hits,
                         self.lidar.max_range)
        self.scan_pts = self.lidar.points(robot.pose, self.ranges, hits)
        front = np.abs(self.lidar.angles) < FRONT_SECTOR
        rear = np.abs(np.abs(self.lidar.angles) - np.pi) < FRONT_SECTOR
        front_min = float(np.min(self.ranges[front]))
        rear_min = float(np.min(self.ranges[rear]))

        if self._steps % 10 == 0:
            self._check_path_blocked()
        if (self.t - self._last_replan_t) > REPLAN_COOLDOWN and (
                self._path_blocked
                or self.pp.deviation(self.track, robot.pos) > 1.0
                or self._recoveries >= 2):
            self.replan()

        target, _ = self.pp.target_point(self.track, robot.pos, robot.v)
        self.target = target

        self.force = self.apf.force(robot.pos, target, self.goal, self.scan_pts)
        f_norm = np.linalg.norm(self.force)
        ld = self.pp.lookahead_dist(robot.v)
        virtual = robot.pos + (self.force / f_norm * ld if f_norm > 1e-9
                               else target - robot.pos)

        delta, alpha = self.pp.steering(robot.pose, virtual, robot.v)
        d_goal = np.linalg.norm(self.goal - robot.pos)

        if self.reverse_ctr == 0 and d_goal > GOAL_TOL and (
                front_min < 0.30 or (abs(alpha) > 2.0 and front_min < 0.5)):
            self.reverse_ctr = REVERSE_STEPS
            self._recoveries += 1
        if self.reverse_ctr > 0 and rear_min < 0.25:
            self.reverse_ctr = 0

        if self.reverse_ctr > 0:
            self.reverse_ctr -= 1
            robot.step(-0.3, -np.sign(alpha) * PurePursuit.MAX_STEER, DT)
        else:
            turn_slow = 1.0 - 0.55 * abs(delta) / PurePursuit.MAX_STEER
            prox = np.clip((front_min - 0.25) / 0.9, 0.0, 1.0)
            v_cmd = robot.v_max * turn_slow * (0.25 + 0.75 * prox) \
                * min(1.0, d_goal / 0.7)
            robot.step(max(v_cmd, 0.08), delta, DT)

        d = self.world.min_distance(robot.pos)
        if d < CarRobot.RADIUS:
            eps = 0.05
            gx = (self.world.min_distance((robot.x + eps, robot.y))
                  - self.world.min_distance((robot.x - eps, robot.y))) / (2 * eps)
            gy = (self.world.min_distance((robot.x, robot.y + eps))
                  - self.world.min_distance((robot.x, robot.y - eps))) / (2 * eps)
            g = np.hypot(gx, gy)
            if g > 1e-6:
                push = CarRobot.RADIUS - d + 0.01
                robot.x += push * gx / g
                robot.y += push * gy / g
            robot.v = 0.0
            self.collisions += 1
            self.reverse_ctr = max(self.reverse_ctr, REVERSE_STEPS // 2)

        self.t += DT
        self.trail.append(robot.pos.copy())
        self.min_clearance = min(self.min_clearance,
                                 self.world.min_distance(robot.pos))

        if d_goal < GOAL_TOL or self.t > MAX_TIME:
            self.done = True
            self.reached = d_goal < GOAL_TOL
            travelled = np.sum(np.linalg.norm(np.diff(self.trail, axis=0),
                                              axis=1))
            status = "GOAL REACHED" if self.reached else "TIMEOUT"
            print(f"[AMRN] {status} at t={self.t:.1f}s | travelled "
                  f"{travelled:.2f} m (path {self.path_len:.2f} m) | "
                  f"min clearance {self.min_clearance:.2f} m | "
                  f"contacts {self.collisions} | replans {self.replans}")

    def travelled(self):
        return float(np.sum(np.linalg.norm(np.diff(self.trail, axis=0),
                                           axis=1)))

    def polar_histogram(self, n_sectors=HIST_SECTORS):
        """Obstacle closeness per bearing sector (0 = clear, 1 = contact)."""
        closeness = np.clip(1.0 - self.ranges / self.lidar.max_range,
                            0.0, 1.0)
        return closeness.reshape(n_sectors, -1).max(axis=1)


INK = "#111111"
GRAY_FILL = "#dcdcdc"
EDGE = "#1a1a1a"
DETAIL = "#9a9a9a"
LABEL_C = "#333333"
MUTED = "#666666"
HIST_BG = "#f2f9f1"
HIST_GRID = "#bcd4bc"
HIST_HEAD = "#1b7f37"
HIST_PATH = "#1565c0"
HIST_CMAP = plt.get_cmap("RdYlGn_r")


def _roof_details(ax, ob):
    cx, cy = ob.center
    if isinstance(ob, Rect):
        sx, sy = ob.hw * 0.74, ob.hh * 0.74
        ax.add_patch(mpatches.Rectangle((cx - sx, cy - sy), 2 * sx, 2 * sy,
                                        fill=False, edgecolor=DETAIL,
                                        lw=0.6, zorder=2.1))
        ax.add_patch(mpatches.Rectangle(
            (cx + 0.40 * ob.hw - 0.05, cy + 0.38 * ob.hh - 0.05), 0.10, 0.10,
            facecolor="#c4c4c4", edgecolor=EDGE, lw=0.4, zorder=2.15))
    elif isinstance(ob, Circle):
        ax.add_patch(mpatches.Circle((cx, cy), ob.r * 0.62, fill=False,
                                     edgecolor=DETAIL, lw=0.6, zorder=2.1))
        ax.add_patch(mpatches.Circle((cx, cy), ob.r * 0.10,
                                     facecolor="#c4c4c4", edgecolor=EDGE,
                                     lw=0.4, zorder=2.15))
    elif isinstance(ob, ConvexPolygon):
        for v in ob.v:
            ax.plot([cx, v[0]], [cy, v[1]], color=DETAIL, lw=0.6, zorder=2.1)


def draw_city_map(ax, world):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, lw=0.4, ls=":", color="#c9c9c9")
    ax.tick_params(labelsize=7, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(1.2)

    for ob in world.obstacles:
        if ob in world.landmarks:
            continue
        ax.add_patch(ob.patch(facecolor=GRAY_FILL, edgecolor=EDGE,
                              lw=1.1, zorder=2))
        _roof_details(ax, ob)
        if ob.name:
            cx, cy = ob.center
            bbox = (dict(fc=GRAY_FILL, ec="none", pad=0.5)
                    if isinstance(ob, ConvexPolygon) else None)
            ax.text(cx, cy, ob.name, ha="center", va="center", fontsize=5.6,
                    weight="bold", color=LABEL_C, zorder=2.5, bbox=bbox)

    for lm in world.landmarks:
        cx, cy = lm.center
        if lm.name == "Fountain":
            ax.add_patch(mpatches.Circle((cx, cy), lm.r, facecolor="white",
                                         edgecolor=EDGE, lw=1.1, zorder=2))
            ax.add_patch(mpatches.Circle((cx, cy), lm.r * 0.60, fill=False,
                                         edgecolor=MUTED, lw=0.7, zorder=2.2))
            ax.plot(cx, cy, "o", ms=2.2, color=INK, zorder=2.4)
        else:
            ax.add_patch(mpatches.Circle((cx, cy), lm.r, facecolor="white",
                                         edgecolor=EDGE, lw=1.1, zorder=2))
            ax.plot(cx, cy, "*", ms=7, color=INK, zorder=2.4)
        ax.text(cx, cy - lm.r - 0.13, lm.name, ha="center", va="top",
                fontsize=5.6, style="italic", color=MUTED, zorder=2.5)

    ax.annotate("", xy=(0.615, 0.982), xytext=(0.615, 0.912),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5), zorder=9)
    ax.text(0.595, 0.945, "N", transform=ax.transAxes, fontsize=8,
            weight="bold", color=INK, ha="right", va="center", zorder=9)
    ax.plot([0.35, 1.35], [0.30, 0.30], color=INK, lw=2.0,
            solid_capstyle="butt", zorder=9)
    ax.text(0.85, 0.38, "1 m", ha="center", fontsize=6.5, color=INK, zorder=9)


def make_figure(sim):
    fig = plt.figure(figsize=(17.0, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 1.08, 0.84],
                          left=0.045, right=0.985, top=0.86, bottom=0.16,
                          wspace=0.24)
    ax_w = fig.add_subplot(gs[0])
    ax_m = fig.add_subplot(gs[1])
    ax_h = fig.add_subplot(gs[2], projection="polar")
    fig.suptitle("Advanced Mobile Robot Navigation — "
                 "Theta* + Intelligent APF + Pure Pursuit + LiDAR SLAM",
                 fontsize=12.5, color=INK, y=0.965)

    ax_w.set_title(f"City map — scenario {sim.scenario}: {sim.name}",
                   fontsize=10, color=INK, weight="bold")
    draw_city_map(ax_w, sim.world)
    for i, ob in enumerate(sim.world.surprises):
        ax_w.add_patch(ob.patch(facecolor="white", edgecolor=EDGE,
                                hatch="////", lw=1.2, zorder=2,
                                label="unknown obstacle" if i == 0 else None))
    path_line, = ax_w.plot(sim.path[:, 0], sim.path[:, 1],
                           ls=(0, (5, 3)), color=INK, lw=1.5,
                           label="Theta* path", zorder=3)
    ax_w.plot(*sim.start, "o", color=INK, ms=8, label="start", zorder=5)
    ax_w.add_patch(mpatches.Circle(sim.goal, GOAL_TOL, fill=False,
                                   edgecolor=INK, ls=":", lw=1.0,
                                   zorder=4.5))
    ax_w.plot(*sim.goal, "*", mfc="white", mec=INK, mew=1.3, ms=16,
              label="goal", zorder=5)

    beams = LineCollection([], colors="#c8c8c8", lw=0.5, alpha=0.5, zorder=1)
    ax_w.add_collection(beams)
    hitpts, = ax_w.plot([], [], ".", color="#404040", ms=2.4, alpha=0.85,
                        zorder=4)
    trail, = ax_w.plot([], [], "-", color="#555555", lw=1.9,
                       label="trajectory", zorder=4)
    body = Polygon(np.zeros((4, 2)), closed=True, facecolor="white",
                   edgecolor=INK, lw=1.2, zorder=6)
    ax_w.add_patch(body)
    wheels = [Polygon(np.zeros((4, 2)), closed=True, facecolor=INK, zorder=7)
              for _ in range(4)]
    for w in wheels:
        ax_w.add_patch(w)
    force_arrow = ax_w.annotate("", xy=(0, 0), xytext=(0, 0),
                                arrowprops=dict(arrowstyle="->",
                                                color=INK, lw=1.4),
                                zorder=8)
    ax_w.set_xlim(0, 10)
    ax_w.set_ylim(0, 10)
    ax_w.set_aspect("equal")
    ax_w.legend(loc="upper center", bbox_to_anchor=(0.5, -0.055), ncol=5,
                fontsize=7.5, frameon=False, handlelength=1.8,
                columnspacing=1.1)

    ax_m.set_title("Live occupancy map (LiDAR SLAM)", fontsize=10,
                   color=INK, weight="bold")
    img = ax_m.imshow(sim.slam.image(), cmap="gray", vmin=0, vmax=1,
                      origin="lower", extent=[0, 10, 0, 10])
    map_path, = ax_m.plot(sim.path[:, 0], sim.path[:, 1], ls=(0, (5, 3)),
                          color=INK, lw=1.1, alpha=0.9)
    map_trail, = ax_m.plot([], [], "-", color="#777777", lw=1.4)
    map_bot, = ax_m.plot([], [], "o", mfc="white", mec=INK, mew=1.1, ms=6)
    ax_m.plot(*sim.goal, "*", mfc="white", mec=INK, mew=1.1, ms=13)
    for lm in sim.world.landmarks:
        ax_m.plot(*lm.center, marker="D", mfc="none", mec=INK, mew=0.9,
                  ms=4, zorder=5)
        ax_m.text(lm.center[0], lm.center[1] - 0.26, lm.name, ha="center",
                  va="top", fontsize=5.4, color="#bbbbbb", zorder=5)
    ax_m.tick_params(labelsize=7, colors=INK)
    for s in ax_m.spines.values():
        s.set_color(INK)
        s.set_linewidth(1.2)
    ax_m.set_xlim(0, 10)
    ax_m.set_ylim(0, 10)
    ax_m.set_aspect("equal")

    ax_h.set_title("Polar obstacle histogram (VFH)", fontsize=10,
                   color=INK, weight="bold", pad=16)
    ax_h.set_facecolor(HIST_BG)
    sector = sim.lidar.angles.reshape(HIST_SECTORS, -1).mean(axis=1)
    bars = ax_h.bar(sector, np.zeros(HIST_SECTORS),
                    width=2 * np.pi / HIST_SECTORS * 0.9,
                    color=HIST_CMAP(0.0), edgecolor="#37503a", lw=0.4,
                    zorder=2)
    ax_h.set_theta_zero_location("N")
    ax_h.set_theta_direction(1)
    ax_h.set_ylim(0, 1)
    ax_h.set_rgrids([0.25, 0.5, 0.75, 1.0], labels=["", "0.5", "", "1.0"],
                    fontsize=6, color=MUTED)
    ax_h.set_thetagrids(np.arange(0, 360, 45),
                        labels=["0°", "45°", "90°", "135°", "±180°",
                                "-135°", "-90°", "-45°"], fontsize=6.5)
    ax_h.grid(color=HIST_GRID, lw=0.6, ls=":")
    sel_line, = ax_h.plot([0, 0], [0, 0.98], color=HIST_HEAD, lw=2.0,
                          zorder=4, label="APF heading")
    tgt_line, = ax_h.plot([0, 0], [0, 0.98], color=HIST_PATH, lw=1.3,
                          ls="--", zorder=3, label="path bearing")
    ax_h.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2,
                fontsize=7.5, frameon=False)

    hud = fig.text(0.5, 0.045, "", ha="center", va="center", fontsize=9.5,
                   family="monospace", color=INK)

    artists = dict(beams=beams, hitpts=hitpts, trail=trail, body=body,
                   wheels=wheels, force=force_arrow, hud=hud, img=img,
                   path=path_line, map_path=map_path,
                   map_trail=map_trail, map_bot=map_bot,
                   bars=bars, sel=sel_line, tgt=tgt_line)
    return fig, artists


def _rel_bearing(vec, theta):
    a = np.arctan2(vec[1], vec[0]) - theta
    return np.arctan2(np.sin(a), np.cos(a))


def update_frame(_, sim, art, steps):
    for _ in range(steps):
        sim.step()

    x, y, th = sim.robot.pose
    segs = []
    for a, r in zip(sim.lidar.angles, sim.ranges):
        segs.append([(x, y), (x + r * np.cos(th + a), y + r * np.sin(th + a))])
    art["beams"].set_segments(segs)
    if len(sim.scan_pts):
        art["hitpts"].set_data(sim.scan_pts[:, 0], sim.scan_pts[:, 1])

    art["path"].set_data(sim.path[:, 0], sim.path[:, 1])
    art["map_path"].set_data(sim.path[:, 0], sim.path[:, 1])

    tr = np.array(sim.trail)
    art["trail"].set_data(tr[:, 0], tr[:, 1])
    b, ws = sim.robot.outline()
    art["body"].set_xy(b)
    for poly, w in zip(art["wheels"], ws):
        poly.set_xy(w)

    f = sim.force
    fn = np.linalg.norm(f)
    tip = sim.robot.pos + (f / fn * 0.9 if fn > 1e-9 else 0)
    art["force"].xy = tuple(tip)
    art["force"].xyann = (x, y)

    art["img"].set_data(sim.slam.image())
    art["map_trail"].set_data(tr[:, 0], tr[:, 1])
    art["map_bot"].set_data([x], [y])

    for bar, h in zip(art["bars"], sim.polar_histogram()):
        bar.set_height(h)
        bar.set_facecolor(HIST_CMAP(float(h)))
    if fn > 1e-9:
        fa = _rel_bearing(f, th)
        art["sel"].set_data([fa, fa], [0, 0.98])
    ta = _rel_bearing(sim.target - sim.robot.pos, th)
    art["tgt"].set_data([ta, ta], [0, 0.98])

    danger = sim.apf.danger_level(sim.scan_pts, sim.robot.pos)
    status = ("GOAL REACHED" if sim.done and sim.reached
              else "TIMEOUT" if sim.done
              else "REVERSING" if sim.reverse_ctr > 0 else "DRIVING")
    art["hud"].set_text(
        f"t = {sim.t:5.1f} s    v = {sim.robot.v:4.2f} m/s    "
        f"danger = {danger:4.2f}    "
        f"explored = {sim.slam.explored_fraction()*100:3.0f} %    "
        f"replans = {sim.replans}    [{status}]")
    return list(art.values())


def _animate(sim, args, steps, interval):
    fig, art = make_figure(sim)

    def gen():
        i = 0
        while not sim.done and i < args.frames:
            yield i
            i += 1
        yield i

    return fig, animation.FuncAnimation(
        fig, update_frame, frames=gen, fargs=(sim, art, steps),
        interval=interval, blit=False, repeat=False, cache_frame_data=False)


def render(scenario, args):
    os.makedirs("media", exist_ok=True)
    suffix = "_surprise" if args.surprise else ""
    base = f"media/scenario{scenario}{suffix}"
    targets = [(base + ".gif", animation.PillowWriter(fps=args.fps))]
    if shutil.which("ffmpeg"):
        targets.append((base + ".mp4",
                        animation.FFMpegWriter(fps=args.fps, bitrate=1800)))
    for path, writer in targets:
        sim = Simulation(scenario, surprise=args.surprise)
        fig, ani = _animate(sim, args, steps=5, interval=1000 // args.fps)
        print(f"[AMRN] rendering {path} ...")
        ani.save(path, writer=writer, dpi=args.dpi)
        plt.close(fig)
        print(f"[AMRN] saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description="AMRN simulation")
    ap.add_argument("--scenario", type=int, default=1, choices=SCENARIOS)
    ap.add_argument("--surprise", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--dpi", type=int, default=70)
    ap.add_argument("--frames", type=int, default=1500)
    args = ap.parse_args()

    if args.save:
        for sc in (list(SCENARIOS) if args.all else [args.scenario]):
            render(sc, args)
    else:
        sim = Simulation(args.scenario, surprise=args.surprise)
        _fig, _ani = _animate(sim, args, steps=STEPS_PER_FRAME, interval=40)
        plt.show()


if __name__ == "__main__":
    main()
