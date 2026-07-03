
import numpy as np


class CarRobot:
    LENGTH = 0.42
    WIDTH = 0.26
    WHEEL_LEN = 0.11
    WHEEL_W = 0.045
    RADIUS = 0.16

    def __init__(self, x, y, theta, wheelbase=0.26, v_max=0.8):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = 0.0
        self.delta = 0.0
        self.L = wheelbase
        self.v_max = v_max

    @property
    def pose(self):
        return (self.x, self.y, self.theta)

    @property
    def pos(self):
        return np.array([self.x, self.y])

    def step(self, v_cmd, delta_cmd, dt):
        self.v += np.clip(v_cmd - self.v, -3.0 * dt, 3.0 * dt)
        self.delta += np.clip(delta_cmd - self.delta, -4.0 * dt, 4.0 * dt)
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.v / self.L * np.tan(self.delta) * dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

    def _rect(self, cx, cy, length, width, angle):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        pts = np.array([[-length / 2, -width / 2], [length / 2, -width / 2],
                        [length / 2, width / 2], [-length / 2, width / 2]])
        return pts @ R.T + [cx, cy]

    def outline(self):
        """(body, wheels) polygon vertex arrays for drawing the car."""
        body = self._rect(self.x, self.y, self.LENGTH, self.WIDTH, self.theta)
        c, s = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[c, -s], [s, c]])
        half_l, half_w = self.L / 2, self.WIDTH / 2 - self.WHEEL_W / 2
        offsets = [(half_l, half_w), (half_l, -half_w),
                   (-half_l, half_w), (-half_l, -half_w)]
        wheels = []
        for i, (ox, oy) in enumerate(offsets):
            wx, wy = np.array([ox, oy]) @ R.T + [self.x, self.y]
            ang = self.theta + (self.delta if i < 2 else 0.0)
            wheels.append(self._rect(wx, wy, self.WHEEL_LEN, self.WHEEL_W, ang))
        return body, wheels
