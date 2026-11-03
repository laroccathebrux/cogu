"""What the brain knows about the world: from the true state (`TruePerception`) or only from the phone (`PhonePerception`).

Both answer the same questions, so the navigation and the commands (pet.py) don't know which one they run on:
- `robot()`  -> (x, y, yaw) of the pelvis;
- `person()` -> ((x, y) or None, seconds since last seen);
- `ball()`   -> ((x, y) or None, seconds since last seen);
- `floor()`  -> the FloorMap of what blocks the robot (without the person: the brain adds them where it thinks they are).

`PhonePerception` uses only what the iPhone will have (ADR-004, item 7):
- the pose from ARKit (`ArkitPose`: late, drifting, noisy); the robot's pelvis is the camera's pose moved by the phone
  mount (fixed, from the CAD). Everything below is in ARKit's frame, so it drifts with it, as on the phone;
- the map from the LiDAR (`Lidar`), 10 times a second, each frame with its own ARKit pose: a 4 cm log-odds grid.
  Two points in a cell between 5 cm and the robot's top (45 cm), closer than 3.5 m, are a hit; a floor point
  (under 3 cm) clears its cell. Cells never seen are passable but cost 3 times more (the reflex and the replanning
  deal with what turns up). The person is cut out of the depth with ARKit's person segmentation, and the robot's own
  body with its known shape;
- the person from that segmentation (ARKit `personSegmentationWithDepth` / Vision): the LiDAR points on the mask,
  their median on the floor. A frame misses them now and then (`miss`), and a few pixels are not a detection;
- the ball from an object detector (Core ML; here the camera's segmentation stands in for it), centred one radius
  behind the points it sees.
In the simulator the segmentation images come from MuJoCo's segmentation render of `phone_cam` (the same render the
LiDAR uses), which is what the detectors would give; their misses are drawn at random.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from navigation import RESOLUTION, ROBOT_TOP, FloorMap
from phone_sensors import ArkitPose, Lidar

FLOOR_Z = 0.03          # m: a LiDAR point lower than this is the floor
HIT_Z = 0.05            # m: higher than this, an obstacle (between the two: neither, the floor's noise)
HIT_RANGE = 3.5         # m: farther hits are left for when it gets closer (the far floor is noisy)
HIT, FREE = 0.9, -0.3   # log-odds per frame
L_MIN, L_MAX, L_OCC = -2.0, 4.0, 0.5
PERSON_DEPTH = 0.10     # m: from the near side of a person's legs to their middle


class TruePerception:
    """The true state: the static map from the model's geoms, the person and the ball where they are."""

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, person_body: str = "person"):
        self.m, self.d = m, d
        self.static = FloorMap.from_model(m, d)
        self.ball_id = m.body("ball").id
        self.person_mocap = m.body_mocapid[m.body(person_body).id]

    def update(self, t: float, dt: float):
        pass

    def close(self):
        pass

    def robot(self) -> tuple[float, float, float]:
        return true_pose(self.d)

    def person(self):
        return self.d.mocap_pos[self.person_mocap, :2].copy(), 0.0

    def ball(self):
        return self.d.xpos[self.ball_id, :2].copy(), 0.0

    def floor(self) -> FloorMap:
        return self.static


def true_pose(d: mujoco.MjData) -> tuple[float, float, float]:
    x, y = d.xpos[1, :2]
    fwd = d.xmat[1].reshape(3, 3)[:, 0]
    return float(x), float(y), math.atan2(fwd[1], fwd[0])


class Track:
    """The last place something was seen and how fast it was going (for a short guess where it went)."""

    def __init__(self):
        self.xy, self.vel, self.t = None, np.zeros(2), -math.inf

    def see(self, xy: np.ndarray, t: float):
        if self.xy is not None and 0.0 < t - self.t < 1.0:
            v = (xy - self.xy) / (t - self.t)
            self.vel = 0.7 * self.vel + 0.3 * v
        elif self.xy is None or t - self.t >= 1.0:
            self.vel = np.zeros(2)
        self.xy, self.t = xy, t

    def get(self, t: float):
        return (None, math.inf) if self.xy is None else (self.xy.copy(), t - self.t)


class PhonePerception:
    """Only the phone's sensors (see the module docstring). `update(t, dt)` every control step."""

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, seed: int = 0, rate: float = 0.1, miss: float = 0.1,
                 min_pixels: int = 60, extent: tuple[float, float, float, float] = (-4.0, 8.0, -6.0, 6.0),
                 res: float = RESOLUTION):
        self.m, self.d, self.rate, self.miss, self.min_pixels = m, d, rate, miss, min_pixels
        self.rng = np.random.default_rng(seed + 100)
        self.lidar = Lidar(m, seed=seed)
        self.arkit = ArkitPose(m, seed=seed)
        cam = m.camera("phone_cam").id
        assert m.cam_bodyid[cam] == 1, "the phone's camera is on the pelvis"
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.cam_quat[cam])
        self.cam_in_pelvis = (m.cam_pos[cam].copy(), rot.reshape(3, 3))
        # whose pixels are whose in the segmentation: the person, the ball, the robot itself
        root = m.body_rootid[m.geom_bodyid]
        self.is_person = m.geom_bodyid == m.body("person").id
        self.is_ball = m.geom_bodyid == m.body("ball").id
        self.is_self = root == 1
        self.ball_radius = float(m.geom_size[m.geom("ball").id, 0])
        x0, x1, y0, y1 = extent
        self.origin, self.res = np.array([x0, y0]), res
        self.logodds = np.zeros((int(round((x1 - x0) / res)), int(round((y1 - y0) / res))), np.float32)
        self.people, self.balls = Track(), Track()
        self.next_frame, self.t = 0.0, 0.0
        self._floor, self._floor_t = None, -math.inf
        self.frames = 0
        self.arkit.update(d, 0.0)

    def close(self):
        self.lidar.close()

    # -- the pose -------------------------------------------------------------------------------------------
    def pelvis(self) -> tuple[np.ndarray, np.ndarray]:
        pos, rot = self.arkit.pose
        p_cp, r_cp = self.cam_in_pelvis
        r = rot @ r_cp.T
        return pos - r @ p_cp, r

    def robot(self) -> tuple[float, float, float]:
        p, r = self.pelvis()
        return float(p[0]), float(p[1]), math.atan2(r[1, 0], r[0, 0])

    # -- sensing --------------------------------------------------------------------------------------------
    def update(self, t: float, dt: float):
        self.t = t
        self.arkit.update(self.d, dt)
        if t + 1e-9 < self.next_frame:
            return
        self.next_frame = t + self.rate
        depth, conf = self.lidar.capture(self.d)
        geom = self.lidar.geom
        seen = geom >= 0
        gid = np.clip(geom, 0, None)
        person = seen & self.is_person[gid]
        ball = seen & self.is_ball[gid]
        mine = seen & self.is_self[gid]
        pose = self.arkit.now       # the frame comes with its own pose (both reach the app `latency` later)
        # the map: everything but the person (segmented out) and the robot's own body
        d_map = np.where(person | mine, 0.0, depth)
        pts = self.lidar.points(d_map, pose, conf=conf)
        self._integrate(pts, pose[0][:2])
        # the person and the ball, when the detectors fire
        for mask, track, depth_in in ((person, self.people, PERSON_DEPTH), (ball, self.balls, self.ball_radius * 0.7)):
            if mask.sum() < self.min_pixels or self.rng.random() < self.miss:
                continue
            on = mask & (depth > 0)
            if on.sum() < self.min_pixels // 3:
                continue
            p = self.lidar.points(np.where(on, depth, 0.0), pose)
            xy = np.median(p[:, :2], axis=0)
            ray = xy - pose[0][:2]                        # the points are its near side: the centre is behind them
            xy = xy + ray / max(np.linalg.norm(ray), 1e-6) * depth_in
            track.see(xy, t)
        self.frames += 1

    def _cells(self, xy: np.ndarray, min_points: int = 1) -> np.ndarray:
        """The grid cells (flat indices) with at least `min_points` of these points."""
        ij = np.floor((xy - self.origin) / self.res).astype(int)
        n, k = self.logodds.shape
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < n) & (ij[:, 1] >= 0) & (ij[:, 1] < k)
        cells, count = np.unique(ij[ok, 0] * k + ij[ok, 1], return_counts=True)
        return cells[count >= min_points]

    def _integrate(self, pts: np.ndarray, cam_xy: np.ndarray):
        z = pts[:, 2]
        near = np.linalg.norm(pts[:, :2] - cam_xy, axis=1) < HIT_RANGE
        hits = self._cells(pts[(z > HIT_Z) & (z < ROBOT_TOP) & near, :2], min_points=2)
        floor = np.setdiff1d(self._cells(pts[z < FLOOR_Z, :2]), hits)
        flat = self.logodds.reshape(-1)
        flat[hits] = np.minimum(flat[hits] + HIT, L_MAX)
        flat[floor] = np.maximum(flat[floor] + FREE, L_MIN)

    # -- what the brain reads -------------------------------------------------------------------------------
    def person(self):
        xy, age = self.people.get(self.t)
        if xy is not None and age > 0.0:                  # out of sight: where they were heading (up to 1.5 s on)
            xy = xy + self.people.vel * min(age, 1.5)
        return xy, age

    def ball(self):
        return self.balls.get(self.t)

    def floor(self) -> FloorMap:
        """The LiDAR map as a FloorMap (rebuilt at most twice a second)."""
        if self.t - self._floor_t >= 0.5 or self._floor is None:
            occ = self.logodds > L_OCC
            occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
            self._floor, self._floor_t = FloorMap(occ, self.origin, self.res, self.logodds == 0.0), self.t
        return self._floor
