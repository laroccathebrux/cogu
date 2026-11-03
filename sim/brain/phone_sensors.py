"""The iPhone's LiDAR and ARKit pose, simulated from the `phone_cam` camera (ADR-004, item 7).

    uv run python sim/brain/phone_sensors.py                  # -> docs/renders/sim_lidar.png (apartment, at rest)
    uv run python sim/brain/phone_sensors.py --at 2.1 0.9 -145   # robot moved to x, y (m), yaw (deg) first

On the robot, depth comes from the iPhone's LiDAR (ARKit `sceneDepth`); the simulator has no LiDAR, so `Lidar` renders
the depth of `phone_cam` and degrades it the way that sensor does:
- 256 × 192 depth map with the camera's field of view, z-depth in metres, 0 = no reading;
- range ~5 m (beyond: no reading), noise growing with distance (~1 cm at 1 m);
- dark surfaces (low albedo; and anything named glass / mirror) mostly return nothing;
- "flying pixels" at depth edges get low confidence, as ARKit's confidence map does (0 low, 1 medium, 2 high).

`ArkitPose` is the camera pose as ARKit reports it: the true one, late by `latency` and drifting slowly (a random walk
in position and yaw), with the 6-DoF noise of visual-inertial odometry.

`Lidar.points()` turns a depth map into world points with a pose (the ARKit one, so the map inherits its error).
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import mujoco
import numpy as np

from apartment import CEILING_GROUP

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
APARTMENT_XML = REPO_ROOT / "build" / "mjcf" / "apartment.xml"


def phone_view() -> mujoco.MjvOption:
    """What the phone sees: the visual geoms and the ceiling (not the collision shapes)."""
    opt = mujoco.MjvOption()
    opt.geomgroup[CEILING_GROUP] = 1
    return opt


class Lidar:
    """`capture(data)` -> (depth m, confidence 0/1/2), both (height, width)."""

    def __init__(self, model: mujoco.MjModel, camera: str = "phone_cam", width: int = 256, height: int = 192,
                 max_range: float = 5.0, noise_abs: float = 0.005, noise_rel: float = 0.006,
                 dark_albedo: float = 0.15, dark_loss: float = 0.85, dropout: float = 0.003, seed: int = 0):
        self.m, self.camera = model, model.camera(camera).id
        self.width, self.height = width, height
        self.max_range, self.noise_abs, self.noise_rel = max_range, noise_abs, noise_rel
        self.dark_loss, self.dropout = dark_loss, dropout
        self.rng = np.random.default_rng(seed)
        self._depth = mujoco.Renderer(model, height, width)
        self._depth.enable_depth_rendering()
        self._seg = mujoco.Renderer(model, height, width)
        self._seg.enable_segmentation_rendering()
        self.scene_option = phone_view()
        # geoms the LiDAR mostly loses: dark ones (by colour or material colour) and glass / mirror by name
        lost = np.zeros(model.ngeom, bool)
        for i in range(model.ngeom):
            rgba = model.geom_rgba[i] if model.geom_matid[i] < 0 else model.mat_rgba[model.geom_matid[i]]
            name = model.geom(i).name
            lost[i] = rgba[:3].mean() < dark_albedo or "glass" in name or "mirror" in name
        self._lost = lost
        fovy = np.radians(model.cam_fovy[self.camera])
        self.f = (height / 2) / np.tan(fovy / 2)              # focal length, px (square pixels)

    def capture(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        self._depth.update_scene(data, self.camera, self.scene_option)
        depth = self._depth.render().astype(np.float32)
        self._seg.update_scene(data, self.camera, self.scene_option)
        seg = self._seg.render()
        geom = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)

        conf = np.full(depth.shape, 2, np.uint8)
        conf[depth > 0.6 * self.max_range] = 1
        # flying pixels: a big depth jump to a neighbour
        jump = np.zeros(depth.shape, bool)
        dz_x, dz_y = np.abs(np.diff(depth, axis=1)), np.abs(np.diff(depth, axis=0))
        edge_x, edge_y = dz_x > 0.08 * depth[:, 1:], dz_y > 0.08 * depth[1:, :]
        jump[:, 1:] |= edge_x
        jump[:, :-1] |= edge_x
        jump[1:, :] |= edge_y
        jump[:-1, :] |= edge_y
        conf[jump] = 0

        sigma = self.noise_abs + self.noise_rel * depth
        depth = depth + self.rng.normal(size=depth.shape).astype(np.float32) * sigma
        lost = (geom >= 0) & self._lost[np.clip(geom, 0, None)] & (self.rng.random(depth.shape) < self.dark_loss)
        invalid = lost | (geom < 0) | (depth > self.max_range) | (self.rng.random(depth.shape) < self.dropout)
        depth[invalid] = 0.0
        conf[invalid] = 0
        return depth, conf

    def points(self, depth: np.ndarray, pose: tuple[np.ndarray, np.ndarray], min_conf: int = 1,
               conf: np.ndarray | None = None) -> np.ndarray:
        """World points (N, 3) from a depth map and a camera pose (position, 3 × 3 rotation; MuJoCo camera axes:
        x right, y up, looking along -z)."""
        v, u = np.nonzero((depth > 0) & ((conf >= min_conf) if conf is not None else True))
        z = depth[v, u]
        cam = np.stack([(u + 0.5 - self.width / 2) / self.f * z, -(v + 0.5 - self.height / 2) / self.f * z, -z], 1)
        pos, rot = pose
        return cam @ rot.T + pos


class ArkitPose:
    """The camera pose as ARKit would report it: late by `latency` s, drifting, noisy. Call `update(data, dt)` every
    control step; `pose` is the latest report (position, rotation)."""

    def __init__(self, model: mujoco.MjModel, camera: str = "phone_cam", latency: float = 0.05,
                 drift_pos: float = 0.004, drift_yaw: float = 0.003, noise_pos: float = 0.003,
                 noise_rot: float = 0.003, seed: int = 0):
        self.camera = model.camera(camera).id
        self.latency, self.drift_pos, self.drift_yaw = latency, drift_pos, drift_yaw
        self.noise_pos, self.noise_rot = noise_pos, noise_rot
        self.rng = np.random.default_rng(seed)
        self.offset = np.zeros(3)                       # m: drift of the position
        self.yaw = 0.0                                  # rad: drift of the heading
        self.queue: collections.deque = collections.deque()
        self.pose = None

    def update(self, data: mujoco.MjData, dt: float):
        # drift: a random walk per sqrt(s) (m, rad), in the floor plane for the position
        s = np.sqrt(dt)
        self.offset[:2] += self.rng.normal(size=2) * self.drift_pos * s
        self.yaw += self.rng.normal() * self.drift_yaw * s
        c, sn = np.cos(self.yaw), np.sin(self.yaw)
        drift_rot = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        pos = drift_rot @ data.cam_xpos[self.camera] + self.offset + self.rng.normal(size=3) * self.noise_pos
        w = self.rng.normal(size=3) * self.noise_rot    # small-angle noise
        noise_rot = np.eye(3) + np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        rot = noise_rot @ drift_rot @ data.cam_xmat[self.camera].reshape(3, 3)
        self.queue.append((data.time + self.latency, (pos, rot)))
        while self.queue and self.queue[0][0] <= data.time:
            self.pose = self.queue.popleft()[1]
        if self.pose is None:
            self.pose = (pos, rot)


def _figure(model, data, lidar, depth, conf, pose, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = mujoco.Renderer(model, 480, 640)
    rgb.update_scene(data, "phone_cam", phone_view())
    image = rgb.render()
    pts = lidar.points(depth, pose, conf=conf)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6), gridspec_kw=dict(width_ratios=[1.33, 1.33, 1.33, 1.1]))
    ax[0].imshow(image)
    ax[0].set_title("câmera (phone_cam)")
    shown = np.where(depth > 0, depth, np.nan)
    im = ax[1].imshow(shown, cmap="turbo", vmin=0, vmax=lidar.max_range)
    ax[1].set_title("LiDAR simulado: profundidade (m), branco = sem leitura")
    fig.colorbar(im, ax=ax[1], fraction=0.035)
    ax[2].imshow(conf, cmap="RdYlGn", vmin=0, vmax=2)
    ax[2].set_title("confiança (0 baixa, 1 média, 2 alta)")
    keep = (pts[:, 2] > 0.03) & (pts[:, 2] < 2.0)         # not the floor, not the ceiling
    ax[3].scatter(pts[keep, 0], pts[keep, 1], s=0.3, c=pts[keep, 2], cmap="viridis", vmin=0, vmax=2)
    ax[3].scatter(*pose[0][:2], c="r", marker="^", s=40)
    ax[3].set_aspect("equal")
    ax[3].set_xlim(-1.6, 5.1)
    ax[3].set_ylim(-1.9, 1.9)
    ax[3].set_title("pontos vistos, de cima (sem chão e teto)")
    for a in ax[:3]:
        a.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", default=str(APARTMENT_XML))
    ap.add_argument("--at", type=float, nargs=3, metavar=("X", "Y", "YAW"), help="move the robot first (m, m, deg)")
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "renders" / "sim_lidar.png"))
    args = ap.parse_args(argv)
    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.keyframe("rest").id)
    if args.at:
        x, y, yaw = args.at
        d.qpos[:2] = x, y
        d.qpos[3:7] = np.cos(np.radians(yaw) / 2), 0.0, 0.0, np.sin(np.radians(yaw) / 2)
    mujoco.mj_forward(m, d)
    lidar, arkit = Lidar(m), ArkitPose(m)
    arkit.update(d, 0.02)
    depth, conf = lidar.capture(d)
    valid = depth > 0
    print(f"depth {depth.shape[1]}×{depth.shape[0]}: {valid.mean():.0%} with a reading, "
          f"{(conf == 2).mean():.0%} high confidence, range {depth[valid].min():.2f}–{depth[valid].max():.2f} m")
    _figure(m, d, lidar, depth, conf, arkit.pose, Path(args.out))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
