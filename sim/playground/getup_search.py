"""Can Cogu get up at all? Search open-loop keyframe sequences with CMA-ES (plain MuJoCo, CPU).

    uv run python sim/playground/getup_search.py                       # all four lying poses
    uv run python sim/playground/getup_search.py --pose back --keyframes 5 --generations 300
    uv run python sim/playground/getup_search.py --replay build/getup_search/back.json --video
    uv run python sim/playground/getup_search.py --replay build/getup_search/back.json --xml build/mjcf/cogu_lower_body.xml
    uv run mjpython sim/playground/getup_search.py --view build/getup_search/*_peak.json     # watch them, looping

A feasibility study before more RL (2026-10-31): the get-up policy did not learn, and Cogu has no arms,
only the legs and the egg-shaped bubble. Here the robot is laid on its front, back or a side in `rest`
and left to settle; then a sequence of K joint-target keyframes runs (each held for its own duration,
0.1-0.8 s), then `rest` is held for 1.5 s. CMA-ES searches the targets (each joint's whole range) and the
durations. Each candidate runs on three variants of the model (nominal; slippery floor with weaker
servos; grippy floor) and scores their mean, so a sequence that only works on one exact floor loses.

Success = upright (pelvis z-axis within ~18° of vertical), pelvis above 18 cm and nearly still at the end
of the hold, on every variant. If a short open-loop sequence gets up, the motion exists, the RL problem is
one of exploration and credit assignment, and the sequence's length says what horizon it needs. If none
does from a pose, no policy will either: the bubble or the legs have to change, not the training.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
GETUP_XML = REPO_ROOT / "build" / "mjcf" / "cogu_lower_body_mjx_getup.xml"
OUT_DIR = REPO_ROOT / "build" / "getup_search"

POSES = {                      # axis, degrees: rotation of the standing robot onto the floor
    "front": ((0, 1, 0), 90),  # face (+x) down
    "back": ((0, 1, 0), -90),
    "left": ((1, 0, 0), 90),   # left side down
    "right": ((1, 0, 0), -90),
    "crouch": ((0, 1, 0), -60),  # second stage: set on its feet, thigh back, tilted 60° face-up; settles at ~25°
}
CROUCH_LEGS = (-80.0, 10.0, 10.0)  # deg, hip pitch / knee / ankle (both legs; yaw and roll 0)
VARIANTS = (                   # (friction scale, servo force scale)
    (1.0, 1.0),
    (0.6, 0.85),
    (1.4, 1.0),
)
SETTLE = 2.0                   # s lying in `rest` before the sequence starts
HOLD = 1.5                     # s holding `rest` after the last keyframe
DUR = (0.1, 0.8)               # s, range of each keyframe's duration
UPRIGHT_COS = 0.95
STAND_HEIGHT = 0.18            # m, pelvis (rest: 0.205)


# --------------------------------------------------------------------------------------------- simulation
class Sim:
    """One model variant, its lying start states, and rollouts of a keyframe sequence."""

    def __init__(self, xml: Path, friction: float = 1.0, force: float = 1.0, objective: str = "getup",
                 tail: list | None = None):
        self.objective = objective
        self.tail = tail or []                       # fixed keyframes after the searched ones (catch, then slow path)
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.m.geom_friction[:, 0] *= friction
        self.m.actuator_forcerange[:] *= force
        self.d = mujoco.MjData(self.m)
        key = self.m.keyframe("rest")
        self.rest_q, self.rest_ctrl = key.qpos.copy(), key.ctrl.copy()
        self.lo, self.hi = self.m.actuator_ctrlrange.T.copy()
        self.dt = self.m.opt.timestep
        self._floor = self.m.geom("floor").id
        self._soles = {self.m.geom("L_foot_sole").id, self.m.geom("R_foot_sole").id}
        self._starts: dict[str, tuple] = {}

    def start(self, pose: str):
        if pose not in self._starts:
            axis, deg = POSES[pose]
            q = np.zeros(4)
            mujoco.mju_axisAngle2Quat(q, np.array(axis, float), np.radians(deg))
            mujoco.mj_resetData(self.m, self.d)
            self.d.qpos[:] = self.rest_q
            self.d.qpos[2] = 0.30
            self.d.qpos[3:7] = q
            self.d.ctrl[:] = self.rest_ctrl
            if pose == "crouch":
                legs = np.radians([0.0, 0.0, *CROUCH_LEGS])
                self.d.qpos[7:] = self.d.ctrl[:] = np.concatenate([legs, legs])
                self._place_on_floor()
            for _ in range(int(SETTLE / self.dt)):
                mujoco.mj_step(self.m, self.d)
            self._starts[pose] = (self.d.qpos.copy(), self.d.qvel.copy(), self.d.act.copy())
        qpos, qvel, act = self._starts[pose]
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:], self.d.qvel[:], self.d.act[:] = qpos, qvel, act
        self.d.ctrl[:] = self.rest_ctrl
        mujoco.mj_forward(self.m, self.d)

    def _place_on_floor(self):
        """Lower the robot until its lowest collision geom touches the floor (no drop)."""
        mujoco.mj_kinematics(self.m, self.d)
        floor = self.m.geom("floor").id
        low = np.inf
        for i in range(self.m.ngeom):
            if i == floor or not (self.m.geom_contype[i] or self.m.geom_conaffinity[i]):
                continue
            r, c, size = self.d.geom_xmat[i].reshape(3, 3), self.d.geom_xpos[i], self.m.geom_size[i]
            t = self.m.geom_type[i]
            if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                z = c[2] - np.sqrt(np.sum((size * r[2]) ** 2))
            elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
                z = c[2] - size[1] * abs(r[2, 2]) - size[0] * np.sqrt(max(0.0, 1 - r[2, 2] ** 2))
            else:
                z = c[2] - np.sum(np.abs(r[2]) * size[:3])
            low = min(low, z)
        self.d.qpos[2] -= low - 0.001

    def decode(self, x: np.ndarray, k: int):
        """Parameters -> (targets K×nu in rad, durations K in s)."""
        u = (np.clip(x, -1.0, 1.0) + 1.0) / 2.0
        u = u.reshape(k, self.m.nu + 1)
        targets = self.lo + (self.hi - self.lo) * u[:, :-1]
        durations = DUR[0] + (DUR[1] - DUR[0]) * u[:, -1]
        return targets, durations

    def encode(self, targets, durations) -> np.ndarray:
        """Inverse of `decode` (for warm starts)."""
        u = np.column_stack([(np.asarray(targets) - self.lo) / (self.hi - self.lo),
                             (np.asarray(durations) - DUR[0]) / (DUR[1] - DUR[0])])
        return (2.0 * u - 1.0).ravel()

    def up(self) -> float:
        return float(self.d.xmat[1][8])               # pelvis z-axis · world z

    def rollout(self, pose: str, targets, durations, frame_cb=None) -> dict:
        self.start(pose)
        best_up = self.up()
        best_land = -1.0
        saturated = steps = 0
        schedule = [(t, dur) for t, dur in zip(targets, durations)] + list(self.tail) + [(self.rest_ctrl, HOLD)]
        catch_at = len(targets) if self.tail else None
        up_catch = on_feet = 0.0
        for n, (target, dur) in enumerate(schedule):
            self.d.ctrl[:] = target
            for _ in range(int(round(dur / self.dt))):
                mujoco.mj_step(self.m, self.d)
                steps += 1
                best_up = max(best_up, self.up())
                if self.objective == "land" and self.d.ncon:   # upright while only the soles touch the floor
                    touching = {g for c in self.d.contact[: self.d.ncon] for g in c.geom} - {self._floor}
                    if touching and touching <= self._soles:
                        best_land = max(best_land, self.up())
                saturated += np.any(np.abs(self.d.actuator_force) >= 0.99 * self.m.actuator_forcerange[:, 1])
                if frame_cb is not None:
                    frame_cb(self)
            if n == catch_at:                         # end of the catch hold: upright, and on the feet only?
                up_catch = self.up()
                touching = {g for c in self.d.contact[: self.d.ncon] for g in c.geom} - {self._floor}
                on_feet = float(bool(touching) and touching <= self._soles)
        up, height = self.up(), float(self.d.qpos[2])
        posture = float(np.exp(-np.sum((self.d.qpos[7:] - self.rest_q[7:]) ** 2) / 0.5))
        speed = float(np.linalg.norm(self.d.qvel[:6]))
        success = up > UPRIGHT_COS and height > STAND_HEIGHT and speed < 0.3
        score = (((1.0 + up) / 2.0) ** 2 + min(height / 0.2, 1.0) + 0.5 * posture
                 + 0.3 * ((1.0 + best_up) / 2.0) ** 2 + 2.0 * success)
        if self.objective == "catch":                 # the searched keyframes only have to land the catch
            score += 2.0 * ((1.0 + up_catch) / 2.0) ** 2 + 1.0 * on_feet
        if self.objective == "land":                  # how upright it gets standing on the feet alone
            score = best_land + 0.2 * best_up
        if self.objective == "peak":                  # how upright it gets at all, whatever happens after
            score = best_up
        return dict(score=score, success=success, up=up, height=height, speed=speed, peak_up=best_up,
                    up_catch=up_catch, on_feet=on_feet, land_up=best_land,
                    saturated=saturated / max(steps, 1), seconds=float(np.sum(durations)))


# ------------------------------------------------------------------------------------------ worker pool
_SIMS: list[Sim] = []


TAIL_FROM = 70.0               # deg: the catch pose is the path's pose at this tilt (80° falls back onto the bubble)


def load_tail(path: str | None, catch_hold: float = 0.8, step: float = 0.4, from_tilt: float = TAIL_FROM) -> list:
    """A quasi-static path from getup_static.py -> fixed keyframes: hold its pose at `from_tilt` (the catch),
    then walk it every 10° of tilt."""
    if not path:
        return []
    poses = [p for p in json.loads(Path(path).read_text()) if abs(p["tilt"]) <= from_tilt]
    tail, last = [], None
    for i, p in enumerate(poses):
        if i and i != len(poses) - 1 and abs(p["tilt"] - last) < 10:
            continue
        legs = np.radians([0.0, 0.0, *p["q"]])
        tail.append((np.concatenate([legs, legs]), catch_hold if not tail else step))
        last = p["tilt"]
    return tail


def _init_worker(xml: str, objective: str, tail_path: str | None):
    global _SIMS
    tail = load_tail(tail_path)
    _SIMS = [Sim(Path(xml), f, s, objective, tail) for f, s in VARIANTS]


def _evaluate(task):
    x, pose, k = task
    results = []
    for sim in _SIMS:
        targets, durations = sim.decode(x, k)
        results.append(sim.rollout(pose, targets, durations))
    excess = float(np.sum(np.maximum(np.abs(x) - 1.0, 0.0) ** 2))   # keep the mean inside the box
    return float(np.mean([r["score"] for r in results])) - excess, results


# --------------------------------------------------------------------------------------------- CMA-ES
class CMAES:
    """Minimal (mu/mu_w, lambda)-CMA-ES (Hansen's tutorial), maximising."""

    def __init__(self, x0, sigma, popsize=None, seed=0):
        n = len(x0)
        self.n, self.mean, self.sigma = n, np.array(x0, float), sigma
        self.rng = np.random.default_rng(seed)
        self.lam = popsize or 4 + int(3 * np.log(n))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.B, self.D, self.C = np.eye(n), np.ones(n), np.eye(n)
        self.gen = 0

    def ask(self) -> np.ndarray:
        z = self.rng.normal(size=(self.lam, self.n))
        return self.mean + self.sigma * (z * self.D) @ self.B.T

    def tell(self, xs: np.ndarray, scores) -> None:
        order = np.argsort(scores)[::-1]
        y = (xs[order[: self.mu]] - self.mean) / self.sigma
        yw = self.w @ y
        self.mean = self.mean + self.sigma * yw
        inv_sqrt_c = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * inv_sqrt_c @ yw
        self.gen += 1
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chi_n
                < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (y.T * self.w) @ y)
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chi_n - 1))
        self.C = (self.C + self.C.T) / 2
        d2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(d2, 1e-20))


# ----------------------------------------------------------------------------------------------- search
def search(pool, pose: str, k: int, generations: int, popsize: int, seed: int, nu: int, sigma: float, x0=None):
    es = CMAES(np.zeros(k * (nu + 1)) if x0 is None else x0, sigma=sigma, popsize=popsize, seed=seed)
    best = (-np.inf, None, None)
    t0 = time.time()
    for g in range(generations):
        xs = es.ask()
        out = pool.map(_evaluate, [(x, pose, k) for x in xs])
        scores = [s for s, _ in out]
        es.tell(xs, scores)
        i = int(np.argmax(scores))
        if scores[i] > best[0]:
            best = (scores[i], xs[i].copy(), out[i][1])
        if g % 20 == 0 or g == generations - 1:
            r = best[2]
            print(f"  [{pose} K={k} seed={seed}] gen {g:3d}  best {best[0]:.2f}  "
                  f"up {' '.join(f'{v['up']:+.2f}' for v in r)}  "
                  f"h {' '.join(f'{v['height'] * 1000:.0f}' for v in r)} mm  "
                  f"peak up {max(v['peak_up'] for v in r):+.2f}  on feet {' '.join(f"{v['land_up']:+.2f}" for v in r)}  catch {' '.join(f"{v['up_catch']:+.2f}{'F' if v['on_feet'] else ''}" for v in r)}  ok {sum(v['success'] for v in r)}/{len(r)}  sigma {es.sigma:.2f}  {time.time() - t0:.0f} s",
                  flush=True)
        if all(v["success"] for v in best[2]) and es.sigma < 0.05:
            break
    return best


def replay(path: Path, xml: Path, video: bool):
    spec = json.loads(path.read_text())
    targets, durations = np.array(spec["targets"]), np.array(spec["durations"])
    print(f"{path.name}: {spec['pose']}, {len(durations)} keyframes, {durations.sum():.2f} s + {HOLD} s hold "
          f"on {xml.name}")
    for f, s in VARIANTS + ((0.8, 0.85), (1.0, 0.7)):
        sim = Sim(xml, f, s)
        r = sim.rollout(spec["pose"], targets, durations)
        print(f"  friction ×{f:.1f} servo ×{s:.2f}: {'UP  ' if r['success'] else 'fail'} up_z {r['up']:+.2f} "
              f"pelvis {r['height'] * 1000:.0f} mm  speed {r['speed']:.2f}  saturated {r['saturated'] * 100:.0f} %")
    if video:
        import imageio.v2 as imageio
        sim = Sim(xml)
        renderer = mujoco.Renderer(sim.m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.type, cam.distance, cam.elevation, cam.azimuth = mujoco.mjtCamera.mjCAMERA_FREE, 0.9, -15, 135
        frames, every = [], int(round(1 / (30 * sim.dt)))
        count = [0]

        def grab(s: Sim):
            count[0] += 1
            if count[0] % every == 0:
                cam.lookat[:] = (s.d.qpos[0], s.d.qpos[1], 0.12)
                renderer.update_scene(s.d, cam)
                frames.append(renderer.render())

        sim.rollout(spec["pose"], targets, durations, frame_cb=grab)
        out = path.with_suffix(".mp4")
        imageio.mimwrite(out, frames, fps=30, codec="libx264", quality=8, macro_block_size=None)
        print(f"video: {out}")


def view(paths: list[Path], xml: Path):
    """Play saved sequences in the MuJoCo viewer, one after the other, forever (macOS: run with mjpython)."""
    import mujoco.viewer
    sim = Sim(xml)
    specs = [(p.stem, json.loads(p.read_text())) for p in paths]
    with mujoco.viewer.launch_passive(sim.m, sim.d) as v:
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 0.9, -15, 135

        count, every = [0], 10                       # draw every 20 ms of sim time, in real time
        t0 = [time.perf_counter()]

        def show(s: Sim):
            count[0] += 1
            if count[0] % every:
                return
            if not v.is_running():
                raise SystemExit
            v.cam.lookat[:] = (s.d.qpos[0], s.d.qpos[1], 0.12)
            v.sync()
            wait = every * s.dt - (time.perf_counter() - t0[0])
            if wait > 0:
                time.sleep(wait)
            t0[0] = time.perf_counter()

        while v.is_running():
            for name, spec in specs:
                print(f"{name}: {spec['pose']}, {len(spec['durations'])} keyframes", flush=True)
                sim.rollout(spec["pose"], np.array(spec["targets"]), np.array(spec["durations"]), frame_cb=show)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", choices=list(POSES) + ["all"], default="all")
    ap.add_argument("--keyframes", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--generations", type=int, default=250)
    ap.add_argument("--popsize", type=int, default=36)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--xml", default=str(GETUP_XML))
    ap.add_argument("--tail", metavar="JSON", help="a getup_static.py path played after the searched keyframes "
                    "(catch, then slow path to rest); use with --objective catch")
    ap.add_argument("--objective", choices=("getup", "peak", "catch", "land"), default="getup",
                    help="peak: only how upright the robot gets at any moment (is the gap crossable at all?)")
    ap.add_argument("--sigma", type=float, default=0.6)
    ap.add_argument("--init", metavar="JSON", help="start the search from a saved sequence (extra keyframes hold `rest`)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--replay", help="a saved sequence (.json) to check instead of searching")
    ap.add_argument("--video", action="store_true", help="with --replay: write an .mp4 next to it")
    ap.add_argument("--view", nargs="+", metavar="JSON", help="saved sequences to watch in the viewer, looping")
    args = ap.parse_args(argv)

    if args.view:
        view([Path(p) for p in args.view], Path(args.xml))
        return 0
    if args.replay:
        replay(Path(args.replay), Path(args.xml), args.video)
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nu = Sim(Path(args.xml)).m.nu
    poses = list(POSES) if args.pose == "all" else [args.pose]
    summary = {}
    with mp.get_context("spawn").Pool(args.workers, _init_worker, (args.xml, args.objective, args.tail)) as pool:
        for pose in poses:
            best_pose = (-np.inf, None, None, None)
            for k in args.keyframes:
                for seed in range(args.seeds):
                    x0 = None
                    if args.init:                     # warm start: a saved sequence, then hold-`rest` keyframes
                        spec = json.loads(Path(args.init).read_text())
                        sim0 = Sim(Path(args.xml))
                        n = len(spec["durations"])
                        targets = np.vstack([spec["targets"]] + [sim0.rest_ctrl] * max(k - n, 0))[:k]
                        durations = np.concatenate([spec["durations"], [0.4] * max(k - n, 0)])[:k]
                        x0 = np.clip(sim0.encode(targets, durations), -0.99, 0.99)
                    score, x, results = search(pool, pose, k, args.generations, args.popsize, seed, nu, args.sigma, x0)
                    if score > best_pose[0]:
                        best_pose = (score, x, results, k)
                    if all(r["success"] for r in results):
                        break
                if all(r["success"] for r in best_pose[2]):
                    break
            score, x, results, k = best_pose
            targets, durations = Sim(Path(args.xml)).decode(x, k)
            tail = load_tail(args.tail)               # saved with the tail, so --replay plays the whole thing
            if tail:
                targets = np.vstack([targets] + [t for t, _ in tail])
                durations = np.concatenate([durations, [dur for _, dur in tail]])
            ok = sum(r["success"] for r in results)
            summary[pose] = f"{ok}/{len(results)} variants up, {k} keyframes, {durations.sum():.2f} s"
            (OUT_DIR / f"{pose}{'' if args.objective == 'getup' else '_' + args.objective}{'_' + Path(args.xml).stem if args.xml != str(GETUP_XML) else ''}.json").write_text(json.dumps(dict(
                pose=pose, score=score, success=[bool(r["success"]) for r in results],
                targets=targets.round(4).tolist(), durations=durations.round(3).tolist()), indent=1))
            summary[pose] += f", peak up_z {min(r['peak_up'] for r in results):+.2f} (worst variant)"
            print(f"{pose}: {summary[pose]}", flush=True)
    print("\nsummary")
    for pose, s in summary.items():
        print(f"  {pose:6s} {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
