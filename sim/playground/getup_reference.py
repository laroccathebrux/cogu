"""A scripted get-up to imitate: an open-loop throw, then a catch that follows the static path by feel (plain MuJoCo).

    uv run python sim/playground/getup_reference.py --seq sim/playground/getup_sequences/back_land.json
    uv run python sim/playground/getup_reference.py --xml build/mjcf/variants/heel_45.xml \
        --path build/getup_search/path_heel45.json --lead 0 5 10 --up 0.4 0.5 0.6
    uv run python sim/playground/getup_reference.py --search --xml build/mjcf/variants/heel_45.xml \
        --path build/getup_search/path_heel45.json --out sim/playground/getup_sequences/reference_heel45.json

The reference the get-up policy will learn around (README, "Levantar por imitação"). The throw is a saved
`getup_search.py` sequence played from lying. The moment the robot is on its soles only and upright enough
(pelvis up_z >= --up), or when the throw ends, the catch starts: every 20 ms the pelvis tilt is measured (as the
IMU would) and the legs are sent to the pose the quasi-static path (`getup_static.py --out`) holds at
`tilt - lead`: a pose that holds at that tilt, a little ahead towards upright. Below 0° the pose is `rest`.
Success as in `getup_search.py` (upright, pelvis above 18 cm, still) 3 s after the throw ends, on each friction /
servo variant.

`--search`: CMA-ES over a symmetric throw (both legs alike, hip yaw and roll 0: the plane the static path lives in)
together with the catch's lead and trigger, scored on the whole get-up, catch included. The saved sequences were
searched without the catch, on all ten joints, and twist the body sideways out of that plane.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import getup_search  # noqa: E402

CTRL_DT = 0.02
AFTER = 3.0                    # s after the throw's last keyframe


class Reference:
    """Throw keyframes by time, then the tilt-indexed catch. `targets(t, tilt, up, on_soles)` -> joint targets.

    `paths`: quasi-static paths (getup_static.py --out), face up (+ tilt) and/or face down (- tilt), joined into one
    tilt -> leg pose table through upright. The catch asks for the pose at `tilt` moved `lead` degrees towards
    upright, and moves its target there at no more than `rate` (rad/s per joint): the path holds only if walked
    slowly (stepping straight to the next pose fell in 3 of 4 starts on the feet)."""

    def __init__(self, seq: dict, paths: list[list[dict]], rest: np.ndarray, lead: float, up_min: float,
                 rate: float = 1.5):
        self.keys = np.asarray(seq["targets"], float)
        self.ends = np.cumsum(seq["durations"])
        nodes = {n["tilt"]: n["q"] for path in paths for n in path}
        tilts = sorted(nodes)
        self.tilts = np.array(tilts, float)
        legs = np.radians(np.array([nodes[t] for t in tilts], float))   # hip pitch, knee, ankle
        self.legs = np.column_stack([np.zeros((len(tilts), 2)), legs])  # hip yaw, roll = 0
        self.rest, self.lead, self.up_min, self.rate = rest, lead, up_min, rate
        self.caught = False
        self.prev: np.ndarray | None = None

    def targets(self, t: float, tilt: float, up: float, on_soles: bool) -> np.ndarray:
        if not self.caught and (t >= self.ends[-1] or (on_soles and up >= self.up_min)):
            self.caught = True
        if not self.caught:
            self.prev = self.keys[min(int(np.searchsorted(self.ends, t, side="right")), len(self.keys) - 1)]
            return self.prev
        want = np.sign(tilt) * max(abs(tilt) - self.lead, 0.0)
        leg = np.array([np.interp(want, self.tilts, self.legs[:, j]) for j in range(5)])
        goal = np.concatenate([leg, leg])
        if self.prev is None:
            self.prev = goal
        step = self.rate * CTRL_DT
        self.prev = self.prev + np.clip(goal - self.prev, -step, step)
        return self.prev


def tilt_deg(d, pelvis: int) -> float:
    """Tilt as getup_static measures it: angle of the pelvis z axis from vertical, + = leaning back (the pelvis x axis
    then points up). Whole angle, so a sideways roll counts too (asin of the x axis alone missed it)."""
    up = np.clip(d.xmat[pelvis][8], -1.0, 1.0)
    return float(np.degrees(np.arccos(up)) * (1.0 if d.xmat[pelvis][6] >= 0.0 else -1.0))


def attempt(sim: getup_search.Sim, pose: str, ref: Reference, frame_cb=None) -> dict:
    m, d = sim.m, sim.d
    sim.start(pose)
    pelvis = m.body("pelvis").id
    every = int(round(CTRL_DT / sim.dt))
    t, caught_at, peak = 0.0, None, sim.up()
    total = ref.ends[-1] + AFTER
    while t < total:
        touching = {g for c in d.contact[: d.ncon] for g in c.geom} - {sim._floor}
        on_soles = bool(touching) and touching <= sim._soles
        d.ctrl[:] = np.clip(ref.targets(t, tilt_deg(d, pelvis), sim.up(), on_soles), sim.lo, sim.hi)
        if ref.caught and caught_at is None:
            caught_at = t
        for _ in range(every):
            mujoco.mj_step(m, d)
            if frame_cb is not None:
                frame_cb(sim)
        peak = max(peak, sim.up())
        t += CTRL_DT
    up, height, speed = sim.up(), float(d.qpos[2]), float(np.linalg.norm(d.qvel[:6]))
    return dict(success=up > getup_search.UPRIGHT_COS and height > getup_search.STAND_HEIGHT and speed < 0.3,
                up=up, height=height, caught_at=caught_at, peak_up=peak)


def argv_has_defaults(args) -> bool:
    return args.lead == [0.0, 5.0, 10.0] and args.up == [0.4, 0.5, 0.6]


# ----------------------------------------------------------------------------------------------- search
LEG = ("hip_pitch", "knee", "ankle")
LEAD = (-10.0, 20.0)           # deg
UP = (0.2, 0.8)
_W: dict = {}


def decode(x: np.ndarray, k: int, sim: getup_search.Sim) -> tuple[dict, float, float]:
    """Parameters in [-1, 1] -> (symmetric sequence, lead, up_min)."""
    u = (np.clip(x, -1.0, 1.0) + 1.0) / 2.0
    idx = [sim.m.actuator(f"L_{j}").id for j in LEG]
    lo, hi = sim.lo[idx], sim.hi[idx]
    rows = u[: 4 * k].reshape(k, 4)
    targets = []
    for r in rows:
        leg = np.concatenate([[0.0, 0.0], lo + (hi - lo) * r[:3]])
        targets.append(np.concatenate([leg, leg]))
    durations = getup_search.DUR[0] + (getup_search.DUR[1] - getup_search.DUR[0]) * rows[:, 3]
    lead = LEAD[0] + (LEAD[1] - LEAD[0]) * u[4 * k]
    up_min = UP[0] + (UP[1] - UP[0]) * u[4 * k + 1]
    seq = dict(pose="back", targets=np.round(targets, 4).tolist(), durations=np.round(durations, 3).tolist())
    return seq, float(lead), float(up_min)


def _init_worker(xml: str, path_file: str):
    _W["sims"] = [getup_search.Sim(Path(xml), f, s) for f, s in getup_search.VARIANTS]
    _W["path"] = json.loads(Path(path_file).read_text())


def _score(task) -> tuple[float, list]:
    x, k = task
    rows = []
    for sim in _W["sims"]:
        seq, lead, up_min = decode(x, k, sim)
        rows.append(attempt(sim, "back", Reference(seq, [_W["path"]], sim.rest_ctrl, lead, up_min)))
    score = float(np.mean([((1 + r["up"]) / 2) ** 2 + min(r["height"] / 0.2, 1.0) + 2.0 * r["success"]
                           + 0.5 * ((1 + r["peak_up"]) / 2) ** 2 for r in rows]))
    return score - float(np.sum(np.maximum(np.abs(x) - 1.0, 0.0) ** 2)), rows


def search(args) -> int:
    import multiprocessing as mp
    import time
    sim0 = getup_search.Sim(Path(args.xml))
    best = (-np.inf, None, None, None)
    with mp.get_context("spawn").Pool(args.workers, _init_worker, (args.xml, args.path)) as pool:
        for k in args.keyframes:
            for seed in range(args.seeds):
                es = getup_search.CMAES(np.zeros(4 * k + 2), sigma=0.6, popsize=args.popsize, seed=seed)
                t0, run = time.time(), (-np.inf, None, None)
                for g in range(args.generations):
                    xs = es.ask()
                    out = pool.map(_score, [(x, k) for x in xs])
                    es.tell(xs, [sc for sc, _ in out])
                    i = int(np.argmax([sc for sc, _ in out]))
                    if out[i][0] > run[0]:
                        run = (out[i][0], xs[i].copy(), out[i][1])
                    if g % 20 == 0 or g == args.generations - 1:
                        r = run[2]
                        print(f"  K={k} seed={seed} gen {g:3d}  best {run[0]:.2f}  ok {sum(v['success'] for v in r)}/"
                              f"{len(r)}  up {' '.join(f'{v['up']:+.2f}' for v in r)}  peak "
                              f"{max(v['peak_up'] for v in r):+.2f}  sigma {es.sigma:.2f}  {time.time() - t0:.0f} s",
                              flush=True)
                    if all(v["success"] for v in run[2]) and es.sigma < 0.05:
                        break
                if run[0] > best[0]:
                    best = (run[0], run[1], run[2], k)
    score, x, rows, k = best
    seq, lead, up_min = decode(x, k, sim0)
    seq.update(score=score, success=[bool(r["success"]) for r in rows], lead=lead, up_min=up_min,
               xml=Path(args.xml).name, path=Path(args.path).name)
    Path(args.out).write_text(json.dumps(seq, indent=1))
    print(f"best: {sum(seq['success'])}/{len(rows)} variants up, {k} keyframes, lead {lead:.1f}°, catch at up_z "
          f"{up_min:.2f} -> {args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", default=str(getup_search.GETUP_XML))
    ap.add_argument("--seq", default=str(HERE / "getup_sequences" / "back_land.json"))
    ap.add_argument("--path", default=str(getup_search.REPO_ROOT / "build" / "getup_search" / "path_heel0.json"))
    ap.add_argument("--lead", type=float, nargs="+", default=[0.0, 5.0, 10.0], help="deg")
    ap.add_argument("--up", type=float, nargs="+", default=[0.4, 0.5, 0.6], help="pelvis up_z that starts the catch")
    ap.add_argument("--video", help="write an .mp4 of the first lead / up on the nominal model")
    ap.add_argument("--search", action="store_true", help="search a symmetric throw + catch (see above)")
    ap.add_argument("--keyframes", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--popsize", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--out", default=str(HERE / "getup_sequences" / "reference.json"))
    args = ap.parse_args(argv)
    if args.search:
        return search(args)
    seq, path = json.loads(Path(args.seq).read_text()), json.loads(Path(args.path).read_text())
    if "lead" in seq and argv_has_defaults(args):           # a searched reference: its own catch settings
        args.lead, args.up = [seq["lead"]], [seq["up_min"]]
    variants = getup_search.VARIANTS + ((0.8, 0.85), (1.0, 0.7))
    sims = [getup_search.Sim(Path(args.xml), f, s) for f, s in variants]
    print(f"{Path(args.seq).stem} on {Path(args.xml).stem}, path {Path(args.path).stem} "
          f"(up to {max(n['tilt'] for n in path):.0f}°); {len(variants)} friction / servo variants")
    for lead in args.lead:
        for up_min in args.up:
            rows = []
            for sim in sims:
                ref = Reference(seq, [path], sim.rest_ctrl, lead, up_min)
                rows.append(attempt(sim, seq["pose"], ref))
            ok = sum(r["success"] for r in rows)
            print(f"  lead {lead:4.1f}°  catch at up_z {up_min:.2f}: {ok}/{len(rows)} up   "
                  + "  ".join(f"{r['up']:+.2f}/{r['height'] * 1000:.0f}mm" for r in rows), flush=True)
    if args.video:
        import imageio.v2 as imageio
        sim = getup_search.Sim(Path(args.xml))
        renderer = mujoco.Renderer(sim.m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.type, cam.distance, cam.elevation, cam.azimuth = mujoco.mjtCamera.mjCAMERA_FREE, 0.9, -15, 135
        frames, count, every = [], [0], int(round(1 / (30 * sim.dt)))

        def grab(s):
            count[0] += 1
            if count[0] % every == 0:
                cam.lookat[:] = (s.d.qpos[0], s.d.qpos[1], 0.12)
                renderer.update_scene(s.d, cam)
                frames.append(renderer.render())

        attempt(sim, seq["pose"], Reference(seq, [path], sim.rest_ctrl, args.lead[0], args.up[0]), grab)
        imageio.mimwrite(args.video, frames, fps=30, codec="libx264", quality=8, macro_block_size=None)
        print(f"video: {args.video}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
