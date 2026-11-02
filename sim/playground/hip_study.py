"""Hip pitch further back? Getting up from the back with no throw, per hip limit (plain MuJoCo).

    uv run python sim/playground/hip_study.py                         # CAD (-85) and -100 .. -140
    uv run python sim/playground/hip_study.py --hip -85 -120 --trials 20
    uv run python sim/playground/hip_study.py --hip -120 --search-roll front      # then --search-roll left
    uv run python sim/playground/hip_study.py --hip -120 -130 --roll              # roll onto the back first
    uv run mjpython sim/playground/hip_study.py --hip -120 --view                 # watch: back, front, side

Lying on the back, the soles reach no closer than ~7 cm to the centre of mass with the CAD's hip (−85°), so getting
up needed a throw (README, "Por que precisa de impulso"). Each variant here is the get-up model with the hip pitch's
backward limit at HIP degrees (`variants.py --hip-min`; the CAD is not touched). For each one:

- window: `getup_static.py`'s quasi-static map, face up and face down (paths -> build/hip_study/);
- drops: as `getup_eval.py drop` (35 cm, random orientation and joint pose, 1 s settle), then no throw at all: the
  legs go back to `rest` at 1 rad/s for `--relax` s (with the legs folded the egg rolls flat onto its back), then
  the tilt-indexed slow catch of `getup_reference.py` walks the model's own static path (10° ahead, 1 rad/s) for 8 s.
  Success = upright, at standing height and still, as `getup_search.py`. Counted by how the robot lay after
  relaxing (back / front / side), on the three floor / servo variants of `getup_search.VARIANTS`.
- `--roll`: lying on the front or a side after relaxing, first a searched open-loop roll onto the back
  (`--search-roll POSE`: CMA-ES over 3 keyframes as `getup_search.py`, then `rest` for 2 s, scored by how flat on
  its back it ends on the three variants; saved as build/hip_study/roll_POSE_HIP.json; the other side = the `left`
  roll mirrored), then the same slow catch.

Results -> build/hip_study/results.json and a table.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import getup_reference  # noqa: E402
import getup_search  # noqa: E402
import getup_starts  # noqa: E402
import getup_static  # noqa: E402
import variants  # noqa: E402

OUT_DIR = getup_search.REPO_ROOT / "build" / "hip_study"
CAD_HIP = -85.0
CTRL_DT = getup_reference.CTRL_DT
ROLL_HOLD = 2.0                # s of `rest` after a roll's keyframes


def model(hip: float) -> Path:
    return getup_search.GETUP_XML if hip == CAD_HIP else variants.build(hip_min=hip)


def paths(xml: Path) -> list[list[dict]]:
    out = []
    for side in ("back", "front"):
        p = OUT_DIR / f"path_{xml.stem}_{side}.json"
        if not p.exists():
            getup_static.main(["--xml", str(xml), "--side", side, "--out", str(p)])
        out.append(json.loads(p.read_text()))
    return out


def roll_file(pose: str, hip: float) -> Path:
    return OUT_DIR / f"roll_{pose}_{hip:g}.json"


def _step(sim: getup_search.Sim, n: int, frame_cb=None) -> None:
    for _ in range(n):
        mujoco.mj_step(sim.m, sim.d)
        if frame_cb is not None:
            frame_cb(sim)


def play(sim: getup_search.Sim, targets, durations, frame_cb=None) -> None:
    for target, dur in list(zip(targets, durations)) + [(sim.rest_ctrl, ROLL_HOLD)]:
        sim.d.ctrl[:] = np.clip(target, sim.lo, sim.hi)
        _step(sim, int(round(dur / sim.dt)), frame_cb)


def relax(sim: getup_search.Sim, seconds: float, frame_cb=None) -> None:
    """Legs back to `rest` at 1 rad/s: with the legs folded the egg rolls onto its back."""
    c = sim.d.ctrl.copy()
    for _ in range(int(seconds / CTRL_DT)):
        c = c + np.clip(sim.rest_ctrl - c, -CTRL_DT, CTRL_DT)
        sim.d.ctrl[:] = c
        _step(sim, int(round(CTRL_DT / sim.dt)), frame_cb)


def roll_over(sim: getup_search.Sim, rolls: dict, left_up: float, frame_cb=None) -> None:
    """Lying on the front or a side: the searched roll onto the back (the other side: the `left` roll mirrored)."""
    pelvis = sim.m.body("pelvis").id
    lying = lay(sim.d, pelvis)
    if lying == "front" and "front" in rolls:
        play(sim, *rolls["front"], frame_cb=frame_cb)
    elif lying == "side" and "left" in rolls:
        targets, durations = rolls["left"]
        same = np.sign(sim.d.xmat[pelvis][7]) == left_up
        play(sim, targets if same else getup_starts.mirrored(targets), durations, frame_cb)


def rise(sim: getup_search.Sim, path_list: list, seconds: float = 8.0, frame_cb=None) -> bool:
    """The slow tilt-indexed catch along the static path, from where the legs are -> upright, high and still?"""
    d = sim.d
    pelvis = sim.m.body("pelvis").id
    c = d.ctrl.copy()
    ref = getup_reference.Reference(dict(targets=[c.tolist()], durations=[CTRL_DT]), path_list, sim.rest_ctrl,
                                    10.0, 0.0, 1.0, trigger=180.0)
    ref.prev, ref.caught = c, True
    t = 0.0
    while t < seconds:
        d.ctrl[:] = np.clip(ref.targets(t, getup_reference.tilt_deg(d, pelvis), sim.up(), False), sim.lo, sim.hi)
        _step(sim, int(round(CTRL_DT / sim.dt)), frame_cb)
        t += CTRL_DT
    return bool(sim.up() > getup_search.UPRIGHT_COS and d.qpos[2] > getup_search.STAND_HEIGHT
                and np.linalg.norm(d.qvel[:6]) < 0.3)


def lay(d, pelvis: int) -> str:
    xz, yz = d.xmat[pelvis][6], d.xmat[pelvis][7]          # pelvis x / y axes, world z component
    if abs(xz) > max(abs(yz), 0.5):
        return "back" if xz > 0 else "front"
    return "side" if abs(yz) > 0.5 else "up"


def drops(task) -> list[tuple[str, bool]]:
    xml, variant, trials, relax_s, path_list, rolls = task
    sim = getup_search.Sim(Path(xml), *getup_search.VARIANTS[variant])
    m, d = sim.m, sim.d
    pelvis = m.body("pelvis").id
    rng = np.random.default_rng(variant)
    sim.start("left")
    left_up = np.sign(d.xmat[pelvis][7])                     # which way the pelvis y axis points in the searched roll
    out = []
    for _ in range(trials):
        mujoco.mj_resetData(m, d)
        quat = rng.normal(size=4)
        q = rng.uniform(m.jnt_range[1:, 0], m.jnt_range[1:, 1])
        d.qpos[:] = np.concatenate([[0.0, 0.0, 0.35], quat / np.linalg.norm(quat), q])
        d.ctrl[:] = q
        for _ in range(int(1.0 / sim.dt)):
            mujoco.mj_step(m, d)
        d.qvel[:] = 0.0
        relax(sim, relax_s)
        lying = lay(d, pelvis)
        roll_over(sim, rolls, left_up)
        out.append((lying, rise(sim, path_list)))
    return out


_SIMS: list[getup_search.Sim] = []


def _init_roll(xml: str):
    global _SIMS
    _SIMS = [getup_search.Sim(Path(xml), *v) for v in getup_search.VARIANTS]


def _roll_score(task) -> tuple[float, list[float]]:
    x, pose, k = task
    flat = []
    for sim in _SIMS:
        sim.start(pose)
        play(sim, *sim.decode(x, k))
        flat.append(float(sim.d.xmat[1][6]))            # pelvis x axis . world z: +1 flat on the back
    excess = float(np.sum(np.maximum(np.abs(x) - 1.0, 0.0) ** 2))
    return float(np.mean(flat) + 0.5 * min(flat)) - excess, flat


def search_roll(xml: Path, pose: str, generations: int, workers: int, out: Path, k: int = 3) -> None:
    with mp.get_context("spawn").Pool(workers, _init_roll, (str(xml),)) as pool:
        nu = _SIMS[0].m.nu if _SIMS else mujoco.MjModel.from_xml_path(str(xml)).nu
        es = getup_search.CMAES(np.zeros(k * (nu + 1)), sigma=0.6, popsize=36, seed=0)
        best = (-np.inf, None, None)
        for g in range(generations):
            xs = es.ask()
            res = pool.map(_roll_score, [(x, pose, k) for x in xs])
            es.tell(xs, [r[0] for r in res])
            i = int(np.argmax([r[0] for r in res]))
            if res[i][0] > best[0]:
                best = (res[i][0], xs[i].copy(), res[i][1])
            if g % 10 == 0 or g == generations - 1:
                print(f"  [{xml.stem} roll {pose}] gen {g:3d}  flat on the back {' '.join(f'{v:+.2f}' for v in best[2])}"
                      f"  sigma {es.sigma:.2f}", flush=True)
    sim = getup_search.Sim(xml)
    targets, durations = sim.decode(best[1], k)
    out.write_text(json.dumps(dict(targets=np.round(targets, 4).tolist(), durations=np.round(durations, 3).tolist())))
    print(f"-> {out}")


def demo(xml: Path, path_list: list, rolls: dict, frame_cb) -> list[tuple[str, bool]]:
    """Lying on the back, the front and a side (getup_search's starts): roll onto the back, rise slowly."""
    sim = getup_search.Sim(xml)
    sim.start("left")
    left_up = np.sign(sim.d.xmat[sim.m.body("pelvis").id][7])
    out = []
    for pose in ("back", "front", "left"):
        sim.start(pose)                                      # the searched rolls start right away (open loop)
        roll_over(sim, rolls, left_up, frame_cb)
        ok = rise(sim, path_list, 6.0, frame_cb)
        _step(sim, int(1.0 / sim.dt), frame_cb)
        out.append((pose, ok))
        print(f"  {pose:5s}: {'stands up' if ok else 'fails'}", flush=True)
    return out


def video(xml: Path, path_list: list, rolls: dict, out: Path) -> None:
    import imageio.v2 as imageio
    m = mujoco.MjModel.from_xml_path(str(xml))
    renderer = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type, cam.distance, cam.elevation, cam.azimuth = mujoco.mjtCamera.mjCAMERA_FREE, 0.9, -15, 135
    frames, every, count = [], int(round(1 / (30 * m.opt.timestep))), [0]

    def grab(sim):
        count[0] += 1
        if count[0] % every == 0:
            cam.lookat[:] = (sim.d.qpos[0], sim.d.qpos[1], 0.12)
            renderer.update_scene(sim.d, cam)
            frames.append(renderer.render())

    demo(xml, path_list, rolls, grab)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=30, codec="libx264", quality=8, macro_block_size=None)
    print(f"video: {out}")


def view(xml: Path, path_list: list, rolls: dict) -> None:
    """The demo in the MuJoCo viewer, in real time, looping (macOS: run with mjpython)."""
    import time
    import mujoco.viewer
    sim0 = getup_search.Sim(xml)
    with mujoco.viewer.launch_passive(sim0.m, sim0.d) as v:
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 0.9, -15, 135
        t0 = [time.perf_counter()]

        def show(sim):
            if not v.is_running():
                raise SystemExit
            if round(sim.d.time / sim.dt) % 10:
                return
            sim0.d.qpos[:] = sim.d.qpos
            mujoco.mj_forward(sim0.m, sim0.d)
            v.cam.lookat[:] = (sim.d.qpos[0], sim.d.qpos[1], 0.12)
            v.sync()
            wait = 10 * sim.dt - (time.perf_counter() - t0[0])
            if wait > 0:
                time.sleep(wait)
            t0[0] = time.perf_counter()

        while v.is_running():
            demo(xml, path_list, rolls, show)


def window(path: list[dict]) -> float:
    return max(abs(n["tilt"]) for n in path if n["supports"] == ["foot_sole"]) if path else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hip", type=float, nargs="+", default=[CAD_HIP, -100.0, -110.0, -115.0, -120.0, -130.0, -140.0],
                    help="deg, hip pitch backward limits (CAD: -85)")
    ap.add_argument("--trials", type=int, default=40, help="drops per floor / servo variant")
    ap.add_argument("--relax", type=float, default=3.0, help="s, legs back to `rest` before the slow catch")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--roll", action="store_true", help="roll onto the back first (searched rolls of each hip)")
    ap.add_argument("--search-roll", choices=("front", "left"), help="search the roll from POSE for each --hip, then exit")
    ap.add_argument("--generations", type=int, default=80)
    ap.add_argument("--video", metavar="MP4", help="film the first --hip lying on the back, front and side: roll + rise")
    ap.add_argument("--view", action="store_true", help="the same in the viewer, looping (macOS: mjpython)")
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    models = {h: model(h) for h in args.hip}
    if args.search_roll:
        for h, xml in models.items():
            search_roll(xml, args.search_roll, args.generations, args.workers, roll_file(args.search_roll, h))
        return 0
    path_lists = {h: paths(x) for h, x in models.items()}
    rolls = {h: {p: tuple(np.asarray(v) for v in json.loads(roll_file(p, h).read_text()).values())
                 for p in ("front", "left") if args.roll and roll_file(p, h).exists()} for h in args.hip}
    if args.video or args.view:
        h = args.hip[0]
        rolls_h = {p: tuple(np.asarray(v) for v in json.loads(roll_file(p, h).read_text()).values())
                   for p in ("front", "left") if roll_file(p, h).exists()}
        if args.view:
            view(models[h], path_lists[h], rolls_h)
        else:
            video(models[h], path_lists[h], rolls_h, Path(args.video))
        return 0
    tasks = [(str(models[h]), v, args.trials, args.relax, path_lists[h], rolls[h])
             for h in args.hip for v in range(len(getup_search.VARIANTS))]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(drops, tasks)

    results = {}
    print(f"\n{'hip':>6s} | window back / front | stands up after a drop (lying after relaxing: back, front, side)")
    for i, h in enumerate(args.hip):
        rows = [r for out in res[i * len(getup_search.VARIANTS):(i + 1) * len(getup_search.VARIANTS)] for r in out]
        count = {k: (sum(ok for lying, ok in rows if lying == k), sum(lying == k for lying, _ in rows))
                 for k in ("back", "front", "side", "up")}
        w = [window(p) for p in path_lists[h]]
        results[f"{h:g}"] = dict(window_back=w[0], window_front=w[1], all=sum(ok for _, ok in rows), n=len(rows),
                                 **{k: list(v) for k, v in count.items()})
        print(f"{h:+6.0f}° | {w[0]:5.0f}° / {w[1]:3.0f}°  | {results[f'{h:g}']['all']:3d}/{len(rows)}  "
              + "  ".join(f"{k} {a}/{b}" for k, (a, b) in count.items() if b), flush=True)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=1))
    print(f"-> {OUT_DIR / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
