"""How often and how fast a get-up policy stands the robot up (plain MuJoCo, sim2sim).

    uv run python sim/playground/getup_eval.py build/checkpoints/cogu_getup_local
    uv run python sim/playground/getup_eval.py CKPT --trials 30 --xml build/mjcf/cogu_lower_body.xml
    uv run python sim/playground/getup_eval.py CKPT --starts feet throw drop --xml build/mjcf/variants/heel_45.xml
    uv run mjpython sim/playground/getup_eval.py CKPT --starts feet throw drop --view     # watch the trials, looping
    uv run python sim/playground/getup_eval.py CKPT --starts drop --trials 4 --video docs/renders/getup.mp4

`drop` trials drop the robot from 35 cm in a random orientation and joint pose (like the training reset), let it
settle for 1 s, then run the policy for up to 10 s. `feet` and `throw` trials start from the `getup_starts.py`
table of the --xml model (on the feet but tilted; mid-throw), as the training does. Success = upright, at standing height, near `rest`
and still for 0.5 s (the moment play.py hands control back to the walking policy). The default model is the
get-up MJX variant; the full model (`cogu_lower_body.xml`, mesh contacts everywhere) is a harder sim2sim test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import cogu_getup  # noqa: E402
import getup_starts  # noqa: E402
from play import GetupPolicy, Sim2Sim  # noqa: E402
from rollout import latest_checkpoint  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--walk", default=str(HERE / "checkpoints" / "cogu_walk_v10"),
                    help="walking checkpoint (only to build the simulator; it does not run)")
    ap.add_argument("--xml", default=str(cogu_getup.GETUP_XML))
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--starts", nargs="+", choices=("drop", "feet", "throw"), default=["drop"])
    ap.add_argument("--verbose", action="store_true", help="one line per failed trial")
    ap.add_argument("--table", default=None, help="getup_starts table (default: the one of --xml)")
    ap.add_argument("--rolls", metavar="DIR", help="play roll_front.json / roll_side.json from DIR instead of the "
                    "checkpoint's rolls (e.g. sim/playground/getup_sequences after a roll_search.py)")
    ap.add_argument("--scripted-residual", choices=("on", "off"), default=None,
                    help="override the checkpoint's: does the policy correct the relax and roll scripts "
                    "(v5, v6 on; v7 off)")
    ap.add_argument("--view", action="store_true", help="play the trials in the viewer, in real time, looping "
                    "(macOS: run with mjpython)")
    ap.add_argument("--video", metavar="MP4", help="film --trials trials of each start kind instead (1 s after up)")
    args = ap.parse_args(argv)

    sim = Sim2Sim(Path(args.xml), latest_checkpoint(Path(args.walk).resolve()))
    getup = GetupPolicy(sim, latest_checkpoint(Path(args.checkpoint).resolve()), rolls=args.rolls)
    cfg = getup.cfg
    if args.scripted_residual:
        cfg.reference.scripted_residual = args.scripted_residual == "on"
    rng = np.random.default_rng(args.seed)
    table = None
    if {"feet", "throw"} & set(args.starts):
        table = np.load(args.table or getup_starts.default_path(args.xml))

    def set_start(kind: str) -> tuple[float, int]:
        """-> where the reference starts: (throw_t, throw_side) of a `throw` state, else (-1, 0)."""
        mujoco.mj_resetData(sim.m, sim.d)
        if kind == "drop":
            quat = rng.normal(size=4)
            q = rng.uniform(sim.lowers, sim.uppers)
            sim.d.qpos[:] = np.concatenate([[0.0, 0.0, 0.35], quat / np.linalg.norm(quat), q])
            sim.d.ctrl[:] = q
            for _ in range(int(cfg.settle_time / sim.m.opt.timestep)):
                mujoco.mj_step(sim.m, sim.d)
            sim.d.qvel[:] = 0.0
            return -1.0, 0
        else:
            i = rng.choice(np.flatnonzero(table["kind"] == ("feet", "throw").index(kind)))
            sim.d.qpos[:], sim.d.qvel[:] = table["qpos"][i], table["qvel"][i]
            sim.d.ctrl[:] = sim.d.qpos[7:]
            mujoco.mj_forward(sim.m, sim.d)
            if "throw_t" in table:
                return float(table["throw_t"][i]), int(table["throw_side"][i])
            return -1.0, 0

    if args.view:
        return view(sim, getup, set_start, args)
    if args.video:
        return video(sim, getup, set_start, args)
    for kind in args.starts:
        times, fails = [], 0
        for t in range(args.trials):
            phase = set_start(kind)
            up0 = sim.d.site_xmat[sim.imu].reshape(3, 3)[2, 2]
            getup.start(*phase)
            done_at = None
            for i in range(int(args.seconds / sim.dt)):
                getup.step()
                if getup.ready():
                    done_at = (i + 1) * sim.dt
                    break
            if done_at is None:
                fails += 1
                up = sim.d.site_xmat[sim.imu].reshape(3, 3)[2, 2]
                if args.verbose:
                    print(f"  {kind} {t:2d}: started with up_z {up0:+.2f} -> not up after {args.seconds:.0f} s "
                          f"(up_z {up:+.2f}, pelvis {sim.d.qpos[2] * 1000:.0f} mm)")
            else:
                times.append(done_at)
        n = args.trials
        print(f"{Path(args.checkpoint).name} [{Path(args.xml).stem}] {kind:5s}: stood up in {n - fails} of {n}"
              + (f", median {np.median(times):.1f} s, slowest {max(times):.1f} s" if times else ""), flush=True)
    return 0


def video(sim, getup, set_start, args) -> int:
    """The trials of each start kind, one after the other, filmed at 30 fps (each ends 1 s after standing up)."""
    import imageio.v2 as imageio
    renderer = mujoco.Renderer(sim.m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type, cam.distance, cam.elevation, cam.azimuth = mujoco.mjtCamera.mjCAMERA_FREE, 0.9, -15, 135
    every = max(1, int(round(1 / (30 * sim.dt))))
    frames = []
    for kind in args.starts:
        for t in range(args.trials):
            getup.start(*set_start(kind))
            t_end = None
            for i in range(int(args.seconds / sim.dt)):
                getup.step()
                if t_end is None and getup.ready():
                    t_end = i
                if i % every == 0:
                    cam.lookat[:] = (sim.d.qpos[0], sim.d.qpos[1], 0.12)
                    renderer.update_scene(sim.d, cam)
                    frames.append(renderer.render())
                if t_end is not None and i - t_end > int(1.0 / sim.dt):
                    break
            print(f"  {kind} {t}: {'up in %.1f s' % ((t_end + 1) * sim.dt) if t_end is not None else 'not up'}", flush=True)
    Path(args.video).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.video, frames, fps=30, codec="libx264", quality=8, macro_block_size=None)
    print(f"video: {args.video}")
    return 0


def view(sim, getup, set_start, args) -> int:
    """Each start kind in turn, one trial after the other, until the window closes."""
    import time

    import mujoco.viewer
    label = f"{Path(args.checkpoint).resolve().name} [{Path(args.xml).stem}]"
    with mujoco.viewer.launch_passive(sim.m, sim.d) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = sim.m.body("pelvis").id
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.0, 135.0, -15.0
        tally = {k: [0, 0] for k in args.starts}
        while v.is_running():
            for kind in args.starts:
                with v.lock():
                    getup.start(*set_start(kind))
                result = "trying"
                t_end = None
                for i in range(int(args.seconds / sim.dt)):
                    t0 = time.perf_counter()
                    if not v.is_running():
                        return 0
                    with v.lock():
                        getup.step()
                    if getup.ready():
                        result = f"UP in {(i + 1) * sim.dt:.1f} s"
                        t_end = i
                    score = "  ".join(f"{k} {up}/{n}" for k, (up, n) in tally.items())
                    v.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "policy\nstart\ntime\nresult\nso far",
                                 f"{label}\n{kind}\n{(i + 1) * sim.dt:4.1f} s\n{result}\n{score}"))
                    v.sync()
                    sleep = sim.dt - (time.perf_counter() - t0)
                    if sleep > 0:
                        time.sleep(sleep)
                    if t_end is not None and i - t_end > 50:      # show 1 s of standing, then the next trial
                        break
                tally[kind][0] += t_end is not None
                tally[kind][1] += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
