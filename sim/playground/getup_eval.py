"""How often and how fast a get-up policy stands the robot up (plain MuJoCo, sim2sim).

    uv run python sim/playground/getup_eval.py build/checkpoints/cogu_getup_local
    uv run python sim/playground/getup_eval.py CKPT --trials 30 --xml build/mjcf/cogu_lower_body.xml

Each trial drops the robot from 35 cm in a random orientation and joint pose (like the training reset), lets
it settle for 1 s, then runs the policy for up to 10 s. Success = upright, at standing height, near `rest`
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
from play import GetupPolicy, Sim2Sim  # noqa: E402
from rollout import latest_checkpoint  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--walk", default=str(HERE / "checkpoints" / "cogu_walk_v5"),
                    help="walking checkpoint (only to build the simulator; it does not run)")
    ap.add_argument("--xml", default=str(cogu_getup.GETUP_XML))
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    sim = Sim2Sim(Path(args.xml), latest_checkpoint(Path(args.walk).resolve()))
    getup = GetupPolicy(sim, latest_checkpoint(Path(args.checkpoint).resolve()))
    cfg = getup.cfg
    rng = np.random.default_rng(args.seed)
    times, fails = [], 0
    for t in range(args.trials):
        mujoco.mj_resetData(sim.m, sim.d)
        quat = rng.normal(size=4)
        q = rng.uniform(sim.lowers, sim.uppers)
        sim.d.qpos[:] = np.concatenate([[0.0, 0.0, cfg.drop_height], quat / np.linalg.norm(quat), q])
        sim.d.ctrl[:] = q
        for _ in range(int(cfg.settle_time / sim.m.opt.timestep)):
            mujoco.mj_step(sim.m, sim.d)
        sim.d.qvel[:] = 0.0
        up0 = sim.d.site_xmat[sim.imu].reshape(3, 3)[2, 2]
        getup.start()
        done_at = None
        for i in range(int(args.seconds / sim.dt)):
            getup.step()
            if getup.ready():
                done_at = (i + 1) * sim.dt
                break
        if done_at is None:
            fails += 1
            up = sim.d.site_xmat[sim.imu].reshape(3, 3)[2, 2]
            print(f"  trial {t:2d}: started with up_z {up0:+.2f} -> not up after {args.seconds:.0f} s "
                  f"(up_z {up:+.2f}, pelvis {sim.d.qpos[2] * 1000:.0f} mm)")
        else:
            times.append(done_at)
    n = args.trials
    print(f"\n{Path(args.checkpoint).name}: stood up in {n - fails} of {n} drops"
          + (f", median {np.median(times):.1f} s, slowest {max(times):.1f} s" if times else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
