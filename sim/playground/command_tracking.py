"""How well a policy follows each kind of command (plain MuJoCo, like play.py).

    uv run python sim/playground/command_tracking.py sim/playground/checkpoints/cogu_walk_v4
    uv run python sim/playground/command_tracking.py CKPT_A CKPT_B --seconds 10

For each command, an 8 s run from the `rest` pose (the first 2 s are skipped) reporting the body-frame
velocity actually reached (vx, vy forward / left in m/s, wz yaw rate in rad/s) and, on the zero command,
the share of time the feet spend off the floor (a robot that stands still keeps both feet down).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import cogu_env  # noqa: E402
from play import Sim2Sim, checkpoint_label  # noqa: E402
from rollout import latest_checkpoint  # noqa: E402

COMMANDS = [(0.15, 0, 0), (0.25, 0, 0), (0.30, 0, 0), (-0.10, 0, 0), (0, 0.10, 0), (0, -0.10, 0),
            (0, 0, 0.6), (0, 0, -0.6), (0, 0, 1.0), (0.10, 0, 0.4), (0, 0, 0)]


def _yaw(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def track(sim: Sim2Sim, cmd, seconds: float, skip: float = 2.0) -> dict:
    sim.reset()
    sim.command[:] = cmd
    m, d = sim.m, sim.d
    found = [m.sensor_adr[m.sensor(f"{s}_foot_sole_floor_found").id] for s in ("L", "R")]
    vel, wz, air = [], [], []
    for i in range(int(seconds / sim.dt)):
        sim.step()
        if sim.fallen():
            return {"fell": (i + 1) * sim.dt}
        if i * sim.dt < skip:
            continue
        c, s = np.cos(_yaw(d.qpos[3:7])), np.sin(_yaw(d.qpos[3:7]))
        v = d.qvel[:2]
        vel.append([c * v[0] + s * v[1], -s * v[0] + c * v[1]])
        wz.append(d.qvel[5])
        air.append(d.sensordata[found] <= 0)
    vx, vy = np.mean(vel, axis=0)
    return {"vx": vx, "vy": vy, "wz": float(np.mean(wz)), "air": 100 * float(np.mean(air))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--xml", default=str(cogu_env.DEFAULT_XML))
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args(argv)
    for c in args.checkpoints:
        ckpt = latest_checkpoint(Path(c).resolve())
        sim = Sim2Sim(Path(args.xml), ckpt)
        print(f"\n{checkpoint_label(ckpt)}")
        print(f"  {'command (vx, vy, wz)':22s} {'vx':>7s} {'vy':>7s} {'wz':>7s}")
        for cmd in COMMANDS:
            r = track(sim, cmd, args.seconds)
            label = str(tuple(float(x) for x in cmd))
            if "fell" in r:
                print(f"  {label:22s}  fell after {r['fell']:.1f} s")
                continue
            extra = f"   feet off the floor {r['air']:.0f} % of the time" if not any(cmd) else ""
            print(f"  {label:22s} {r['vx']:7.3f} {r['vy']:7.3f} {r['wz']:7.3f}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
