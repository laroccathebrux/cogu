"""How hard a push a policy survives, standing and walking (plain MuJoCo, like play.py).

    uv run python sim/playground/push_recovery.py sim/playground/checkpoints/cogu_walk_v5
    uv run python sim/playground/push_recovery.py CKPT_A CKPT_B

The push is the one training uses: a sudden change of the body's horizontal velocity (m/s), here in the
robot's frame, forward / backward / left / right, after 2 s of standing (zero command) or walking at
0.15 m/s. Each cell shows whether the robot was still up 3 s later and the widest the feet got apart
(their horizontal distance; 55 mm is the stance width at rest). A robot that recovers with a step shows
a wider spread; one that only leans keeps ~55 mm and falls sooner.
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

DIRECTIONS = {"forward": (1, 0), "backward": (-1, 0), "left": (0, 1), "right": (0, -1)}
MAGNITUDES = (0.2, 0.3, 0.4, 0.5, 0.6)


def push(sim: Sim2Sim, vx: float, direction, magnitude: float, settle: float = 2.0, after: float = 3.0):
    """(survived, widest foot spread in m)."""
    feet = [sim.m.site(n).id for n in ("L_foot_site", "R_foot_site")]
    sim.reset()
    sim.command[:] = (vx, 0.0, 0.0)
    for _ in range(int(settle / sim.dt)):
        sim.step()
    if sim.fallen():
        return False, 0.0
    r = sim.d.xmat[1].reshape(3, 3)
    yaw = np.arctan2(r[1, 0], r[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    kx, ky = direction
    sim.d.qvel[0] += magnitude * (c * kx - s * ky)
    sim.d.qvel[1] += magnitude * (s * kx + c * ky)
    spread = 0.0
    for _ in range(int(after / sim.dt)):
        sim.step()
        p = sim.d.site_xpos[feet]
        spread = max(spread, float(np.linalg.norm(p[0, :2] - p[1, :2])))
        if sim.fallen():
            return False, spread
    return True, spread


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--xml", default=str(cogu_env.DEFAULT_XML))
    args = ap.parse_args(argv)
    for c in args.checkpoints:
        ckpt = latest_checkpoint(Path(c).resolve())
        sim = Sim2Sim(Path(args.xml), ckpt)
        print(f"\n{checkpoint_label(ckpt)} — push (m/s): up or FELL, widest foot spread (mm)")
        survived = 0
        for vx, label in ((0.0, "standing"), (0.15, "walking 0.15 m/s")):
            print(f"  {label}")
            for name, d in DIRECTIONS.items():
                cells = []
                for mag in MAGNITUDES:
                    ok, spread = push(sim, vx, d, mag)
                    survived += ok
                    cells.append(f"{mag:.1f} {'up  ' if ok else 'FELL'} {spread * 1000:3.0f}")
                print(f"    {name:8s} " + " | ".join(cells))
        print(f"  survived {survived} of {2 * len(DIRECTIONS) * len(MAGNITUDES)} pushes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
