"""Get-up = an open-loop push onto the feet, then the walking policy catches the robot (plain MuJoCo).

    uv run python sim/playground/getup_handoff.py build/getup_search/back_land.json
    uv run python sim/playground/getup_handoff.py build/getup_search/back_land.json --walk sim/playground/checkpoints/cogu_walk_v7 --up 0.6 0.7 0.8

The sequence (from `getup_search.py --objective land`) throws the robot from lying onto its feet; the moment it is
upright enough (pelvis up_z >= --up) with only the soles on the floor, the walking policy takes over with a zero
command (`Sim2Sim.handover`, as `play.py --getup` does) and has to stop the swing. Success = still up and at
standing height 4 s later. Each sequence runs on the same friction / servo variants as the search.
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
from play import Sim2Sim  # noqa: E402
from rollout import latest_checkpoint  # noqa: E402


def attempt(sim: Sim2Sim, base: getup_search.Sim, pose: str, targets, durations, up_min: float,
            friction: float, force: float, after: float = 4.0, frame_cb=None):
    """-> (stood up, handover time or None, final up_z, final pelvis height)."""
    m, d = sim.m, sim.d
    m.geom_friction[:, 0] = base.m.geom_friction[:, 0]
    m.actuator_forcerange[:] = base.m.actuator_forcerange
    base.start(pose)                                   # lying, settled (cached per pose and variant)
    d.qpos[:], d.qvel[:] = base.d.qpos, base.d.qvel
    d.time = 0.0
    mujoco.mj_forward(m, d)
    soles = {m.geom("L_foot_sole").id, m.geom("R_foot_sole").id}
    floor = m.geom("floor").id
    handed = None
    for target, dur in zip(targets, durations):
        d.ctrl[:] = target
        for i in range(int(round(dur / m.opt.timestep))):
            mujoco.mj_step(m, d)
            if frame_cb is not None:
                frame_cb(m, d)
            if i % sim.n_substeps:
                continue
            touching = {g for c in d.contact[: d.ncon] for g in c.geom} - {floor}
            if d.site_xmat[sim.imu][8] >= up_min and touching and touching <= soles:
                handed = d.time
                break
        if handed is not None:
            break
    if handed is None:
        return False, None, float(d.site_xmat[sim.imu][8]), float(d.qpos[2])
    sim.command[:] = 0.0
    sim.handover()
    for _ in range(int(after / sim.dt)):
        sim.step()
        if frame_cb is not None:
            for _ in range(sim.n_substeps):           # one call per physics step, as in the open-loop phase
                frame_cb(m, d)
    up, height = float(d.site_xmat[sim.imu][8]), float(d.qpos[2])
    return up > getup_search.UPRIGHT_COS and height > getup_search.STAND_HEIGHT, handed, up, height


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequences", nargs="+")
    ap.add_argument("--walk", default=str(HERE / "checkpoints" / "cogu_walk_v7"))
    ap.add_argument("--xml", default=str(getup_search.GETUP_XML))
    ap.add_argument("--up", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9],
                    help="pelvis up_z at which the walking policy takes over")
    ap.add_argument("--video", action="store_true", help="record the first variant at the first --up value")
    args = ap.parse_args(argv)

    sim = Sim2Sim(Path(args.xml), latest_checkpoint(Path(args.walk).resolve()))
    variants = getup_search.VARIANTS + ((0.8, 0.85),)
    bases = [getup_search.Sim(Path(args.xml), f, s) for f, s in variants]
    for seq in args.sequences:
        spec = json.loads(Path(seq).read_text())
        targets, durations = np.array(spec["targets"]), np.array(spec["durations"])
        print(f"{Path(seq).name}: from {spec['pose']}, {len(durations)} keyframes, walker {Path(args.walk).name}")
        for up_min in args.up:
            cells = []
            for base, (f, s) in zip(bases, variants):
                ok, t, up, h = attempt(sim, base, spec["pose"], targets, durations, up_min, f, s)
                cells.append(("UP " if ok else "-- ") + (f"@{t:.2f}s" if t is not None else "never ") + f" {up:+.2f}")
            print(f"  hand over at up_z {up_min:.1f}:  " + " | ".join(cells))
        if args.video:
            import imageio.v2 as imageio
            renderer = mujoco.Renderer(sim.m, 480, 640)
            cam = mujoco.MjvCamera()
            cam.type, cam.distance, cam.elevation, cam.azimuth = mujoco.mjtCamera.mjCAMERA_FREE, 0.9, -15, 135
            frames, count = [], [0]

            def grab(m, d):
                count[0] += 1
                if count[0] % 17 == 0:
                    cam.lookat[:] = (d.qpos[0], d.qpos[1], 0.12)
                    renderer.update_scene(d, cam)
                    frames.append(renderer.render())

            attempt(sim, bases[0], spec["pose"], targets, durations, args.up[0], *variants[0], frame_cb=grab)
            out = Path(seq).with_name(Path(seq).stem + "_handoff.mp4")
            imageio.mimwrite(out, frames, fps=30, codec="libx264", quality=8, macro_block_size=None)
            print(f"video: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
