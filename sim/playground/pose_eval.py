"""Sit / lie-down sim2sim check: the pose policy (or the scripted reference alone) on plain MuJoCo.

    uv run python sim/playground/pose_eval.py sim/playground/checkpoints/cogu_pose_v1
    uv run python sim/playground/pose_eval.py script --xml build/mjcf/variants/full_hip_-120.xml   # reference alone
    uv run python sim/playground/pose_eval.py build/checkpoints/cogu_pose_v1 --trials 40 --delay 2 --push 0.6

Each trial, from standing in `rest` (walk v10 holding it first, as on the robot), runs the three moves the robot
will be asked for, with a random push (up to `--push` m/s, random direction) in the middle of each hold:
  1. sit down and hold 3 s, 2. stand up and get back to `rest` (ready for the walker), 3. sit again, then lie down.
Scored per move: sitting = leaning as the config's sit (35 ± 8°, pelvis 165 ± 20 mm) at the end of the hold; standing =
the policy says ready within 6 s; lying = on the back (tilt above 90°) without falling over on the way.
Friction (0.6–1.2), servo strength (85–100 %) and the latency (0 to `--delay` steps) are drawn per trial.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import play  # noqa: E402
from rollout import latest_checkpoint  # noqa: E402

WALK = HERE / "checkpoints" / "cogu_walk_v10"


def trial(xml: Path, pose_ckpt: Path | None, rng: np.random.Generator, max_delay: int, push: float) -> dict:
    sim = play.Sim2Sim(xml, latest_checkpoint(WALK), delay_steps=int(rng.integers(0, max_delay + 1)))
    sim.m.geom_friction[sim.m.geom("floor").id, 0] = rng.uniform(0.6, 1.2)
    strength = rng.uniform(0.85, 1.0)
    sim.m.actuator_forcerange[:] *= strength
    pose = play.PosePolicy(sim, pose_ckpt, seed=int(rng.integers(1 << 30)))
    for _ in range(50):
        sim.step()

    def run(seconds: float, push_at: float | None = None) -> bool:
        for i in range(int(seconds / sim.dt)):
            if push_at is not None and i == int(push_at / sim.dt) and push > 0:
                theta = rng.uniform(0, 2 * np.pi)
                sim.d.qvel[:2] += rng.uniform(0.0, push) * np.array([np.cos(theta), np.sin(theta)])
            pose.step()
            if pose.fallen():
                return False
        return True

    def sitting() -> bool:
        return (abs(pose.tilt() - pose.cfg.sit.pelvis_tilt) < 8.0 and abs(sim.d.qpos[2] - pose.cfg.sit.height) < 0.02)

    out = dict(sit=False, stand=False, lie=False)
    pose.start(1)
    out["sit"] = run(2.0 + 3.0, push_at=3.5) and sitting()
    if not out["sit"]:
        return out
    pose.mode = 0
    for _ in range(int(6.0 / sim.dt)):
        pose.step()
        if pose.ready() or pose.fallen():
            break
    out["stand"] = pose.ready()
    if not out["stand"]:
        return out
    sim.handover()
    for _ in range(25):
        sim.step()
    pose.start(1)
    if not run(3.0):
        return out
    pose.mode = 2
    run(5.0)
    out["lie"] = pose.tilt() > 90.0
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="pose checkpoint (step dir or run folder), or 'script' for the reference alone")
    ap.add_argument("--xml", default=str(play.cogu_pose.POSE_XML))
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--delay", type=int, default=2, help="max actuator latency, 20 ms steps (training: 2)")
    ap.add_argument("--push", type=float, default=0.4, help="max push in each hold, m/s (training: 0.5)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    ckpt = None if args.checkpoint == "script" else latest_checkpoint(Path(args.checkpoint).resolve())
    rng = np.random.default_rng(args.seed)
    results = [trial(Path(args.xml), ckpt, rng, args.delay, args.push) for _ in range(args.trials)]
    n = len(results)
    label = "scripted reference" if ckpt is None else play.checkpoint_label(ckpt)
    print(f"{label} on {Path(args.xml).name}, {n} trials (push ≤ {args.push} m/s, delay ≤ {args.delay}): "
          + "  ".join(f"{k} {sum(r[k] for r in results)}/{n}" for k in ("sit", "stand", "lie")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
