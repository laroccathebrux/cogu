"""Is there a slow (quasi-static) way up? Map the poses where Cogu rests in balance and search a path through them.

    uv run python sim/playground/getup_static.py                                  # current model
    uv run python sim/playground/getup_static.py --xml build/mjcf/variants/heel_30.xml --out build/getup_search/path_heel30.json

The robot is reduced to its sagittal plane (both legs alike, hip yaw and roll 0): a node is a body tilt (0 =
upright, + = leaning back / face up, - = face down) and a hip-pitch / knee / ankle triple on a grid. The robot is
lowered onto the floor and everything within 2 mm of it is a support; the node is *stable* when the centre of mass
lies over the supports (between the rearmost and foremost support points, 5 mm inside) and no joint needs more
than `--max-torque` to hold it (each leg carries the share of the weight its foot takes, levers measured
horizontally). Two nodes are neighbours when the tilt and every joint differ by at most one grid step.

A path of neighbouring stable nodes from a lying tilt to upright in `rest` is a get-up that can be done as slowly
as wanted: every intermediate pose holds by itself. No path means getting up needs a dynamic throw somewhere.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
GETUP_XML = REPO_ROOT / "build" / "mjcf" / "cogu_lower_body_mjx_getup.xml"
JOINTS = ("hip_pitch", "knee", "ankle")
TOUCH = 0.002                  # m: within this of the lowest point counts as touching
INSIDE = 0.005                 # m: centre of mass this far inside the supports

_M = _D = None


def _init(xml: str):
    global _M, _D
    _M = mujoco.MjModel.from_xml_path(xml)
    _D = mujoco.MjData(_M)


def _low_points(m, d, i):
    """Points of collision geom i near its lowest point (box corners, cylinder rim, ellipsoid bottom)."""
    r, c, s = d.geom_xmat[i].reshape(3, 3), d.geom_xpos[i], m.geom_size[i]
    t = m.geom_type[i]
    if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        n = np.sqrt(np.sum((s * r[2]) ** 2))
        return [c - r @ (s ** 2 * r[2]) / n]
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return [c + r @ np.array([s[0] * np.cos(a), s[0] * np.sin(a), h * s[1]])
                for h in (-1, 1) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
    return [c + r @ (np.array(sg) * s[:3]) for sg in itertools.product((-1, 1), repeat=3)]


def _evaluate(task):
    """(tilt_deg, pitch, knee, ankle in deg) -> None or (supports, max torque, foot load share, pelvis height)."""
    tilt, p, k, a = task
    m, d = _M, _D
    q = np.radians([0.0, 0.0, p, k, a])
    beta = -np.radians(tilt)                           # + tilt = leaning back (rotation about -y)
    d.qpos[:] = 0.0
    d.qpos[3:7] = (np.cos(beta / 2), 0.0, np.sin(beta / 2), 0.0)
    d.qpos[7:] = np.concatenate([q, q])
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    pts = []
    for i in range(m.ngeom):
        if m.geom(i).name == "floor" or not (m.geom_contype[i] or m.geom_conaffinity[i]):
            continue
        pts += [(i, p_) for p_ in _low_points(m, d, i)]
    floor = min(p_[2] for _, p_ in pts)
    support = [(i, p_) for i, p_ in pts if p_[2] < floor + TOUCH]
    xs = [p_[0] for _, p_ in support]
    com = d.subtree_com[1][0]
    if not (min(xs) + INSIDE < com < max(xs) - INSIDE):
        return None
    soles = {m.geom("L_foot_sole").id, m.geom("R_foot_sole").id}
    feet = [p_[0] for i, p_ in support if i in soles]
    others = [p_[0] for i, p_ in support if i not in soles]
    weight = m.body_subtreemass[1] * 9.81
    if not feet:
        load, foot_x = 0.0, 0.0
    elif not others:
        load, foot_x = 1.0, com
    else:
        foot_x, other_x = float(np.mean(feet)), float(np.mean(others))
        load = float(np.clip((com - other_x) / (foot_x - other_x), 0.0, 1.0)) if abs(foot_x - other_x) > 1e-3 else 0.5
    tau = 0.0
    for j in JOINTS:
        x = d.xanchor[m.joint("L_" + j).id][0]
        tau = max(tau, load * weight / 2 * abs(foot_x - x))
    names = sorted({m.geom(i).name.replace("L_", "").replace("R_", "") for i, _ in support})
    return names, tau, load, -floor


def stable_map(xml: str, side: str = "back", tilt_step: float = 5.0, joint_step: float = 5.0,
               max_torque: float = 1.2, max_tilt: float = 115.0):
    """-> (tilts, stable {(tilt, pitch, knee, ankle): (supports, torque, foot load, pelvis height)},
    nxt {pose: next pose towards upright near `rest`, None at the goal} for the poses a slow path connects)."""
    m = mujoco.MjModel.from_xml_path(xml)
    rng = {j: np.degrees(m.jnt_range[m.joint("L_" + j).id]) for j in JOINTS}
    axes = [np.arange(np.ceil(lo / joint_step) * joint_step, hi + 1e-6, joint_step)
            for lo, hi in (rng[j] for j in JOINTS)]
    sign = 1 if side == "back" else -1
    tilts = sign * np.arange(0.0, max_tilt + 1e-6, tilt_step)
    tasks = [(float(t), *map(float, q)) for t in tilts for q in itertools.product(*axes)]
    print(f"{Path(xml).name}: {len(tasks)} poses, lying on the {side}", flush=True)
    with mp.get_context("spawn").Pool(max(1, mp.cpu_count() - 1), _init, (xml,)) as pool:
        results = pool.map(_evaluate, tasks, chunksize=2000)
    stable = {t[:4]: r for t, r in zip(tasks, results) if r is not None and r[1] <= max_torque}

    # breadth-first search from upright near `rest` through neighbouring stable poses (the graph is undirected)
    ts, js = tilt_step, joint_step
    rest = np.array([-35.0, 70.0, -35.0])
    goal = [key for key in stable if key[0] == 0 and np.all(np.abs(np.array(key[1:]) - rest) <= 2 * js)]
    nxt = {key: None for key in goal}                # pose -> next pose towards upright
    queue = deque(goal)
    while queue:
        key = queue.popleft()
        t, p, k, a = key
        for dt, dp, dk, da in itertools.product((-ts, 0, ts), (-js, 0, js), (-js, 0, js), (-js, 0, js)):
            n = (t + dt, p + dp, k + dk, a + da)
            if n in stable and n not in nxt:
                nxt[n] = key
                queue.append(n)
    return tilts, stable, nxt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", default=str(GETUP_XML))
    ap.add_argument("--tilt-step", type=float, default=5.0, help="deg (2.5 leaves few flat-foot poses: grid artifact)")
    ap.add_argument("--joint-step", type=float, default=5.0)
    ap.add_argument("--max-torque", type=float, default=1.2, help="N·m per joint (stall is 1.9)")
    ap.add_argument("--side", choices=("back", "front"), default="back", help="lying face up or face down")
    ap.add_argument("--out", help="write the path (tilt, joint targets) as JSON")
    args = ap.parse_args(argv)

    tilts, stable, nxt = stable_map(args.xml, args.side, args.tilt_step, args.joint_step, args.max_torque)

    print("stable poses by tilt (supports of the easiest one):")
    for t in tilts:
        here = [(key, r) for key, r in stable.items() if key[0] == t]
        if here:
            kinds = sorted({"+".join(r[0]) for _, r in here})
            print(f"  {t:+6.1f}°  {len(here):5d}  " + ", ".join(k.replace("_hull", "").replace("foot_sole", "feet")
                                                           for k in kinds[:4]))
        else:
            print(f"  {t:+6.1f}°      0")

    on_feet = [key for key in nxt if stable[key][0] == ["foot_sole"]]
    window = max(on_feet, key=lambda key: (abs(key[0]), -stable[key][1]), default=None)
    lying = [key for key in nxt if abs(key[0]) >= 85 and "bubble_hull" in stable[key][0]]
    if window is not None:
        print(f"\ncatch window: standing on the feet alone at up to {abs(window[0]):.1f}°, a slow path leads to "
              f"`rest` (pitch/knee/ankle {window[1]:+.0f} {window[2]:+.0f} {window[3]:+.0f})")
    if lying:
        start = max(lying, key=lambda key: abs(key[0]))
        print(f"quasi-static path from lying ({abs(start[0]):.1f}°, on the bubble) to standing: getting up needs no throw")
    else:
        print("no quasi-static path from lying: getting onto the feet needs a dynamic push"
              + (f" that lands on them at <= {abs(window[0]):.1f}°" if window else ""))
        start = window
    if start is None:
        return 1
    path = []
    while start is not None:
        path.append(start)
        start = nxt[start]
    print(f"path: {len(path)} poses")
    for key in path[:: max(1, len(path) // 25)] + [path[-1]]:
        r = stable[key]
        print(f"  tilt {key[0]:+6.1f}°  pitch/knee/ankle {key[1]:+5.0f} {key[2]:+5.0f} {key[3]:+5.0f}  "
              f"{'+'.join(r[0]):40s} feet carry {r[2] * 100:3.0f} %  {r[1]:.2f} N·m  pelvis {r[3] * 1000:.0f} mm")
    if args.out:
        Path(args.out).write_text(json.dumps([dict(tilt=key[0], q=key[1:], supports=stable[key][0]) for key in path]))
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
