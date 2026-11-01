"""Which foot? Bigger soles in simulation: the landing window, a scripted catch, and the walking policy (plain MuJoCo).

    uv run python sim/playground/foot_study.py                          # the default set of feet
    uv run python sim/playground/foot_study.py --feet 0,0,0 45,20,0 --walk sim/playground/checkpoints/cogu_walk_v7

Getting up fails at the catch: the poses where the feet alone hold the tilted robot have ~5 mm of margin (the old
65 × 52 mm sole under a centre of mass 191 mm up), so a landing with any speed tips over (README, "Força, pescoço,
joelhos e pegada roteirizada"). Each foot is HEEL,TOE,WIDTH in mm added to the sole box (`variants.py`; heel
backwards, toe forwards, width outwards), on the get-up model for the get-up measures and on the walking model for the
walking ones. Since 2026-11-01 the CAD foot is the 45,20,15 sole of this study, so the feet now add to that one:

- window: the most tilted pose, face up / face down, where the feet alone hold the robot and a slow path leads to
  `rest` (`getup_static.py`);
- catch: from landings on the feet (the `getup_starts.py` feet states of that foot: 15° up to its window, random
  heading, uneven legs) the scripted catch of `getup_reference.py` (walk the static path by feel, 1 rad/s, 10° ahead)
  for 5 s; % that end upright at standing height and still, on the nominal model and with slippery floor + weak servos;
- walking: the walking policy (not retrained) for 8 s at 0.15 and 0.25 m/s and turning at 0.8 rad/s (distance, or
  the time it fell; turning: degrees turned), and `push_recovery.py`'s 40 pushes.

Results -> build/foot_study/results.json and a table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import cogu_env  # noqa: E402
import getup_reference  # noqa: E402
import getup_search  # noqa: E402
import getup_starts  # noqa: E402
import getup_static  # noqa: E402
import push_recovery  # noqa: E402
import variants  # noqa: E402
from play import Sim2Sim  # noqa: E402
from rollout import latest_checkpoint  # noqa: E402

OUT_DIR = getup_search.REPO_ROOT / "build" / "foot_study"
FEET = ("0,0,0", "45,0,0", "30,20,0", "45,20,0", "60,20,0", "45,40,0", "45,20,15", "60,40,15")
CATCH_VARIANTS = ((1.0, 1.0), (0.6, 0.85))   # (friction, servo) scales
CATCH_SECONDS = 5.0


def catch_rate(xml: Path, paths: list, states: list, friction: float, force: float) -> tuple[float, float, float]:
    """-> (% up overall, face up, face down)."""
    sim = getup_search.Sim(xml, friction, force)
    m, d = sim.m, sim.d
    pelvis = m.body("pelvis").id
    seq = dict(pose="back", targets=[[0.0] * m.nu], durations=[0.0])
    every = int(round(getup_reference.CTRL_DT / sim.dt))
    ok, side = [], []
    for qpos, qvel, _, _ in states:
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:] = qpos, qvel
        mujoco.mj_forward(m, d)
        ref = getup_reference.Reference(seq, paths, sim.rest_ctrl, lead=10.0, up_min=0.0, rate=1.0)
        ref.caught, ref.prev = True, d.qpos[7:].copy()
        side.append(getup_reference.tilt_deg(d, pelvis) > 0)
        for _ in range(int(CATCH_SECONDS / getup_reference.CTRL_DT)):
            d.ctrl[:] = np.clip(ref.targets(0.0, getup_reference.tilt_deg(d, pelvis), sim.up(), True), sim.lo, sim.hi)
            for _ in range(every):
                mujoco.mj_step(m, d)
        ok.append(sim.up() > getup_search.UPRIGHT_COS and d.qpos[2] > getup_search.STAND_HEIGHT
                  and np.linalg.norm(d.qvel[:6]) < 0.3)
    ok, side = np.array(ok), np.array(side)
    pct = lambda a: float(100 * a.mean()) if a.size else float("nan")
    return pct(ok), pct(ok[side]), pct(ok[~side])


def walk(xml: Path, ckpt: Path) -> dict:
    sim = Sim2Sim(xml, ckpt)
    out = {}
    for label, cmd in (("0.15", (0.15, 0.0, 0.0)), ("0.25", (0.25, 0.0, 0.0)), ("turn", (0.0, 0.0, 0.8))):
        sim.reset()
        sim.command[:] = cmd
        x0 = sim.d.qpos[:2].copy()
        yaw, last, fell = 0.0, None, None
        for i in range(int(8.0 / sim.dt)):
            sim.step()
            r = sim.d.xmat[1]
            heading = np.arctan2(r[3], r[0])
            if last is not None:
                yaw += (heading - last + np.pi) % (2 * np.pi) - np.pi
            last = heading
            if sim.fallen():
                fell = (i + 1) * sim.dt
                break
        if fell:
            out[label] = f"fell {fell:.1f} s"
        elif label == "turn":
            out[label] = f"{np.degrees(yaw):.0f}°"
        else:
            out[label] = f"{np.linalg.norm(sim.d.qpos[:2] - x0):.2f} m"
    survived = 0
    for vx in (0.0, 0.15):
        for d in push_recovery.DIRECTIONS.values():
            for mag in push_recovery.MAGNITUDES:
                survived += push_recovery.push(sim, vx, d, mag)[0]
    out["pushes"] = survived
    return out


def study(foot: str, ckpt: Path, rng: np.random.Generator) -> dict:
    heel, toe, width = (float(v) for v in foot.split(","))
    t0 = time.time()
    getup_xml = variants.build(heel=heel, toe=toe, width=width)
    walk_xml = variants.build(heel=heel, toe=toe, width=width, src=cogu_env.DEFAULT_XML,
                              out=variants.OUT_DIR / f"walk_{getup_xml.stem}.xml")
    maps = {side: getup_static.stable_map(str(getup_xml), side) for side in ("back", "front")}   # to 115°, as the README
    windows, paths = {}, []
    for side, (_, stable, nxt) in maps.items():
        w = getup_static.catch_window(stable, nxt)
        windows[side] = abs(w[0]) if w else 0.0
        if w:
            paths.append(getup_static.path_from(w, stable, nxt))
    states = getup_starts.feet_states(str(getup_xml), rng, maps)
    pick = rng.choice(len(states), size=min(80, len(states)), replace=False)
    states = [states[i] for i in pick]
    catch = {f"{f:g}/{s:g}": catch_rate(getup_xml, paths, states, f, s) for f, s in CATCH_VARIANTS}
    row = dict(foot=foot, length=65 + heel + toe, width=52 + width, window=windows, catch=catch,
               walk=walk(walk_xml, ckpt), seconds=round(time.time() - t0))
    print(json.dumps(row), flush=True)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feet", nargs="+", default=list(FEET), help="HEEL,TOE,WIDTH in mm")
    ap.add_argument("--walk", default=str(HERE / "checkpoints" / "cogu_walk_v7"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    ckpt = latest_checkpoint(Path(args.walk).resolve())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [study(f, ckpt, np.random.default_rng(args.seed)) for f in args.feet]
    (OUT_DIR / "results.json").write_text(json.dumps(rows, indent=1))
    print("\n| Foot (heel, toe, width mm) | Sole (mm) | Window up / down | Catch nominal (up / down) | Catch slippery+weak "
          "| Walk 0.15 / 0.25 / turn | Pushes |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        (n, nu, nd), (s, _, _) = r["catch"]["1/1"], r["catch"]["0.6/0.85"]
        w = r["walk"]
        print(f"| {r['foot']} | {r['length']:g} × {r['width']:g} | {r['window']['back']:.0f}° / {r['window']['front']:.0f}° "
              f"| {n:.0f} % ({nu:.0f} / {nd:.0f}) | {s:.0f} % | {w['0.15']} / {w['0.25']} / {w['turn']} | {w['pushes']}/40 |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
