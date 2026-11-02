"""Start states for the get-up training: on the feet but tilted, and in the middle of a throw (plain MuJoCo, CPU).

    uv run python sim/playground/getup_starts.py                                         # -> build/getup_starts/<xml stem>.npz
    uv run python sim/playground/getup_starts.py --xml build/mjcf/variants/heel_45.xml

Getting up from lying needs a throw onto the feet and then a catch (README, "Calcanhar e impulso + equilíbrio"). A
policy that only starts lying never sees a landing, so the get-up env also starts episodes from these states:

- `feet` (kind 0): the poses of the quasi-static map (`getup_static.stable_map`) where the feet alone hold the robot
  and a slow path leads to `rest`, from 15° up to the edge of that window, face up and face down, as many at each
  tilt. Each is set down on the floor at a random heading, with the legs a little uneven, held for 0.1 s and kept
  if still on the soles only.
- `throw` (kind 1): snapshots every 20 ms of the env's scripted throws (`getup_sequences/throw_back.json`,
  `throw_front.json`, from `getup_reference.py --search --full`), played on this model with the search's friction /
  servo variants until the catch takes over, while the pelvis is less than ~73° from vertical (up_z > 0.3): rising
  or in the air. Each keeps its moment in the throw (`throw_t`, s) and side (`throw_side`, 0 back / 1 front), where
  the env's reference starts. The lying part before is what the env's drops give. (Before 2026-11-02: snapshots of
  all the open-loop throws of `getup_search.py` in `getup_sequences/`.)

The env samples them (`start_mix`) and adds noise to the velocities.

The file also holds the catch path (`catch_tilt` deg, `catch_q` hip pitch / knee / ankle deg): the slow paths from
the edge of the catch window to `rest`, face up (+ tilt) and face down (- tilt), joined through upright. The env's
reference (`cogu_getup`, "reference") walks it by feel, as `getup_reference.py` does.
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
import getup_static  # noqa: E402

REPO_ROOT = HERE.parents[1]
OUT_DIR = REPO_ROOT / "build" / "getup_starts"
SEQUENCES = HERE / "getup_sequences"
THROWS = {"back": SEQUENCES / "throw_back.json", "front": SEQUENCES / "throw_front.json"}   # the env's throws
MIN_TILT = 15.0                # deg: below this the tilted-`rest` starts already cover it
PER_SIDE = 800                 # feet poses kept per side
THROW_UP = 0.3                 # up_z above which a throw snapshot is kept
THROW_EVERY = 0.02             # s
REST_LEGS = (-35.0, 70.0, -35.0)   # deg, hip pitch / knee / ankle of the `rest` keyframe


def default_path(xml: str | Path) -> Path:
    return OUT_DIR / f"{Path(xml).stem}.npz"


def feet_states(xml: str, rng: np.random.Generator, maps: dict | None = None) -> list[tuple]:
    """`maps`: {side: getup_static.stable_map(xml, side, max_tilt=90.0)} if already computed."""
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    sim = getup_search.Sim(Path(xml))                 # for _place_on_floor
    floor, soles = m.geom("floor").id, {m.geom("L_foot_sole").id, m.geom("R_foot_sole").id}
    out = []
    for side in ("back", "front"):
        _, stable, nxt = maps[side] if maps else getup_static.stable_map(xml, side, max_tilt=90.0)
        poses = [key for key in nxt if stable[key][0] == ["foot_sole"] and abs(key[0]) >= MIN_TILT]
        if not poses:
            continue
        print(f"  {side}: {len(poses)} feet-only poses connected to `rest`, "
              f"{min(abs(p[0]) for p in poses):.0f}°-{max(abs(p[0]) for p in poses):.0f}°", flush=True)
        by_tilt: dict[float, list] = {}
        for key in poses:
            by_tilt.setdefault(key[0], []).append(key)
        tilts = sorted(by_tilt)
        kept = 0
        for _ in range(3 * PER_SIDE):                   # as many at each tilt (most poses are near upright)
            if kept >= PER_SIDE:
                break
            group = by_tilt[tilts[rng.integers(len(tilts))]]
            tilt, p, k, a = group[rng.integers(len(group))]
            beta = -np.radians(tilt)                    # as in getup_static: + tilt = leaning back
            pitch = np.array([np.cos(beta / 2), 0.0, np.sin(beta / 2), 0.0])
            yaw = rng.uniform(-np.pi, np.pi)
            heading = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, heading, pitch)
            leg = np.radians([0.0, 0.0, p, k, a])
            q = np.concatenate([leg, leg]) + rng.uniform(-0.05, 0.05, 10)
            q = np.clip(q, m.jnt_range[1:, 0], m.jnt_range[1:, 1])
            mujoco.mj_resetData(m, d)
            d.qpos[:3] = 0.0, 0.0, 0.3
            d.qpos[3:7] = quat
            d.qpos[7:] = d.ctrl[:] = q
            sim.m, sim.d = m, d
            sim._place_on_floor()
            for _ in range(int(0.1 / m.opt.timestep)):
                mujoco.mj_step(m, d)
            touching = {g for c in d.contact[: d.ncon] for g in c.geom} - {floor}
            if touching and touching <= soles:
                out.append((d.qpos.copy(), d.qvel.copy(), 0, abs(tilt)))
                kept += 1
        print(f"  {side}: kept {kept} (on the soles only after 0.1 s)", flush=True)
    return out


def catch_path(maps: dict) -> tuple[np.ndarray, np.ndarray]:
    """-> (tilts deg ascending, leg poses deg (n, 3)): getup_static's slow paths from each side's catch window."""
    nodes = {}
    for _, stable, nxt in maps.values():
        window = getup_static.catch_window(stable, nxt)
        if window is not None:
            nodes.update({n["tilt"]: n["q"] for n in getup_static.path_from(window, stable, nxt)})
    nodes[0.0] = list(REST_LEGS)        # end in `rest` itself (the path's upright node is within 10° of it)
    tilts = sorted(nodes)
    return np.array(tilts, float), np.array([nodes[t] for t in tilts], float)


def throw_states(xml: str, catch: tuple[np.ndarray, np.ndarray]) -> list[tuple]:
    """Snapshots of the env's throws, each -> (qpos, qvel, 1, tilt, throw_t, side). `catch`: catch_path's table."""
    import getup_reference
    path = [dict(tilt=float(t), q=[float(v) for v in q]) for t, q in zip(*catch)]
    out = []
    for side, (name, file) in enumerate(THROWS.items()):
        if not file.exists():
            print(f"  no {file.name}: no {name} throw", flush=True)
            continue
        spec = json.loads(file.read_text())
        n = 0
        for friction, force in getup_search.VARIANTS:
            sim = getup_search.Sim(Path(xml), friction, force)
            ref = getup_reference.Reference(spec, [path], sim.rest_ctrl, spec["lead"], 0.0, spec["rate"],
                                            spec["trigger"])
            pelvis = sim.m.body("pelvis").id
            sim.start(spec["pose"])
            every = int(round(THROW_EVERY / sim.dt))
            t = 0.0
            while True:
                target = ref.targets(t, getup_reference.tilt_deg(sim.d, pelvis), sim.up(), False)
                if ref.caught:
                    break
                if sim.up() > THROW_UP:
                    out.append((sim.d.qpos.copy(), sim.d.qvel.copy(), 1, float(np.degrees(np.arccos(sim.up()))),
                                t, side))
                    n += 1
                sim.d.ctrl[:] = np.clip(target, sim.lo, sim.hi)
                for _ in range(every):
                    mujoco.mj_step(sim.m, sim.d)
                t += THROW_EVERY
        print(f"  {file.stem}: {n} snapshots", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", default=str(getup_search.GETUP_XML))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    print(f"start states for {Path(args.xml).name}")
    maps = {side: getup_static.stable_map(args.xml, side, max_tilt=90.0) for side in ("back", "front")}
    catch_tilt, catch_q = catch_path(maps)
    states = [s + (-1.0, 0) for s in feet_states(args.xml, rng, maps)] + throw_states(args.xml, (catch_tilt, catch_q))
    qpos, qvel, kind, tilt, throw_t, throw_side = (np.array(x) for x in zip(*states))
    out = Path(args.out) if args.out else default_path(args.xml)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, qpos=qpos.astype(np.float32), qvel=qvel.astype(np.float32), kind=kind.astype(np.int32),
             tilt=tilt.astype(np.float32), throw_t=throw_t.astype(np.float32),
             throw_side=throw_side.astype(np.int32), catch_tilt=catch_tilt, catch_q=catch_q)
    for k, name in enumerate(("feet", "throw")):
        sel = kind == k
        if sel.any():
            print(f"{name}: {sel.sum()} states, tilt {tilt[sel].min():.0f}°-{tilt[sel].max():.0f}° "
                  f"(median {np.median(tilt[sel]):.0f}°)")
    print(f"catch path: {len(catch_tilt)} poses, {catch_tilt.min():+.0f}° to {catch_tilt.max():+.0f}°")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
