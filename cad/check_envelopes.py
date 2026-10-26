"""Joint envelope check: sweep each joint and measure EXACT solid overlaps between the
printed parts (build123d booleans). MuJoCo collides convex hulls and skips parent-child
pairs, so it cannot see these. Prints the collision-free interval of every joint and
fails if parameters.JOINT_RANGES_DEG is not inside it (with a safety margin).

    uv run python -m cad.check_envelopes [--step 5] [--margin 3]
"""

from __future__ import annotations

import argparse
import sys

from build123d import Axis, Location, Part, Plane

from cad import parameters as P
from cad.parts import hip
from cad.parts.back_hub import build_back_hub
from cad.parts.foot import build_foot
from cad.parts.link import build_shin, build_thigh

PITCH_AXIS = (0, -1, 0)   # positive = distal end forward (see assembly.py)


def _vol(a: Part, b: Part) -> float:
    i = a.intersect(b)
    if i is None:
        return 0.0
    return sum(s.volume for s in i) if isinstance(i, (list, tuple)) else i.volume


def _rot(s: Part, axis_dir, deg: float, pos=(0, 0, 0)) -> Part:
    return s.rotate(Axis((0, 0, 0), axis_dir), deg).moved(Location(pos))


class Parts:
    def __init__(self):
        self.pelvis = hip.build_pelvis().solid
        self.yaw = hip.build_yaw_link().solid
        self.roll = hip.build_roll_link().solid
        self.thigh = build_thigh().solid.moved(Location((0, 0, -P.THIGH_LENGTH)))   # in the hip-pitch frame
        self.shin = build_shin().solid.moved(Location((0, 0, -P.SHIN_LENGTH)))      # in the knee frame
        self.foot = build_foot().solid                                               # in the ankle frame
        self.rest = {"hip_pitch": P.REST_HIP_PITCH_DEG, "knee": P.REST_KNEE_DEG, "ankle": P.REST_ANKLE_DEG}

    # chain in the hip-pitch frame
    def thigh_chain(self, pitch: float, knee: float, ankle: float) -> tuple[Part, Part, Part]:
        foot = _rot(self.foot, PITCH_AXIS, ankle, (0, 0, -P.SHIN_LENGTH))
        shin = self.shin
        shin, foot = (_rot(s, PITCH_AXIS, knee, (0, 0, -P.THIGH_LENGTH)) for s in (shin, foot))
        thigh = self.thigh
        thigh, shin, foot = (_rot(s, PITCH_AXIS, pitch) for s in (thigh, shin, foot))
        return thigh, shin, foot

    def leg_in_yaw_frame(self, roll: float, pitch: float, knee: float, ankle: float) -> list[Part]:
        thigh, shin, foot = self.thigh_chain(pitch, knee, ankle)
        pa = (hip.PITCH_AXIS_X, hip.PITCH_AXIS_Y, hip.PITCH_AXIS_Z)
        parts = [self.roll] + [s.moved(Location(pa)) for s in (thigh, shin, foot)]
        ra = (hip.ROLL_AXIS_X, 0, hip.ROLL_AXIS_Z)
        return [_rot(s, (1, 0, 0), roll, ra) for s in parts]


def step_of(angles) -> float:
    return angles[1] - angles[0]


def sweep(name, angles, test, rest):
    collides = []
    for a in angles:
        v = test(a)
        collides.append(v > 1.0)
    # largest free run containing rest
    s = step_of(angles)
    idx = min(range(len(angles)), key=lambda i: abs(angles[i] - rest))
    lo = hi = idx
    if collides[idx]:
        print(f"  {name:10s}: COLLIDES AT REST ({rest}°)")
        return None
    while lo - 1 >= 0 and not collides[lo - 1]:
        lo -= 1
    while hi + 1 < len(angles) and not collides[hi + 1]:
        hi += 1
    interval = (angles[lo], angles[hi])
    edge = ("(sweep limit)" if lo == 0 else "") + ("(sweep limit)" if hi == len(angles) - 1 else "")
    print(f"  {name:10s}: free {interval[0]:+6.0f} .. {interval[1]:+6.0f}°  rest {rest:+.0f}° {edge}")
    return interval


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=5.0)
    ap.add_argument("--margin", type=float, default=3.0)
    args = ap.parse_args(argv)
    st = args.step
    rng = lambda lo, hi: [lo + i * st for i in range(int(round((hi - lo) / st)) + 1)]
    p = Parts()
    r = p.rest
    print("exact overlap sweep (build123d), step", st, "deg")
    results = {}

    # hip pitch: roll link vs thigh+shin (knee & ankle at rest)
    def t_pitch(a):
        th, sh, _ = p.thigh_chain(a, r["knee"], r["ankle"])
        pa = (hip.PITCH_AXIS_X, hip.PITCH_AXIS_Y, hip.PITCH_AXIS_Z)
        return _vol(p.roll, th.moved(Location(pa))) + _vol(p.roll, sh.moved(Location(pa)))
    results["hip_pitch"] = sweep("hip_pitch", rng(-100, 100), t_pitch, r["hip_pitch"])

    # knee: thigh vs shin
    def t_knee(a):
        th, sh, _ = p.thigh_chain(0, a, r["ankle"])
        return _vol(th, sh)
    results["knee"] = sweep("knee", rng(-30, 150), t_knee, r["knee"])

    # ankle: shin vs foot
    def t_ankle(a):
        _, sh, ft = p.thigh_chain(0, 0, a)
        return _vol(sh, ft)
    results["ankle"] = sweep("ankle", rng(-100, 100), t_ankle, r["ankle"])

    # roll: yaw link vs (roll link + leg at rest)
    def t_roll(a):
        return sum(_vol(p.yaw, s) for s in p.leg_in_yaw_frame(a, r["hip_pitch"], r["knee"], r["ankle"]))
    results["hip_roll"] = sweep("hip_roll", rng(-60, 60), t_roll, 0.0)

    # yaw: one leg turning vs the pelvis and vs the other leg held straight
    right = [p.yaw] + p.leg_in_yaw_frame(0, r["hip_pitch"], r["knee"], r["ankle"])
    yaw_y = P.HIP_SPACING / 2
    other = [s.mirror(Plane.XZ).moved(Location((0, +yaw_y, 0))) for s in right]  # left leg, mirrored, at +y
    def t_yaw(a):
        turned = [_rot(s, (0, 0, 1), a, (0, -yaw_y, 0)) for s in right]
        v = sum(_vol(p.pelvis, s) for s in turned)
        v += sum(_vol(o, s) for o in other[:3] for s in turned[:3])              # hip parts only (legs are far apart below)
        return v
    results["hip_yaw"] = sweep("hip_yaw", rng(-60, 60), t_yaw, 0.0)

    ok = True
    # back hubs: the hub plate spins inside the hole of the proximal part's back wall at every joint
    hub = build_back_hub().solid
    hub_pairs = {
        "hip_roll": (p.yaw, hub.rotate(Axis.Z, -90).moved(Location((hip.ROLL_AXIS_X, 0, hip.ROLL_AXIS_Z)))),
        "hip_pitch": (p.roll, hub.moved(Location((hip.PITCH_AXIS_X, hip.PITCH_AXIS_Y, hip.PITCH_AXIS_Z)))),
        "knee": (p.thigh, hub.moved(Location((0, 0, -P.THIGH_LENGTH)))),
        "ankle": (p.shin, hub.moved(Location((0, 0, -P.SHIN_LENGTH)))),
    }
    print("\nback hub vs the housing it spins in:")
    for j, (housing, h) in hub_pairs.items():
        v = _vol(housing, h)
        good = v < 0.01
        ok &= good
        print(f"  {j:10s}: overlap {v:5.2f} mm³  {'ok' if good else 'FAIL'}")
    print("\nparameters.JOINT_RANGES_DEG vs measured free interval (margin %.0f°):" % args.margin)
    for j, iv in results.items():
        want = P.JOINT_RANGES_DEG[j]
        if iv is None:
            ok = False
            print(f"  {j:10s}: FAIL (collides at rest)")
            continue
        lo_ok = want[0] >= iv[0] + args.margin or iv[0] <= -99
        hi_ok = want[1] <= iv[1] - args.margin or iv[1] >= 99
        status = "ok" if (lo_ok and hi_ok) else "FAIL"
        ok &= lo_ok and hi_ok
        print(f"  {j:10s}: range {want[0]:+5.0f}..{want[1]:+5.0f}  free {iv[0]:+5.0f}..{iv[1]:+5.0f}  {status}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
