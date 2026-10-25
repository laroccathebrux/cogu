"""Body / neck envelope checks (ADR-001), run in CI after cad.check_envelopes:

  1. the body shell (with bezel and collar) never touches a leg anywhere in the hip
     yaw × roll × pitch × knee box of parameters.JOINT_RANGES_DEG;
  2. the shell clears the hardware bolted inside it at rest (pelvis, yaw servos, yaw links);
  3. the cap never touches the knees / shins when crouching or sitting (hip pitch and knee at
     their limits) for any cap pitch / roll in NECK_*_RANGE_DEG — the brief's mandatory check;
  4. the neck ranges themselves: the bracket vs the collar + body, and the cap vs the
     bracket, collar and body, swept in pitch and roll (the free interval must contain the
     configured range with a margin).

All overlaps are exact build123d booleans on the printed solids.

    uv run python -m cad.check_body [--step 5] [--margin 3]
"""

from __future__ import annotations

import argparse
import itertools
import sys

from build123d import Axis, Location, Part, Plane

from cad import assembly
from cad import parameters as P
from cad.check_envelopes import Parts, _rot, _vol, sweep
from cad.parts import cap as capm
from cad.parts import collar as collarm
from cad.parts import hip


def _leg_parts_in_pelvis_frame(p: Parts, side: str, yaw: float, roll: float, pitch: float, knee: float,
                               ankle: float) -> list[Part]:
    """yaw link, roll link, thigh, shin, foot of one leg, posed, in the pelvis frame."""
    s = 1.0 if side == "L" else -1.0
    parts = [p.yaw] + p.leg_in_yaw_frame(roll, pitch, knee, ankle)
    if side == "L":
        parts = [q.mirror(Plane.XZ) for q in parts]
        yaw = -yaw                                   # right-hand convention mirrored, like assembly.leg()
    return [_rot(q, (0, 0, 1), yaw, (0, s * P.HIP_SPACING / 2, 0)) for q in parts]


class Shells:
    def __init__(self, parts: dict):
        self.body = parts["body"].solid + parts["bezel"].solid
        self.collar = parts["collar"].solid
        self.bracket = parts["cap_bracket"].solid
        self.cap = parts["cap"].solid
        self.pelvis = parts["pelvis"].solid
        # the BNO055 sits on 2 pegs that pass through its holes (the reference box has none): skip it
        self.hardware = [r.solid for r in assembly.reference_solids() if r.body == "pelvis" and r.name != "bno055"]

    def neck_pose(self, pitch: float, roll: float) -> tuple[Part, Part]:
        """(bracket, cap) in the pelvis frame for the given neck angles (deg)."""
        np_ = (collarm.NECK_PITCH_X, collarm.NECK_PITCH_Y, collarm.NECK_PITCH_Z)
        cap = _rot(self.cap, (1, 0, 0), roll, (capm.ROLL_X, capm.ROLL_Y, capm.ROLL_Z))
        bracket, cap = (_rot(q, (0, -1, 0), pitch, np_) for q in (self.bracket, cap))
        return bracket, cap


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=5.0)
    ap.add_argument("--margin", type=float, default=3.0)
    args = ap.parse_args(argv)
    st = args.step
    rng = lambda lo, hi: [lo + i * st for i in range(int(round((hi - lo) / st)) + 1)]
    parts = assembly.build_parts()
    sh = Shells(parts)
    p = Parts()
    r = p.rest
    J = P.JOINT_RANGES_DEG
    ok = True

    # 1. body shell vs the leg over the hip box (both legs; the left leg uses the mirrored yaw range)
    print("1. body + bezel + collar vs leg over hip yaw × roll × pitch × knee (exact overlaps) …")
    yaws = (J["hip_yaw"][0], 0.0, J["hip_yaw"][1])
    rolls = (J["hip_roll"][0], 0.0, J["hip_roll"][1])
    pitches = (J["hip_pitch"][0], P.REST_HIP_PITCH_DEG, J["hip_pitch"][1])
    knees = (J["knee"][0], P.REST_KNEE_DEG, J["knee"][1])
    worst = (0.0, None)
    shell = sh.body + sh.collar
    for yaw, roll, pitch, knee in itertools.product(yaws, rolls, pitches, knees):
        legs = _leg_parts_in_pelvis_frame(p, "R", yaw, roll, pitch, knee, r["ankle"])
        v = sum(_vol(shell, q) for q in legs)
        if v > worst[0]:
            worst = (v, (yaw, roll, pitch, knee))
    if worst[0] > 1.0:
        ok = False
        print(f"   FAIL: body overlaps the leg by {worst[0]:.0f} mm³ at yaw/roll/pitch/knee = {worst[1]}")
    else:
        print(f"   ok (max overlap {worst[0]:.2f} mm³)")

    # 2. shell vs the hardware inside it at rest
    print("2. body vs pelvis, yaw servos and electronics envelopes at rest …")
    v = _vol(sh.body, sh.pelvis) + sum(_vol(sh.body, h) for h in sh.hardware)
    v += sum(_vol(sh.collar, h) for h in sh.hardware)
    if v > 1.0:
        ok = False
        print(f"   FAIL: {v:.0f} mm³ of overlap")
    else:
        print("   ok")

    # 3. cap vs knees / shins / feet when crouching or sitting, any cap pose
    print("3. cap vs legs at the crouch / sit extremes for all neck poses …")
    worst = (0.0, None)
    for npitch, nroll in itertools.product(P.NECK_PITCH_RANGE_DEG + (0.0,), P.NECK_ROLL_RANGE_DEG + (0.0,)):
        _, cap = sh.neck_pose(npitch, nroll)
        for pitch, knee in itertools.product((J["hip_pitch"][0], J["hip_pitch"][1]), (J["knee"][0], J["knee"][1])):
            for roll in rolls:
                legs = _leg_parts_in_pelvis_frame(p, "R", 0.0, roll, pitch, knee, r["ankle"])
                v = sum(_vol(cap, q) for q in legs[2:])          # thigh, shin, foot
                if v > worst[0]:
                    worst = (v, (npitch, nroll, pitch, knee, roll))
    if worst[0] > 1.0:
        ok = False
        print(f"   FAIL: cap overlaps a leg by {worst[0]:.0f} mm³ at neck pitch/roll, hip pitch, knee, hip roll = {worst[1]}")
    else:
        print(f"   ok (max overlap {worst[0]:.2f} mm³)")

    # 4. neck ranges
    print("4. neck sweeps (exact overlaps):")
    fixed = sh.body + sh.collar

    def t_pitch(a):
        bracket, cap = sh.neck_pose(a, 0.0)
        return _vol(fixed, bracket) + _vol(fixed, cap)

    def t_roll(a):
        bracket, cap = sh.neck_pose(0.0, a)
        return _vol(cap, bracket) + _vol(fixed, cap)

    def t_pitch_rolled(a):
        v = 0.0
        for nr in P.NECK_ROLL_RANGE_DEG:
            bracket, cap = sh.neck_pose(a, nr)
            v += _vol(fixed, bracket) + _vol(fixed, cap) + _vol(cap, bracket)
        return v

    results = {
        "neck_pitch": sweep("neck_pitch", rng(-60, 60), t_pitch, 0.0),
        "neck_roll": sweep("neck_roll", rng(-60, 60), t_roll, 0.0),
        "pitch@roll±": sweep("pitch@roll±", rng(-60, 60), t_pitch_rolled, 0.0),
    }
    want = {"neck_pitch": P.NECK_PITCH_RANGE_DEG, "neck_roll": P.NECK_ROLL_RANGE_DEG, "pitch@roll±": P.NECK_PITCH_RANGE_DEG}
    print(f"\n   configured ranges vs measured free interval (margin {args.margin:.0f}°):")
    for j, iv in results.items():
        w = want[j]
        if iv is None:
            ok = False
            print(f"   {j:12s}: FAIL (collides at rest)")
            continue
        good = (w[0] >= iv[0] + args.margin or iv[0] <= -59) and (w[1] <= iv[1] - args.margin or iv[1] >= 59)
        ok &= good
        print(f"   {j:12s}: range {w[0]:+5.0f}..{w[1]:+5.0f}  free {iv[0]:+5.0f}..{iv[1]:+5.0f}  {'ok' if good else 'FAIL'}")
    print("\nBODY / NECK CHECKS:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
