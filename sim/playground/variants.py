"""Model variants for simulation studies, made from the CAD-generated get-up MJCF with MjSpec (the CAD is not touched).

    uv run python sim/playground/variants.py --heel 45                  # -> build/mjcf/variants/heel_45.xml
    uv run python sim/playground/variants.py --heel 45 --knee-max 130   # -> build/mjcf/variants/heel_45_knee_130.xml
    uv run python sim/playground/variants.py --servo 1.5                # -> build/mjcf/variants/servo_1.5.xml
    uv run python sim/playground/variants.py --neck 60                  # -> build/mjcf/variants/neck_60.xml
    uv run python sim/playground/variants.py --heel 45 --toe 20 --width 15   # -> .../heel_45_toe_20_width_15.xml

`--heel MM`: the foot sole box lengthened MM millimetres backwards (a longer heel), both feet. It widens the tilt at
which the robot can stand on its feet alone (README, "Calcanhar"). `--toe MM`: the same at the front (a longer toe).
`--width MM`: each sole MM wider, outwards only (the feet are 9 mm apart at rest). `--knee-max DEG`: the knees fold further than the
CAD's 90° (joint range and servo target range); whether the leg shells allow it is a CAD question. `--servo X`: every
servo's torque limit times X (a stronger servo). `--neck DEG`: the phone's mass (NECK_MASS, phone + guard share) moves
to a head body on a pitch hinge with the same servo, ±DEG; it swings the phone forward and back to shift the centre of
mass, as Open Duck Mini's head does. The bubble's collision shape stays on the pelvis: a first-order test of the mass
shift, not of a moving shell. The neck is the 11th joint and actuator (after the legs). If a variant wins, the change
goes into the CAD.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
GETUP_XML = REPO_ROOT / "build" / "mjcf" / "cogu_lower_body_mjx_getup.xml"
OUT_DIR = REPO_ROOT / "build" / "mjcf" / "variants"
NECK_MASS = 0.25               # kg: iPhone 15 Pro Max (221 g) + a share of cradle and guard
NECK_PIVOT = (0.0, 0.0, 0.03)  # m, pelvis frame: on the hip cover, under the phone
NECK_COM = (0.012, 0.0, 0.12)  # m, pelvis frame: the phone's centre (PHONE_X ahead of the yaw axes)


def name(heel: float = 0.0, knee_max: float | None = None, servo: float = 1.0, neck: float = 0.0,
         toe: float = 0.0, width: float = 0.0) -> str:
    plain = knee_max is None and servo == 1.0 and not neck and not toe and not width
    parts = [f"heel_{heel:g}"] if heel or plain else []
    if toe:
        parts.append(f"toe_{toe:g}")
    if width:
        parts.append(f"width_{width:g}")
    if knee_max is not None:
        parts.append(f"knee_{knee_max:g}")
    if servo != 1.0:
        parts.append(f"servo_{servo:g}")
    if neck:
        parts.append(f"neck_{neck:g}")
    return "_".join(parts)


def build(heel: float = 0.0, knee_max: float | None = None, servo: float = 1.0, neck: float = 0.0,
          src: Path = GETUP_XML, out: Path | None = None, toe: float = 0.0, width: float = 0.0) -> Path:
    spec = mujoco.MjSpec.from_file(str(src))
    if not Path(spec.meshdir).is_absolute():          # the variant lives in another folder
        spec.meshdir = str((src.parent / spec.meshdir).resolve())
    deg = (lambda a: a) if spec.compiler.degree else math.radians   # joint ranges follow the compiler's unit
    for side in "LR":
        g = spec.geom(f"{side}_foot_sole")
        g.size[0] += (heel + toe) / 2000                # half-length
        g.pos[0] += (toe - heel) / 2000                 # heel grows backwards, toe forwards
        g.size[1] += width / 2000
        g.pos[1] += (width / 2000) * (1 if side == "L" else -1)   # outwards: the inner edges stay 9 mm apart
        if knee_max is not None:
            spec.joint(f"{side}_knee").range[1] = deg(knee_max)
            spec.actuator(f"{side}_knee").ctrlrange[1] = math.radians(knee_max)   # ctrlrange is always radians
    for a in spec.actuators:
        a.forcerange = np.asarray(a.forcerange) * servo
    if neck:
        pelvis = spec.body("pelvis")
        m0, c0 = pelvis.mass, np.asarray(pelvis.ipos, float)
        com = np.asarray(NECK_COM)
        pelvis.mass = m0 - NECK_MASS                    # the rest keeps its place: same total CoM at neck 0
        pelvis.ipos = (m0 * c0 - NECK_MASS * com) / pelvis.mass
        pelvis.fullinertia = np.asarray(pelvis.fullinertia) * pelvis.mass / m0
        head = pelvis.add_body(name="head", pos=NECK_PIVOT)
        head.mass, head.ipos = NECK_MASS, com - np.asarray(NECK_PIVOT)
        head.inertia = [5.3e-4, 5.3e-4, 1.5e-4]         # a 160 × 77 mm plate
        head.explicitinertial = True
        head.add_joint(name="neck", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-deg(neck), deg(neck)])
        head.add_geom(name="head_mark", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.015, 0, 0], pos=head.ipos,
                      contype=0, conaffinity=0, rgba=[0.9, 0.4, 0.1, 1])
        legs = spec.actuator("L_knee")
        a = spec.add_actuator(name="neck", target="neck", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype, a.biastype = legs.gaintype, legs.biastype
        a.gainprm, a.biasprm = legs.gainprm, legs.biasprm
        a.ctrllimited, a.ctrlrange = True, [-math.radians(neck), math.radians(neck)]
        a.forcelimited, a.forcerange = True, legs.forcerange
        for key in spec.keys:
            key.qpos = list(key.qpos) + [0.0]
            key.ctrl = list(key.ctrl) + [0.0]
    spec.compile()
    out = out or OUT_DIR / f"{name(heel, knee_max, servo, neck, toe, width)}.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(spec.to_xml())
    return out


def heel(mm: float, src: Path = GETUP_XML, out: Path | None = None) -> Path:
    return build(heel=mm, src=src, out=out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heel", type=float, default=0.0, help="mm added to the back of each sole")
    ap.add_argument("--toe", type=float, default=0.0, help="mm added to the front of each sole")
    ap.add_argument("--width", type=float, default=0.0, help="mm added to the outer side of each sole")
    ap.add_argument("--knee-max", type=float, default=None, help="deg, knee fold limit (CAD: 90)")
    ap.add_argument("--servo", type=float, default=1.0, help="servo torque limit scale")
    ap.add_argument("--neck", type=float, default=0.0, help="deg, a neck pitch joint carrying the phone (0 = none)")
    ap.add_argument("--src", default=str(GETUP_XML))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    print(build(args.heel, args.knee_max, args.servo, args.neck, Path(args.src), Path(args.out) if args.out else None,
                args.toe, args.width))
    return 0


if __name__ == "__main__":
    sys.exit(main())
