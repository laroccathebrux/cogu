"""Kinematic tree of the Cogu robot: pelvis + two 5-DOF legs, body shell, collar and
the 2-DOF cap (neck pitch + roll).

Everything here is derived from parameters.py and the part modules; the MJCF
exporter walks this tree. Frames are axis-aligned with the pelvis frame at
q = 0 (straight legs); each body's origin sits on its joint axis.

Joint sign convention (MuJoCo hinge, right-hand rule about `axis`):
  hip_yaw   axis +Z   positive = foot turns to the robot's left (range mirrored per leg:
                      the knee may swing ~30° outboard but only ~8° inboard)
  hip_roll  axis +X   positive = foot swings to the robot's left
  hip_pitch axis −Y   positive = thigh swings FORWARD
  knee      axis −Y   positive = shin swings forward (digitigrade rest pose is negative)
  ankle     axis −Y   positive = toes up
  neck_pitch axis −Y  positive = cap nose down (tilts forward)
  neck_roll  axis +X  positive = cap rolls to the robot's left
"""

from __future__ import annotations

from dataclasses import dataclass, field

from build123d import Align, Axis, Box, Location, Part, Plane, Vector

from cad import parameters as P
from cad import stack as K
from cad.part import CoguPart
from cad.parts import cap as capm
from cad.parts import collar as collarm
from cad.parts import hip
from cad.parts.back_hub import build_back_hub
from cad.parts.bezel import build_bezel
from cad.parts.body import build_body
from cad.parts.foot import build_foot
from cad.parts.link import build_shin, build_thigh


@dataclass
class PartInstance:
    part_name: str              # key into the parts dict
    pos: tuple                  # position of the part frame in the body frame (mm)
    mirrored: bool = False      # left-leg parts are mirrored across the XZ plane
    rot_z: float = 0.0          # extra rotation about Z (deg) before placement


@dataclass
class Body:
    name: str
    pos: tuple                  # body origin in the parent frame (mm)
    joint: str | None           # joint name or None for the root
    joint_axis: tuple | None
    joint_range_deg: tuple | None
    parts: list[PartInstance] = field(default_factory=list)
    children: list["Body"] = field(default_factory=list)


def build_parts() -> dict[str, CoguPart]:
    """All unique printed parts (right-hand versions)."""
    return {
        "pelvis": hip.build_pelvis(),
        "hip_yaw_link": hip.build_yaw_link(),
        "hip_roll_link": hip.build_roll_link(),
        "thigh": build_thigh(),
        "shin": build_shin(),
        "foot": build_foot(),
        "back_hub": build_back_hub(),
        "body": build_body(),
        "bezel": build_bezel(),
        "collar": collarm.build_collar(),
        "cap_bracket": capm.build_cap_bracket(),
        "cap": capm.build_cap(),
    }


# parts whose geometry is mirror-symmetric about the XZ plane: one mesh serves both sides
SYMMETRIC_PARTS = ("pelvis", "body", "bezel", "collar", "cap_bracket", "cap")
# parts printed from a single STL (the back hub is asymmetric but its mirror image is the same part turned over)
ONE_PRINT_FILE = SYMMETRIC_PARTS + ("back_hub",)


def mirrored(part: Part) -> Part:
    return part.mirror(Plane.XZ)


def leg(side: str) -> Body:
    s = 1.0 if side == "L" else -1.0        # left leg is at +Y
    m = side == "L"
    yaw_y = s * P.HIP_SPACING / 2
    rng = P.JOINT_RANGES_DEG

    foot = Body(f"{side}_foot", (0, 0, -P.SHIN_LENGTH), f"{side}_ankle", (0, -1, 0), rng["ankle"],
                parts=[PartInstance("foot", (0, 0, 0), m), PartInstance("back_hub", (0, 0, 0), m)])
    shin = Body(f"{side}_shin", (0, 0, -P.THIGH_LENGTH), f"{side}_knee", (0, -1, 0), rng["knee"],
                parts=[PartInstance("shin", (0, 0, -P.SHIN_LENGTH), m), PartInstance("back_hub", (0, 0, 0), m)],
                children=[foot])
    # the pitch stack is offset towards +Y in the (right-hand) roll link; mirrored for the left leg
    thigh = Body(f"{side}_thigh", (hip.PITCH_AXIS_X, -s * hip.PITCH_AXIS_Y, hip.PITCH_AXIS_Z),
                 f"{side}_hip_pitch", (0, -1, 0), rng["hip_pitch"],
                 parts=[PartInstance("thigh", (0, 0, -P.THIGH_LENGTH), m), PartInstance("back_hub", (0, 0, 0), m)],
                 children=[shin])
    roll = Body(f"{side}_hip_roll", (hip.ROLL_AXIS_X, 0, hip.ROLL_AXIS_Z), f"{side}_hip_roll", (1, 0, 0), rng["hip_roll"],
                parts=[PartInstance("hip_roll_link", (0, 0, 0), m),
                       # hub axis Y -> X. Never mirrored: the roll horn points +X on BOTH legs, and
                       # mirroring (Y flip) before the Z rotation would send the hub to +X, in front.
                       PartInstance("back_hub", (0, 0, 0), False, rot_z=-90)],
                children=[thigh])
    yaw_range = rng["hip_yaw"] if side == "R" else (-rng["hip_yaw"][1], -rng["hip_yaw"][0])   # mirrored
    yaw = Body(f"{side}_hip_yaw", (0, yaw_y, 0), f"{side}_hip_yaw", (0, 0, 1), yaw_range,
               parts=[PartInstance("hip_yaw_link", (0, 0, 0), m)],
               children=[roll])
    return yaw


def neck() -> Body:
    """Collar is fixed to the body; the bracket pitches, the cap rolls on the bracket."""
    cap = Body("cap", (capm.ROLL_X, capm.ROLL_Y, capm.ROLL_Z), "neck_roll", (1, 0, 0), P.NECK_ROLL_RANGE_DEG,
               parts=[PartInstance("cap", (0, 0, 0)),
                      PartInstance("back_hub", (0, 0, 0), False, rot_z=-90)])   # hub axis Y -> X
    return Body("neck", (collarm.NECK_PITCH_X, collarm.NECK_PITCH_Y, collarm.NECK_PITCH_Z), "neck_pitch", (0, -1, 0),
                P.NECK_PITCH_RANGE_DEG,
                parts=[PartInstance("cap_bracket", (0, 0, 0)), PartInstance("back_hub", (0, 0, 0))],
                children=[cap])


def lower_body() -> Body:
    """Legs only (the historical name; `robot()` adds the body, collar and cap)."""
    return Body("pelvis", (0, 0, 0), None, None, None,
                parts=[PartInstance("pelvis", (0, 0, 0))],
                children=[leg("L"), leg("R")])


def robot() -> Body:
    """The complete robot: pelvis body carries the shell, bezel and collar; legs and neck hang off it."""
    root = lower_body()
    root.parts += [PartInstance("body", (0, 0, 0)), PartInstance("bezel", (0, 0, 0)), PartInstance("collar", (0, 0, 0))]
    root.children.append(neck())
    return root


def rest_pose() -> dict[str, float]:
    """Digitigrade rest angles (deg) per joint, both legs."""
    out = {}
    for side in ("L", "R"):
        out[f"{side}_hip_yaw"] = 0.0
        out[f"{side}_hip_roll"] = 0.0
        out[f"{side}_hip_pitch"] = P.REST_HIP_PITCH_DEG
        out[f"{side}_knee"] = P.REST_KNEE_DEG
        out[f"{side}_ankle"] = P.REST_ANKLE_DEG
    out["neck_pitch"] = 0.0
    out["neck_roll"] = 0.0
    return out


def standing_pelvis_height() -> float:
    """Height of the pelvis origin above the ground at the rest pose (mm), from planar FK."""
    import math
    hp = math.radians(P.REST_HIP_PITCH_DEG)
    kn = math.radians(P.REST_KNEE_DEG)
    hip_z = hip.ROLL_AXIS_Z + hip.PITCH_AXIS_Z
    knee_z = hip_z - P.THIGH_LENGTH * math.cos(hp)
    ankle_z = knee_z - P.SHIN_LENGTH * math.cos(hp + kn)
    ground_z = ankle_z - P.FOOT_HEIGHT
    return -ground_z


def total_height() -> float:
    """Ground -> top of the cap at the rest pose (mm)."""
    return standing_pelvis_height() + P.CAP_RIM_Z + P.CAP_HEIGHT


def part_counts(root: Body | None = None) -> dict[str, int]:
    """How many times each printed part appears in the assembly."""
    counts: dict[str, int] = {}

    def walk(b: Body):
        for inst in b.parts:
            counts[inst.part_name] = counts.get(inst.part_name, 0) + 1
        for c in b.children:
            walk(c)

    walk(root or robot())
    return counts


def total_mass_g(parts: dict[str, CoguPart]) -> float:
    return sum(parts[n].total_mass_g * k for n, k in part_counts().items())


@dataclass
class ReferenceSolid:
    """Bought part shown in renders / MJCF for context (visual only, mass is in attached_masses)."""
    name: str
    solid: Part
    body: str                   # kinematic body it moves with
    rgba: str


def reference_solids() -> list[ReferenceSolid]:
    from cad.parts import sts3215
    from cad.parts import body as bodym
    out: list[ReferenceSolid] = []
    grey, dark, blue, green = "0.25 0.25 0.27 1", "0.08 0.08 0.1 1", "0.15 0.2 0.5 1", "0.1 0.45 0.2 1"
    servo = sts3215.servo_solid()
    # yaw servos in the pelvis: servo X -> -X, shaft (Y) -> -Z, Z -> Y
    yaw_servo = servo.rotate(Axis.X, -90).mirror(Plane.YZ)
    for side, sy in (("L", P.HIP_SPACING / 2), ("R", -P.HIP_SPACING / 2)):
        out.append(ReferenceSolid(f"yaw_servo_{side}", yaw_servo.moved(Location((0, sy, 0))), "pelvis", grey))
    # neck pitch servo (collar) and neck roll servo (bracket): same placement as their housings
    # the servo solid goes through the same transforms as its housing (servo frame -> housing frame -> part)
    pitch_servo = sts3215_to_housing(servo).rotate(Axis.Y, -90).moved(
        Location((collarm.NECK_PITCH_X, collarm.NECK_PITCH_Y, collarm.NECK_PITCH_Z)))
    out.append(ReferenceSolid("neck_pitch_servo", pitch_servo, "pelvis", grey))
    roll_servo = capm._roll_housing_place(sts3215_to_housing(servo))
    out.append(ReferenceSolid("neck_roll_servo", roll_servo, "neck", grey))
    # CYD: PCB + glass, tilted with the face
    pcb_l, pcb_w, pcb_t = P.CYD["pcb"]
    cyd = Box(pcb_t, pcb_l, pcb_w, align=(Align.MIN, Align.MIN, Align.CENTER)).moved(Location((0, bodym.PCB_Y0, 0)))
    glass = Box(P.CYD["glass_thickness"], P.CYD["glass_length"], pcb_w, align=(Align.MIN, Align.MIN, Align.CENTER)).moved(
        Location((pcb_t, bodym.GLASS_Y0, 0)))
    dx = P.CYD_PCB_X - P.BODY_FACE_X
    tilt = Location((P.BODY_FACE_X, 0, P.BODY_FACE_CENTRE_Z), (0, -P.BODY_FACE_TILT_DEG, 0))
    out.append(ReferenceSolid("cyd_pcb", cyd.moved(tilt * Location((dx, 0, 0))), "pelvis", green))
    out.append(ReferenceSolid("cyd_glass", glass.moved(tilt * Location((dx, 0, 0))), "pelvis", dark))
    pw, ph, pt = P.PI_ZERO["size"]
    out.append(ReferenceSolid("pi_zero", Box(pt, pw, ph, align=(Align.MIN, Align.CENTER, Align.CENTER)).moved(
        Location((P.PI_ZERO_X, 0, P.PI_ZERO_Z))), "pelvis", green))
    sw, sh, st = P.SERVO_BOARD["size"]
    out.append(ReferenceSolid("servo_board", Box(st, sw, sh, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(
        Location((P.SERVO_BOARD_X, 0, P.SERVO_BOARD_Z))), "pelvis", blue))
    bx, by, bz = P.BATTERY_2S["size"]
    out.append(ReferenceSolid("battery_2s", Box(bx, by, bz).moved(Location(P.BATTERY_POS)), "pelvis", blue))
    iw, il, it = P.BNO055["size"]
    out.append(ReferenceSolid("bno055", Box(iw, il, it).moved(Location(P.IMU_POS)), "pelvis", blue))
    return out


def sts3215_to_housing(servo: Part) -> Part:
    """Servo frame -> housing frame (see cad.parts.link._servo_to_housing)."""
    return servo.rotate(Axis.Y, -90)
