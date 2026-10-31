"""Kinematic tree of the Cogu robot (ADR-002): pelvis + two 5-DOF digitigrade legs, the hip
cover and the phone cradle with the phone standing on it (rear cameras forward).

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
"""

from __future__ import annotations

from dataclasses import dataclass, field

from build123d import Align, Axis, Box, Location, Part, Plane, Vector

from cad import parameters as P
from cad import stack as K
from cad.part import CoguPart
from cad.parts import hip
from cad.parts import phone_mount
from cad.parts.back_hub import build_back_hub, build_back_hub_roll
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
        "back_hub_roll": build_back_hub_roll(),
        "hip_cover": phone_mount.build_hip_cover(),
        "phone_cradle": phone_mount.build_phone_cradle(),
        **({f"phone_guard_{P.PHONE_GUARD}": phone_mount.GUARDS[P.PHONE_GUARD]()} if P.PHONE_GUARD else {}),
        **({"phone_guard_tpu": phone_mount.build_guard_tpu()} if P.PHONE_GUARD == "bubble" and P.GUARD_TPU else {}),
    }


# parts that are never mirrored (pelvis-frame parts on the centre line): one mesh / one STL
SYMMETRIC_PARTS = ("pelvis", "hip_cover", "phone_cradle", "phone_guard_frame", "phone_guard_bubble", "phone_guard_tpu")
# parts printed from a single STL (the back hub is asymmetric but its mirror image is the same part turned over)
ONE_PRINT_FILE = SYMMETRIC_PARTS + ("back_hub", "back_hub_roll")


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
                       PartInstance("back_hub_roll", (0, 0, 0), False, rot_z=-90)],
                children=[thigh])
    yaw_range = rng["hip_yaw"] if side == "R" else (-rng["hip_yaw"][1], -rng["hip_yaw"][0])   # mirrored
    yaw = Body(f"{side}_hip_yaw", (0, yaw_y, 0), f"{side}_hip_yaw", (0, 0, 1), yaw_range,
               parts=[PartInstance("hip_yaw_link", (0, 0, 0), m)],
               children=[roll])
    return yaw


def lower_body() -> Body:
    """Legs only (the historical name; `robot()` adds the hip cover and the phone cradle)."""
    return Body("pelvis", (0, 0, 0), None, None, None,
                parts=[PartInstance("pelvis", (0, 0, 0))],
                children=[leg("L"), leg("R")])


def robot() -> Body:
    """The complete robot: the pelvis body carries the hip cover and the phone cradle (the phone's
    mass is attached to the cradle, the battery and electronics to the cover); the legs hang off it."""
    root = lower_body()
    root.parts += [PartInstance("hip_cover", (0, 0, 0)), PartInstance("phone_cradle", (0, 0, 0))]
    if P.PHONE_GUARD:
        root.parts.append(PartInstance(f"phone_guard_{P.PHONE_GUARD}", (0, 0, 0)))
    if P.PHONE_GUARD == "bubble" and P.GUARD_TPU:
        root.parts.append(PartInstance("phone_guard_tpu", (0, 0, 0)))
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
    """Ground -> top of the phone (or of its guard) at the rest pose (mm)."""
    top = phone_mount.PHONE_Z0 + P.PHONE["height"]
    if P.PHONE_GUARD:
        top = max(top, phone_mount.GUARDS[P.PHONE_GUARD]().solid.bounding_box().max.Z)
    if P.PHONE_GUARD == "bubble" and P.GUARD_TPU:
        top = max(top, phone_mount.build_guard_tpu().solid.bounding_box().max.Z)
    return standing_pelvis_height() + top


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
    """Bought part shown in renders / MJCF for context (visual only, mass is in attached_masses).
    `collide`: in the full MJCF it touches the floor (the phone, for falls)."""
    name: str
    solid: Part
    body: str                   # kinematic body it moves with
    rgba: str
    collide: bool = False


def reference_solids() -> list[ReferenceSolid]:
    from cad.parts import sts3215
    out: list[ReferenceSolid] = []
    grey, dark, blue, green = "0.25 0.25 0.27 1", "0.08 0.08 0.1 1", "0.15 0.2 0.5 1", "0.1 0.45 0.2 1"
    servo = sts3215.servo_solid()
    # yaw servos in the pelvis: servo X -> -X, shaft (Y) -> -Z, Z -> Y
    yaw_servo = servo.rotate(Axis.X, -90).mirror(Plane.YZ)
    for side, sy in (("L", P.HIP_SPACING / 2), ("R", -P.HIP_SPACING / 2)):
        out.append(ReferenceSolid(f"yaw_servo_{side}", yaw_servo.moved(Location((0, sy, 0))), "pelvis", P.PALETTE["servo"]))
    # leg servos (each link houses the servo of its distal joint), placed like their housings
    from cad.parts.link import _servo_to_housing
    in_housing = _servo_to_housing(servo)
    roll_servo = in_housing.rotate(Axis.Z, -90).rotate(Axis.X, 180).moved(Location((hip.ROLL_AXIS_X, 0, hip.ROLL_AXIS_Z)))
    pitch_servo = in_housing.rotate(Axis.Y, 90).moved(Location((hip.PITCH_AXIS_X, hip.PITCH_AXIS_Y, hip.PITCH_AXIS_Z)))
    link_servo = in_housing.moved(Location((0, 0, -P.THIGH_LENGTH)))
    for side in ("L", "R"):
        m = (lambda s: mirrored(s)) if side == "L" else (lambda s: s)
        out.append(ReferenceSolid(f"roll_servo_{side}", m(roll_servo), f"{side}_hip_yaw", P.PALETTE["servo"]))
        out.append(ReferenceSolid(f"pitch_servo_{side}", m(pitch_servo), f"{side}_hip_roll", P.PALETTE["servo"]))
        out.append(ReferenceSolid(f"knee_servo_{side}", m(link_servo), f"{side}_thigh", P.PALETTE["servo"]))
        out.append(ReferenceSolid(f"ankle_servo_{side}", m(link_servo), f"{side}_shin", P.PALETTE["servo"]))
    phone, bump = phone_mount.phone_solid()
    out.append(ReferenceSolid("phone", phone, "pelvis", P.PALETTE["phone"], collide=True))
    out.append(ReferenceSolid("phone_camera", bump, "pelvis", "0.35 0.35 0.38 1"))
    bx, by, bz = P.BATTERY_2S["size"]
    out.append(ReferenceSolid("battery_2s", Box(bx, by, bz).moved(Location(P.BATTERY_POS)), "pelvis", blue))
    ix, iy, iz = P.IMU_BOARD["size"]
    out.append(ReferenceSolid("imu_board", Box(ix, iy, iz).moved(Location(P.IMU_POS)), "pelvis", green))
    sd = P.SERVO_DRIVER
    t, l, h = sd["pcb"]
    x_front = sd["pcb_front_x"]
    board = Box(t + sd["components"], l, h, align=(Align.MAX, Align.CENTER, Align.MIN)).moved(Location((x_front, 0, sd["z0"])))
    out.append(ReferenceSolid("servo_driver", board, "pelvis", "0.05 0.05 0.06 1"))
    return out


def sts3215_to_housing(servo: Part) -> Part:
    """Servo frame -> housing frame (see cad.parts.link._servo_to_housing)."""
    return servo.rotate(Axis.Y, -90)
