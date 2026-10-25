"""Kinematic tree of the Cogu lower body (pelvis + two 5-DOF legs).

Everything here is derived from parameters.py and the part modules; the MJCF
exporter walks this tree. Frames are axis-aligned with the pelvis frame at
q = 0 (straight legs); each body's origin sits on its joint axis.

Joint sign convention (MuJoCo hinge, right-hand rule about `axis`):
  hip_yaw   axis +Z   positive = foot turns to the robot's left
  hip_roll  axis +X   positive = foot swings to the robot's left
  hip_pitch axis −Y   positive = thigh swings FORWARD
  knee      axis −Y   positive = shin swings forward (digitigrade rest pose is negative)
  ankle     axis −Y   positive = toes up
"""

from __future__ import annotations

from dataclasses import dataclass, field

from build123d import Axis, Location, Part, Plane, Vector

from cad import parameters as P
from cad import stack as K
from cad.part import CoguPart
from cad.parts import hip
from cad.parts.back_hub import build_back_hub
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
    }


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
                       PartInstance("back_hub", (0, 0, 0), m, rot_z=-90)],   # hub axis Y -> X
                children=[thigh])
    yaw = Body(f"{side}_hip_yaw", (0, yaw_y, 0), f"{side}_hip_yaw", (0, 0, 1), rng["hip_yaw"],
               parts=[PartInstance("hip_yaw_link", (0, 0, 0), m)],
               children=[roll])
    return yaw


def lower_body() -> Body:
    return Body("pelvis", (0, 0, 0), None, None, None,
                parts=[PartInstance("pelvis", (0, 0, 0))],
                children=[leg("L"), leg("R")])


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
