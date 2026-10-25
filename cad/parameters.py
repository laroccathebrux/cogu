"""Cogu — all dimensions in millimetres, masses in grams, angles in degrees.

Single source of truth. Every part, export and design-rule check reads from here.
Values come from docs/decisions/ADR-000-project-brief.md (§3 hardware constitution,
§4 anatomy) and from measuring the Feetech ST3215 STEP model published by Waveshare
(https://www.waveshare.com/wiki/ST3215_Servo, ST3215-3D.zip, drawing SCS215 2022-06-08).
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Robot anatomy (§4)
# ---------------------------------------------------------------------------
CAP_DIAMETER = 160.0        # ceiling of the A1 Mini bed (with brim margin)
CAP_HEIGHT = 55.0
COLLAR_HEIGHT = 18.0
BODY_DIAMETER = 95.0
THIGH_LENGTH = 60.0         # hip-pitch axis -> knee axis
SHIN_LENGTH = 60.0          # knee axis -> ankle axis
FOOT_HEIGHT = 26.0          # ground -> ankle axis (brief said 15; the shin housing corner sweeps r≈19.8 mm around the ankle, plus 5 mm sole)
HIP_SPACING = 40.0          # distance between the two hip-yaw axes
LEG_GAP = 10.0              # minimum lateral gap between the two legs' inner fork plates at rest
TOTAL_HEIGHT_TARGET = 280.0
MASS_TARGET_G = 1200.0

# Digitigrade ("gull-wing" / bird) rest pose: thigh goes BACKWARD and down from the hip,
# the knee sticks out BEHIND the hip, the shin comes forward and down, the ankle sits
# behind the foot centre. Positive pitch = distal end swings forward (see assembly.py).
REST_HIP_PITCH_DEG = -35.0
REST_KNEE_DEG = 70.0
REST_ANKLE_DEG = -35.0

# Joint ranges used by the simulator (absolute, 0 = straight leg). Limits come from the
# collision envelopes of the printed parts: thigh vs hip-pitch beam (≈ ±47° where the beam
# is), shin cheeks vs thigh housing (≈ 115° of knee flexion).
JOINT_RANGES_DEG = {
    "hip_yaw": (-45.0, 45.0),
    "hip_roll": (-35.0, 35.0),
    "hip_pitch": (-90.0, 45.0),
    "knee": (0.0, 115.0),
    "ankle": (-75.0, 45.0),
}

# ---------------------------------------------------------------------------
# Printing constraints (§3) — Bambu Lab A1 Mini, 0.4 mm nozzle
# ---------------------------------------------------------------------------
PRINTER_BED = (180.0, 180.0, 180.0)
MAX_PART_DIMENSION = 160.0
NOZZLE = 0.4
LAYER_HEIGHT = 0.2
WALL_MIN = 1.2              # 3 perimeters × 0.4 mm
WALL_STRUCTURAL = 2.4       # 6 perimeters — around bearings and horns
CLEARANCE_TIGHT = 0.2       # press-ish fits (bearing seats, horn recess)
CLEARANCE_LOOSE = 0.3       # sliding fits (fork vs servo body)
MATERIALS = ("PETG", "PLA")
PETG_DENSITY_G_CM3 = 1.27
INFILL_MASS_FACTOR = 0.55   # printed solid mass ≈ CAD volume × density × this (walls + 25 % infill)

# ---------------------------------------------------------------------------
# Fasteners (§3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HeatInsert:
    name: str = "M3 x 5.7 (OD 4.6)"
    thread: str = "M3"
    outer_diameter: float = 4.6
    length: float = 5.7
    hole_diameter: float = 4.0
    hole_depth: float = 6.5
    chamfer: float = 0.5
    wall_min: float = 2.0       # minimum material around the hole


INSERT_M3 = HeatInsert()

M3_SCREW_LENGTHS = (8.0, 12.0, 16.0)   # the ONLY M3 lengths allowed in the BOM
M3_CLEARANCE_HOLE = 3.4
M3_HEAD_DIAMETER = 5.5          # socket head cap
M3_HEAD_HEIGHT = 3.0

# Servo screws: the original Feetech M2 screws (self-tapping into PETG)
M2_PILOT_HOLE = 1.8             # for M2 self-tapping into PETG/PLA
M2_CLEARANCE_HOLE = 2.4
M2_HEAD_DIAMETER = 3.8
M2_HEAD_HEIGHT = 1.6

# ---------------------------------------------------------------------------
# Bearing 688ZZ (§3) — on the side opposite the servo horn
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Bearing:
    name: str = "688ZZ"
    outer_diameter: float = 16.0
    inner_diameter: float = 8.0
    width: float = 5.0
    seat_diameter: float = 16.0 + CLEARANCE_TIGHT   # printed pocket
    stub_diameter: float = 8.0 - 0.15               # printed stub inside the bore


BEARING_688 = Bearing()

# ---------------------------------------------------------------------------
# O-rings for foot grip (§3) — nitrile, off the shelf
# ---------------------------------------------------------------------------
ORING_CS = 3.0                  # cross-section
ORING_ID = 25.0                 # inner diameter (25 × 3 mm NBR, very common)
ORING_GROOVE_WIDTH = ORING_CS + 0.2
ORING_GROOVE_DEPTH = 2.0        # ring stands 1 mm proud of the sole
ORINGS_PER_FOOT = 2

# ---------------------------------------------------------------------------
# Feetech STS3215 — measured from ST3215.step (Waveshare). Servo-local frame:
#   origin = output-shaft axis ∩ horn-side case face
#   +Y = shaft axis, pointing out through the horn
#   +X = along the body, from the horn end towards the back end
#   Z  = across the body width
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ServoSTS3215:
    name: str = "Feetech STS3215"
    mass_g: float = 55.0
    body_length: float = 45.22          # X
    body_width: float = 24.72           # Z
    case_height: float = 28.75          # Y, back case face -> horn-side case face
    shaft_from_horn_end: float = 10.11  # X distance horn-end face -> shaft axis
    corner_radius: float = 2.0
    # horn side (+Y)
    horn_hub_diameter: float = 6.0
    horn_hub_height: float = 2.2        # case face -> underside of the horn disc
    horn_disc_diameter: float = 19.2
    horn_disc_thickness: float = 2.5
    horn_hole_pcd: float = 14.0         # 4 holes at 0/90/180/270°
    horn_hole_diameter: float = 2.5     # clearance for the M2 horn screws
    horn_top_height: float = 4.7        # case face -> top of horn disc (2.2 + 2.5)
    # back side (−Y): idler disc with the same 4-hole pattern
    back_boss_height: float = 1.55      # case face -> mounting bosses on the back cover
    back_disc_diameter: float = 19.2
    back_disc_thickness: float = 2.2
    back_disc_bottom: float = 3.8       # back case face -> underside of back disc (incl. bosses)
    # mounting pilot holes (Ø1.6 in the servo, for M2 self-tapping screws), axis along Y
    mount_hole_diameter: float = 1.6
    mount_hole_z: float = 10.25         # ± across the width (20.5 mm apart)
    mount_holes_x_horn_side: tuple = (8.3, 29.0)    # X from shaft axis, on the horn-side face
    mount_holes_x_back_side: tuple = (8.3, 32.75)   # X from shaft axis, on the back face
    # overall envelope along Y (from the horn-side case face)
    total_height: float = 28.75 + 3.8 + 4.7  # back disc bottom -> horn disc top = 37.25

    @property
    def x_min(self) -> float:
        return -self.shaft_from_horn_end

    @property
    def x_max(self) -> float:
        return self.body_length - self.shaft_from_horn_end

    @property
    def y_back_face(self) -> float:
        return -self.case_height

    @property
    def y_back_disc_bottom(self) -> float:
        return -self.case_height - self.back_disc_bottom


SERVO = ServoSTS3215()

# ---------------------------------------------------------------------------
# Leg link layout (thigh and shin share the same design)
#   - each link houses the servo of its DISTAL joint (knee servo in the thigh,
#     ankle servo in the shin), shaft at the link's lower axis, body pointing up;
#   - the link's upper end is a fork: a horn plate on one side, a 688ZZ plate on the
#     other, straddling the servo of the PROXIMAL joint (housed in the part above).
# ---------------------------------------------------------------------------
LINK_WALL = 2.4                 # wall around the servo pocket
HORN_PLATE_THICKNESS = 3.0      # plate screwed to the servo horn (M2 self-tapping)
BEARING_PLATE_THICKNESS = 6.0   # plate holding the 688ZZ (5 mm) + 1 mm lip
BACK_HUB_PLATE_THICKNESS = 4.0  # printed hub bolted to the back disc (M2 heads counterbored 2 mm)
BACK_HUB_STUB_LENGTH = 5.0      # Ø(8 − 0.15) stub that enters the bearing
FORK_CLEARANCE = 0.8            # gap between fork plates and servo discs
CABLE_CHANNEL_WIDTH = 6.0       # for the 3-wire servo lead (JST 2.54 3P connector ≈ 8 × 3.5 mm)
CABLE_CHANNEL_DEPTH = 4.0

# ---------------------------------------------------------------------------
# Hip (pelvis + yaw link + roll link)
# ---------------------------------------------------------------------------
PELVIS_WIDTH = 80.0
PELVIS_DEPTH = 50.0
PELVIS_THICKNESS = 6.0
YAW_LINK_THICKNESS = 4.0

# ---------------------------------------------------------------------------
# Foot
# ---------------------------------------------------------------------------
FOOT_LENGTH = 65.0
FOOT_WIDTH = 52.0               # brief said ~45; must cover the 51 mm ankle fork (horn plate .. bearing plate)
FOOT_CORNER_RADIUS = 15.0
FOOT_SOLE_THICKNESS = 5.0
ANKLE_X_FROM_HEEL = 22.0        # ankle axis sits behind the foot centre
FOOT_SWITCH_SIZE = (6.0, 6.0, 3.5)   # pocket for a 6×6 mm tactile switch, 2 per foot
FOOT_SWITCHES_PER_FOOT = 2
