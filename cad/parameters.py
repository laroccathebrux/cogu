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
BODY_DIAMETER = 110.0       # at the equator; the skirt flares to 115. Brief said ~95: see docs/decisions/ADR-001-body-shape.md
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

# Joint ranges used by the simulator (absolute, 0 = straight leg). They must stay inside the
# collision-free intervals measured by `uv run python -m cad.check_envelopes` (exact solid
# overlaps, run in CI). Measured on 2026-10-25: hip_pitch -90..+20, knee -30..+95,
# ankle -85..+85, hip_roll beyond ±60, hip_yaw ≈ ±40 (one leg turning, the other straight).
JOINT_RANGES_DEG = {
    # hip_yaw is ASYMMETRIC: the knee sits ~65 mm behind the yaw axis and the legs are only
    # LEG_GAP apart, so turning the knee INBOARD collides with the other leg after ~10°, while
    # turning it outboard is free to ~35°. Given for the RIGHT leg (+ = knee outboard); the
    # assembly mirrors it for the left leg.
    "hip_yaw": (-7.0, 30.0),
    "hip_roll": (-35.0, 35.0),
    "hip_pitch": (-85.0, 15.0),   # forward limit: thigh housing corners meet the hip-pitch beam
    "knee": (0.0, 90.0),          # shin cheeks meet the thigh housing beyond ~95°
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
PELVIS_WIDTH = 85.0            # the 4 body bosses must clear the yaw servo bodies (|y| ≤ 32.4)
PELVIS_DEPTH = 50.0
PELVIS_THICKNESS = 6.0
YAW_LINK_THICKNESS = 4.0

PELVIS_CORNER_TRIM_R = 50.0     # plate corners trimmed to this radius about the body axis
PELVIS_BOSS_Y = 37.7            # body-screw bosses (Ø9.2) start 0.7 mm outboard of the servo bodies
PELVIS_BOSS_X = 17.0

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

# ---------------------------------------------------------------------------
# Body shell (ADR-001). Pelvis frame: origin between the hip-yaw axes on the yaw servos'
# horn face, +Z up, +X forward. The pelvis sits INSIDE the shell; the shell is open at the
# bottom (skirt rim) so the hip stack can swing. Numbers come from the measured envelope
# of the hip hardware over JOINT_RANGES_DEG (cad/check_envelopes.py --body).
# ---------------------------------------------------------------------------
BODY_AXIS_X = -5.0              # vertical axis of the shell, slightly behind the hip-yaw axes
BODY_WALL = 2.0                 # shell wall (5 perimeters); ribs add stiffness
BODY_RIM_Z = -37.0              # skirt rim: 2.4 mm above the highest point the thigh reaches at hip_pitch -85°; the skirt
                                # flares to Ø115 there because the hip-roll arms reach r = 54.4 mm at roll -35° / yaw +30°
BODY_NECK_Z = 46.0              # top of the shell: flat neck ledge the collar bolts onto
BODY_NECK_HOLE_R = 36.3         # opening in the ledge: the collar ring (inner R 36) and the neck bracket swing inside it
BODY_NECK_LEDGE_R = 45.0        # outer radius of the flat ledge (shell profile reaches it)
BODY_NECK_INSERT_R = 40.0       # 3 M3 inserts under the ledge, just outside the collar ring
# outer profile (z, radius): spline through these, from the rim to the ledge
BODY_PROFILE = ((-37.0, 57.5), (-28.0, 56.5), (-16.0, 55.5), (-4.0, 55.0), (8.0, 54.8),
                (20.0, 53.4), (30.0, 51.0), (38.0, 47.2), (43.0, 46.0), (46.0, 45.0))
BODY_RIBS = 5                   # vertical ribs inside the shell (1.2 × 4 mm), above BODY_RIB_Z0 only, 75°..285° (clear of the visor)
BODY_RIB_Z0 = -8.0              # the hip-roll paddles sweep r ≤ 46 mm below this height: the skirt stays smooth there
BODY_RIB_SIZE = (1.2, 4.0)
BODY_SCREW_LENGTH = 8.0         # M3 × 8 through the 3 mm ears into the pelvis inserts
BODY_INSERTS_NECK = 3           # M3 inserts under the neck ledge for the collar
BODY_INSERTS_FACE = 4           # M3 inserts in the face box for the bezel
# Face: the CYD's 86 × 50 PCB does not fit inside a Ø110 sphere anywhere in front of the hip stack,
# so the screen lives in a flat visor box as wide as the body, standing proud of the sphere at its
# sides (ADR-001). The box starts behind the PCB and outside the hip-roll paddles' sweep (x ≥ 28).
BODY_FACE_CENTRE_Z = 1.0        # centre of the screen on the shell (visor bottom-rear corner stays above the hip-roll sweep)
BODY_FACE_X = 42.0 + 1.6 + 4.5  # 48.1: front of the glass = CYD_PCB_X + PCB + glass, where the bezel sits (sphere apex at x = 50)
BODY_FACE_TILT_DEG = 8.0        # the face looks slightly up (people look down at a 32 cm robot)
BODY_FACE_WIDTH = 104.0         # visor outline (y): holds the PCB with its off-centre ESP32 end
BODY_FACE_HEIGHT = 71.0         # visor outline (z): the 4 bezel bosses sit above and below the PCB
BODY_FACE_BOX_X0 = 28.0         # rear of the visor box (measured along the face normal like BODY_FACE_X)
BODY_FACE_CORNER_R = 12.0
BODY_SWITCH_HOLE_D = 12.0       # rocker/push power switch, rear right
BODY_USBC_SLOT = (10.0, 5.0)    # USB-C panel-mount slot, rear right

# Electronics envelopes (mm) and masses (g). Positions are in the pelvis frame.
CYD = dict(pcb=(86.0, 50.0, 1.6), glass_length=69.0, glass_thickness=4.5, back_components=7.0,
           active=(57.6, 43.2), active_centre_from_glass_end=34.6, mass_g=55.0)
CYD_ACTIVE_CENTRE_Y = -3.0      # the active area is off-centre on the PCB; 3 mm keeps the ESP32 end inside the shell
CYD_PCB_X = 42.0                # rear face of the PCB, measured along the (tilted) face normal like BODY_FACE_X
PI_ZERO = dict(size=(65.0, 30.0, 1.6), holes=(58.0, 23.0), hole_d=2.75, components=5.0, mass_g=11.0)
PI_ZERO_X = 16.5                # PCB plane, standing vertically in front of the yaw servos, component side +X (0.7 mm ahead of the neck housing)
PI_ZERO_Z = 26.0                # centre height: dropped in through the neck hole, rests on tabs above the pelvis bosses
BNO055 = dict(size=(20.0, 27.0, 4.0), holes=20.0, hole_d=2.5, mass_g=3.0)   # Adafruit-style breakout, 2 holes 20 mm apart
IMU_POS = (-44.0, 0.0, 30.0)    # BNO055 centre (also the MJCF `imu` site): rear ledge above the battery
SERVO_BOARD = dict(size=(42.0, 33.0, 7.0), holes=(37.0, 28.0), hole_d=2.5, mass_g=12.0)  # Waveshare Bus Servo Adapter (A), DC jack removed
SERVO_BOARD_X = 30.0            # standing between the Pi and the screen, centre of its 7 mm envelope (PCB at the rear)
SERVO_BOARD_Z = 19.0            # rails start above the pelvis plate
BATTERY_2S = dict(size=(14.0, 31.0, 58.0), mass_g=60.0)   # x × y × z, standing in the rear bay (2S 1000 mAh soft pack)
BATTERY_POS = (-43.0, 0.0, -6.0)
MISC_BODY_MASS_G = 40.0         # 5 V regulator, switch, USB-C, cables
CAP_SENSORS_MASS_G = 25.0       # camera, VL53L5CX ToF, LED ring, cables

# ---------------------------------------------------------------------------
# Neck (2 DOF under the cap: pitch servo in the collar, roll servo in the bracket above)
# ---------------------------------------------------------------------------
COLLAR_INNER_R = 36.0           # the bracket's paddle discs (12.6 mm about an axis 8 mm ahead of the body axis) swing inside
COLLAR_OUTER_R = 39.0
COLLAR_Z0 = BODY_NECK_Z                      # 46: sits on the neck ledge
COLLAR_Z1 = COLLAR_Z0 + COLLAR_HEIGHT        # 64
COLLAR_DECK_THICKNESS = 3.0
NECK_PITCH_AXIS_Z = 49.0        # housing bottom (−15.06) clears the yaw servos' back discs (+32.55)
NECK_PITCH_AXIS_X_OFFSET = 8.0  # pitch axis ahead of the body axis: trades housing rear overhang against paddle reach inside the collar
NECK_ARM_HALF_WIDTH = 8.0       # bracket arms narrow above the paddle disc so they clear the collar ring at ±20°
NECK_PITCH_RANGE_DEG = (-15.0, 15.0)   # positive = cap nose down (axis −Y, like the hip pitch). Limited by the cap rim
NECK_ROLL_RANGE_DEG = (-12.0, 12.0)    # positive = cap rolls to the robot's left (axis +X).  vs the visor's top corners
CAP_PADDLE_FLAT = 10.6          # the cap's fork paddles are cut flat this far below the roll axis (688ZZ seat r 8.1 + 2.5 mm)
NECK_BEAM_CLEARANCE = 25.0      # bracket beam underside above the pitch axis: its end corners dip when pitched ±20° and must clear the pitch housing top (+15.06)
CAP_RIM_Z = COLLAR_Z1 + 1.0     # 65: cap rim just above the collar
CAP_WALL = 1.6
CAP_RIBS = 8                    # radial ribs, outer half of the dome only (the neck bracket swings under the centre)
CAP_RIB_SIZE = (1.2, 5.0)
CAP_SPOTS = ((0.0, 0.52, 15.0), (125.0, 0.62, 12.0), (235.0, 0.45, 10.0))   # (azimuth °, r/R, radius) cream spots
CAP_SPOT_DEPTH = 0.8            # recess for a press-in cream disc (or colour change when printed dome-down)
CAP_SLOT = dict(z0=6.0, height=4.0, depth=1.5, window_half_angle=35.0)   # 360° sensor groove; through at the front
