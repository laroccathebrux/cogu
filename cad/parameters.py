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
THIGH_LENGTH = 60.0         # hip-pitch axis -> knee axis
SHIN_LENGTH = 60.0          # knee axis -> ankle axis
FOOT_HEIGHT = 26.0          # ground -> ankle axis (brief said 15; the shin housing corner sweeps r≈19.8 mm around the ankle, plus 5 mm sole)
HIP_SPACING = 40.0          # distance between the two hip-yaw axes
LEG_GAP = 10.0              # minimum lateral gap between the two legs' inner fork plates at rest
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
# Exceptions to the PLA/PETG rule, per part (ADR-002, 2026-10-30), only for the phone bubble:
#   ribs in PCTG (a tougher copolyester that prints like PETG on the A1 Mini; silver),
#   ground-contact covers in TPU 95A (impact absorption and grip for getting up; black, replaceable).
MATERIAL_EXCEPTIONS = {"phone_guard_bubble": "PCTG", "phone_guard_tpu": "TPU"}
PCTG_DENSITY_G_CM3 = 1.23
TPU_DENSITY_G_CM3 = 1.21
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

PELVIS_TRIM_X = -5.0            # plate corners trimmed to PELVIS_CORNER_TRIM_R about this vertical axis
PELVIS_CORNER_TRIM_R = 50.0
PELVIS_BOSS_Y = 37.7            # hip-cover screw bosses (Ø9.2) start 0.7 mm outboard of the yaw servo bodies
PELVIS_BOSS_X = 8.0             # close to the yaw axes so the cover's front battery bay is free (x > 13)

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
# Phone mount (ADR-002). Pelvis frame: origin between the hip-yaw axes on the yaw servos'
# horn face, +Z up, +X forward. Above the pelvis plate (z > 0) nothing of the legs swings,
# so the hip cover and the phone cradle only have to clear the two yaw servo bodies
# (x −35.1 … +10.1, |y| 7.6 … 32.4, z ≤ 28.75; back discs Ø19.2 on the yaw axes up to z 32.55).
# ---------------------------------------------------------------------------
# The phone the robot carries. Defaults: an iPhone 15/16-class phone without a case; edit to
# yours (the cradle is generated from these numbers). The training env randomises mass and
# centre of mass over PHONE_ENVELOPE.
PHONE = dict(height=147.6, width=71.6, thickness=8.3, mass_g=190.0, corner_r=9.0)
PHONE_ENVELOPE = dict(height=(145.0, 165.0), width=(70.0, 78.0), thickness=(7.5, 9.5), mass_g=(165.0, 230.0))
PHONE_X = 12.0                  # phone mid-plane: ahead of the yaw axes, which pulls the CoM over the feet
PHONE_CAMERA = dict(from_top=22.0, from_left_edge=20.0, bump=(36.0, 36.0, 3.0))   # rear camera module, robot's left
# Hip cover: open-bottom box over the yaw servos, bolted to the 4 pelvis bosses through screw tubes
# (M3 × 8 from the top, heads at the bottom of Ø6.5 counterbores). Holds the battery (front) and the
# "spinal cord" board (rear).
COVER_SHAPE = "round"           # "round": a drum under the bubble (its base ring sits flush on the rim); "box": the first cover
COVER_CENTER_X = -2.0           # drum axis (= the bubble's centre)
COVER_RADIUS = None             # None: the bubble's outer radius at the deck height (53.6 mm for the 150 mm bubble)
COVER_X = (-48.5, 34.0)         # box cover: rear / front outer faces
COVER_TOP_Z = 40.0
COVER_WALL = 2.4
COVER_DECK = 3.0
COVER_CORNER_R = 10.0
COVER_TUBE_WEB = 3.0            # material under the screw head, above the pelvis boss
COVER_CRADLE_INSERTS_X = (-14.0, 0.0)   # 2 M3 inserts on the centre line (between the yaw servos)
COVER_SWITCH = None             # the drum leaves no side spot clear of the screw tubes: switch position to settle in hardware
# COVER_SWITCH = dict(d=12.0, x=-41.0, z=22.0)   # box cover: power switch through the left wall
# Phone cradle: base plate on the cover deck + two C-rails gripping the phone's long edges.
# The phone slides in from the top; the top third stays free for the camera module and buttons.
CRADLE_BASE = 6.0
CRADLE_BASE_X = (-20.0, 22.0)
# The servo driver board rides on a tongue of the cradle base behind the phone, inside the bubble:
CRADLE_TONGUE = dict(x0=-34.0, half_width=33.0)       # base extension behind the cradle (clear of the bubble's base ring)
CRADLE_DRIVER_POSTS = dict(y=29.0, width=6.0, thickness=4.0)   # two posts the board screws onto
CRADLE_RAIL_HEIGHT = 85.0       # above the base
CRADLE_WALL = 2.4
CRADLE_LIP = 3.0                # rail lips overlap the phone faces by this much
CRADLE_CLEARANCE = 0.4
CRADLE_CABLE_SLOT = 16.0        # slot under the phone's USB-C port

# Phone guard (fall protection, under study 2026-10-30). None = no guard (current robot);
# "frame" = printed frame around the phone; "bubble" = egg of printed ribs around the phone.
# `uv run python -m cad.export --guard frame|bubble` builds a variant without changing this default.
PHONE_GUARD = "bubble"          # chosen 2026-10-30 (ADR-002): 150 mm rib bubble
GUARD_FRAME = dict(bar=5.0, depth_beyond_phone=8.0, top_clearance=12.0, side_clearance=0.5)
GUARD_BUBBLE = dict(center_x=-2.0, center_z=110.0, radius_xy=75.0, radius_z=100.0, wall=3.0,
                    rib_width=6.0, rib_azimuths=(30.0, 90.0, 150.0, 210.0, 270.0, 330.0),
                    hoop_z=(110.0,), hoop_height=6.0, base_ring=6.0)
# TPU covers on the bubble's ground-contact parts (press-fit "tyres", printed in TPU 95A):
# C-section on each hoop (lips over its top and bottom edges), outer sleeve + top lip on the base ring,
# and a cap over the crown. None to leave them out.
GUARD_TPU = dict(thickness=2.0, lip=1.5, crown_cap=12.0)   # 2.5 mm weighed 50 g

# Print colours (2026-10-30, "option 3"): light printed parts so the black servos read as joints, one coral
# accent on the feet and on the drum's rim (a filament change), silver PCTG bubble, black TPU covers.
# rgba strings for the MJCF; the 3D comparison page uses the same palette.
PALETTE = dict(printed="0.91 0.92 0.93 1", accent="0.88 0.40 0.29 1", servo="0.09 0.09 0.10 1",
               pctg="0.78 0.80 0.82 1", tpu="0.08 0.08 0.09 1", phone="0.07 0.07 0.08 1")
ACCENT_PARTS = ("foot",)        # printed in the accent colour (the drum's rim too, as a colour change)

# Electronics (bought parts). Positions are box centres in the pelvis frame.
BATTERY_2S = dict(size=(16.0, 70.0, 34.0), mass_g=65.0)   # x × y × z: 2S 1000 mAh pack on its long edge, front bay
BATTERY_POS = (21.5, 0.0, 19.0)
BATTERY_CAGE_WALL = 1.6
# "Spinal cord" = Waveshare Servo Driver with ESP32 (bought 2026-10-30): ESP32 + ST/SC bus-servo interface (2 ports,
# one per leg) + 0.91" OLED, 6–12 V DC jack 5.5 × 2.1 mm, mounting holes Ø2.75 at 23 × 58 mm (board ≈ 30 × 65 mm).
# It stands on two posts of the cradle, behind the phone, OLED facing backwards (visible between the rear ribs),
# fixed with spare M2 self-tapping servo screws.
SERVO_DRIVER = dict(pcb=(1.6, 65.0, 30.0), components=10.0, holes=(58.0, 23.0), hole_d=2.75, hole_inset=3.5,
                    mass_g=20.0, pcb_front_x=-30.0, z0=48.0)
# 6-axis IMU breakout (ICM-42688-P / BMI270 class), on I2C, inside the drum at the rear: the old spine-board slot.
IMU_BOARD = dict(size=(4.0, 25.0, 20.0), mass_g=3.0)
IMU_POS = (-41.5, 0.0, 19.0)    # also the MJCF `imu` site
MISC_ELECTRONICS_MASS_G = 20.0  # switch, fuse, cables
