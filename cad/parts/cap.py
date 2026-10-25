"""Cap (the mushroom hat) and its 2-DOF bracket.

Neck-pitch frame (bracket): origin on the pitch axis at the pitch servo's horn face, +Y
towards the horn, +Z up, +X forward. The bracket is a fork around the pitch servo (horn
paddle on +Y, 688ZZ paddle on −Y) whose top beam is the housing of the neck-ROLL servo,
lying along Y with its shaft along X.

Neck-roll frame (cap): origin on the roll axis at the roll servo's horn face, +X towards
the horn (forward). The cap is a low half-ellipsoid shell (Ø CAP_DIAMETER × CAP_HEIGHT)
with 3 cream spots, a 360° sensor groove (open at the front for camera + ToF), a rim ring
with an LED channel, radial ribs, and an internal fork around the roll servo.
"""

from __future__ import annotations

import math

from build123d import (
    Align,
    Axis,
    Box,
    Cylinder,
    Line,
    Location,
    Part,
    Plane,
    Polyline,
    Rotation,
    Spline,
    Vector,
    extrude,
    make_face,
    revolve,
)

from cad import parameters as P
from cad import stack as K
from cad.part import CoguPart
from cad.parts import collar as C
from cad.parts.link import bearing_paddle_cutter, horn_paddle_cutters, paddle, servo_housing

S = P.SERVO

# --- roll axis in the neck-pitch frame -------------------------------------------------
ROLL_Z = P.NECK_BEAM_CLEARANCE + K.HOUSING_X_HALF             # 37.06 above the pitch axis
ROLL_X = -K.HOUSING_Y0 / 2                                      # 16.65: housing (shaft X) centred on the PITCH axis, so both its ends clear the pitch housing when tilted
ROLL_Y = -C.NECK_PITCH_Y + (S.x_min + S.x_max) / 2              # housing (body -Y) centred on y = 0 of the pelvis
ROLL_AXIS_Y_PELVIS = C.NECK_PITCH_Y + ROLL_Y                    # 12.5: the roll axis is 12.5 mm left of the midline
ROLL_AXIS_Z_PELVIS = C.NECK_PITCH_Z + ROLL_Z                    # 86.06
# --- cap geometry in the neck-roll frame -----------------------------------------------
DOME_CX = P.BODY_AXIS_X - (C.NECK_PITCH_X + ROLL_X)            # dome centre x (−16.65)
DOME_CY = -ROLL_AXIS_Y_PELVIS                                   # dome centre y (−12.5)
RIM_Z = P.CAP_RIM_Z - ROLL_AXIS_Z_PELVIS                        # −21.06
R_CAP = P.CAP_DIAMETER / 2
H_CAP = P.CAP_HEIGHT


def _roll_housing_place(p: Part) -> Part:
    """housing frame (Y shaft, +Z body, X across) -> pitch frame: shaft +X, body −Y, across Z."""
    p = p.rotate(Axis.Z, -90)       # (x, y, z) -> (y, -x, z): Y -> X
    p = p.rotate(Axis.X, 90)        # (x, y, z) -> (x, -z, y): Z -> -Y, (-X) -> -Z
    return p.moved(Location((ROLL_X, ROLL_Y, ROLL_Z)))


def build_cap_bracket() -> CoguPart:
    beam_top = ROLL_Z + K.HOUSING_X_HALF
    horn = paddle(K.HORN_PLATE_Y0, K.HORN_PLATE_Y1, -beam_top, 0.0).mirror(Plane.XY)
    bear = paddle(K.BEARING_PLATE_Y0, K.BEARING_PLATE_Y1, -beam_top, 0.0).mirror(Plane.XY)
    pcut = horn_paddle_cutters(K.HORN_PLATE_Y0, K.HORN_PLATE_Y1, 0.0) + bearing_paddle_cutter(K.BEARING_PLATE_Y1, 0.0)
    # narrow the arms above the paddle discs so they clear the collar ring when pitched
    hw = P.NECK_ARM_HALF_WIDTH
    for sx in (1, -1):
        trim = Box(K.PADDLE_RADIUS, 60, beam_top, align=(Align.MIN if sx > 0 else Align.MAX, Align.CENTER, Align.MIN)).moved(
            Location((sx * hw, (K.HORN_PLATE_Y1 + K.BEARING_PLATE_Y0) / 2, K.PADDLE_RADIUS * 0.5)))   # both arms
        pcut = pcut + trim
    housing, cutters = servo_housing()
    channel = Box(P.CABLE_CHANNEL_WIDTH + 4, P.CABLE_CHANNEL_DEPTH + 2, 10,
                  align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(
        Location((0, S.y_back_face / 2, S.x_max + P.CLEARANCE_LOOSE - 1.0)))
    housing, cutters = _roll_housing_place(housing), _roll_housing_place(cutters + channel)
    solid = horn + bear + housing - (pcut + cutters)
    servo_com = Vector(ROLL_X + S.y_back_face / 2, ROLL_Y - (S.x_min + S.x_max) / 2, ROLL_Z)
    return CoguPart(
        name="cap_bracket",
        solid=solid,
        print_rotation=Rotation(180, 0, 0),
        print_notes="Beam (roll housing) flat on the bed, paddles pointing up; tree supports under the "
                    "paddle discs only. Roll servo slides in from the front (+X), 4 × M2 through the back wall. PETG.",
        attached_masses=[("neck_roll_servo", S.mass_g, servo_com)],
    )


def _ellipsoid(a: float, c: float, z0: float, extra_below: float = 0.0) -> Part:
    """Upper half-ellipsoid (semi-axes a horizontal, c vertical) with its base at z0, centred on
    the dome axis; `extra_below` adds a straight cylinder below the base."""
    n = 24
    pts = [Vector(a * math.cos(t), 0, z0 + c * math.sin(t)) for t in (math.pi / 2 * i / n for i in range(n + 1))]
    edges = [Spline(*pts)]
    edges.append(Line(pts[-1], Vector(0, 0, z0 - extra_below)))
    edges.append(Line(Vector(0, 0, z0 - extra_below), Vector(a, 0, z0 - extra_below)))
    if extra_below > 0:
        edges.append(Line(Vector(a, 0, z0 - extra_below), pts[0]))
    face = make_face(edges)
    return revolve(face, Axis.Z, 360).moved(Location((DOME_CX, DOME_CY, 0)))


def _sector(half_angle_deg: float, radius: float, z0: float, z1: float) -> Part:
    """Prism covering azimuths ±half_angle about +X, from the dome axis out to `radius`."""
    a = math.radians(half_angle_deg)
    pts = [(0, 0), (radius * math.cos(a), -radius * math.sin(a)), (radius * 1.2, -radius * math.sin(a)),
           (radius * 1.2, radius * math.sin(a)), (radius * math.cos(a), radius * math.sin(a)), (0, 0)]
    face = make_face(Polyline(*[Vector(x, y, 0) for x, y in pts]))
    return extrude(face, z1 - z0).moved(Location((DOME_CX, DOME_CY, z0)))


def build_cap() -> CoguPart:
    w = P.CAP_WALL
    outer = _ellipsoid(R_CAP, H_CAP, RIM_Z)
    inner = _ellipsoid(R_CAP - w, H_CAP - w, RIM_Z, extra_below=5.0)
    shell = outer - inner
    # rim ring with the LED channel on its underside
    ring = Cylinder(R_CAP, 4.0, align=(Align.CENTER, Align.CENTER, Align.MIN)) \
        - Cylinder(R_CAP - 10.0, 6.0, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((0, 0, -1)))
    led = Cylinder(R_CAP - 1.5, 2.0, align=(Align.CENTER, Align.CENTER, Align.MIN)) \
        - Cylinder(R_CAP - 8.5, 4.0, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((0, 0, -1)))
    ring = (ring - led).moved(Location((DOME_CX, DOME_CY, RIM_Z)))
    # radial ribs on the inner surface, outer half of the dome only (the bracket swings under the centre)
    rib_band = _ellipsoid(R_CAP - w, H_CAP - w, RIM_Z, 1.0) - _ellipsoid(R_CAP - w - P.CAP_RIB_SIZE[1], H_CAP - w - P.CAP_RIB_SIZE[1], RIM_Z, 1.0)
    ribs = None
    for i in range(P.CAP_RIBS):
        az = 360.0 * i / P.CAP_RIBS + 22.5
        bar = Box(R_CAP - 10.0 - 45.0, P.CAP_RIB_SIZE[0], H_CAP, align=(Align.MIN, Align.CENTER, Align.MIN))
        bar = bar.moved(Location((45.0, 0, 0))).rotate(Axis.Z, az).moved(Location((DOME_CX, DOME_CY, RIM_Z)))
        ribs = bar if ribs is None else ribs + bar
    ribs = ribs & rib_band
    # internal fork around the roll servo: horn paddle (+X) and bearing paddle (−X), plates ⊥ X,
    # joined at the top by a saddle block that conforms to the dome
    top = RIM_Z + H_CAP
    horn = paddle(K.HORN_PLATE_Y0, K.HORN_PLATE_Y1, -top, 0.0).mirror(Plane.XY).rotate(Axis.Z, -90)
    bear = paddle(K.BEARING_PLATE_Y0, K.BEARING_PLATE_Y1, -top, 0.0).mirror(Plane.XY).rotate(Axis.Z, -90)
    pcut = (horn_paddle_cutters(K.HORN_PLATE_Y0, K.HORN_PLATE_Y1, 0.0) + bearing_paddle_cutter(K.BEARING_PLATE_Y1, 0.0)).rotate(Axis.Z, -90)
    # flat bottom on the paddles: they swing over the collar when the cap pitches
    pcut = pcut + Box(120, 60, 20, align=(Align.CENTER, Align.CENTER, Align.MAX)).moved(Location((-18, 0, -P.CAP_PADDLE_FLAT)))
    saddle_z0 = K.HOUSING_X_HALF + 8.0        # clears the roll housing swept by ±15° (checked in check_envelopes)
    saddle = Box(K.HORN_PLATE_Y1 - K.BEARING_PLATE_Y0 + 6.0, 2 * K.PADDLE_RADIUS + 8.0, top - saddle_z0,
                 align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(
        Location(((K.HORN_PLATE_Y1 + K.BEARING_PLATE_Y0) / 2, 0, saddle_z0)))
    fork = (horn + bear + saddle) & outer
    solid = shell + ring + ribs + fork
    # --- cosmetic / sensor cuts ---
    cutters = pcut
    # 360° groove, through at the front
    sl = P.CAP_SLOT
    groove = (outer - _ellipsoid(R_CAP - sl["depth"], H_CAP - sl["depth"], RIM_Z)) \
        & Box(400, 400, sl["height"], align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((0, 0, RIM_Z + sl["z0"])))
    window = _sector(sl["window_half_angle"], R_CAP + 5, RIM_Z + sl["z0"], RIM_Z + sl["z0"] + sl["height"]) \
        & (outer - _ellipsoid(R_CAP - 12.0, H_CAP - 12.0, RIM_Z))
    cutters = cutters + groove + window
    # 3 cream spots: shallow recesses for press-in discs
    recess_band = outer - _ellipsoid(R_CAP - P.CAP_SPOT_DEPTH, H_CAP - P.CAP_SPOT_DEPTH, RIM_Z)
    for az, frac, rad in P.CAP_SPOTS:
        a = math.radians(az)
        cyl = Cylinder(rad, 100, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(
            Location((DOME_CX + frac * R_CAP * math.cos(a), DOME_CY + frac * R_CAP * math.sin(a), RIM_Z + H_CAP / 2)))
        cutters = cutters + (recess_band & cyl)
    solid = solid - cutters
    sensors_pos = Vector(DOME_CX + R_CAP - 15.0, DOME_CY, RIM_Z + 8.0)
    return CoguPart(
        name="cap",
        solid=solid,
        print_rotation=Rotation(0, 0, 0),
        print_notes="Rim down on the bed (Ø160 ring, use a brim: this is the bed's limit). The dome closes "
                    "gradually; the last ~15 mm around the apex are a shallow roof — print slowly, 1.6 mm "
                    "walls = 4 perimeters. Tree supports only under the fork saddle. PETG, coral.",
        attached_masses=[("cap_sensors_leds", P.CAP_SENSORS_MASS_G, sensors_pos)],
        infill_factor=0.95,
    )
