"""Collar: the short cream ring between body and cap, which is also the neck-pitch servo mount.

Pelvis frame (bolted to the body's neck ledge with 3 × M3 × 8 from above). The pitch servo
lies horizontally, shaft along Y, body pointing backwards, centred on the body axis; its
housing hangs below the collar deck into the top of the body and its upper half is hidden
by the ring. The cap bracket (cad/parts/cap.py) is the fork around it.

Neck-pitch frame: origin on the pitch axis at the servo's horn-side case face
(x = NECK_PITCH_X, y = NECK_PITCH_Y, z = NECK_PITCH_AXIS_Z), +Y towards the horn.
"""

from __future__ import annotations

import math

from build123d import Align, Axis, Box, Cylinder, Location, Part, Rotation, Vector

from cad import parameters as P
from cad import stack as K
from cad.part import CoguPart
from cad.parts import body as B
from cad.parts.link import servo_housing

S = P.SERVO

NECK_PITCH_X = P.BODY_AXIS_X + P.NECK_PITCH_AXIS_X_OFFSET  # 3.0: housing (body backwards) spans x = −34.8 .. +15.8
NECK_PITCH_Y = -(K.HORN_PLATE_Y1 + K.BEARING_PLATE_Y0) / 2  # 17.8: fork (horn plate .. bearing plate) centred on y = 0
NECK_PITCH_Z = P.NECK_PITCH_AXIS_Z
FLANGE_R1 = P.BODY_NECK_LEDGE_R
CABLE_HOLE = (P.BODY_AXIS_X - 22.0, 24.0, 5.0)            # (x, y, radius): cap servo leads pass here
PADDLE_SLOT_HALF_X = 14.0                                  # deck slots the bracket paddles and the back hub swing through


def housing_in_pelvis_frame(open_top: float = 0.0) -> tuple[Part, Part]:
    """(housing, cutters) of the neck-pitch servo placed in the pelvis frame."""
    housing, cutters = servo_housing(open_top=open_top)
    # cable channel out of the back end of the housing (the servo lead leaves the far end)
    channel = Box(P.CABLE_CHANNEL_WIDTH + 4, P.CABLE_CHANNEL_DEPTH + 2, 10,
                  align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(
        Location((0, S.y_back_face / 2, S.x_max + P.CLEARANCE_LOOSE - 1.0)))
    cutters = cutters + channel

    def place(p: Part) -> Part:
        # housing frame (Y shaft, +Z body, X across) -> pelvis: body backwards (-X), across -> Z
        p = p.rotate(Axis.Y, -90)        # (x, y, z) -> (-z, y, x)
        return p.moved(Location((NECK_PITCH_X, NECK_PITCH_Y, NECK_PITCH_Z)))

    return place(housing), place(cutters)


def build_collar() -> CoguPart:
    xc = P.BODY_AXIS_X
    ring = Cylinder(P.COLLAR_OUTER_R, P.COLLAR_Z1 - P.COLLAR_Z0, align=(Align.CENTER, Align.CENTER, Align.MIN)) \
        - Cylinder(P.COLLAR_INNER_R, P.COLLAR_Z1 - P.COLLAR_Z0 + 2, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(
            Location((0, 0, -1)))
    ring = ring.moved(Location((xc, 0, P.COLLAR_Z0)))
    # deck: flange over the body ledge (screw holes) + plate inside the ring carrying the housing
    deck = Cylinder(FLANGE_R1, P.COLLAR_DECK_THICKNESS, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(
        Location((xc, 0, P.COLLAR_Z0)))
    housing, cutters = housing_in_pelvis_frame()
    solid = ring + deck + housing
    # screws down into the body's neck inserts
    for az in B.NECK_INSERT_AZIMUTHS:
        a = math.radians(az)
        x, y = xc + B.NECK_INSERT_R * math.cos(a), B.NECK_INSERT_R * math.sin(a)
        cutters = cutters + Cylinder(P.M3_CLEARANCE_HOLE / 2, 20, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(
            Location((x, y, P.COLLAR_Z0)))
    # slots in the deck for the bracket's paddle discs (horn side) and the back hub + bearing paddle
    for y0, y1 in ((NECK_PITCH_Y - 0.01, 28.0), (-28.0, NECK_PITCH_Y + K.HOUSING_Y0 - 0.01)):
        cutters = cutters + Box(2 * PADDLE_SLOT_HALF_X, y1 - y0, 20, align=(Align.CENTER, Align.MIN, Align.CENTER)).moved(
            Location((NECK_PITCH_X, y0, P.COLLAR_Z0)))
    # cable hole through the deck for the cap servos' leads
    hx, hy, hr = CABLE_HOLE
    cutters = cutters + Cylinder(hr, 20, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(Location((hx, hy, P.COLLAR_Z0)))
    solid = solid - cutters
    servo_com = Vector(NECK_PITCH_X - (S.x_min + S.x_max) / 2, NECK_PITCH_Y + S.y_back_face / 2, NECK_PITCH_Z)
    return CoguPart(
        name="collar",
        solid=solid,
        print_rotation=Rotation(180, 0, 0),
        print_notes="Upside down: ring top on the bed, servo housing printing as a tower with its open "
                    "side sideways (the 25 mm pocket roof bridges). No supports. Slide the pitch servo in "
                    "from the horn side (+Y), 4 × M2 through the back wall. 3 × M3 × 8 down into the body.",
        attached_masses=[("neck_pitch_servo", S.mass_g, servo_com)],
        infill_factor=0.8,
    )
