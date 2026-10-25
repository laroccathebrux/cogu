"""Foot: flat rounded sole with two O-ring grooves, 2 contact switches and the ankle fork.

Foot frame: origin on the ANKLE axis, +Y along the axis (horn side), +Z up, +X forward.
Ground is at z = -FOOT_HEIGHT.
"""

from __future__ import annotations

from build123d import (
    Align,
    Axis,
    Box,
    Cylinder,
    Location,
    Part,
    Rotation,
    SlotCenterToCenter,
    extrude,
    Plane,
)

from cad import parameters as P
from cad import stack as K
from cad.part import CoguPart, rounded_box
from cad.parts.link import bearing_paddle_cutter, horn_paddle_cutters, paddle

LATERAL_CENTRE = (K.HORN_PLATE_Y1 + K.BEARING_PLATE_Y0) / 2     # fork centre line


def _oring_groove(x_centre: float) -> Part:
    """Stadium-shaped groove whose centreline matches the O-ring's mean circumference."""
    mean_d = P.ORING_ID + P.ORING_CS
    circumference = 3.141592653589793 * mean_d
    r = 6.0                                       # end radius of the stadium centreline
    straight = (circumference - 2 * 3.141592653589793 * r) / 2
    with_slot = SlotCenterToCenter(straight, 2 * r + P.ORING_GROOVE_WIDTH)
    inner = SlotCenterToCenter(straight, 2 * r - P.ORING_GROOVE_WIDTH)
    ring2d = with_slot - inner
    groove = extrude(ring2d, P.ORING_GROOVE_DEPTH + 1)      # along +Z from z=0
    # slot straights along Y (lateral): rotate 90° about Z, then place on the sole underside
    groove = groove.rotate(Axis.Z, 90)
    return groove.moved(Location((x_centre, LATERAL_CENTRE, -P.FOOT_HEIGHT - 1)))


def build_foot() -> CoguPart:
    sole_z0 = -P.FOOT_HEIGHT
    sole = rounded_box(P.FOOT_LENGTH, P.FOOT_WIDTH, P.FOOT_SOLE_THICKNESS, P.FOOT_CORNER_RADIUS)
    sole = sole.moved(Location((P.FOOT_LENGTH / 2 - P.ANKLE_X_FROM_HEEL, LATERAL_CENTRE, sole_z0)))
    sole_top = sole_z0 + P.FOOT_SOLE_THICKNESS

    # ankle fork: paddles rise from the sole to the ankle axis (z=0)
    horn_pad = paddle(K.HORN_PLATE_Y0, K.HORN_PLATE_Y1, sole_top - 0.01, 0.0)
    bear_pad = paddle(K.BEARING_PLATE_Y0, K.BEARING_PLATE_Y1, sole_top - 0.01, 0.0)
    # gussets: thicker roots so the 3 mm paddles don't snap off the sole
    gusset_h = 6.0
    g1 = Box(2 * K.PADDLE_RADIUS + 6, K.HORN_PLATE_Y1 - K.HORN_PLATE_Y0 + 3, gusset_h,
             align=(Align.CENTER, Align.MIN, Align.MIN)).moved(Location((0, K.HORN_PLATE_Y0 - 3, sole_top - 0.01)))
    g2 = Box(2 * K.PADDLE_RADIUS + 6, K.BEARING_PLATE_Y1 - K.BEARING_PLATE_Y0 + 3, gusset_h,
             align=(Align.CENTER, Align.MIN, Align.MIN)).moved(Location((0, K.BEARING_PLATE_Y0, sole_top - 0.01)))
    solid = sole + horn_pad + bear_pad + g1 + g2

    cutters = horn_paddle_cutters(K.HORN_PLATE_Y0, K.HORN_PLATE_Y1, 0.0) \
        + bearing_paddle_cutter(K.BEARING_PLATE_Y1, 0.0)
    # O-ring grooves, front and back of the sole
    x_front = P.FOOT_LENGTH - P.ANKLE_X_FROM_HEEL - 16
    x_back = -P.ANKLE_X_FROM_HEEL + 10
    cutters = cutters + _oring_groove(x_front) + _oring_groove(x_back)
    # contact-switch pockets (6×6 tactile), from the top of the sole, with a Ø2.5 plunger hole
    sw_l, sw_w, sw_h = P.FOOT_SWITCH_SIZE
    for x in (x_front + 14, x_back - 8):
        pocket = Box(sw_l + 0.4, sw_w + 0.4, sw_h, align=(Align.CENTER, Align.CENTER, Align.MAX))
        pocket = pocket.moved(Location((x, LATERAL_CENTRE, sole_top + 0.01)))
        plunger = Cylinder(1.25, 10, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(
            Location((x, LATERAL_CENTRE, sole_z0)))
        cutters = cutters + pocket + plunger
    solid = solid - cutters

    return CoguPart(
        name="foot",
        solid=solid,
        print_rotation=Rotation(0, 0, 0),
        print_notes="Sole down on the bed. Fork paddles print vertically. O-ring grooves are on the "
                    "bed side (first layers) — use a brim. PETG.",
    )
