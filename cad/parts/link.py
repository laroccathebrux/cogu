"""Leg link (thigh and shin share this design) and the generic servo housing.

Link frame: origin on the DISTAL joint axis (the servo housed in this link),
+Y along the joint axes (towards the horn side), +Z up the link towards the
proximal axis at z = length, +X forward.

The link houses the distal servo (shaft at the origin, body pointing +Z) and
ends in a fork around the proximal servo housed in the part above: a horn
paddle on +Y and a 688ZZ bearing paddle on −Y.
"""

from __future__ import annotations

from build123d import (
    Align,
    Axis,
    Box,
    Circle,
    Cylinder,
    Location,
    Part,
    Plane,
    Rectangle,
    Rotation,
    Vector,
    extrude,
)

from cad import parameters as P
from cad import stack as K
from cad.part import CoguPart, bearing_seat_cutter, m2_clearance_cutter
from cad.parts import sts3215

S = P.SERVO

# ---------------------------------------------------------------------------
# Generic servo housing (housing frame = link frame around the distal axis)
# ---------------------------------------------------------------------------

def _servo_to_housing(part: Part) -> Part:
    """Servo frame -> housing frame: servo X (along body) -> +Z, servo Y (shaft) -> +Y, servo Z -> -X."""
    return part.rotate(Axis.Y, -90)


def servo_housing(open_top: float = 0.0, bus_window: bool = False) -> tuple[Part, Part]:
    """Return (housing_solid, cutters) in the housing frame. Subtract cutters after unions.
    `bus_window`: open the back wall over the servo's two bus sockets, so the plugs pass through."""
    z0, z1 = K.HOUSING_Z_BOTTOM, K.HOUSING_Z_TOP + open_top
    housing = Box(2 * K.HOUSING_X_HALF, K.HOUSING_Y1 - K.HOUSING_Y0, z1 - z0,
                  align=(Align.CENTER, Align.MIN, Align.MIN))
    housing = housing.fillet(3.0, housing.edges().filter_by(Axis.Y))
    housing = housing.moved(Location((0, K.HOUSING_Y0, z0)))

    pocket = _servo_to_housing(sts3215.servo_pocket(open_top=open_top))
    # hole in the back wall for the back disc + printed hub
    back_hole = Cylinder(K.BACK_DISC_HOLE_D / 2, P.LINK_WALL + 2 * P.CLEARANCE_LOOSE + 2,
                         align=(Align.CENTER, Align.CENTER, Align.CENTER))
    back_hole = back_hole.rotate(Axis.X, 90).moved(Location((0, K.HOUSING_Y0 + P.LINK_WALL / 2, 0)))
    # 4 × M2 clearance holes through the back wall into the servo's back-face pilots
    screws = None
    for loc in sts3215.mount_hole_locations("back"):
        x_s, _, z_s = loc.position
        # servo (x_s, z_s) -> housing (x=-z_s, z=x_s)
        c = Cylinder(P.M2_CLEARANCE_HOLE / 2, 12, align=(Align.CENTER, Align.CENTER, Align.CENTER))
        c = c.rotate(Axis.X, 90).moved(Location((-z_s, K.HOUSING_Y0 + P.LINK_WALL / 2, x_s)))
        screws = c if screws is None else screws + c
    cutters = pocket + back_hole + screws
    if bus_window:
        # starts inside the back-disc hole (r 10.4) so no sliver is left between them; |x| stops short of the
        # heads of the screws at servo (8.3, ±10.25)
        x0, x1 = 10.3, S.bus_socket_x[1] + 0.5
        half = S.bus_socket_half_width - 0.3
        window = Box(2 * half, P.LINK_WALL + 2 * P.CLEARANCE_LOOSE + 2, x1 - x0,
                     align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((0, K.HOUSING_Y0 + P.LINK_WALL / 2, x0)))
        cutters = cutters + window
    return housing, cutters


def paddle(y0: float, y1: float, z_from: float, z_axis: float, radius: float = K.PADDLE_RADIUS) -> Part:
    """Vertical plate in the XZ plane between y0..y1: a rectangle from z_from up to z_axis
    capped by a circle of `radius` centred on the proximal axis."""
    thickness = y1 - y0
    with_rect = Box(2 * radius, thickness, z_axis - z_from, align=(Align.CENTER, Align.MIN, Align.MIN))
    with_rect = with_rect.moved(Location((0, y0, z_from)))
    cap = Cylinder(radius, thickness, align=(Align.CENTER, Align.CENTER, Align.MIN))
    cap = cap.rotate(Axis.X, -90).moved(Location((0, y0, z_axis)))
    return with_rect + cap


def horn_paddle_cutters(y0: float, y1: float, z_axis: float) -> Part:
    """4 × M2 clearance holes (PCD 14) through a horn paddle."""
    out = None
    r = S.horn_hole_pcd / 2
    for dx, dz in ((r, 0), (-r, 0), (0, r), (0, -r)):
        c = Cylinder(P.M2_CLEARANCE_HOLE / 2, (y1 - y0) + 2, align=(Align.CENTER, Align.CENTER, Align.CENTER))
        c = c.rotate(Axis.X, 90).moved(Location((dx, (y0 + y1) / 2, z_axis + dz)))
        out = c if out is None else out + c
    return out


def bearing_paddle_cutter(y_inner: float, z_axis: float) -> Part:
    """688ZZ seat opening on the paddle's inner face (towards +Y) at the proximal axis."""
    seat = Cylinder(P.BEARING_688.seat_diameter / 2, P.BEARING_688.width,
                    align=(Align.CENTER, Align.CENTER, Align.MIN))
    seat = seat.rotate(Axis.X, 90)   # +Z -> -Y: pocket goes from the inner face outward (−Y)
    through = Cylinder(4.5, 30, align=(Align.CENTER, Align.CENTER, Align.CENTER)).rotate(Axis.X, 90)
    return (seat + through).moved(Location((0, y_inner, z_axis)))


# ---------------------------------------------------------------------------
# The link
# ---------------------------------------------------------------------------

CHEEK_Z0 = 14.0   # cheeks start above the distal paddle sweep (radius 12.6) + clearance


def build_link(name: str, length: float) -> CoguPart:
    housing, cutters = servo_housing()

    # horn-side cheek: two strips beside the pocket mouth + the top cap, extended to the horn plate
    strip_w = K.HOUSING_X_HALF - (S.body_width / 2 + P.CLEARANCE_LOOSE)
    cheek_h = K.HOUSING_Z_TOP - CHEEK_Z0
    strips = None
    for sx in (K.HOUSING_X_HALF - strip_w / 2, -(K.HOUSING_X_HALF - strip_w / 2)):
        b = Box(strip_w, K.HORN_PLATE_Y1, cheek_h, align=(Align.CENTER, Align.MIN, Align.MIN))
        b = b.moved(Location((sx, 0, CHEEK_Z0)))
        strips = b if strips is None else strips + b
    pocket_top = S.x_max + P.CLEARANCE_LOOSE
    top_cap = Box(2 * K.HOUSING_X_HALF, K.HORN_PLATE_Y1, K.HOUSING_Z_TOP - pocket_top,
                  align=(Align.CENTER, Align.MIN, Align.MIN)).moved(Location((0, 0, pocket_top)))
    # bearing-side cheek: solid block behind the back wall
    back_cheek = Box(2 * K.HOUSING_X_HALF, K.HOUSING_Y0 - K.BEARING_PLATE_Y0, cheek_h,
                     align=(Align.CENTER, Align.MIN, Align.MIN)).moved(Location((0, K.BEARING_PLATE_Y0, CHEEK_Z0)))

    paddle_z_from = CHEEK_Z0 + 8.0
    horn_pad = paddle(K.HORN_PLATE_Y0, K.HORN_PLATE_Y1, paddle_z_from, length)
    bear_pad = paddle(K.BEARING_PLATE_Y0, K.BEARING_PLATE_Y1, paddle_z_from, length)

    solid = housing + strips + top_cap + back_cheek + horn_pad + bear_pad

    # cable channel: from the top of the pocket (servo connector end) up through the top cap
    channel = Box(P.CABLE_CHANNEL_WIDTH + 4, P.CABLE_CHANNEL_DEPTH + 2, 30,
                  align=(Align.CENTER, Align.CENTER, Align.MIN))
    channel = channel.moved(Location((0, S.y_back_face / 2, pocket_top - 1)))
    # driver access holes through the back cheek for the upper pair of servo screws
    access = None
    for loc in sts3215.mount_hole_locations("back"):
        x_s, _, z_s = loc.position
        if x_s < CHEEK_Z0:
            continue
        a = Cylinder(2.0, 40, align=(Align.CENTER, Align.CENTER, Align.CENTER)).rotate(Axis.X, 90)
        a = a.moved(Location((-z_s, K.HOUSING_Y0 - 5, x_s)))
        access = a if access is None else access + a

    all_cutters = cutters + channel + access \
        + horn_paddle_cutters(K.HORN_PLATE_Y0, K.HORN_PLATE_Y1, length) \
        + bearing_paddle_cutter(K.BEARING_PLATE_Y1, length)
    solid = solid - all_cutters

    servo_com = Vector(0, S.y_back_face / 2, (S.x_min + S.x_max) / 2)
    return CoguPart(
        name=name,
        solid=solid,
        # stands on the housing bottom (distal end): pocket is a vertical cavity, paddles vertical.
        print_rotation=Rotation(0, 0, 0),
        print_notes="Stand on the housing bottom (distal end). Tree supports only under the two cheeks "
                    "(10 mm overhangs at z = 14). Bearing seat and horn holes are horizontal. "
                    "PETG, 3+ perimeters.",
        attached_masses=[(f"{name}_servo", S.mass_g, servo_com)],
    )


def build_thigh() -> CoguPart:
    return build_link("thigh", P.THIGH_LENGTH)


def build_shin() -> CoguPart:
    return build_link("shin", P.SHIN_LENGTH)
