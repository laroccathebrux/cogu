"""Simplified Feetech STS3215 model: visual solid + pocket cutter.

Servo-local frame (see parameters.ServoSTS3215): origin on the shaft axis at the
horn-side case face, +Y = shaft axis out through the horn, +X along the body
towards the back end, Z across the width.
"""

from __future__ import annotations

from build123d import Align, Axis, Box, Cylinder, Location, Part, Plane, Rotation

from cad import parameters as P

S = P.SERVO


def _cyl_y(radius: float, y0: float, y1: float, x: float = 0.0, z: float = 0.0) -> Part:
    """Cylinder along +Y from y0 to y1 at (x, z)."""
    c = Cylinder(radius, y1 - y0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    # Cylinder is built along Z; rotate so its axis is +Y, then move
    c = c.rotate(Axis.X, -90)          # local +Z -> +Y
    return c.moved(Location((x, y0, z)))


def servo_solid(with_horn: bool = True, with_back_disc: bool = True) -> Part:
    """Visual/reference solid of the servo (case + horn + back disc + hubs)."""
    body = Box(S.body_length, S.case_height, S.body_width,
               align=(Align.MIN, Align.MAX, Align.CENTER))
    body = body.fillet(S.corner_radius, body.edges().filter_by(Axis.Y))
    body = body.moved(Location((S.x_min, 0, 0)))
    solid = body
    if with_horn:
        hub = _cyl_y(S.horn_hub_diameter / 2, 0, S.horn_hub_height)
        disc = _cyl_y(S.horn_disc_diameter / 2, S.horn_hub_height, S.horn_top_height)
        solid = solid + hub + disc
    if with_back_disc:
        yb = S.y_back_face
        bosses = Box(S.body_length - 2, S.back_boss_height, S.body_width - 2,
                     align=(Align.MIN, Align.MAX, Align.CENTER)).moved(Location((S.x_min + 1, yb, 0)))
        hub = _cyl_y(S.horn_hub_diameter / 2, yb - S.back_disc_bottom, yb)
        disc = _cyl_y(S.back_disc_diameter / 2, yb - S.back_disc_bottom,
                      yb - S.back_disc_bottom + S.back_disc_thickness)
        solid = solid + bosses + hub + disc
    return solid


def servo_pocket(clearance: float = P.CLEARANCE_LOOSE, open_top: float = 0.0, open_back: bool = False) -> Part:
    """Cutter for a servo pocket: the case envelope grown by `clearance` on every side.

    `open_top` extends the pocket by that amount along +X (the back end of the servo),
    so the servo can slide into the pocket from the end of a link.
    The pocket spans the case height PLUS the back-cover bosses along Y (horn face to the
    top of the bosses, where the back disc starts = cad.stack.BACK_DISC_Y1), so the housing's
    back wall really begins at HOUSING_BACK_WALL_Y1. The horn hub/disc and the back disc are
    NOT included (they sit outside the link walls).
    """
    length = S.body_length + 2 * clearance + open_top
    pocket = Box(length, S.case_height + S.back_boss_height + 2 * clearance, S.body_width + 2 * clearance,
                 align=(Align.MIN, Align.MAX, Align.CENTER))
    pocket = pocket.moved(Location((S.x_min - clearance, clearance, 0)))
    return pocket


def horn_screw_locations(y: float = 0.0) -> list[Location]:
    """The 4 horn/back-disc screw positions (PCD 14) in the servo frame, axis along Y.

    Returned locations have local +Z along −Y (pointing from the horn into the mating
    plate), so they can be used directly as cutter locations on a plate whose face
    touches the horn at height `y`.
    """
    r = S.horn_hole_pcd / 2
    locs = []
    for x, z in ((r, 0), (-r, 0), (0, r), (0, -r)):
        locs.append(Location((x, y, z)) * Location(Rotation(90, 0, 0)))
    return locs


def mount_hole_locations(side: str = "horn") -> list[Location]:
    """Mounting pilot-hole positions on the horn-side (+Y face) or back-side (−Y face)."""
    xs = S.mount_holes_x_horn_side if side == "horn" else S.mount_holes_x_back_side
    y = 0.0 if side == "horn" else S.y_back_face
    out = []
    for x in xs:
        for z in (S.mount_hole_z, -S.mount_hole_z):
            out.append(Location((x, y, z)))
    return out
