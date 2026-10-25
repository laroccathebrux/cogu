"""Screen bezel: the printed frame with 4 visible M3 screws that holds the CYD on the body's face.

Pelvis frame (like the body). The plate sits on the flat face (x = BODY_FACE_X) and clamps
the display glass against the ledges inside the face box.
"""

from __future__ import annotations

from build123d import Align, Axis, Box, Cylinder, Location, Rotation, Solid, Vector

from cad import parameters as P
from cad.part import CoguPart, rounded_box
from cad.parts import body as B

THICKNESS = 3.0


def build_bezel() -> CoguPart:
    aw, ah = P.CYD["active"]
    plate = rounded_box(B.FACE_W, B.FACE_H, THICKNESS, P.BODY_FACE_CORNER_R, align=(Align.CENTER, Align.CENTER, Align.MIN))
    window = rounded_box(aw + 1.0, ah + 1.0, THICKNESS + 2, 1.0, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(
        Location((P.CYD_ACTIVE_CENTRE_Y, 0, -1)))
    # 45° chamfer around the window on the outside
    cham = Solid.extrude_taper(rounded_box(aw + 1.0, ah + 1.0, 0.1, 1.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
                               .faces().sort_by(Axis.Z)[0], (0, 0, 1.5), taper=-45)
    plate = plate - window - cham.moved(Location((P.CYD_ACTIVE_CENTRE_Y, 0, THICKNESS - 1.5)))
    for sy in (1, -1):
        for sz in (1, -1):
            hole = Cylinder(P.M3_CLEARANCE_HOLE / 2, 10, align=(Align.CENTER, Align.CENTER, Align.CENTER))
            cbore = Cylinder(P.M3_HEAD_DIAMETER / 2 + 0.3, 2.0, align=(Align.CENTER, Align.CENTER, Align.MAX))
            plate = plate - (hole + cbore).moved(Location((sy * B.FACE_SCREW[0], sz * B.FACE_SCREW[1], THICKNESS)))
    # built in the XY plane with +Z outwards; the face points +X: Z -> X, X (lateral) -> Y, Y (vertical) -> Z
    solid = plate.rotate(Axis.X, 90).rotate(Axis.Z, 90).rotate(Axis.Y, -P.BODY_FACE_TILT_DEG).moved(
        Location((P.BODY_FACE_X, 0, P.BODY_FACE_CENTRE_Z)))
    return CoguPart(
        name="bezel",
        solid=solid,
        print_rotation=Rotation(0, 90 + P.BODY_FACE_TILT_DEG, 0),   # outside face down, flat
        print_notes="Flat, outside face down (the chamfer prints as an overhang of 45°). 4 × M3 × 8 into the "
                    "face-box inserts. Cream or a contrasting colour.",
        infill_factor=0.9,
    )
