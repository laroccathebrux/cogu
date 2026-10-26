"""Printed hub bolted to the servo's back (idler) disc; its stub runs in the 688ZZ.

Joint-local frame (same as cad.stack): origin on the joint axis at the horn-side
case face, +Y towards the horn. The hub lives on the −Y side.
"""

from __future__ import annotations

from build123d import Align, Axis, Cylinder, Location, Rotation, Vector

from cad import parameters as P
from cad import stack as K
from cad.part import CoguPart

S = P.SERVO


def build_back_hub() -> CoguPart:
    plate = Cylinder(K.HUB_PLATE_RADIUS, P.BACK_HUB_PLATE_THICKNESS,
                     align=(Align.CENTER, Align.CENTER, Align.MIN))
    stub = Cylinder(P.BEARING_688.stub_diameter / 2, P.BACK_HUB_STUB_LENGTH + 0.01,
                    align=(Align.CENTER, Align.CENTER, Align.MAX))
    hub = plate + stub          # built along +Z: plate z∈[0,4], stub z∈[-5,0]
    # 4 × M2 clearance holes on PCD 14 with 2 mm counterbores on the stub side (z=0 face)
    r = S.horn_hole_pcd / 2
    for dx, dy in ((r, 0), (-r, 0), (0, r), (0, -r)):
        hole = Cylinder(P.M2_CLEARANCE_HOLE / 2, 20, align=(Align.CENTER, Align.CENTER, Align.CENTER))
        cbore = Cylinder(P.M2_HEAD_DIAMETER / 2 + 0.2, 2.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
        hub = hub - (hole + cbore).moved(Location((dx, dy, 0)))
    # orient into the joint frame: local +Z -> +Y (towards the servo), plate top at HUB_PLATE_Y1
    hub = hub.rotate(Axis.X, -90).moved(Location((0, K.HUB_PLATE_Y0, 0)))
    return CoguPart(
        name="back_hub",
        solid=hub,
        print_rotation=Rotation(90, 0, 0),
        print_notes="Plate face (disc side) on the bed, stub up. 100 % infill recommended (small part).",
    )
