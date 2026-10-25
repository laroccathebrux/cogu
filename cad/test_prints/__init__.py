"""Calibration prints. Print these BEFORE any robot part.

- insert_test: block with 4 × M3 insert holes (§3 geometry) to dial the soldering-iron
  temperature (PLA ≈ 200 °C, PETG ≈ 230 °C) and check the insert sits flush.
- fit_test: pins and holes at 0.1 / 0.2 / 0.3 / 0.4 mm clearance so you know which
  clearance your printer actually produces (the robot assumes 0.2–0.3 mm).
- bearing_test: one 688ZZ seat (Ø16 + CLEARANCE_TIGHT) and one Ø(8 − 0.15) stub.
"""

from __future__ import annotations

from build123d import Align, Box, Cylinder, Location, Part, Rotation

from cad import parameters as P
from cad.part import CoguPart, InsertHole, insert_hole_cutter


def build_insert_test() -> CoguPart:
    size, h = 30.0, 10.0
    block = Box(size, size, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
    inserts = []
    for x in (8, -8):
        for y in (8, -8):
            loc = Location((x, y, h)) * Location((0, 0, 0), (180, 0, 0))
            inserts.append(InsertHole(loc))
            block = block - insert_hole_cutter(loc)
    return CoguPart("insert_test", block, print_notes="Flat on the bed. Heat-set 4 inserts, check flushness.",
                    insert_holes=inserts)


def build_fit_test() -> CoguPart:
    clearances = (0.1, 0.2, 0.3, 0.4)
    pitch = 14.0
    length = pitch * len(clearances) + 6
    plate = Box(length, 30, 4, align=(Align.MIN, Align.CENTER, Align.MIN))
    pin_d = 6.0
    for i, c in enumerate(clearances):
        x = 3 + pitch * i + pitch / 2
        # hole at Ø(pin + clearance) in the top row, a pin of Ø pin_d in the bottom row
        plate = plate - Cylinder((pin_d + c) / 2, 10, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(
            Location((x, 7, 2)))
        plate = plate + Cylinder(pin_d / 2, 8, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(
            Location((x, -7, 4)))
        # engrave the clearance as a notch count (i+1 notches) so no text is needed
        for n in range(i + 1):
            plate = plate - Box(1.0, 3.0, 1.0, align=(Align.CENTER, Align.CENTER, Align.MAX)).moved(
                Location((x - 3 + n * 2, 13.5, 4)))
    return CoguPart("fit_test", plate,
                    print_notes="Flat on the bed. Pins Ø6, holes Ø6 + 0.1/0.2/0.3/0.4 (notches = tenths). "
                                "The robot assumes 0.2–0.3 mm: the pin must slide into the 0.2 hole by hand.")


def build_bearing_test() -> CoguPart:
    b = P.BEARING_688
    block = Box(40, 22, 8, align=(Align.MIN, Align.CENTER, Align.MIN))
    seat = Cylinder(b.seat_diameter / 2, b.width, align=(Align.CENTER, Align.CENTER, Align.MAX)).moved(Location((11, 0, 8)))
    through = Cylinder(4.5, 20, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(Location((11, 0, 4)))
    stub = Cylinder(b.stub_diameter / 2, P.BACK_HUB_STUB_LENGTH, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(
        Location((30, 0, 8)))
    return CoguPart("bearing_test", block - seat - through + stub,
                    print_notes="Flat on the bed. The 688ZZ must press into the seat by thumb; the stub must "
                                "turn freely inside the bearing bore.")


def build_test_prints() -> dict[str, CoguPart]:
    return {p.name: p for p in (build_insert_test(), build_fit_test(), build_bearing_test())}
