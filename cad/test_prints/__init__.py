"""Calibration prints. Print these BEFORE any robot part.

- insert_test: block with 4 × M3 insert holes (§3 geometry) to dial the soldering-iron
  temperature (PLA ≈ 200 °C, PETG ≈ 230 °C) and check the insert sits flush.
- fit_test: pins and holes at 0.1 / 0.2 / 0.3 / 0.4 mm clearance so you know which
  clearance your printer actually produces (the A1 Mini: snug at 0.1, free at 0.3).
- bearing_test: three 688ZZ seats, Ø16 + 0.05 / CLEARANCE_TIGHT / 0.15 (1 / 2 / 3 notches), and one Ø(8 − 0.15)
  stub: the bearing must press in by thumb in the middle one.
- servo_fit_test: the leg links' servo housing alone (pocket, back wall with the bus window, screw holes): an
  STS3215 must slide in, the back disc sit in its hole, the 4 M2 holes line up and the plugs go in.
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
                                "The robot assumes a snug fit at 0.1 and a free one at 0.3.")


def build_bearing_test() -> CoguPart:
    b = P.BEARING_688
    seats = (b.seat_diameter - 0.05, b.seat_diameter, b.seat_diameter + 0.05)
    pitch = 21.0
    block = Box(pitch * len(seats) + 19, 24, 8, align=(Align.MIN, Align.CENTER, Align.MIN))
    for i, d in enumerate(seats):
        x = 11 + pitch * i
        block = block - Cylinder(d / 2, b.width, align=(Align.CENTER, Align.CENTER, Align.MAX)).moved(
            Location((x, 0, 8)))
        block = block - Cylinder(4.5, 20, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(Location((x, 0, 4)))
        for n in range(i + 1):                       # i+1 notches on the edge: 1 = tighter, 2 = the CAD's, 3 = looser
            block = block - Box(1.0, 3.0, 1.0, align=(Align.CENTER, Align.CENTER, Align.MAX)).moved(
                Location((x - 2 + n * 2, 10.5, 8)))
    stub = Cylinder(b.stub_diameter / 2, P.BACK_HUB_STUB_LENGTH, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(
        Location((11 + pitch * len(seats) - 2, 0, 8)))
    return CoguPart("bearing_test", block + stub,
                    print_notes="Flat on the bed. Seats Ø%.2f / %.2f / %.2f (1 / 2 / 3 notches): the 688ZZ must press "
                                "into the middle one by thumb. The stub must turn freely inside the bearing bore."
                                % seats)


def build_servo_fit_test() -> CoguPart:
    from cad.parts.link import servo_housing
    housing, cutters = servo_housing(bus_window=True)
    return CoguPart("servo_fit_test", housing - cutters,
                    print_notes="Stand on the housing bottom, like the thigh. Slide an STS3215 in: snug, no force; "
                                "the back disc in its hole, the 4 M2 holes over the servo's, both bus plugs in.")


def build_test_prints() -> dict[str, CoguPart]:
    return {p.name: p for p in (build_insert_test(), build_fit_test(), build_bearing_test(), build_servo_fit_test())}
