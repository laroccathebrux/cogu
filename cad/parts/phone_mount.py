"""Phone mount (ADR-002): hip cover over the yaw servos + cradle that holds the phone upright,
rear cameras facing forward (+X).

Both parts are built directly in the pelvis frame (origin between the hip-yaw axes on the yaw
servos' horn face, +Z up, +X forward) and placed at the origin of the pelvis body.

  hip_cover    open-bottom box from the pelvis plate (z = 0) to COVER_TOP_Z, bolted to the 4 pelvis
               bosses through screw tubes. Front bay: the 2S battery in a cage hanging from the
               deck. Rear bay: the "spinal cord" board (ESP32-S3 + IMU + bus-servo interface).
               2 M3 inserts on the centre line take the cradle. Printed deck-down: no supports.
  phone_cradle base plate on the deck + two C-rails gripping the phone's long edges. The rails
               are straight extrusions along Z, so printed base-down they need no supports.
"""

from __future__ import annotations

from build123d import Align, Axis, Box, Cylinder, Location, Part, Rotation, Vector

from cad import parameters as P
from cad.part import CoguPart, InsertHole, insert_hole_cutter, rounded_box

PH = P.PHONE
DECK_UNDER_Z = P.COVER_TOP_Z - P.COVER_DECK
BOSS_TOP_Z = P.INSERT_M3.hole_depth + 2.0            # top of the pelvis bosses (cad.parts.hip)
COUNTERBORE_D = 6.5                                    # M3 socket head Ø5.5 + clearance
PHONE_Z0 = P.COVER_TOP_Z + P.CRADLE_BASE               # phone bottom edge
SLOT_HALF_X = PH["thickness"] / 2 + P.CRADLE_CLEARANCE  # half width of the rails' slot (phone thickness)
RAIL_INNER_Y = PH["width"] / 2 + P.CRADLE_CLEARANCE    # rail slot bottom (phone edge + clearance)


def _cover_boss_positions() -> list[tuple[float, float]]:
    return [(bx, by) for bx in (P.PELVIS_BOSS_X, -P.PELVIS_BOSS_X) for by in (P.PELVIS_BOSS_Y, -P.PELVIS_BOSS_Y)]


def build_hip_cover() -> CoguPart:
    x0, x1 = P.COVER_X
    length, width = x1 - x0, P.PELVIS_WIDTH
    xc = (x0 + x1) / 2
    w = P.COVER_WALL
    outer = rounded_box(length, width, P.COVER_TOP_Z, P.COVER_CORNER_R).moved(Location((xc, 0, 0)))
    outer = outer.fillet(3.0, outer.edges().group_by(Axis.Z)[-1])            # soft top edge
    inner = rounded_box(length - 2 * w, width - 2 * w, DECK_UNDER_Z + 1.0, P.COVER_CORNER_R - w).moved(
        Location((xc, 0, -1.0)))
    solid: Part = outer - inner
    cutters: Part | None = None

    # screw tubes down to the pelvis bosses: Ø6.5 counterbore from the deck, 3 mm web, Ø3.4 through
    tube_r = COUNTERBORE_D / 2 + 1.2
    for bx, by in _cover_boss_positions():
        tube = Cylinder(tube_r, DECK_UNDER_Z - BOSS_TOP_Z + 0.01, align=(Align.CENTER, Align.CENTER, Align.MIN))
        solid = solid + tube.moved(Location((bx, by, BOSS_TOP_Z)))
        cb = Cylinder(COUNTERBORE_D / 2, P.COVER_TOP_Z - BOSS_TOP_Z - P.COVER_TUBE_WEB + 1.0,
                      align=(Align.CENTER, Align.CENTER, Align.MAX)).moved(Location((bx, by, P.COVER_TOP_Z + 1.0)))
        through = Cylinder(P.M3_CLEARANCE_HOLE / 2, 20, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(
            Location((bx, by, BOSS_TOP_Z)))
        c = cb + through
        cutters = c if cutters is None else cutters + c

    # 2 M3 inserts for the cradle, on bosses hanging from the deck between the yaw servos
    inserts: list[InsertHole] = []
    ins_r = P.INSERT_M3.hole_diameter / 2 + P.INSERT_M3.wall_min + 0.3
    for ix in P.COVER_CRADLE_INSERTS_X:
        boss_h = P.INSERT_M3.hole_depth + 2.0 - P.COVER_DECK
        solid = solid + Cylinder(ins_r, boss_h + 0.01, align=(Align.CENTER, Align.CENTER, Align.MAX)).moved(
            Location((ix, 0, DECK_UNDER_Z + 0.01)))
        loc = Location((ix, 0, P.COVER_TOP_Z)) * Location((0, 0, 0), (180, 0, 0))   # axis pointing down
        inserts.append(InsertHole(loc))
        cutters = cutters + insert_hole_cutter(loc)

    # battery cage hanging from the deck (walls + floor), front bay
    bx, by, bz = P.BATTERY_2S["size"]
    cx, cy, cz = P.BATTERY_POS
    cw = P.BATTERY_CAGE_WALL
    floor_z = cz - bz / 2
    cage_out = Box(bx + 0.6 + 2 * cw, by + 0.6 + 2 * cw, DECK_UNDER_Z - (floor_z - cw) + 0.01,
                   align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((cx, cy, floor_z - cw)))
    cage_in = Box(bx + 0.6, by + 0.6, DECK_UNDER_Z - floor_z + 1.0,
                  align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((cx, cy, floor_z)))
    # strap window in the floor (a hook-and-loop strap holds the pack) and a lead exit at the rear
    window = Box(bx - 4, by - 30, 10, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(
        Location((cx, cy, floor_z - cw / 2)))
    lead = Box(4 * cw, 14, 10, align=(Align.CENTER, Align.CENTER, Align.MAX)).moved(
        Location((cx - bx / 2 - cw / 2, cy, DECK_UNDER_Z - 2)))
    solid = solid + (cage_out - cage_in)
    cutters = cutters + window + lead

    # cable pass from the cradle slot down into the cover, power switch through the left wall
    cutters = cutters + Box(2 * SLOT_HALF_X, P.CRADLE_CABLE_SLOT, 20, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(
        Location((P.PHONE_X, 0, P.COVER_TOP_Z)))
    sw = P.COVER_SWITCH
    cutters = cutters + Cylinder(sw["d"] / 2, 10, rotation=(90, 0, 0)).moved(Location((sw["x"], width / 2, sw["z"])))
    solid = solid - cutters

    masses = [
        ("battery_2s", P.BATTERY_2S["mass_g"], Vector(*P.BATTERY_POS)),
        ("spine_board", P.SPINE["mass_g"], Vector(*P.SPINE_POS)),
        ("misc_electronics", P.MISC_ELECTRONICS_MASS_G, Vector(-20.0, 0.0, 20.0)),
    ]
    return CoguPart(
        name="hip_cover",
        solid=solid,
        print_rotation=Rotation(180, 0, 0),
        print_notes="Deck down on the bed, walls and tubes up: no supports. Heat-set the 2 M3 inserts "
                    "from the deck side. The battery cage floor bridges 13 mm.",
        insert_holes=inserts,
        attached_masses=masses,
        infill_factor=0.9,           # thin walls: almost all perimeters
    )


def build_phone_cradle() -> CoguPart:
    x0, x1 = P.CRADLE_BASE_X
    w, lip = P.CRADLE_WALL, P.CRADLE_LIP
    rail_out_y = RAIL_INNER_Y + w
    base = rounded_box(x1 - x0, 2 * rail_out_y, P.CRADLE_BASE, 6.0).moved(
        Location(((x0 + x1) / 2, 0, P.COVER_TOP_Z)))
    solid: Part = base
    rail_half_x = SLOT_HALF_X + w
    for s in (1.0, -1.0):
        y_in = s * (PH["width"] / 2 - lip)                # lips reach this far over the phone faces
        y_out = s * rail_out_y
        y_lo, y_hi = min(y_in, y_out), max(y_in, y_out)
        rail = Box(2 * rail_half_x, y_hi - y_lo, P.CRADLE_RAIL_HEIGHT + 0.01,
                   align=(Align.CENTER, Align.MIN, Align.MIN)).moved(Location((P.PHONE_X, y_lo, PHONE_Z0 - 0.01)))
        slot_lo, slot_hi = sorted((s * (PH["width"] / 2 - lip - 1.0), s * RAIL_INNER_Y))
        slot = Box(2 * SLOT_HALF_X, slot_hi - slot_lo, P.CRADLE_RAIL_HEIGHT + 2,
                   align=(Align.CENTER, Align.MIN, Align.MIN)).moved(Location((P.PHONE_X, slot_lo, PHONE_Z0 - 0.01)))
        rail = rail - slot
        try:
            rail = rail.fillet(1.0, rail.edges().group_by(Axis.Z)[-1])    # rounded rail tops (bumpers)
        except ValueError:
            pass                                                         # thin lips: keep sharp tops
        solid = solid + rail
    cutters: Part | None = None
    for ix in P.COVER_CRADLE_INSERTS_X:                   # M3 × 8: 3 mm head in a counterbore, 5 mm into the insert
        cb = Cylinder(COUNTERBORE_D / 2, 4.0, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(
            Location((ix, 0, PHONE_Z0 - 3.0)))
        th = Cylinder(P.M3_CLEARANCE_HOLE / 2, 20, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(
            Location((ix, 0, P.COVER_TOP_Z)))
        c = cb + th
        cutters = c if cutters is None else cutters + c
    cutters = cutters + Box(2 * SLOT_HALF_X, P.CRADLE_CABLE_SLOT, 20, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(
        Location((P.PHONE_X, 0, P.COVER_TOP_Z)))
    solid = solid - cutters
    return CoguPart(
        name="phone_cradle",
        solid=solid,
        print_rotation=Rotation(0, 0, 0),
        print_notes="Base on the bed, rails standing: straight extrusions, no supports. Generated for the "
                    "phone in parameters.PHONE — reprint for another phone.",
        attached_masses=[("phone", PH["mass_g"], Vector(P.PHONE_X, 0.0, PHONE_Z0 + PH["height"] / 2))],
    )


def phone_solid() -> tuple[Part, Part]:
    """(phone body, rear camera bump) in the pelvis frame, for renders and the MJCF (visual)."""
    t, wd, h, r = PH["thickness"], PH["width"], PH["height"], PH["corner_r"]
    body = Box(t, wd, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
    body = body.fillet(r, body.edges().filter_by(Axis.X))
    body = body.moved(Location((P.PHONE_X, 0, PHONE_Z0)))
    cam = P.PHONE_CAMERA
    bw, bh, bt = cam["bump"]
    bump = Box(bt, bw, bh, align=(Align.MIN, Align.MAX, Align.MAX))
    bump = bump.fillet(6.0, bump.edges().filter_by(Axis.X))
    bump = bump.moved(Location((P.PHONE_X + t / 2 - 0.01, wd / 2 - 4.0, PHONE_Z0 + h - 4.0)))
    return body, bump


def phone_camera_pos() -> tuple[float, float, float]:
    """Centre of the main rear camera lens (pelvis frame, mm)."""
    cam = P.PHONE_CAMERA
    return (P.PHONE_X + PH["thickness"] / 2 + cam["bump"][2],
            PH["width"] / 2 - cam["from_left_edge"],
            PHONE_Z0 + PH["height"] - cam["from_top"])
