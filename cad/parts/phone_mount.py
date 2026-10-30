"""Phone mount (ADR-002): hip cover over the yaw servos + cradle that holds the phone upright,
rear cameras facing forward (+X).

Both parts are built directly in the pelvis frame (origin between the hip-yaw axes on the yaw
servos' horn face, +Z up, +X forward) and placed at the origin of the pelvis body.

  hip_cover    open-bottom drum (or box) from the pelvis plate (z = 0) to COVER_TOP_Z, bolted to the 4 pelvis
               bosses through screw tubes. Front bay: the 2S battery in a cage hanging from the
               deck. Rear bay: the "spinal cord" board (ESP32-S3 + IMU + bus-servo interface).
               2 M3 inserts on the centre line take the cradle. Printed deck-down: no supports.
  phone_cradle base plate on the deck + two C-rails gripping the phone's long edges. The rails
               are straight extrusions along Z, so printed base-down they need no supports.
"""

from __future__ import annotations

from build123d import Align, Axis, Box, Compound, Cylinder, Location, Part, Rotation, Vector

from cad import parameters as P
from cad.part import CoguPart, InsertHole, insert_hole_cutter, rounded_box

PH = P.PHONE
DECK_UNDER_Z = P.COVER_TOP_Z - P.COVER_DECK
BOSS_TOP_Z = P.INSERT_M3.hole_depth + 2.0            # top of the pelvis bosses (cad.parts.hip)
COUNTERBORE_D = 6.5                                    # M3 socket head Ø5.5 + clearance
PHONE_Z0 = P.COVER_TOP_Z + P.CRADLE_BASE               # phone bottom edge
SLOT_HALF_X = PH["thickness"] / 2 + P.CRADLE_CLEARANCE  # half width of the rails' slot (phone thickness)
RAIL_INNER_Y = PH["width"] / 2 + P.CRADLE_CLEARANCE    # rail slot bottom (phone edge + clearance)
DRIVER_CABLE_HOLE = ((5.0, 16.0), -23.0)              # (x size, y size), x centre: through the cradle tongue and the deck


def _cover_boss_positions() -> list[tuple[float, float]]:
    return [(bx, by) for bx in (P.PELVIS_BOSS_X, -P.PELVIS_BOSS_X) for by in (P.PELVIS_BOSS_Y, -P.PELVIS_BOSS_Y)]


def cover_radius() -> float:
    """Drum cover radius: parameters.COVER_RADIUS, or the bubble's outer radius at the deck height so the
    bubble's base ring sits flush on the drum's rim."""
    import math
    if P.COVER_RADIUS:
        return P.COVER_RADIUS
    g = P.GUARD_BUBBLE
    u = (P.COVER_TOP_Z - g["center_z"]) / g["radius_z"]
    return g["radius_xy"] * math.sqrt(max(0.0, 1.0 - u * u))


def build_hip_cover() -> CoguPart:
    w = P.COVER_WALL
    width = P.PELVIS_WIDTH
    if P.COVER_SHAPE == "round":                                          # drum under the bubble
        R = cover_radius()
        outer = Cylinder(R, P.COVER_TOP_Z, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(
            Location((P.COVER_CENTER_X, 0, 0)))
        outer = outer.fillet(2.0, outer.edges().group_by(Axis.Z)[-1])
        inner = Cylinder(R - w, DECK_UNDER_Z + 1.0, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(
            Location((P.COVER_CENTER_X, 0, -1.0)))
    else:
        x0, x1 = P.COVER_X
        length, xc = x1 - x0, (x0 + x1) / 2
        outer = rounded_box(length, width, P.COVER_TOP_Z, P.COVER_CORNER_R).moved(Location((xc, 0, 0)))
        outer = outer.fillet(3.0, outer.edges().group_by(Axis.Z)[-1])        # soft top edge
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
    gap = 2 * P.BATTERY_CAGE_CLEARANCE
    floor_z = cz - bz / 2
    cage_out = Box(bx + gap + 2 * cw, by + gap + 2 * cw, DECK_UNDER_Z - (floor_z - cw) + 0.01,
                   align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((cx, cy, floor_z - cw)))
    cage_in = Box(bx + gap, by + gap, DECK_UNDER_Z - floor_z + 1.0,
                  align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((cx, cy, floor_z)))
    # strap window in the floor (a hook-and-loop strap holds the pack) and a lead exit at the rear
    window = Box(bx - 4, by - 30, 10, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(
        Location((cx, cy, floor_z - cw / 2)))
    lead = Box(4 * cw, 14, 10, align=(Align.CENTER, Align.CENTER, Align.MAX)).moved(
        Location((cx - bx / 2 - cw / 2, cy, DECK_UNDER_Z - 2)))
    solid = solid + (cage_out - cage_in)
    cutters = cutters + window + lead

    # servo leads from the driver board, through the cradle tongue
    cutters = cutters + Box(*DRIVER_CABLE_HOLE[0], 20, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(
        Location((DRIVER_CABLE_HOLE[1], 0, P.COVER_TOP_Z)))
    # cable pass from the cradle slot down into the cover, power switch through the left wall
    cutters = cutters + Box(2 * SLOT_HALF_X, P.CRADLE_CABLE_SLOT, 20, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(
        Location((P.PHONE_X, 0, P.COVER_TOP_Z)))
    sw = P.COVER_SWITCH
    if sw:
        cutters = cutters + Cylinder(sw["d"] / 2, 10, rotation=(90, 0, 0)).moved(Location((sw["x"], width / 2, sw["z"])))
    solid = solid - cutters

    masses = [
        ("battery_2s", P.BATTERY_2S["mass_g"], Vector(*P.BATTERY_POS)),
        ("imu_board", P.IMU_BOARD["mass_g"], Vector(*P.IMU_POS)),
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
    # tongue behind the cradle + two posts carrying the servo driver board (OLED facing backwards)
    tg, sd, pp = P.CRADLE_TONGUE, P.SERVO_DRIVER, P.CRADLE_DRIVER_POSTS
    tongue = rounded_box(x0 + 6.0 - tg["x0"], 2 * tg["half_width"], P.CRADLE_BASE, 4.0).moved(
        Location(((tg["x0"] + x0 + 6.0) / 2, 0, P.COVER_TOP_Z)))
    solid = solid + tongue
    _, board_l, board_h = sd["pcb"]
    post_top = sd["z0"] + board_h + 2.0
    for s in (1.0, -1.0):
        post = Box(pp["thickness"], pp["width"], post_top - PHONE_Z0 + 0.01, align=(Align.MIN, Align.CENTER, Align.MIN))
        solid = solid + post.moved(Location((sd["pcb_front_x"], s * pp["y"], PHONE_Z0 - 0.01)))
    cutters: Part | None = None
    for s in (1.0, -1.0):                                 # M2 pilot holes for the board's four corner holes
        for hz in (sd["z0"] + sd["hole_inset"], sd["z0"] + sd["hole_inset"] + sd["holes"][1]):
            c = Cylinder(P.M2_PILOT_HOLE / 2, pp["thickness"] + 2, rotation=(0, 90, 0)).moved(
                Location((sd["pcb_front_x"] + pp["thickness"] / 2, s * sd["holes"][0] / 2, hz)))
            cutters = c if cutters is None else cutters + c
    cable = Box(*DRIVER_CABLE_HOLE[0], 20, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(
        Location((DRIVER_CABLE_HOLE[1], 0, P.COVER_TOP_Z)))
    cutters = cutters + cable                             # servo leads go down into the drum, to both legs
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
        attached_masses=[("phone", PH["mass_g"], Vector(P.PHONE_X, 0.0, PHONE_Z0 + PH["height"] / 2)),
                         ("servo_driver", P.SERVO_DRIVER["mass_g"],
                          Vector(P.SERVO_DRIVER["pcb_front_x"] - (P.SERVO_DRIVER["pcb"][0] + P.SERVO_DRIVER["components"]) / 2,
                                 0.0, P.SERVO_DRIVER["z0"] + P.SERVO_DRIVER["pcb"][2] / 2))],
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


# ---------------------------------------------------------------------------
# Phone guards (fall protection, under study). Both are concepts: built as one solid each,
# printed in several pieces later. Pelvis frame, like the parts above.
# ---------------------------------------------------------------------------

def build_guard_frame() -> CoguPart:
    """Rectangular frame around the phone: two posts outside the cradle rails and a top bar above the
    phone, deeper than the phone so that a fall on either face lands on the frame, not on the glass."""
    g = P.GUARD_FRAME
    bar = g["bar"]
    depth = PH["thickness"] + 2 * g["depth_beyond_phone"]
    y_in = RAIL_INNER_Y + P.CRADLE_WALL + g["side_clearance"]          # outside the cradle rails
    z0 = P.COVER_TOP_Z
    z_top = PHONE_Z0 + PH["height"] + g["top_clearance"]
    solid: Part | None = None
    for s in (1.0, -1.0):
        post = Box(depth, bar, z_top + bar - z0, align=(Align.CENTER, Align.MIN if s > 0 else Align.MAX, Align.MIN))
        post = post.moved(Location((P.PHONE_X, s * y_in, z0)))
        solid = post if solid is None else solid + post
    top = Box(depth, 2 * (y_in + bar), bar, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(
        Location((P.PHONE_X, 0, z_top)))
    solid = solid + top
    solid = solid.fillet(2.0, solid.edges().filter_by(Axis.Y))          # rounded corners on the bumpers
    # lightening windows through the posts (they only have to stop the phone, not carry load)
    for s in (1.0, -1.0):
        win = Box(depth - 2 * 3.0, bar + 2, (z_top - z0) - 40, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(
            Location((P.PHONE_X, s * (y_in + bar / 2), z0 + 20)))
        solid = solid - win
    return CoguPart(
        name="phone_guard_frame",
        solid=solid,
        print_rotation=Rotation(0, 0, 0),
        print_notes="Concept: frame around the phone, printed flat (posts and top bar as separate pieces).",
        infill_factor=0.9,
        printed_in_pieces=3,
    )


def build_guard_bubble() -> CoguPart:
    """Egg of printed ribs around the phone (like a bubble-football suit): meridian ribs from a base ring on
    the hip cover up to a small crown, an equator hoop, wide gaps between the ribs (the rear cameras look
    through the front one).

    Built as ONE fused solid: the shell is intersected once with a union of plain boxes (rib slabs, hoop,
    base and crown bands). Two other ways failed silently: a union of the curved pieces lost ~45 % of the
    volume, and slabs along the ellipsoid's y axis came out empty when cut one by one; the result is checked
    against a Monte Carlo volume of the same analytic shape. For printing it is cut at the first hoop into a
    lower bowl (no supports) and an upper dome; the TPU hoop tyre, whose lips grip the hoop's top and bottom
    edges, clamps the two halves together."""
    from build123d import Sphere, scale

    g = P.GUARD_BUBBLE
    rx, rz, wall = g["radius_xy"], g["radius_z"], g["wall"]
    cz = g["center_z"]
    z_base = P.COVER_TOP_Z - cz                                            # all in the ellipsoid's frame
    crown_z = rz - 10.0                                                    # small crown where the ribs meet
    shell = scale(Sphere(1.0), (rx, rx, rz)) - scale(Sphere(1.0), (rx - wall, rx - wall, rz - wall))
    mask = Box(400, 400, 20, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((0, 0, crown_z)))
    mask = mask + Box(400, 400, g["base_ring"], align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(
        Location((0, 0, z_base)))
    for hz in g["hoop_z"]:
        mask = mask + Box(400, 400, g["hoop_height"]).moved(Location((0, 0, hz - cz)))
    for az in g["rib_azimuths"]:
        mask = mask + Box(rx + 20, g["rib_width"], 2 * rz + 20,
                          align=(Align.MIN, Align.CENTER, Align.CENTER)).rotate(Axis.Z, az)
    above_deck = Box(400, 400, 400, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((0, 0, z_base)))
    fused = shell & mask & above_deck
    expected = _bubble_volume_mc()
    if len(fused.solids()) != 1 or abs(fused.volume - expected) > 0.03 * expected:
        raise RuntimeError(f"bubble: {len(fused.solids())} solids, {fused.volume:.0f} mm3, expected ~{expected:.0f} mm3")

    cut = (g["hoop_z"][0] if g["hoop_z"] else cz) - cz
    lower = fused & Box(400, 400, 400, align=(Align.CENTER, Align.CENTER, Align.MAX)).moved(Location((0, 0, cut)))
    upper = fused & Box(400, 400, 400, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((0, 0, cut)))
    if abs(lower.volume + upper.volume - fused.volume) > 0.01 * fused.volume:
        raise RuntimeError("bubble: the split at the hoop lost volume")
    centre = Location((g["center_x"], 0.0, cz))
    lower, upper = lower.moved(centre), upper.moved(centre)
    flat = Rotation(0, 0, 0)
    return CoguPart(
        name="phone_guard_bubble",
        solid=Part(Compound(children=[lower, upper]).wrapped),
        material="PCTG",
        density_g_cm3=P.PCTG_DENSITY_G_CM3,
        print_rotation=flat,
        print_pieces=[("lower", lower, flat), ("upper", upper, flat)],
        print_notes="Two pieces split at the hoop. Lower bowl: base ring on the bed, no supports (walls lean out "
                    "≤ 36°). Upper dome: cut face on the bed, tree supports only under the crown and the top "
                    "of the ribs (steeper than 45° above ~80 mm). 4 walls (the shell is 3 mm). The TPU hoop "
                    "tyre clamps the halves; a drop of CA glue on the joint is optional.",
        infill_factor=0.95,
    )


def _bubble_volume_mc(n: int = 2_000_000) -> float:
    """Volume of the rib cage from its analytic definition (Monte Carlo, ±~0.7 %), to catch OCC booleans that
    silently drop material."""
    import math

    import numpy as np

    g = P.GUARD_BUBBLE
    rx, rz, wall, cz, w = g["radius_xy"], g["radius_z"], g["wall"], g["center_z"], g["rib_width"]
    z0 = P.COVER_TOP_Z - cz
    lo, hi = np.array([-rx, -rx, z0]), np.array([rx, rx, rz])
    x, y, z = np.random.default_rng(0).uniform(lo, hi, size=(n, 3)).T
    shell = ((x * x + y * y) / rx**2 + z * z / rz**2 <= 1) & ~((x * x + y * y) / (rx - wall)**2 + z * z / (rz - wall)**2 < 1)
    keep = (z >= rz - 10.0) | (z <= z0 + g["base_ring"])
    for hz in g["hoop_z"]:
        keep |= np.abs(z - (hz - cz)) <= g["hoop_height"] / 2
    for az in g["rib_azimuths"]:
        a = math.radians(az)
        keep |= (x * math.cos(a) + y * math.sin(a) >= 0) & (np.abs(-x * math.sin(a) + y * math.cos(a)) <= w / 2)
    return float(np.prod(hi - lo) * (shell & keep).mean())


GUARDS = {"frame": build_guard_frame, "bubble": build_guard_bubble}


def _ellipsoid(dr: float) -> Part:
    """The bubble's ellipsoid grown (dr > 0) or shrunk (dr < 0) by dr, in the pelvis frame."""
    from build123d import Sphere, scale
    g = P.GUARD_BUBBLE
    return scale(Sphere(1.0), (g["radius_xy"] + dr, g["radius_xy"] + dr, g["radius_z"] + dr)).moved(
        Location((g["center_x"], 0.0, g["center_z"])))


def _zband(z0: float, z1: float) -> Part:
    return Box(400, 400, z1 - z0, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((0, 0, z0)))


def build_guard_tpu() -> CoguPart:
    """TPU 95A covers on the bubble's ground-contact parts, press-fitted ("tyres"):
    a C-section on each hoop (outer sleeve + lips over its top and bottom edges), an outer sleeve with a top
    lip on the base ring, and a cap over the crown. Kept as separate solids (one print each)."""
    g, c = P.GUARD_BUBBLE, P.GUARD_TPU
    tt, lip, wall = c["thickness"], c["lip"], g["wall"]
    sleeve_shell = _ellipsoid(tt) - _ellipsoid(0.0)              # outside the PETG/PCTG surface
    lip_shell = _ellipsoid(0.0) - _ellipsoid(-min(lip, wall))      # over the part's edges, inside its outline
    pieces: list[Part] = []
    names: list[str] = []
    for i, hz in enumerate(g["hoop_z"]):
        names.append(f"hoop_tyre{i + 1 if len(g['hoop_z']) > 1 else ''}")
        h = g["hoop_height"]
        tyre = (sleeve_shell & _zband(hz - h / 2 - tt, hz + h / 2 + tt)) \
            + (lip_shell & _zband(hz + h / 2, hz + h / 2 + tt)) + (lip_shell & _zband(hz - h / 2 - tt, hz - h / 2))
        pieces.append(tyre)
    zb = P.COVER_TOP_Z
    base = (sleeve_shell & _zband(zb, zb + g["base_ring"] + tt)) + (lip_shell & _zband(zb + g["base_ring"], zb + g["base_ring"] + tt))
    pieces.append(base)
    names.append("base_sleeve")
    top = g["center_z"] + g["radius_z"]
    pieces.append(sleeve_shell & _zband(top - c["crown_cap"], top + tt + 1))
    names.append("crown_cap")
    solids = [so for pc in pieces for so in pc.solids()]
    for pc in pieces:
        if pc.volume < 1.0:
            raise RuntimeError("a TPU cover came out empty")
    flat = Rotation(0, 0, 0)
    return CoguPart(
        name="phone_guard_tpu",
        solid=Part(Compound(children=solids).wrapped),
        material="TPU",
        density_g_cm3=P.TPU_DENSITY_G_CM3,
        infill_factor=1.0,
        print_rotation=flat,
        print_pieces=[(n, pc, flat) for n, pc in zip(names, pieces)],
        print_notes="TPU 95A, black, slow (~25 mm/s), no retraction tuning needed for these shapes. Hoop tyre "
                    "and base sleeve stand as printed (axis vertical), no supports: the lips overhang 1.5 mm. "
                    "Crown cap: dome up, tree supports under its top. Press-fitted over the PCTG; the hoop tyre "
                    "also clamps the two halves of the bubble. Replaceable wear parts.",
    )
