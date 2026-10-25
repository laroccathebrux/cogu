"""Body shell: the cream "egg" that is torso and head (ADR-001).

Pelvis frame (the shell is bolted to the pelvis): origin between the hip-yaw axes on the
yaw servos' horn face, +Z up, +X forward, +Y left. The shell's vertical axis is at
x = BODY_AXIS_X. It is open at the bottom (skirt rim at BODY_RIM_Z) so the hip stack can
swing, and open at the top (neck hole) for the collar / neck servo.

Inside: 4 ears bolted to the pelvis bosses, 3 inserts under the neck ledge (collar), a
flat face box at the front for the CYD screen with 4 inserts (bezel), rails for the
Pi Zero and the servo board, a battery bay at the back, an IMU ledge, ribs.
"""

from __future__ import annotations

import math

from build123d import (
    Align,
    Axis,
    Box,
    Cylinder,
    Line,
    Location,
    Part,
    Plane,
    Rotation,
    Spline,
    Vector,
    make_face,
    revolve,
)

from cad import parameters as P
from cad.part import CoguPart, InsertHole, insert_hole_cutter, rounded_box
from cad.parts import hip

XC = P.BODY_AXIS_X
WALL = P.BODY_WALL
CAVITY_TOP_Z = P.BODY_NECK_Z - 3.0                 # underside of the neck ledge
FACE_W = P.BODY_FACE_WIDTH                         # 104: visor outline (y)
FACE_H = P.BODY_FACE_HEIGHT                        # 71: visor outline (z)
FACE_SCREW = (44.0, P.CYD["pcb"][1] / 2 + 4.5)     # (±44, ±29.5) bezel screw / insert positions, above and below the PCB
FACE_BOX_X0 = P.BODY_FACE_BOX_X0                   # 28: rear wall of the visor box (clears the hip-roll paddles)
GLASS_END_Y = P.CYD_ACTIVE_CENTRE_Y - P.CYD["active_centre_from_glass_end"]   # -37.6: PCB and glass end
PCB_Y0, PCB_Y1 = GLASS_END_Y, GLASS_END_Y + P.CYD["pcb"][0]                    # -37.6 .. 48.4
GLASS_Y0, GLASS_Y1 = GLASS_END_Y, GLASS_END_Y + P.CYD["glass_length"]          # -37.6 .. 31.4
PCB_X0 = P.CYD_PCB_X                                                           # 32 (rear face)
PCB_X1 = PCB_X0 + P.CYD["pcb"][2]                                              # 33.6
NECK_INSERT_R = P.BODY_NECK_INSERT_R
NECK_INSERT_AZIMUTHS = (90.0, 210.0, 330.0)


def _axis_loc(x: float, y: float, z: float) -> Location:
    return Location((x + XC, y, z))


def _profile_solid(points, z_bottom_extra: float = 0.0) -> Part:
    """Revolve a (z, r) spline profile about the shell axis into a solid.

    The profile runs from the rim up to the last point; it is closed with straight lines to
    the axis. `z_bottom_extra` extends the solid straight down below the first point.
    """
    pts = [Vector(r, 0, z) for z, r in points]
    edges = [Spline(*pts)]
    top = pts[-1]
    bot = pts[0]
    edges.append(Line(top, Vector(0, 0, top.Z)))
    z_low = bot.Z - z_bottom_extra
    edges.append(Line(Vector(0, 0, top.Z), Vector(0, 0, z_low)))
    if z_bottom_extra > 0:
        edges.append(Line(Vector(0, 0, z_low), Vector(bot.X, 0, z_low)))
        edges.append(Line(Vector(bot.X, 0, z_low), bot))
    else:
        edges.append(Line(Vector(0, 0, z_low), bot))
    face = make_face(edges)
    return revolve(face, Axis.Z, 360).moved(Location((XC, 0, 0)))


def outer_solid() -> Part:
    """Solid of revolution bounded by the outer skin (no face box)."""
    return _profile_solid(P.BODY_PROFILE)


def inner_solid() -> Part:
    """Cavity: outer profile shrunk by the wall, closed at the neck-ledge underside,
    extended 6 mm below the rim so the bottom is open."""
    inner = [(z, r - WALL) for z, r in P.BODY_PROFILE if z <= CAVITY_TOP_Z + 1e-6]
    return _profile_solid(inner, z_bottom_extra=6.0)


def radius_at(z: float) -> float:
    """Outer radius of the revolved profile at height z (linear interpolation of the control points)."""
    pts = P.BODY_PROFILE
    for (z0, r0), (z1, r1) in zip(pts, pts[1:]):
        if z0 <= z <= z1:
            return r0 + (r1 - r0) * (z - z0) / (z1 - z0)
    return pts[0][1] if z < pts[0][0] else pts[-1][1]


def _radial_cutter(azimuth_deg: float, z: float, cutter: Part) -> Part:
    """Place a cutter built along +X (pointing out of the shell) at the given azimuth on the skin."""
    r = radius_at(z)
    a = math.radians(azimuth_deg)
    return cutter.rotate(Axis.Z, azimuth_deg).moved(Location((XC + r * math.cos(a), r * math.sin(a), z)))


def face_box(shell_cavity: Part) -> tuple[Part, Part]:
    """(frame solid, cutter) of the flat face: a rounded-rectangle tube from FACE_BOX_X0 to
    BODY_FACE_X whose front lip holds the display glass flush with the face."""
    cz = P.BODY_FACE_CENTRE_Z
    length = P.BODY_FACE_X - FACE_BOX_X0
    # the tube is built along +Z (rounded_box fillets the Z edges) and rotated into place below
    outer = rounded_box(FACE_W, FACE_H, length, P.BODY_FACE_CORNER_R, align=(Align.CENTER, Align.CENTER, Align.MIN))
    inner = rounded_box(FACE_W - 2 * WALL, FACE_H - 2 * WALL, length + 2, P.BODY_FACE_CORNER_R - WALL,
                        align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((0, 0, -1)))
    # front lip: plate at the face with the glass opening, starting 0.3 mm in front of the PCB
    lip = rounded_box(FACE_W, FACE_H, P.BODY_FACE_X - (PCB_X1 + 0.3), P.BODY_FACE_CORNER_R,
                      align=(Align.CENTER, Align.CENTER, Align.MAX)).moved(Location((0, 0, length)))
    glass_w = P.CYD["glass_length"] + 0.6
    glass_h = P.CYD["pcb"][1] + 0.6
    glass_cy = (GLASS_Y0 + GLASS_Y1) / 2
    glass_open = rounded_box(glass_w, glass_h, 20, 1.5, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(
        Location((glass_cy, 0, length)))
    # ledges the PCB rests on (behind the PCB's rear face), top and bottom
    pcb_h = P.CYD["pcb"][1]
    ledge_z = PCB_X0 - FACE_BOX_X0           # along the tube axis (becomes X)
    ledges = None
    for sgn in (1, -1):
        lg = Box(FACE_W - 2 * WALL, 2.0, 3.0, align=(Align.CENTER, Align.CENTER, Align.MAX)).moved(
            Location((0, sgn * (pcb_h / 2 + 1.0 + 0.3), ledge_z)))
        ledges = lg if ledges is None else ledges + lg
    # bezel insert bosses in the 4 corners, axis along the tube (X), from the face inward
    bosses = None
    boss_r = P.INSERT_M3.hole_diameter / 2 + P.INSERT_M3.wall_min + 0.3
    boss_len = P.INSERT_M3.hole_depth + 2.0
    for sy in (1, -1):
        for sz in (1, -1):
            b = Cylinder(boss_r, boss_len, align=(Align.CENTER, Align.CENTER, Align.MAX)).moved(
                Location((sy * FACE_SCREW[0], sz * FACE_SCREW[1], length)))
            bosses = b if bosses is None else bosses + b
    back = rounded_box(FACE_W, FACE_H, WALL, P.BODY_FACE_CORNER_R, align=(Align.CENTER, Align.CENTER, Align.MIN))
    tube = outer - inner
    front = lip + ledges + bosses - glass_open
    # tube built along +Z: rotate so +Z -> +X (rotate about Y by +90: (x,y,z) -> (z,y,-x)), then
    # built coordinates: x was lateral (Y), y was vertical (Z). So first map (x,y,z) -> (?,x,y) with z->X.
    def to_body(p: Part) -> Part:
        p = p.moved(Location((0, 0, -length)))   # face plane at the origin
        p = p.rotate(Axis.X, 90)      # (x, y, z) -> (x, -z, y): y->Z, z->-Y
        p = p.rotate(Axis.Z, 90)      # (x, y, z) -> (-y, x, z): -Y -> X, x -> Y
        p = p.rotate(Axis.Y, -P.BODY_FACE_TILT_DEG)   # nose up
        return p.moved(Location((P.BODY_FACE_X, 0, cz)))
    # tube walls and rear wall only where the box stands outside the sphere; the lip, ledges and
    # bosses stay whole (the lip's centre is inside the sphere's front bulge)
    frame = (to_body(tube + back) - shell_cavity) + to_body(front)
    # cutter: the tube interior all the way out of the shell (so the shell in front of the face is removed)
    cut = rounded_box(FACE_W - 2 * WALL, FACE_H - 2 * WALL, 60, P.BODY_FACE_CORNER_R - WALL,
                      align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((0, 0, -0.5)))
    cut = to_body(cut)
    return frame, cut


def build_body() -> CoguPart:
    outer = outer_solid()
    cavity = inner_solid()
    shell = outer - cavity
    cutters: Part | None = None
    inserts: list[InsertHole] = []

    def add_cut(c: Part):
        nonlocal cutters
        cutters = c if cutters is None else cutters + c

    # --- neck hole + ledge inserts (collar screws come from above) ---
    add_cut(Cylinder(P.BODY_NECK_HOLE_R, 20, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(
        _axis_loc(0, 0, P.BODY_NECK_Z)))
    boss_r = P.INSERT_M3.hole_diameter / 2 + P.INSERT_M3.wall_min + 0.3
    boss_len = P.INSERT_M3.hole_depth + 2.0
    for az in NECK_INSERT_AZIMUTHS:
        a = math.radians(az)
        x, y = XC + NECK_INSERT_R * math.cos(a), NECK_INSERT_R * math.sin(a)
        shell = shell + Cylinder(boss_r, boss_len, align=(Align.CENTER, Align.CENTER, Align.MAX)).moved(
            Location((x, y, P.BODY_NECK_Z)))
        loc = Location((x, y, P.BODY_NECK_Z)) * Location((0, 0, 0), (180, 0, 0))
        inserts.append(InsertHole(loc))
        add_cut(insert_hole_cutter(loc))

    # --- ears on the pelvis bosses (M3 x 8 from above into the pelvis inserts) ---
    ear_z0 = hip.BOSS_TOP_Z
    ear_t = 3.0
    for bx, by in hip.body_boss_positions():
        ear = Cylinder(5.0, ear_t, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((bx, by, ear_z0)))
        # rib from the ear to the wall, radial from the shell axis
        r_ear = math.hypot(bx - XC, by)
        r_wall = radius_at(ear_z0) - WALL + 1.0
        az = math.degrees(math.atan2(by, bx - XC))
        rib = Box(r_wall - r_ear + 2.0, 3.0, ear_t + 3.0, align=(Align.MIN, Align.CENTER, Align.MIN))
        rib = rib.rotate(Axis.Z, az).moved(Location((bx, by, ear_z0)))
        shell = shell + ear + rib
        add_cut(Cylinder(P.M3_CLEARANCE_HOLE / 2, 30, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(
            Location((bx, by, ear_z0))))
        add_cut(Cylinder(P.M3_HEAD_DIAMETER / 2 + 0.3, 20, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(
            Location((bx, by, ear_z0 + ear_t))))

    # --- face box for the CYD ---
    frame, face_cut = face_box(cavity)
    shell = (shell - face_cut) + frame
    for sy in (1, -1):
        for sz in (1, -1):
            loc = Location((P.BODY_FACE_X, 0, P.BODY_FACE_CENTRE_Z), (0, -P.BODY_FACE_TILT_DEG, 0)) \
                * Location((0, sy * FACE_SCREW[0], sz * FACE_SCREW[1]), (0, -90, 0))   # insert axis -> into the boss
            inserts.append(InsertHole(loc))
            add_cut(insert_hole_cutter(loc))

    # --- internal features: built oversize, then trimmed to the outer skin ---
    internal: list[Part] = []

    # vertical ribs on the inner wall
    rw, rd = P.BODY_RIB_SIZE
    for i in range(P.BODY_RIBS):
        az = 75.0 + i * (210.0 / (P.BODY_RIBS - 1))      # 75 .. 285: clear of the visor box
        z0, z1 = P.BODY_RIB_Z0, CAVITY_TOP_Z - 1.0
        n = 8
        for k in range(n):
            za = z0 + (z1 - z0) * k / n
            zb = z0 + (z1 - z0) * (k + 1) / n
            r = radius_at((za + zb) / 2) - WALL
            seg = Box(rd + 3.0, rw, zb - za + 0.01, align=(Align.MAX, Align.CENTER, Align.MIN))
            internal.append(seg.moved(Location((r + 3.0, 0, za))).rotate(Axis.Z, az).moved(Location((XC, 0, 0))))

    def rails(x_pcb: float, pcb_t: float, width: float, height: float, z: float, depth: float,
              extra_below: float = 2.0, floor_tab: bool = False):
        """Two vertical U channels at y = ±width/2 holding a PCB standing in the YZ plane,
        each bridged outwards to the shell. The channels run from `extra_below` under the
        board to 2 mm above it; with `floor_tab` the board rests on a tab at its bottom edge
        (it is dropped in from above). Returns (solids, cutters)."""
        sol, cut = [], []
        rail_h = height + 2.0 + extra_below
        rail_zc = z + (2.0 - extra_below) / 2
        for sy in (1, -1):
            y = sy * (width / 2 + 0.3)
            a_y = Align.MIN if sy > 0 else Align.MAX
            rail = Box(depth, 3.0, rail_h, align=(Align.CENTER, a_y, Align.CENTER)).moved(Location((x_pcb, y, rail_zc)))
            # groove: open at the bottom (board slides up) unless it rests on a tab, then open at the top
            g_z0 = z - height / 2 - (0.0 if floor_tab else extra_below + 1.0)
            g_z1 = z + height / 2 + (3.0 if floor_tab else 1.0)
            groove = Box(pcb_t + 0.4, 4.0, g_z1 - g_z0, align=(Align.CENTER, a_y, Align.MIN)).moved(
                Location((x_pcb, y - sy * 2.0, g_z0)))
            bridge = Box(2.0, 70.0, rail_h, align=(Align.CENTER, a_y, Align.CENTER)).moved(Location((x_pcb, y, rail_zc)))
            sol += [rail, bridge]
            cut.append(groove)
            if floor_tab:
                sol.append(Box(depth, 6.0, 2.0, align=(Align.CENTER, a_y, Align.MAX)).moved(
                    Location((x_pcb, y - sy * 3.0, z - height / 2))))
        return sol, cut

    pw, ph, pt = P.PI_ZERO["size"]
    sol, cut = rails(P.PI_ZERO_X + pt / 2, pt, pw, ph, P.PI_ZERO_Z, pt + 4.0, extra_below=0.0, floor_tab=True)
    internal += sol
    for c in cut:
        add_cut(c)
    sw, sh, st = P.SERVO_BOARD["size"]
    sol, cut = rails(P.SERVO_BOARD_X - st / 2 + 0.8, 1.6, sw, sh, P.SERVO_BOARD_Z, 5.6)
    internal += sol
    for c in cut:
        add_cut(c)

    # battery bay at the back: two side walls and a floor tab reaching the shell, strap slots in the skin
    bx_, by_, bz_ = P.BATTERY_2S["size"]
    bpx, bpy, bpz = P.BATTERY_POS
    for sy in (1, -1):
        internal.append(Box(bx_ + 14.0, 2.0, bz_ * 0.6, align=(Align.MAX, Align.CENTER, Align.CENTER)).moved(
            Location((bpx + bx_ / 2 - 0.5, bpy + sy * (by_ / 2 + 1.3), bpz))))       # ends 0.6 mm behind the yaw servos
    internal.append(Box(bx_ + 14.0, by_ + 4.6, 2.0, align=(Align.MAX, Align.CENTER, Align.MAX)).moved(
        Location((bpx + bx_ / 2 - 0.5, bpy, bpz - bz_ / 2 - 0.3))))
    for sy in (1, -1):
        slot = Box(10.0, 3.0, 12.0, align=(Align.CENTER, Align.CENTER, Align.CENTER))
        add_cut(_radial_cutter(180.0 + sy * 25.0, bpz, slot))

    # IMU ledge (BNO055 on 2 pegs), cantilevered from the rear wall
    iw, il, it = P.BNO055["size"]
    ix, iy, iz = P.IMU_POS
    internal.append(Box(30.0, il + 2.0, 2.0, align=(Align.MIN, Align.CENTER, Align.MAX)).moved(
        Location((ix - 2.0 - 22.0, iy, iz - it / 2))))
    for sy in (1, -1):
        internal.append(Cylinder(P.BNO055["hole_d"] / 2 - 0.1, 3.0, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(
            Location((ix - 2.0, iy + sy * P.BNO055["holes"] / 2, iz - it / 2))))

    feats = internal[0]
    for f in internal[1:]:
        feats = feats + f
    shell = shell + (feats & outer)

    # --- power switch and USB-C at the back right ---
    add_cut(_radial_cutter(220.0, -18.0, Cylinder(P.BODY_SWITCH_HOLE_D / 2, 20, align=(Align.CENTER, Align.CENTER, Align.CENTER))
                           .rotate(Axis.Y, 90)))
    usb_w, usb_h = P.BODY_USBC_SLOT
    add_cut(_radial_cutter(235.0, 2.0, Box(20, usb_w, usb_h, align=(Align.CENTER, Align.CENTER, Align.CENTER))))

    solid = shell - cutters

    masses = [
        ("cyd", P.CYD["mass_g"], Vector(PCB_X0 + 3.0, (PCB_Y0 + PCB_Y1) / 2, P.BODY_FACE_CENTRE_Z)),
        ("pi_zero", P.PI_ZERO["mass_g"], Vector(P.PI_ZERO_X, 0, P.PI_ZERO_Z)),
        ("servo_board", P.SERVO_BOARD["mass_g"], Vector(P.SERVO_BOARD_X, 0, P.SERVO_BOARD_Z)),
        ("battery_2s", P.BATTERY_2S["mass_g"], Vector(*P.BATTERY_POS)),
        ("bno055", P.BNO055["mass_g"], Vector(*P.IMU_POS)),
        ("body_misc", P.MISC_BODY_MASS_G, Vector(XC, 0, 0)),
    ]
    return CoguPart(
        name="body",
        solid=solid,
        print_rotation=Rotation(0, 0, 0),
        print_notes="Skirt rim down on the bed (Ø99 ring), neck up. 2 mm walls = 5 perimeters, 0 % infill "
                    "needed. Tree supports only under the neck ledge, the face-box top wall and the IMU ledge. "
                    "Heat-set 3 inserts under the neck ledge and 4 in the face box. PETG or PLA.",
        insert_holes=inserts,
        attached_masses=masses,
        infill_factor=0.95,
    )
