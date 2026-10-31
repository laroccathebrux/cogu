"""Hip parts: pelvis, yaw link and roll link (U bracket housing the hip-pitch servo).

Frames (one leg, right side; the left leg is mirrored at assembly):
  pelvis frame : origin at the midpoint between the two hip-yaw axes, on the yaw
                 servos' horn-side case face. +Z up, +X forward, +Y left.
  yaw-link frame: origin on the yaw axis at the yaw servo's horn-side face, +Z up.
                 The yaw horn points DOWN (−Z). The roll servo hangs below, shaft
                 along +X (horn at the front).
  roll-link frame: origin on the roll axis at the roll servo's horn-side face,
                 +X along the axis towards the horn (front). The U bracket carries
                 the hip-pitch servo (shaft along Y, horn on +Y).
"""

from __future__ import annotations

from build123d import Align, Axis, Box, Cylinder, Location, Part, Rotation, Vector

from cad import parameters as P
from cad import stack as K
from cad.part import CoguPart, InsertHole, insert_hole_cutter
from cad.parts import sts3215
from cad.parts.link import bearing_paddle_cutter, horn_paddle_cutters, paddle, servo_housing

S = P.SERVO

YAW_PADDLE_RADIUS = S.horn_disc_diameter / 2 + 1.0      # 10.6: must pass inside the pelvis screw heads
PELVIS_THICKNESS = 6.0
PELVIS_RECESS_R = 11.3                                   # recess for the yaw paddle, keeps the heads clear
PELVIS_RECESS_CEILING = -3.9                             # 0.8 mm above the paddle top (−4.7)
YAW_PLATE_Z0, YAW_PLATE_Z1 = -11.5, -8.5                 # plate under the paddle, below the screw heads
ROLL_AXIS_X = -K.HOUSING_Y0 / 2                          # 16.5: roll servo horn face, housing centred under the yaw axis
ROLL_AXIS_Z = YAW_PLATE_Z0 - (-K.HOUSING_Z_BOTTOM)       # -24.31: housing top flush with the plate
ROLL_LINK_BEAM_TOP = -((K.HOUSING_X_HALF ** 2 + K.HOUSING_Z_TOP ** 2) ** 0.5 + 1.0)   # -41.7
PITCH_AXIS_Z = ROLL_LINK_BEAM_TOP - K.HOUSING_X_HALF     # beam = pitch housing, centred on its X half-width
PITCH_HOUSING_OPEN = 1.0
# The pitch servo body points FORWARD in the beam, so the thigh is free to swing backwards
# (digitigrade rest pose and crouch); the beam's rear end is flush with the rear paddle.
# The U's rear paddle is pushed back ROLL_REAR_PADDLE_SHIFT, clear of the roll servo's bus sockets. The pitch beam
# stays put (moving it back would leave the front paddle hanging off its open front end, where the servo slides
# in): a root block behind the beam carries the paddle instead, and the leg does not move.
ROLL_REAR_Y0 = K.BEARING_PLATE_Y0 - P.ROLL_REAR_PADDLE_SHIFT
ROLL_REAR_Y1 = K.BEARING_PLATE_Y1 - P.ROLL_REAR_PADDLE_SHIFT
PITCH_AXIS_X = K.BEARING_PLATE_Y0 - P.CLEARANCE_LOOSE - K.HOUSING_Z_BOTTOM   # -30.84
BEAM_REAR_X = PITCH_AXIS_X + K.HOUSING_Z_BOTTOM                              # -43.65: rear end of the pitch beam
# Lateral position of the pitch stack. The leg's fork (horn plate .. bearing plate) is wider
# than HIP_SPACING, so the fork centre is pushed OUTBOARD of the roll/yaw axis until the two
# legs' inner plates are LEG_GAP apart at the midline.
FORK_SPAN = K.HORN_PLATE_Y1 - K.BEARING_PLATE_Y0                                  # 51.05
LEG_LATERAL_OFFSET = max(0.0, (FORK_SPAN + P.LEG_GAP) / 2 - P.HIP_SPACING / 2)     # 10.5 outboard
PITCH_AXIS_Y = -(K.HORN_PLATE_Y1 + K.BEARING_PLATE_Y0) / 2 - LEG_LATERAL_OFFSET    # 17.8 - 10.5 = 7.3


def cover_boss_positions() -> list[tuple[float, float]]:
    """(x, y) of the 4 bosses the hip cover bolts onto (insert tops at z = BOSS_TOP_Z)."""
    return [(bx, by) for bx in (P.PELVIS_BOSS_X, -P.PELVIS_BOSS_X) for by in (P.PELVIS_BOSS_Y, -P.PELVIS_BOSS_Y)]


BOSS_TOP_Z = P.INSERT_M3.hole_depth + 2.0      # 8.5
BOSS_R = P.INSERT_M3.hole_diameter / 2 + P.INSERT_M3.wall_min + 0.3   # 4.6 (Ø9.2)


def build_pelvis() -> CoguPart:
    """Flat plate holding the two yaw servos (bodies up, horns down) with 4 M3 inserts
    on bosses to bolt the hip cover on. Boss positions clear the yaw servo bodies."""
    plate = Box(P.PELVIS_DEPTH, P.PELVIS_WIDTH, PELVIS_THICKNESS, align=(Align.CENTER, Align.CENTER, Align.MAX))
    plate = plate.fillet(6.0, plate.edges().filter_by(Axis.Z))
    # corners trimmed (they would only catch on things)
    plate = plate & Cylinder(P.PELVIS_CORNER_TRIM_R, 40, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(
        Location((P.PELVIS_TRIM_X, 0, 0)))
    boss_h = BOSS_TOP_Z
    boss_r = BOSS_R
    inserts: list[InsertHole] = []
    solid: Part = plate
    cutters: Part | None = None
    for bx in (P.PELVIS_BOSS_X, -P.PELVIS_BOSS_X):
        for by in (P.PELVIS_BOSS_Y, -P.PELVIS_BOSS_Y):
            boss = Cylinder(boss_r, boss_h, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((bx, by, 0)))
            solid = solid + boss
            loc = Location((bx, by, boss_h)) * Location((0, 0, 0), (180, 0, 0))   # insert axis pointing down
            inserts.append(InsertHole(loc))
            c = insert_hole_cutter(loc)
            cutters = c if cutters is None else cutters + c
    for sy in (P.HIP_SPACING / 2, -P.HIP_SPACING / 2):
        # yaw servo: servo frame rotated so +Y (shaft) -> -Z, X (body length) -> +X
        horn_hole = Cylinder(K.HORN_HOLE_D / 2, 20, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(Location((0, sy, 0)))
        recess = Cylinder(PELVIS_RECESS_R, PELVIS_THICKNESS + PELVIS_RECESS_CEILING + 0.01,
                          align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((0, sy, -PELVIS_THICKNESS - 0.01)))
        cutters = cutters + horn_hole + recess
        for loc in sts3215.mount_hole_locations("horn"):
            x_s, _, z_s = loc.position
            # servo X -> pelvis X (body runs backwards from the axis: flip sign so the body is behind), Z -> Y
            hole = Cylinder(P.M2_CLEARANCE_HOLE / 2, 20, align=(Align.CENTER, Align.CENTER, Align.CENTER))
            cutters = cutters + hole.moved(Location((-x_s, sy + z_s, 0)))
    solid = solid - cutters
    masses = [(f"hip_yaw_servo_{i}", S.mass_g, Vector(-((S.x_min + S.x_max) / 2), sy, S.case_height / 2))
              for i, sy in enumerate((P.HIP_SPACING / 2, -P.HIP_SPACING / 2))]
    return CoguPart(
        name="pelvis",
        solid=solid,
        print_rotation=Rotation(0, 0, 0),
        print_notes="Bosses up, flat face on the bed. Heat-set the 4 M3 inserts after printing.",
        insert_holes=inserts,
        attached_masses=masses,
    )


def build_yaw_link() -> CoguPart:
    """Paddle under the yaw horn + plate + housing for the hip-roll servo (shaft along +X)."""
    # horn paddle, horizontal: built as a vertical paddle then laid flat
    pad = Cylinder(YAW_PADDLE_RADIUS, (-K.HORN_PLATE_Y0) - YAW_PLATE_Z1 + 0.01,
                   align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((0, 0, YAW_PLATE_Z1 - 0.01)))
    plate = Box(2 * K.HOUSING_X_HALF + 2, 2 * K.HOUSING_X_HALF + 2, YAW_PLATE_Z1 - YAW_PLATE_Z0,
                align=(Align.CENTER, Align.CENTER, Align.MIN))
    plate = plate.fillet(4.0, plate.edges().filter_by(Axis.Z)).moved(Location((0, 0, YAW_PLATE_Z0)))
    # roll servo housing: housing frame (Y = shaft, Z = body) -> yaw-link frame (X = shaft, −Z = body).
    # Window over the bus sockets: the plugs stand in the gap to the U's shifted rear paddle.
    housing, cutters = servo_housing(bus_window=True)
    def to_yaw(p: Part) -> Part:
        """housing +Y -> +X (shaft forward), housing +Z -> -Z (body hangs down), housing +X -> +Y."""
        p = p.rotate(Axis.Z, -90)      # +Y -> +X, +X -> -Y
        p = p.rotate(Axis.X, 180)      # +Z -> -Z, -Y -> +Y
        return p.moved(Location((ROLL_AXIS_X, 0, ROLL_AXIS_Z)))
    roll_axis_x = ROLL_AXIS_X
    housing = to_yaw(housing)
    cutters = to_yaw(cutters)
    solid = pad + plate + housing
    # 4 × M2 holes through the paddle into the yaw horn (vertical)
    r = S.horn_hole_pcd / 2
    for dx, dy in ((r, 0), (-r, 0), (0, r), (0, -r)):
        cutters = cutters + Cylinder(P.M2_CLEARANCE_HOLE / 2, 12, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(
            Location((dx, dy, (-K.HORN_PLATE_Y0 + YAW_PLATE_Z1) / 2)))
    solid = solid - cutters
    servo_com = Vector(roll_axis_x + S.y_back_face / 2, 0, ROLL_AXIS_Z - (S.x_min + S.x_max) / 2)
    return CoguPart(
        name="hip_yaw_link",
        solid=solid,
        print_rotation=Rotation(180, 0, 0),
        print_notes="Upside down: horn-disc face on the bed, housing prints as a tower. Supports under "
                    "the plate ring around the disc (3.8 mm gap).",
        attached_masses=[("hip_roll_servo", S.mass_g, servo_com)],
    )


def build_roll_link() -> CoguPart:
    """U bracket around the roll servo (horn paddle front, bearing paddle rear) whose
    bottom beam houses the hip-pitch servo (shaft along +Y)."""
    front = paddle(K.HORN_PLATE_Y0, K.HORN_PLATE_Y1, ROLL_LINK_BEAM_TOP - 2.0, 0.0)
    rear = paddle(ROLL_REAR_Y0, ROLL_REAR_Y1, ROLL_LINK_BEAM_TOP - 2.0, 0.0)
    pcut = horn_paddle_cutters(K.HORN_PLATE_Y0, K.HORN_PLATE_Y1, 0.0) + bearing_paddle_cutter(ROLL_REAR_Y1, 0.0)
    # paddles were built with Y as the axis; the roll axis is X: rotate −90° about Z (+Y -> +X)
    front, rear, pcut = (s.rotate(Axis.Z, -90) for s in (front, rear, pcut))

    # pitch housing: housing frame (Y shaft, Z body, X across) -> roll frame (Y shaft, +X body, Z across)
    housing, cutters = servo_housing(open_top=PITCH_HOUSING_OPEN)
    housing = housing.rotate(Axis.Y, 90).moved(Location((PITCH_AXIS_X, PITCH_AXIS_Y, PITCH_AXIS_Z)))
    cutters = cutters.rotate(Axis.Y, 90).moved(Location((PITCH_AXIS_X, PITCH_AXIS_Y, PITCH_AXIS_Z)))
    # root of the shifted rear paddle, behind the beam; stops 1 mm short of the thigh's horn plate, which swings
    # past it at hip pitch < -60°
    root_y0, root_y1 = -K.PADDLE_RADIUS, PITCH_AXIS_Y + K.HORN_PLATE_Y0 - 1.0
    root = Box(BEAM_REAR_X + 2.0 - ROLL_REAR_Y0, root_y1 - root_y0, 8.0,
               align=(Align.MIN, Align.MIN, Align.MAX)).moved(Location((ROLL_REAR_Y0, root_y0, ROLL_LINK_BEAM_TOP)))
    solid = front + rear + root + housing - (pcut + cutters)
    servo_com = Vector(PITCH_AXIS_X + (S.x_min + S.x_max) / 2, PITCH_AXIS_Y + S.y_back_face / 2, PITCH_AXIS_Z)
    return CoguPart(
        name="hip_roll_link",
        solid=solid,
        print_rotation=Rotation(0, 0, 0),
        print_notes="Beam (pitch housing) flat on the bed, U arms up. Tree supports under the arm roots only.",
        attached_masses=[("hip_pitch_servo", S.mass_g, servo_com)],
    )
