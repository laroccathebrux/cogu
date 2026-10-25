"""Derived positions along a joint axis (the "joint stack"), shared by every part.

Joint-local frame: origin on the joint axis at the servo's horn-side case face,
+Y along the axis pointing out through the horn. All numbers in mm.

      bearing plate | hub | back disc | servo case | horn disc | horn plate
   y: -41.85 .. -35.85     -32.55 ..  -28.75 .. 0     2.2 .. 4.7   4.7 .. 7.7
"""

from cad import parameters as P

S = P.SERVO

# horn side (+Y)
HORN_PLATE_Y0 = S.horn_top_height                        # 4.7: inner face touches the horn disc
HORN_PLATE_Y1 = HORN_PLATE_Y0 + P.HORN_PLATE_THICKNESS   # 7.7

# back side (−Y)
BACK_DISC_Y1 = S.y_back_face - S.back_boss_height        # -30.3: top of the back disc (against the bosses)
BACK_DISC_Y0 = S.y_back_disc_bottom                      # -32.55: underside of the back disc
HUB_PLATE_Y1 = BACK_DISC_Y0                              # hub plate sits on the back disc
HUB_PLATE_Y0 = HUB_PLATE_Y1 - P.BACK_HUB_PLATE_THICKNESS # -36.55
HUB_STUB_Y0 = HUB_PLATE_Y0 - P.BACK_HUB_STUB_LENGTH      # -41.55
BEARING_PLATE_Y1 = HUB_PLATE_Y0 - P.FORK_CLEARANCE       # -37.35: inner face of the bearing plate
BEARING_PLATE_Y0 = BEARING_PLATE_Y1 - P.BEARING_PLATE_THICKNESS  # -43.35

# housing that holds the servo (walls around the pocket)
HOUSING_Y1 = 0.0                                         # flush with the horn-side face (open side)
HOUSING_BACK_WALL_Y1 = BACK_DISC_Y1 - P.CLEARANCE_LOOSE  # inner face of the back wall
HOUSING_Y0 = HOUSING_BACK_WALL_Y1 - P.LINK_WALL          # outer face of the back wall
HOUSING_X_HALF = S.body_width / 2 + P.CLEARANCE_LOOSE + P.LINK_WALL   # fore-aft half width (15.06)
HOUSING_Z_BOTTOM = S.x_min - P.CLEARANCE_LOOSE - P.LINK_WALL          # below the lower axis (-12.81)
HOUSING_Z_TOP = S.x_max + P.CLEARANCE_LOOSE + P.LINK_WALL             # above the lower axis (37.81)

# fork paddles around the proximal joint
PADDLE_RADIUS = S.horn_disc_diameter / 2 + 3.0           # 12.6
# the proximal servo's case corner sweeps a circle of this radius around the proximal axis
PROXIMAL_SERVO_SWEEP = (S.shaft_from_horn_end ** 2 + (S.body_width / 2) ** 2) ** 0.5  # ~16.0

BACK_DISC_HOLE_D = S.back_disc_diameter + 2 * P.CLEARANCE_LOOSE + 1.0   # hole in the back wall (20.8)
HORN_HOLE_D = S.horn_disc_diameter + 2 * P.CLEARANCE_LOOSE + 1.0        # hole in a plate the horn passes through
