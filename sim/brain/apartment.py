"""A small apartment around the robot, for the brain (ADR-004): made with MjSpec from a robot MJCF (not touched).

    uv run python sim/brain/apartment.py                      # -> build/mjcf/apartment.xml (from the get-up MJCF)
    uv run python sim/brain/apartment.py --ball 0.095         # a size-4 ball (radius, m)
    uv run mjpython sim/playground/play.py --checkpoint sim/playground/checkpoints/cogu_walk_v10 \
        --getup sim/playground/checkpoints/cogu_getup_v8 --xml build/mjcf/apartment.xml

Living room (4 × 3.5 m) with a sofa, a coffee table, a TV stand, a plant and a rug; a doorway in the far wall to a
corridor; a soccer ball (size 5: 22 cm, 430 g, a hollow shell that rolls) the robot can push; a person (a mocap body:
the brain moves it) who pushes the robot if they walk into it. The robot starts at the origin facing +x.

Everything solid collides like the floor (contype 3, conaffinity 1: the robot's bubble, drum and leg boxes touch it);
the ball also touches the soles (explicit pairs, as the floor does). The robot stays first in qpos (its free joint and
joints), so play.py and the policies read it as before; the ball's free joint comes after it.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
ROBOT_XML = REPO_ROOT / "build" / "mjcf" / "cogu_lower_body_mjx_getup.xml"   # hip to -130°, ankle to +82°
OUT = REPO_ROOT / "build" / "mjcf" / "apartment.xml"

ROOM = (-1.5, 2.5, -1.75, 1.75)      # m: x0, x1, y0, y1 of the living room
CORRIDOR = (2.5, 5.0, -1.75, -0.75)  # m: through the doorway in the x1 wall
WALL_H, WALL_T = 2.4, 0.1
BALL_R, BALL_MASS = 0.11, 0.43       # size 5; sizes 3 / 4: r 0.095 / 0.1, 0.30 / 0.36 kg
BALL_XY = (1.2, -0.6)
PERSON_XY = (-0.9, -0.9)

SOLID = dict(contype=3, conaffinity=1)
CEILING_GROUP = 5


def _box(parent, name, center, half, rgba, solid=True, material=""):
    g = parent.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(center), size=list(half),
                        rgba=list(rgba), group=0)
    if material:
        g.material = material
    if solid:
        g.contype, g.conaffinity = SOLID["contype"], SOLID["conaffinity"]
    else:
        g.contype = g.conaffinity = 0
    return g


def _walls(wb):
    x0, x1, y0, y1 = ROOM
    cx0, cx1, cy0, cy1 = CORRIDOR
    h, t = WALL_H / 2, WALL_T / 2
    wall = (0.93, 0.91, 0.86, 1)
    _box(wb, "wall_back", ((x0 + x1) / 2, y1 + t, h), ((x1 - x0) / 2 + 2 * t, t, h), wall)
    _box(wb, "wall_front", ((x0 + x1) / 2, y0 - t, h), ((x1 - x0) / 2 + 2 * t, t, h), wall)
    _box(wb, "wall_left", (x0 - t, (y0 + y1) / 2, h), (t, (y1 - y0) / 2, h), wall)
    # the x1 wall, with the doorway to the corridor (corridor width) on its y0 side
    _box(wb, "wall_right", (x1 + t, (cy1 + y1) / 2, h), (t, (y1 - cy1) / 2, h), wall)
    _box(wb, "wall_corridor_back", ((cx0 + cx1) / 2 + t, cy1 + t, h), ((cx1 - cx0) / 2, t, h), wall)
    _box(wb, "wall_corridor_front", ((cx0 + cx1) / 2 + t, cy0 - t, h), ((cx1 - cx0) / 2, t, h), wall)
    _box(wb, "wall_corridor_end", (cx1 + 2 * t, (cy0 + cy1) / 2, h), (t, (cy1 - cy0) / 2 + 2 * t, h), wall)
    # the ceiling, in group 5: the phone's camera and LiDAR see it (phone_sensors.py turns the group on), the
    # viewer, looking from outside, does not
    c = _box(wb, "ceiling", ((x0 + cx1) / 2, (y0 + y1) / 2, WALL_H + t), ((cx1 - x0) / 2 + 3 * t, (y1 - y0) / 2 + 2 * t, t),
             (0.97, 0.97, 0.95, 1), solid=False)
    c.group = CEILING_GROUP
    # the door leaf, open against the corridor's back wall (hinged at the doorway's y1 jamb)
    _box(wb, "door", (cx0 + 0.45, cy1 - 0.03, 1.0), (0.42, 0.02, 1.0), (0.55, 0.38, 0.24, 1))


def _furniture(wb):
    x0, x1, y0, y1 = ROOM
    fabric, wood, dark = (0.32, 0.42, 0.55, 1), (0.52, 0.36, 0.22, 1), (0.12, 0.12, 0.13, 1)
    # sofa against the back wall: seat, back, arms
    sx, sy = 0.4, y1 - 0.45
    _box(wb, "sofa_seat", (sx, sy, 0.21), (0.9, 0.4, 0.21), fabric)
    _box(wb, "sofa_back", (sx, y1 - 0.08, 0.45), (0.9, 0.08, 0.45), fabric)
    for side, dx in (("L", -0.98), ("R", 0.98)):
        _box(wb, f"sofa_arm_{side}", (sx + dx, sy, 0.31), (0.08, 0.4, 0.31), fabric)
    # coffee table in front of it: top on four legs (the robot sees under it, and can't pass: 40 cm high)
    tx, ty = 0.4, 0.35
    _box(wb, "table_top", (tx, ty, 0.39), (0.5, 0.25, 0.015), wood)
    for i, (dx, dy) in enumerate(((-0.45, -0.2), (-0.45, 0.2), (0.45, -0.2), (0.45, 0.2))):
        _box(wb, f"table_leg_{i}", (tx + dx, ty + dy, 0.19), (0.02, 0.02, 0.19), wood)
    # TV stand on the front wall, the TV on it (black: the LiDAR loses it)
    _box(wb, "tv_stand", (0.4, y0 + 0.2, 0.25), (0.7, 0.2, 0.25), wood)
    _box(wb, "tv", (0.4, y0 + 0.12, 0.85), (0.6, 0.03, 0.35), dark)
    # a plant in the corner: pot and leaves
    wb.add_geom(name="plant_pot", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[x0 + 0.3, y1 - 0.3, 0.15],
                size=[0.14, 0.15, 0], rgba=[0.6, 0.35, 0.25, 1], **SOLID)
    wb.add_geom(name="plant_leaves", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, pos=[x0 + 0.3, y1 - 0.3, 0.6],
                size=[0.25, 0.25, 0.32], rgba=[0.25, 0.5, 0.25, 1], **SOLID)
    # a rug under the table (visual only: flat)
    _box(wb, "rug", (0.4, 0.3, 0.001), (1.1, 0.8, 0.001), (0.62, 0.25, 0.25, 1), solid=False)


def _ball(spec, wb, radius: float, mass: float):
    spec.add_texture(name="ball", type=mujoco.mjtTexture.mjTEXTURE_CUBE, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.95, 0.95, 0.95], rgb2=[0.08, 0.08, 0.08], width=64, height=64)
    spec.add_material(name="ball", textures=["", "ball"])
    body = wb.add_body(name="ball", pos=[*BALL_XY, radius])
    body.add_freejoint(name="ball")
    g = body.add_geom(name="ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[radius, 0, 0], mass=mass,
                      material="ball", contype=3, conaffinity=3, condim=6, friction=[0.7, 0.01, 0.002])
    g.typeinertia = mujoco.mjtGeomInertia.mjINERTIA_SHELL     # a hollow ball: I = 2/3 m r²
    for side in "LR":                                        # the soles only touch what they are paired with
        spec.add_pair(name=f"{side}_foot_sole_ball", geomname1=f"{side}_foot_sole", geomname2="ball",
                      condim=3, friction=[0.9, 0.9, 0.005, 0.0001, 0.0001])
    return body


def _person(wb):
    """A standing adult, 1.7 m, as a mocap body (it moves where the brain puts it; it can push the robot)."""
    body = wb.add_body(name="person", mocap=True, pos=[*PERSON_XY, 0.0])
    skin, shirt, jeans = (0.85, 0.68, 0.55, 1), (0.85, 0.45, 0.15, 1), (0.2, 0.25, 0.45, 1)
    cap = mujoco.mjtGeom.mjGEOM_CAPSULE
    for side, dy in (("L", 0.1), ("R", -0.1)):
        body.add_geom(name=f"person_leg_{side}", type=cap, fromto=[0, dy, 0.06, 0, dy, 0.85], size=[0.065, 0, 0],
                      rgba=list(jeans), **SOLID)
        body.add_geom(name=f"person_foot_{side}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.05, dy, 0.03],
                      size=[0.12, 0.05, 0.03], rgba=[0.15, 0.15, 0.15, 1], **SOLID)
        body.add_geom(name=f"person_arm_{side}", type=cap, fromto=[0, 2.4 * dy, 1.42, 0, 2.5 * dy, 0.85],
                      size=[0.045, 0, 0], rgba=list(skin), **SOLID)
    body.add_geom(name="person_torso", type=cap, fromto=[0, 0, 0.95, 0, 0, 1.4], size=[0.17, 0, 0],
                  rgba=list(shirt), **SOLID)
    body.add_geom(name="person_head", type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[0, 0, 1.6], size=[0.11, 0, 0],
                  rgba=list(skin), **SOLID)
    return body


def build(src: Path = ROBOT_XML, out: Path = OUT, ball_radius: float = BALL_R, ball_mass: float = BALL_MASS) -> Path:
    spec = mujoco.MjSpec.from_file(str(src))
    if not Path(spec.meshdir).is_absolute():
        spec.meshdir = str((src.parent / spec.meshdir).resolve())
    spec.modelname = "cogu_apartment"
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1600, 1200
    wb = spec.worldbody

    spec.add_texture(name="wood_floor", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.62, 0.47, 0.32], rgb2=[0.58, 0.43, 0.29],
                     width=512, height=512)
    spec.add_material(name="wood_floor", textures=["", "wood_floor"], texrepeat=[12, 12], reflectance=0.05)
    spec.geom("floor").material = "wood_floor"
    _walls(wb)
    _furniture(wb)
    _ball(spec, wb, ball_radius, ball_mass)
    _person(wb)
    for i, (x, y) in enumerate(((0.5, 0.0), (3.7, -1.25))):    # ceiling lights: room and corridor
        wb.add_light(name=f"ceiling_{i}", pos=[x, y, WALL_H - 0.05], dir=[0, 0, -1], diffuse=[0.55, 0.55, 0.5],
                     castshadow=i == 0)
    wb.add_camera(name="overview", pos=[ROOM[0] + 0.15, ROOM[3] - 0.15, 2.2],   # a corner, looking at the robot
                  mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody="pelvis")
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]

    # the keyframe: the robot's `rest` as before, then the ball resting on the floor
    key = spec.key("rest")
    key.qpos = list(key.qpos) + [*BALL_XY, ball_radius, 1.0, 0.0, 0.0, 0.0]
    model = spec.compile()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(spec.to_xml())
    assert model.body(1).name == "pelvis" and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", default=str(ROBOT_XML), help="robot MJCF (default: the -120 hip get-up model)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--ball", type=float, default=BALL_R, help="ball radius, m (size 5: 0.11)")
    ap.add_argument("--ball-mass", type=float, default=BALL_MASS, help="kg (size 5: 0.43)")
    args = ap.parse_args(argv)
    out = build(Path(args.xml), Path(args.out), args.ball, args.ball_mass)
    m = mujoco.MjModel.from_xml_path(str(out))
    print(f"{out}  ({m.nbody} bodies, {m.ngeom} geoms, nq {m.nq}, ball r {args.ball:g} m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
