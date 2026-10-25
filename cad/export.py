"""Generate every artefact from the CAD: STL per part (print orientation), STEP,
MJCF with mass/inertia computed from the CAD, and run the design rules.

    uv run python -m cad.export [--out build] [--no-step]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom

import numpy as np
from build123d import Axis, Compound, Location, Part, export_step, export_stl

from cad import assembly, design_rules
from cad import parameters as P
from cad.part import CoguPart

MM = 1e-3
G = 1e-3


# ---------------------------------------------------------------------------
# Mass properties
# ---------------------------------------------------------------------------

def _inertia_about_origin(solid: Part) -> tuple[float, np.ndarray, np.ndarray]:
    """(volume mm³, COM mm, inertia tensor mm⁵ about the part origin) of a unit-density solid."""
    vol = solid.volume
    com = np.array(tuple(solid.center()), dtype=float)
    m = solid.matrix_of_inertia      # about the centre of mass, unit density, mm^5
    i_com = np.array([[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]], dtype=float)
    # parallel axis to the origin
    r = com
    i_org = i_com + vol * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return vol, com, i_org


def body_inertial(instances: list[tuple[CoguPart, Location, bool, float]]) -> tuple[float, np.ndarray, np.ndarray]:
    """Combine printed solids and their attached point masses into (mass kg, com m, inertia kg·m² about com)."""
    total_m = 0.0
    first = np.zeros(3)
    i_org = np.zeros((3, 3))
    density = P.PETG_DENSITY_G_CM3 * P.INFILL_MASS_FACTOR / 1000.0   # g/mm³
    for part, loc, mirror, rot_z in instances:
        solid = part.solid
        if mirror:
            solid = assembly.mirrored(solid)
        if rot_z:
            solid = solid.rotate(Axis.Z, rot_z)
        solid = solid.moved(loc)
        vol, com, i_o = _inertia_about_origin(solid)
        m = vol * density
        total_m += m
        first += m * com
        i_org += density * i_o
        for name, mg, pos in part.attached_masses:
            p = np.array(tuple(pos), dtype=float)
            if mirror:
                p[1] = -p[1]
            if rot_z:
                a = math.radians(rot_z)
                p = np.array([p[0] * math.cos(a) - p[1] * math.sin(a), p[0] * math.sin(a) + p[1] * math.cos(a), p[2]])
            p = p + np.array(tuple(loc.position), dtype=float)
            total_m += mg
            first += mg * p
            i_org += mg * (np.dot(p, p) * np.eye(3) - np.outer(p, p))
    com = first / total_m
    i_com = i_org - total_m * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
    # units: g·mm² -> kg·m²
    return total_m * G, com * MM, i_com * G * MM * MM


# ---------------------------------------------------------------------------
# MJCF
# ---------------------------------------------------------------------------

def _fmt(v, nd=6) -> str:
    return " ".join(f"{float(x):.{nd}g}" for x in v)


def export_meshes(parts: dict[str, CoguPart], mesh_dir: Path) -> dict[tuple[str, bool], str]:
    """One visual/collision STL per (part, side). Returns {(part, mirrored): mesh name}."""
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_names: dict[tuple[str, bool], str] = {}
    for name, part in parts.items():
        for mirror in (False, True):
            mname = f"{name}_{'L' if mirror else 'R'}"
            solid = assembly.mirrored(part.solid) if mirror else part.solid
            export_stl(solid, str(mesh_dir / f"{mname}.stl"), tolerance=0.05, angular_tolerance=0.2)
            mesh_names[(name, mirror)] = mname
    return mesh_names


def _sole_box(parts: dict[str, CoguPart], mirrored: bool) -> tuple[list[float], list[float]]:
    """(centre m, half-sizes m) of a box matching the foot sole, in the foot (ankle) frame."""
    solid = assembly.mirrored(parts["foot"].solid) if mirrored else parts["foot"].solid
    bb = solid.bounding_box()
    centre = [(bb.min.X + bb.max.X) / 2, (bb.min.Y + bb.max.Y) / 2, bb.min.Z + P.FOOT_SOLE_THICKNESS / 2]
    half = [(bb.max.X - bb.min.X) / 2, (bb.max.Y - bb.min.Y) / 2, P.FOOT_SOLE_THICKNESS / 2]
    return [c * MM for c in centre], [h * MM for h in half]


def build_mjcf(parts: dict[str, CoguPart], mesh_names: dict[tuple[str, bool], str], mjx: bool) -> ET.Element:
    """MJCF tree for the lower body.

    mjx=False: every printed part is a collision mesh (faithful contacts, for `check_mjcf.py`
               and the interactive viewer).
    mjx=True : the training variant for MuJoCo Playground / MJX. Same bodies, joints, masses
               and actuators, but meshes are visual only and the only collision geometry is
               one box per foot sole (sole-floor and sole-sole contacts, nothing else), plus the
               solver settings MJX likes. Both variants carry the same IMU / foot sites and sensors.
    """
    root = ET.Element("mujoco", model="cogu_lower_body" + ("_mjx" if mjx else ""))
    ET.SubElement(root, "compiler", angle="degree", meshdir="meshes", autolimits="true")
    option = ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81")
    if mjx:
        option.set("iterations", "3")
        option.set("ls_iterations", "5")
        ET.SubElement(option, "flag", eulerdamp="disable")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="0.05", armature="0.005", frictionloss="0.02")
    if mjx:
        ET.SubElement(default, "geom", type="mesh", contype="0", conaffinity="0", group="2",
                      rgba="0.93 0.86 0.72 1")
        coll = ET.SubElement(default, "default", **{"class": "collision"})
        ET.SubElement(coll, "geom", type="box", contype="1", conaffinity="1", group="3", condim="3",
                      friction="0.9 0.005 0.0001", rgba="0.3 0.3 0.8 0.4")
    else:
        ET.SubElement(default, "geom", type="mesh", contype="1", conaffinity="1", friction="0.9 0.005 0.0001",
                      rgba="0.93 0.86 0.72 1", group="0")
    ET.SubElement(default, "site", size="0.004", rgba="1 0 0 1", group="4")
    ET.SubElement(default, "position", kp="8", kv="0.3", forcerange="-1.9 1.9", ctrllimited="true")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", rgb1=".2 .3 .4", rgb2=".3 .4 .5",
                  width="300", height="300")
    ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="8 8", reflectance="0.1")
    for (name, mirror), mname in mesh_names.items():
        ET.SubElement(asset, "mesh", name=mname, file=f"{mname}.stl", scale="0.001 0.001 0.001")

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0.3 0 1.2", dir="-0.3 0 -1", diffuse="0.9 0.9 0.9")
    if mjx:
        ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05", material="grid",
                      contype="1", conaffinity="0", priority="1", friction="0.8", condim="3")
    else:
        ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 0.05", material="grid",
                      contype="1", conaffinity="1")

    joints: list[tuple[str, tuple]] = []
    foot_contact_geoms: list[str] = []
    pelvis_h = assembly.standing_pelvis_height() * MM

    def emit(body: assembly.Body, parent_el: ET.Element, root_body: bool = False):
        el = ET.SubElement(parent_el, "body", name=body.name, pos=_fmt([v * MM for v in body.pos]))
        if root_body:
            el.set("pos", f"0 0 {pelvis_h:.5f}")
            ET.SubElement(el, "freejoint", name="root")
            # IMU frame: pelvis origin, axes aligned with the pelvis (the BNO055 goes in the body above)
            ET.SubElement(el, "site", name="imu", pos="0 0 0")
            ET.SubElement(el, "camera", name="track", mode="trackcom", pos="0.0 -0.5 0.1",
                          xyaxes="1 0 0 0 0.25 1")
            ET.SubElement(el, "light", name="track_light", mode="trackcom", pos="0.2 -0.3 1.0", dir="-0.2 0.3 -1",
                          diffuse="0.7 0.7 0.7")
        elif body.joint:
            ET.SubElement(el, "joint", name=body.joint, type="hinge", axis=_fmt(body.joint_axis),
                          range=_fmt(body.joint_range_deg))
            joints.append((body.joint, body.joint_range_deg))
        instances = []
        for inst in body.parts:
            part = parts[inst.part_name]
            loc = Location(inst.pos)
            instances.append((part, loc, inst.mirrored, inst.rot_z))
            ET.SubElement(el, "geom", name=f"{body.name}_{inst.part_name}", mesh=mesh_names[(inst.part_name, inst.mirrored)],
                          pos=_fmt([v * MM for v in inst.pos]),
                          **({"euler": f"0 0 {inst.rot_z}"} if inst.rot_z else {}))
        mass, com, inertia = body_inertial(instances)
        ET.SubElement(el, "inertial", pos=_fmt(com), mass=f"{mass:.5f}",
                      fullinertia=_fmt([inertia[0, 0], inertia[1, 1], inertia[2, 2],
                                        inertia[0, 1], inertia[0, 2], inertia[1, 2]], 5))
        if body.name.endswith("_foot"):
            mirrored = body.name.startswith("L")
            centre, half = _sole_box(parts, mirrored)
            sole_bottom = [centre[0], centre[1], centre[2] - half[2]]
            ET.SubElement(el, "site", name=f"{body.name}_site", pos=_fmt(sole_bottom))
            if mjx:
                ET.SubElement(el, "geom", name=f"{body.name}_sole", pos=_fmt(centre), size=_fmt(half),
                              **{"class": "collision"})
                foot_contact_geoms.append(f"{body.name}_sole")
            else:
                foot_contact_geoms.append(f"{body.name}_foot")
        for child in body.children:
            emit(child, el)

    emit(assembly.lower_body(), world, root_body=True)

    act = ET.SubElement(root, "actuator")
    for jname, rng in joints:
        # ctrlrange is NOT converted by compiler/angle: always radians
        ET.SubElement(act, "position", name=jname, joint=jname, ctrlrange=_fmt([math.radians(v) for v in rng]))

    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "gyro", name="gyro", site="imu")
    ET.SubElement(sensor, "accelerometer", name="accelerometer", site="imu")
    ET.SubElement(sensor, "velocimeter", name="local_linvel", site="imu")
    ET.SubElement(sensor, "framezaxis", name="upvector", objtype="site", objname="imu")
    ET.SubElement(sensor, "framelinvel", name="global_linvel", objtype="site", objname="imu")
    ET.SubElement(sensor, "frameangvel", name="global_angvel", objtype="site", objname="imu")
    for side in ("L", "R"):
        ET.SubElement(sensor, "framepos", name=f"{side}_foot_pos", objtype="site", objname=f"{side}_foot_site")
        ET.SubElement(sensor, "framelinvel", name=f"{side}_foot_global_linvel", objtype="site",
                      objname=f"{side}_foot_site")
    for g in foot_contact_geoms:
        ET.SubElement(sensor, "contact", name=f"{g}_floor_found", geom1=g, geom2="floor", reduce="mindist",
                      num="1", data="found")

    # keyframe qpos/ctrl are always radians, whatever compiler/angle says
    rest = {j: math.radians(a) for j, a in assembly.rest_pose().items()}
    key = ET.SubElement(root, "keyframe")
    qpos = ["0", "0", f"{pelvis_h:.5f}", "1", "0", "0", "0"] + [f"{rest[j]:.5f}" for j, _ in joints]
    ctrl = [f"{rest[j]:.5f}" for j, _ in joints]
    ET.SubElement(key, "key", name="rest", qpos=" ".join(qpos), ctrl=" ".join(ctrl))
    return root


def write_mjcf(parts: dict[str, CoguPart], out_dir: Path) -> list[Path]:
    """Write `cogu_lower_body.xml` (full collision meshes) and `cogu_lower_body_mjx.xml`
    (training variant) sharing one `meshes/` folder."""
    mesh_names = export_meshes(parts, out_dir / "meshes")
    written = []
    for mjx, fname in ((False, "cogu_lower_body.xml"), (True, "cogu_lower_body_mjx.xml")):
        xml = minidom.parseString(ET.tostring(build_mjcf(parts, mesh_names, mjx))).toprettyxml(indent="  ")
        path = out_dir / fname
        path.write_text(xml)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# STL / STEP
# ---------------------------------------------------------------------------

def write_print_files(parts: dict[str, CoguPart], out_dir: Path) -> list[Path]:
    stl_dir = out_dir / "stl"
    stl_dir.mkdir(parents=True, exist_ok=True)
    written = []
    notes = ["# Print orientation notes (generated by cad/export.py)", ""]
    for name, part in parts.items():
        for mirror in (False, True):
            suffix = "L" if mirror else "R"
            solid = assembly.mirrored(part.solid) if mirror else part.solid
            oriented = CoguPart(name, solid, print_rotation=part.print_rotation).oriented_for_print()
            path = stl_dir / f"{name}_{suffix}.stl"
            export_stl(oriented, str(path), tolerance=0.02, angular_tolerance=0.1)
            written.append(path)
            if name in ("pelvis", "back_hub"):
                break      # symmetric parts: one file
        notes.append(f"- **{name}** ({part.material}, ~{part.printed_mass_g:.0f} g printed): {part.print_notes}")
    (stl_dir / "PRINT_NOTES.md").write_text("\n".join(notes) + "\n")
    return written


def write_step(parts: dict[str, CoguPart], out_dir: Path) -> Path:
    step_dir = out_dir / "step"
    step_dir.mkdir(parents=True, exist_ok=True)
    # assembled right leg + pelvis at q = 0, in the pelvis frame
    placed = []

    def walk(body: assembly.Body, origin: tuple):
        o = tuple(a + b for a, b in zip(origin, body.pos))
        for inst in body.parts:
            solid = parts[inst.part_name].solid
            if inst.mirrored:
                solid = assembly.mirrored(solid)
            if inst.rot_z:
                solid = solid.rotate(Axis.Z, inst.rot_z)
            placed.append(solid.moved(Location(tuple(a + b for a, b in zip(o, inst.pos)))))
        for c in body.children:
            walk(c, o)

    walk(assembly.lower_body(), (0, 0, 0))
    path = step_dir / "cogu_lower_body.step"
    export_step(Compound(children=placed), str(path))
    for name, part in parts.items():
        export_step(part.solid, str(step_dir / f"{name}.step"))
    return path


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build")
    ap.add_argument("--no-step", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("building parts …")
    parts = assembly.build_parts()
    from cad.test_prints import build_test_prints
    tests = build_test_prints()

    print("design rules …")
    violations = design_rules.check_all(list(parts.values()) + list(tests.values()))
    for v in violations:
        print("  VIOLATION:", v)

    print("STL (print orientation) …")
    write_print_files(parts, out)
    tdir = out / "stl" / "test_prints"
    tdir.mkdir(parents=True, exist_ok=True)
    for name, part in tests.items():
        export_stl(part.oriented_for_print(), str(tdir / f"{name}.stl"), tolerance=0.02, angular_tolerance=0.1)

    print("MJCF …")
    for mjcf in write_mjcf(parts, out / "mjcf"):
        print("  ", mjcf)

    if not args.no_step:
        print("STEP …")
        print("  ", write_step(parts, out))

    print()
    print(f"{'part':16s} {'printed g':>10s} {'with servos g':>14s}")
    for name, part in parts.items():
        print(f"{name:16s} {part.printed_mass_g:10.1f} {part.total_mass_g:14.1f}")
    print(f"standing pelvis height: {assembly.standing_pelvis_height():.1f} mm")
    if violations:
        print(f"\n{len(violations)} design-rule violation(s)")
        return 1
    print("\ndesign rules: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
