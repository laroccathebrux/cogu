"""Compare phone-guard variants (ADR-002 follow-up): mass, CoM, height and what happens to the phone in falls.

    uv run python -m cad.export --out build/guard_study/none   --no-step --guard none
    uv run python -m cad.export --out build/guard_study/frame  --no-step --guard frame
    uv run python -m cad.export --out build/guard_study/bubble --no-step --guard bubble
    uv run python sim/guard_study.py            # reads the three exports above

For each variant (full-contact MJCF, servos holding the rest pose, no policy) the robot is tipped over in
8 directions (an initial 4 rad/s roll-over about a horizontal axis) and simulated for 2.5 s. Reported per
variant: how many of the 8 falls let the phone itself touch the floor, and the phone's fastest speed at
contact. Also writes build/guard_study/{base,guard_frame,guard_bubble}.glb (the assembled robot at rest,
for the 3D comparison page) and summary.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "guard_study"
VARIANTS = {"none": OUT / "none" / "mjcf" / "cogu_lower_body.xml",
            "frame": OUT / "frame" / "mjcf" / "cogu_lower_body.xml",
            "bubble": OUT / "bubble" / "mjcf" / "cogu_lower_body.xml"}

COLORS = {"legs": (232, 235, 237), "hip": (232, 235, 237), "cover": (232, 235, 237), "feet": (225, 102, 74),
          "servo": (23, 23, 25), "phone": (18, 18, 22), "camera": (90, 92, 98), "guard": (198, 203, 209),
          "tpu": (22, 22, 24)}


def category(name: str) -> str | None:
    if name == "pelvis_phone_guard_tpu":
        return "tpu"
    if name.startswith("pelvis_phone_guard"):
        return "guard"
    if name == "pelvis_hip_cover":
        return "cover"
    if name in ("pelvis_phone_cradle", "pelvis_pelvis"):
        return "hip"
    if name.startswith("ref_") and "_servo_" in name and not name.startswith("ref_yaw"):
        return "servo"
    if name.endswith("_foot_foot"):
        return "feet"
    if name == "ref_phone":
        return "phone"
    if name == "ref_phone_camera":
        return "camera"
    if name.startswith(("L_", "R_")):
        return "legs"
    if name == "ref_servo_driver":
        return "servo"              # black PCB on the cradle, behind the phone
    return None                     # yaw servos, battery and IMU are hidden inside the cover


def rest(m: mujoco.MjModel) -> mujoco.MjData:
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.keyframe("rest").id)
    mujoco.mj_forward(m, d)
    return d


def static_numbers(m: mujoco.MjModel) -> dict:
    d = rest(m)
    top = 0.0
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        top = max(top, float((d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].max()))
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    up = d.xmat[1].reshape(3, 3)[:, 2]
    return {"mass_kg": float(m.body_subtreemass[1]), "com_height_mm": float(rest(m).subtree_com[1][2] * 1000),
            "height_mm": top * 1000, "stands": bool(up[2] > math.cos(math.radians(15)))}


def fall_test(m: mujoco.MjModel, n_dirs: int = 8, seconds: float = 2.5) -> dict:
    floor = m.geom("floor").id
    phone = m.geom("ref_phone").id
    pelvis = m.body("pelvis").id
    results = []
    for k in range(n_dirs):
        th = 2 * math.pi * k / n_dirs                    # 0 = falls forward (+x), pi/2 = to the robot's left
        d = rest(m)
        axis = np.array([-math.sin(th), math.cos(th), 0.0])
        d.qvel[3:6] = 4.0 * axis                         # tip over towards direction th
        d.qvel[0:2] = 0.3 * np.array([math.cos(th), math.sin(th)])
        touched, v_hit = False, 0.0
        for _ in range(int(seconds / m.opt.timestep)):
            mujoco.mj_step(m, d)
            for c in d.contact[: d.ncon]:
                if {c.geom1, c.geom2} == {floor, phone}:
                    vel = np.zeros(6)
                    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, pelvis, vel, 0)
                    touched, v_hit = True, max(v_hit, float(np.linalg.norm(vel[3:])))
        results.append({"direction_deg": round(math.degrees(th)), "phone_touched": touched, "speed_m_s": v_hit})
    return {"falls": results, "phone_hits": sum(r["phone_touched"] for r in results),
            "worst_body_speed_m_s": max(r["speed_m_s"] for r in results)}


def export_glb(m: mujoco.MjModel, cats: set[str], path: Path):
    d = rest(m)
    groups: dict[str, list[trimesh.Trimesh]] = {}
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        cat = category(name)
        if cat not in cats or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        f = m.mesh_face[m.mesh_faceadr[mid]: m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
        groups.setdefault(cat, []).append(trimesh.Trimesh(w[:, [0, 2, 1]] * [1, 1, -1], f, process=False))
    scene = trimesh.Scene()
    for cat, meshes in groups.items():
        mesh = trimesh.util.concatenate(meshes)
        mesh.merge_vertices()
        mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=[*COLORS[cat], 255])
        scene.add_geometry(mesh, node_name=cat, geom_name=cat)
    path.write_bytes(scene.export(file_type="glb"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, xml in VARIANTS.items():
        if not xml.exists():
            print(f"missing {xml}: run cad.export first (see the docstring)", file=sys.stderr)
            return 1
        m = mujoco.MjModel.from_xml_path(str(xml))
        s = static_numbers(m)
        s.update(fall_test(m))
        summary[name] = s
        hits = s["phone_hits"]
        print(f"{name:7s} mass {s['mass_kg']*1000:6.0f} g  CoM {s['com_height_mm']:5.0f} mm  height {s['height_mm']:4.0f} mm  "
              f"stands {s['stands']}  phone hit the floor in {hits}/8 falls  hits from {[f["direction_deg"] for f in s["falls"] if f["phone_touched"]]}°")
        if name == "none":
            export_glb(m, {"legs", "hip", "cover", "feet", "servo", "phone", "camera"}, OUT / "base.glb")
        else:
            export_glb(m, {"guard", "tpu"}, OUT / f"guard_{name}.glb")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    for f in sorted(OUT.glob("*.glb")):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
