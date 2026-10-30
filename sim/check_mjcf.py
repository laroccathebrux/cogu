"""Load the generated MJCF, sanity-check it and let the robot stand at the rest pose.

    uv run python sim/check_mjcf.py [build/mjcf/cogu_lower_body.xml] [--render out.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else Path("build/mjcf/cogu_lower_body.xml")
    render_to = None
    if "--render" in sys.argv:
        render_to = Path(sys.argv[sys.argv.index("--render") + 1])

    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    print(f"model: {model.nbody} bodies, {model.njnt} joints, {model.nu} actuators, {model.nmesh} meshes")
    print(f"total mass: {sum(model.body_mass):.3f} kg")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"  {name:14s} {model.body_mass[i]*1000:7.1f} g  inertia diag {np.array2string(model.body_inertia[i], precision=2)}")

    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "rest")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    z0 = data.xpos[pelvis][2]
    # lowest mesh vertex of the foot soles at rest (should be ~ 0 = ground)
    def lowest_z(geom_id: int) -> float:
        mid = model.geom_dataid[geom_id]
        v = model.mesh_vert[model.mesh_vertadr[mid]: model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        w = data.geom_xpos[geom_id] + v @ data.geom_xmat[geom_id].reshape(3, 3).T
        return float(w[:, 2].min())
    feet = [i for i in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").endswith("_foot_foot")]
    low = min(lowest_z(i) for i in feet)
    print(f"rest pose: pelvis z = {z0*1000:.1f} mm, foot bound min z = {low*1000:.1f} mm")

    # simulate 3 s holding the rest pose with the position actuators
    steps = int(3.0 / model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    z1 = data.xpos[pelvis][2]
    quat = data.xquat[pelvis]
    tilt_deg = float(np.degrees(2 * np.arccos(min(1.0, abs(quat[0])))))
    print(f"after 3 s: pelvis z = {z1*1000:.1f} mm (drop {(z0-z1)*1000:.1f} mm), tilt {tilt_deg:.1f}°, "
          f"contacts {data.ncon}")
    ok = (z0 - z1) < 0.03 and tilt_deg < 15
    print("STANDING: OK" if ok else "STANDING: FAILED (fell or sank)")

    if render_to:
        try:
            renderer = mujoco.Renderer(model, height=720, width=960)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0, 0, 0.19]          # mid-height of the ~400 mm robot with the phone
            cam.distance, cam.azimuth, cam.elevation = 0.85, 135, -12
            renderer.update_scene(data, camera=cam)
            import PIL.Image
            PIL.Image.fromarray(renderer.render()).save(render_to)
            print("rendered", render_to)
        except Exception as e:   # offscreen GL is platform dependent
            print("render skipped:", e)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
