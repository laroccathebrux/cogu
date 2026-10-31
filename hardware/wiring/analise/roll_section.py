"""Section of the hip-roll joint (plane through the roll axis, perpendicular to the leg's lateral axis)."""
import os, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fk import frames
from meshes import body_meshes, servo_frames, SV
from cad import assembly as A, parameters as P
from matplotlib.collections import LineCollection

def slice_y(tris, y):
    """(x, z) segments where the triangles cross the plane Y = y (no networkx needed)"""
    d = tris[:, :, 1] - y
    segs = []
    for t, dd in zip(tris, d):
        if dd.min() > 0 or dd.max() < 0: continue
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if (dd[i] > 0) != (dd[j] > 0):
                f = dd[i] / (dd[i] - dd[j]); q = t[i] + f * (t[j] - t[i]); pts.append((q[0], q[2]))
        if len(pts) == 2: segs.append(pts)
    return segs
out = sys.argv[1]
fr = frames(A.rest_pose()); sf = servo_frames()
bm = body_meshes(exclude=("pitch_servo_R", "knee_servo_R", "ankle_servo_R", "yaw_servo_R"))
fig, ax = plt.subplots(figsize=(8, 8))
cols = {"pelvis": "0.6", "R_hip_yaw": "tab:blue", "R_hip_roll": "tab:orange"}
for b, c in cols.items():
    R, t, _ = fr[b]; m = bm[b].copy()
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; m.apply_transform(T)
    for y, ls in ((-20.0, "-"), (-26.0, ":")):
        ax.add_collection(LineCollection(slice_y(m.triangles, y), colors=c, linestyles=ls, lw=1.2 if ls == "-" else 0.8))
# socket footprint and a plug + bend envelope (8 + 3 mm) on the roll servo's back face
b, Rs, ts = sf["roll_servo_R"]; Rb, tb, _ = fr[b]
for x in (SV.bus_socket_x[0], SV.bus_socket_x[1]):
    p0 = tb + Rb @ (ts + Rs @ [x, -SV.case_height, 0]); p1 = tb + Rb @ (ts + Rs @ [x, -SV.case_height - 11, 0])
    ax.plot([p0[0], p1[0]], [p0[2], p1[2]], "r-", lw=2)
ax.text(p1[0] - 2, p1[2] - 3, "plugue + curva\n(11 mm, estimado)", color="r", fontsize=8, ha="right")
ax.set_aspect("equal"); ax.set_xlim(-60, 45); ax.set_ylim(-80, 15); ax.grid(alpha=.2)
ax.set_xlabel("x (mm, + = frente)"); ax.set_ylabel("z (mm)")
ax.set_title("Corte da junta do roll, perna direita (linha cheia: plano do eixo, y = −20; pontilhada: y = −26)\n"
             "azul = yaw link + servo do roll, laranja = roll link + hub, cinza = pelve", fontsize=9)
fig.tight_layout(); fig.savefig(f"{out}/roll_section.png", dpi=120)
