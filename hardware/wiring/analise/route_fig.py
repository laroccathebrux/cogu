import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fk import frames
from meshes import body_meshes, servo_frames, SV
from cad import assembly as A, parameters as P
rest = A.rest_pose(); fr = frames(rest); sf = servo_frames()
bm = body_meshes(exclude=())
def wv(body, v):
    R, t, _ = fr[body]; return (R @ v.T).T + t
TOP = np.array([14.0, -SV.case_height - 8, 0])
def conn(n):
    b, R, t = sf[n]; Rb, tb, _ = fr[b]; return tb + Rb @ (t + R @ TOP)
ax_pos = {b: fr[b][1] for b in ("R_hip_yaw","R_hip_roll","R_thigh","R_shin","R_foot")}
YO = -12.7 - 43.35 - 1.0           # outboard faces of the right leg (bearing plates), + 1 mm
C = {n: conn(n + "_servo_R") for n in ("yaw","roll","pitch","knee","ankle")}
P_ = lambda *a: np.array(a, float)
routes = {
 "placa → yaw":         [P_(-36,-4,52), P_(-23,-4,40), P_(-18,-14,36)] + [C["yaw"]],
 "yaw → roll":          [C["yaw"], P_(-34,-20,36), P_(-34,-20,-20), P_(-26,-20,-38), C["roll"]],
 "roll → pitch":        [C["roll"], P_(-22,-34,-45), P_(-14,YO,-62), P_(-0.3,YO,-81.1), C["pitch"]],
 "pitch → joelho":      [C["pitch"], P_(-6,YO,-88), ax_pos["R_thigh"] + [0, YO + 12.7, -10], C["knee"]],
 "joelho → tornozelo":  [C["knee"], ax_pos["R_shin"] + [0, YO + 12.7, -10], C["ankle"]],
 "chaves do pé":        [P_(-14.3+27,-30.5,-200), P_(-14.3,-30.5,-200), P_(-14.3,YO,-200), P_(-14.3,YO,-189), ax_pos["R_foot"] + [-3, YO + 12.7, 0], wv("R_shin", np.array([[0, -44.35, -46.0]]))[0], C["ankle"]],
}
def plen(pts): return sum(np.linalg.norm(b - a) for a, b in zip(pts, pts[1:]))
for k, v in routes.items(): print(f"{k:20s} {plen(v):5.0f} mm (rota esquemática)")
print("pé -> placa", plen(routes["chaves do pé"]) + sum(plen(v) for k, v in routes.items() if k != "chaves do pé"))
cols = dict(zip(routes, ["#444", "#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]))
fig, axs = plt.subplots(1, 2, figsize=(13, 8.5))
for ax, (i, j, lab) in zip(axs, ((0, 2, "lado (x, z) — perna direita"), (1, 2, "frente (y, z) — perna direita"))):
    for b, m in bm.items():
        if b.startswith("L_"): continue
        v = wv(b, m.vertices)[m.faces]
        if i == 1 and b == "pelvis": v = v[(v[:, :, 1].mean(axis=1) < 5)]
        ax.add_collection(PolyCollection(v[:, :, [i, j]], fc="0.85", ec="none", alpha=0.25))
    for k, pts in routes.items():
        pts = np.array(pts); ax.plot(pts[:, i], pts[:, j], "-", color=cols[k], lw=2.5, label=k)
    for n, p in C.items():
        ax.plot(p[i], p[j], "o", ms=8, mfc="w", mec="k"); ax.annotate(n, (p[i], p[j]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    for b, p in ax_pos.items():
        ax.plot(p[i], p[j], "k+", ms=10)
    ax.set_aspect("equal"); ax.autoscale(); ax.set_title(lab, fontsize=10); ax.grid(alpha=.2)
axs[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Cogu: cabos da perna direita na pose de repouso (○ = topo do plugue na face traseira do servo, + = eixos)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{sys.argv[1]}/cable_route.png", dpi=120)
