"""Free depth in front of the connectors on each servo's BACK face (servo −Y), over the joint range.
Sockets (photo, 2026-10-30): two 3-pin 5264 side by side across the width, x ≈ 11.6..16.5 from the shaft."""
import sys, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fk import frames
from meshes import body_meshes, servo_frames, SV
from cad import assembly as A, parameters as P

sf = servo_frames(); rest = A.rest_pose(); rng = P.JOINT_RANGES_DEG
CASES = {
    "yaw_servo_R": (None, []),
    "roll_servo_R": ("R_hip_roll", ["R_hip_roll"]),
    "pitch_servo_R": ("R_hip_pitch", ["R_thigh"]),
    "knee_servo_R": ("R_knee", ["R_shin"]),
    "ankle_servo_R": ("R_ankle", ["R_foot"]),
}
X0, X1 = 11.6, 16.5
xs = np.arange(X0, X1 + 0.01, 0.7); zs = np.arange(-10, 10.01, 1.0)
GX, GZ = np.meshgrid(xs, zs)
YF = -SV.case_height
def depthmap(tris):
    h = np.full(GX.size, 99.0); px, pz = GX.ravel(), GZ.ravel()
    tris = tris[tris[:, :, 1].min(axis=1) < YF + 0.01]
    for t in tris:
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = t
        m = (px >= min(x0, x1, x2)) & (px <= max(x0, x1, x2)) & (pz >= min(z0, z1, z2)) & (pz <= max(z0, z1, z2))
        if not m.any(): continue
        det = (z1 - z2) * (x0 - x2) + (x2 - x1) * (z0 - z2)
        if abs(det) < 1e-9: continue
        qx, qz = px[m], pz[m]
        l0 = ((z1 - z2) * (qx - x2) + (x2 - x1) * (qz - z2)) / det
        l1 = ((z2 - z0) * (qx - x2) + (x0 - x2) * (qz - z2)) / det
        l2 = 1 - l0 - l1
        ok = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        y = l0 * y0 + l1 * y1 + l2 * y2
        ok &= y <= YF + 0.01
        idx = np.nonzero(m)[0][ok]
        h[idx] = np.minimum(h[idx], YF - y[ok])
    return h
def in_servo(mesh, fr, body, sbody, R, t):
    Rb, tb, _ = fr[body]; Rs, ts, _ = fr[sbody]
    v = (Rb @ mesh.vertices.T).T + tb
    v = Rs.T @ (v - ts).T
    v = (R.T @ (v.T - t).T).T
    return v[mesh.faces]
for name, (joint, others) in CASES.items():
    sbody, R, t = sf[name]
    bm = body_meshes(exclude=(name,))
    own = depthmap(in_servo(bm[sbody], frames(rest), sbody, sbody, R, t)).reshape(GZ.shape)
    worst = np.full(GX.size, 99.0); wa = np.full(GX.size, np.nan)
    if joint:
        jn = joint.split("_", 1)[1]
        for a in np.arange(rng[jn][0], rng[jn][1] + 0.1, 5.0):
            q = dict(rest); q[joint] = a; fr = frames(q)
            h = np.full(GX.size, 99.0)
            for ob in others:
                h = np.minimum(h, depthmap(in_servo(bm[ob], fr, ob, sbody, R, t)))
            b = h < worst; worst[b] = h[b]; wa[b] = a
    w = worst.reshape(GZ.shape); wa = wa.reshape(GZ.shape)
    print(f"{name:14s} body={sbody:11s} back face points {np.round(R @ [0,-1,0] ,2)} (body frame)")
    print(f"   own part: first wall at {own.min():.1f}..{own.max():.1f} mm; beyond the own walls, next own surface: see map")
    if joint:
        print(f"   moving neighbour ({joint}): free depth min {w.min():.1f} (at {wa.flat[np.argmin(w)]}°, x={GX.flat[np.argmin(w)]:.1f} z={GZ.flat[np.argmin(w)]:.0f}), median {np.median(w):.1f}")
    np.save(f"{sys.argv[1]}/head2_{name}.npy", np.stack([GX, GZ, own, w, wa]))
