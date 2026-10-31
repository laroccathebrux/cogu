"""Clearance above the foot sole top under the swept shin, over the ankle range."""
import numpy as np, trimesh, sys
from build123d import export_stl
from cad import parameters as P
from cad.parts.link import build_shin
from cad.parts.foot import build_foot, LATERAL_CENTRE
from cad.parts.back_hub import build_back_hub
out = sys.argv[1]
export_stl(build_shin().solid, f"{out}/shin.stl", tolerance=0.05)
export_stl(build_foot().solid, f"{out}/foot.stl", tolerance=0.05)
shin = trimesh.load(f"{out}/shin.stl")
lo, hi = P.JOINT_RANGES_DEG["ankle"]
angles = np.arange(lo - 3, hi + 3.1, 1.5)
sole_top = -P.FOOT_HEIGHT + P.FOOT_SOLE_THICKNESS
xs = np.arange(-P.ANKLE_X_FROM_HEEL, P.FOOT_LENGTH - P.ANKLE_X_FROM_HEEL + 0.1, 1.0)
ys = np.arange(LATERAL_CENTRE - P.FOOT_WIDTH/2, LATERAL_CENTRE + P.FOOT_WIDTH/2 + 0.1, 1.0)
X, Y = np.meshgrid(xs, ys)
orig = np.c_[X.ravel(), Y.ravel(), np.full(X.size, sole_top)]
dirs = np.tile([0, 0, 1.0], (X.size, 1))
clear = np.full(X.size, 99.0)
def heightmap(tris):
    """lowest z of the mesh above each grid point (99 if none)"""
    h = np.full(X.size, 99.0)
    px, py = orig[:, 0], orig[:, 1]
    for t in tris:
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = t
        m = (px >= min(x0, x1, x2)) & (px <= max(x0, x1, x2)) & (py >= min(y0, y1, y2)) & (py <= max(y0, y1, y2))
        if not m.any():
            continue
        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(det) < 1e-9:
            continue
        qx, qy = px[m], py[m]
        l0 = ((y1 - y2) * (qx - x2) + (x2 - x1) * (qy - y2)) / det
        l1 = ((y2 - y0) * (qx - x2) + (x0 - x2) * (qy - y2)) / det
        l2 = 1 - l0 - l1
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        z = l0 * z0 + l1 * z1 + l2 * z2
        idx = np.nonzero(m)[0][inside]
        h[idx] = np.minimum(h[idx], z[inside] - sole_top)
    return h
worst_angle = np.full(X.size, np.nan)
for a in angles:
    m = shin.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(a), [0, 1, 0]))
    tris = m.triangles
    tris = tris[tris[:, :, 2].min(axis=1) < sole_top + 15]
    h = heightmap(tris)
    better = h < clear
    worst_angle[better] = a
    clear = np.minimum(clear, h)
C = clear.reshape(X.shape)
np.save(f"{out}/clear.npy", np.stack([X, Y, C, worst_angle.reshape(X.shape)]))
print("sole_top", sole_top, "lateral centre", LATERAL_CENTRE, "y", ys[0], ys[-1])
for x in range(-22, 44, 3):
    row = C[:, np.argmin(abs(xs - x))]
    print(f"x={x:4d} ", " ".join(f"{v:4.1f}" if v < 30 else "  . " for v in row[::3]))
print("y cols:", " ".join(f"{y:5.1f}" for y in ys[::3]))
