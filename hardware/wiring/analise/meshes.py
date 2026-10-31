"""Per-body trimesh of the right leg + pelvis (printed parts + reference solids), cached as STL."""
import os, numpy as np, trimesh
from build123d import export_stl, Box, Location
from cad import parameters as P
from cad import assembly as A
from cad.parts import sts3215
SV = P.SERVO
D = os.path.join("build", "wiring_stl")   # cache, run from the repo root
os.makedirs(D, exist_ok=True)

def to_mesh(solid, key):
    f = os.path.join(D, f"{key}_{solid.volume:.0f}.stl")   # the volume in the name: a CAD change misses the cache
    if not os.path.exists(f):
        export_stl(solid, f, tolerance=0.1)
    return trimesh.load(f)

def body_meshes(side="R", exclude=()):
    parts = A.build_parts()
    out = {}
    def walk(b):
        ms = []
        for inst in b.parts:
            if inst.part_name.startswith("phone_guard"): continue
            s = parts[inst.part_name].solid
            if inst.mirrored: s = A.mirrored(s)
            if inst.rot_z: s = s.rotate(__import__("build123d").Axis.Z, inst.rot_z)
            s = s.moved(Location(inst.pos))
            ms.append(to_mesh(s, f"{b.name}_{inst.part_name}"))
        out[b.name] = ms
        for c in b.children:
            if c.name.startswith(("L_", "R_")) and not c.name.startswith(side): continue
            walk(c)
    walk(A.robot())
    for r in A.reference_solids():
        if r.body in out and not r.name.endswith("_L") and not r.name.startswith("phone") and r.name not in exclude:
            out[r.body].append(to_mesh(r.solid, "ref_" + r.name))
    return {k: trimesh.util.concatenate(v) for k, v in out.items() if v}

def servo_frames():
    """{servo name: (body, R, t)} servo frame -> body frame"""
    orig = sts3215.servo_solid
    pts = {}
    for key, p in (("o", (0, 0, 0)), ("x", (10, 0, 0)), ("y", (0, 10, 0)), ("z", (0, 0, 10))):
        sts3215.servo_solid = lambda **kw: Box(0.2, 0.2, 0.2).moved(Location(p))
        try:
            for r in A.reference_solids():
                if "servo" in r.name and "driver" not in r.name and not r.name.endswith("_L"):
                    c = r.solid.bounding_box().center()
                    pts.setdefault(r.name, {"body": r.body})[key] = np.array([c.X, c.Y, c.Z])
        finally:
            sts3215.servo_solid = orig
    res = {}
    for n, d in pts.items():
        R = np.c_[(d["x"] - d["o"]) / 10, (d["y"] - d["o"]) / 10, (d["z"] - d["o"]) / 10]
        res[n] = (d["body"], R, d["o"])
    return res
