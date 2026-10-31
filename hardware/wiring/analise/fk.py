"""FK of the Cogu body tree + STS3215 connector positions (assumed on the servo's back-end face, +X)."""
import numpy as np
from build123d import Box, Location
from cad import parameters as P
from cad import assembly as A
from cad.parts import sts3215

SV = P.SERVO
PLUG = 8.0   # plug + cable bend beyond the case end

def rot(axis, deg):
    a = np.asarray(axis, float); a /= np.linalg.norm(a); t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K

def frames(q: dict, root=None):
    """world (R, t) of every body for joint angles q (deg, missing = 0)"""
    root = root or A.robot()
    out = {}
    def walk(b, R, t):
        t2 = t + R @ np.asarray(b.pos, float)
        R2 = R @ (rot(b.joint_axis, q.get(b.joint, 0.0)) if b.joint else np.eye(3))
        out[b.name] = (R2, t2, b)
        for c in b.children:
            walk(c, R2, t2)
    walk(root, np.eye(3), np.zeros(3))
    return out

def connector_points():
    """{servo name: (body, conn point, exit point)} in the body frame"""
    def marker(x):
        return Box(0.2, 0.2, 0.2).moved(Location((x, -SV.case_height / 2, 0)))
    res = {}
    orig = sts3215.servo_solid
    pts = {}
    for key, x in (("c", SV.x_max), ("e", SV.x_max + PLUG), ("s", 0.0)):
        sts3215.servo_solid = lambda **kw: marker(x)
        try:
            for r in A.reference_solids():
                if "servo" in r.name and "driver" not in r.name:
                    c = r.solid.bounding_box().center()
                    pts.setdefault(r.name, {"body": r.body})[key] = np.array([c.X, c.Y, c.Z])
        finally:
            sts3215.servo_solid = orig
    return pts

def world(fr, body, p):
    R, t, _ = fr[body]
    return t + R @ p

if __name__ == "__main__":
    pts = connector_points()
    fr = frames(A.rest_pose())
    for n, d in pts.items():
        if n.endswith("_L"): continue
        print(f"{n:15s} body={d['body']:12s} conn(body)={np.round(d['c'],1)}  conn(world,rest)={np.round(world(fr, d['body'], d['c']),1)}  exit dir={np.round(d['e']-d['c'],1)}")
    for b in ("R_hip_yaw", "R_hip_roll", "R_thigh", "R_shin", "R_foot"):
        R, t, body = fr[b]
        print(f"joint {body.joint:12s} axis world={np.round(R @ np.asarray(body.joint_axis,float),2)} at {np.round(t,1)}")
