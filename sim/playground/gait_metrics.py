"""Measure how a policy walks, not just whether it stays up (plain MuJoCo, like play.py).

    uv run python sim/playground/gait_metrics.py sim/playground/checkpoints/cogu_walk_v1 build/checkpoints/cogu_walk_v3
    uv run python sim/playground/gait_metrics.py CKPT --vx 0.2 --seconds 10

For each checkpoint, an 8 s walk from the `rest` pose at the given command (the first second is
skipped), reporting the numbers that separate a gait from a shuffle:

    step freq    dominant frequency of the knee motion (Hz). A gait at this size: 1.5–2.5 Hz.
    jitter       share of joint-velocity power above 5 Hz. Vibrating policies sit near 1.
    joint swing  mean peak-to-peak joint motion (deg) over hip pitch, knee and ankle.
    foot lift    mean of each step's swing apex (mm). The training target is max_foot_height.
    air time     mean duration of a swing (s); duty = share of time a foot is on the floor.
    slip         foot speed while touching the floor (m/s); ideally ~0, compare with body speed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import cogu_env  # noqa: E402
from play import Sim2Sim, checkpoint_label  # noqa: E402
from rollout import latest_checkpoint  # noqa: E402

GOOD = {"step freq": "1.5–2.5 Hz", "jitter": "< 0.2", "joint swing": "15–40°", "foot lift": "≥ 20 mm",
        "air time": "0.2–0.4 s", "duty": "0.5–0.7", "slip": "< 0.03 m/s"}


def measure(sim: Sim2Sim, seconds: float = 8.0, skip: float = 1.0) -> dict:
    m, d = sim.m, sim.d
    feet = [m.site(n).id for n in ("L_foot_site", "R_foot_site")]
    found = [m.sensor_adr[m.sensor(f"{s}_foot_sole_floor_found").id] for s in ("L", "R")]
    n, n_skip = int(round(seconds / sim.dt)), int(round(skip / sim.dt))
    q, contact, fz, slip, x = [], [], [], [], []
    prev = d.site_xpos[feet].copy()
    fell = None
    for i in range(n):
        sim.step()
        pos = d.site_xpos[feet].copy()
        vel = (pos - prev) / sim.dt
        prev = pos
        if sim.fallen():
            fell = (i + 1) * sim.dt
            break
        if i < n_skip:
            continue
        c = d.sensordata[found] > 0
        q.append(d.qpos[7:].copy())
        contact.append(c)
        fz.append(pos[:, 2])
        slip.append(np.where(c, np.linalg.norm(vel[:, :2], axis=1), np.nan))
        x.append(d.qpos[:2].copy())
    out = {"fell": fell}
    if len(q) < 50:
        return out
    q, contact, fz, slip, x = map(np.array, (q, contact, fz, slip, x))
    dt = sim.dt
    # knee motion spectrum -> dominant frequency
    k = q[:, [3, 8]].mean(1)
    k = k - k.mean()
    freqs = np.fft.rfftfreq(len(k), dt)
    power = np.abs(np.fft.rfft(k)) ** 2
    out["step freq"] = float(freqs[1:][power[1:].argmax()])
    # jitter: share of joint-velocity power above 5 Hz, all joints
    qd = np.diff(q, axis=0) / dt
    pw = np.abs(np.fft.rfft(qd - qd.mean(0), axis=0)) ** 2
    fq = np.fft.rfftfreq(len(qd), dt)
    out["jitter"] = float(pw[fq > 5.0].sum() / max(pw[1:].sum(), 1e-12))
    sagittal = [2, 3, 4, 7, 8, 9]
    out["joint swing"] = float(np.degrees(q[:, sagittal].max(0) - q[:, sagittal].min(0)).mean())
    # swings: runs of no contact per foot
    apexes, airs = [], []
    for f in range(2):
        c = contact[:, f]
        start = None
        for t in range(1, len(c)):
            if c[t - 1] and not c[t]:
                start = t
            elif start is not None and not c[t - 1] and c[t]:
                airs.append((t - start) * dt)
                apexes.append(fz[start:t, f].max())
                start = None
    out["foot lift"] = float(np.mean(apexes) * 1000) if apexes else 0.0
    out["air time"] = float(np.mean(airs)) if airs else 0.0
    out["steps"] = len(airs)
    out["duty"] = float(contact.mean())
    out["slip"] = float(np.nanmean(slip))
    out["speed"] = float(np.linalg.norm(x[-1] - x[0]) / (len(x) * dt))
    return out


def fmt(key: str, v) -> str:
    if v is None:
        return "-"
    return {"step freq": f"{v:.2f} Hz", "jitter": f"{v:.2f}", "joint swing": f"{v:.1f}°",
            "foot lift": f"{v:.1f} mm", "air time": f"{v:.2f} s", "duty": f"{v:.2f}", "slip": f"{v:.3f} m/s",
            "speed": f"{v:.3f} m/s", "steps": f"{v}"}.get(key, str(v))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--xml", default=str(cogu_env.DEFAULT_XML))
    ap.add_argument("--vx", type=float, default=0.15)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args(argv)

    rows, labels = [], []
    for c in args.checkpoints:
        ckpt = latest_checkpoint(Path(c).resolve())
        sim = Sim2Sim(Path(args.xml), ckpt)
        sim.command[:] = (args.vx, args.vy, args.wz)
        rows.append(measure(sim, args.seconds))
        labels.append(checkpoint_label(ckpt))
    keys = ["speed", "step freq", "jitter", "joint swing", "foot lift", "air time", "steps", "duty", "slip"]
    w = max(22, *(len(l) for l in labels))
    print(f"command vx={args.vx} vy={args.vy} wz={args.wz}, {args.seconds:.0f} s from rest\n")
    print(f"{'':12s}" + "".join(f"{l:>{w}s}" for l in labels) + f"{'a gait looks like':>22s}")
    for k in keys:
        print(f"{k:12s}" + "".join(f"{fmt(k, r.get(k)):>{w}s}" for r in rows) + f"{GOOD.get(k, ''):>22s}")
    fell = [f"{r['fell']:.1f} s" if r.get("fell") else "no" for r in rows]
    print(f"{'fell':12s}" + "".join(f"{f:>{w}s}" for f in fell))
    return 0


if __name__ == "__main__":
    sys.exit(main())
