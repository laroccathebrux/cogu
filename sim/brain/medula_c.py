"""The medula firmware (firmware/medula, C++) in the simulator: software in the loop. MuJoCo is the body, the C++ is
the code that will run on the ESP32, built as a shared library; `MedulaC` is a drop-in for medula.Medula, so the
phone side (pet.py, behavior.py, conversation.py) runs on it unchanged with `--medula c`.

    uv run python sim/brain/medula_c.py --test     # the firmware against the Python medula, then on its own

What the checks are:
1. **the policies**: the C++ MLPs against the JAX ones, on observations from a real run;
2. **in the shadow of the Python medula**: both get the same sensor readings and packets every step (the Python one
   drives the body); the servo targets, the state and the telemetry must match all along a sequence that walks,
   turns, stops, sits, lies down, gets up, is pushed over and gets up again;
3. **on its own**: medula.py's scenarios (silent phone, repeated sit, flat battery) with the C++ driving the body;
4. **like the robot**: the same, sensing only what the robot can (no pelvis height or speed), and with the gravity
   from the firmware's IMU filter fed by the simulated gyro and accelerometer instead of the perfect one.
"""

from __future__ import annotations

import ctypes
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import mujoco  # noqa: E402

import protocol as proto  # noqa: E402
from medula import Battery  # noqa: E402

FIRMWARE = ROOT / "firmware" / "medula"
LIB = ROOT / "build" / "medula" / ("libcogu_medula.dylib" if sys.platform == "darwin" else "libcogu_medula.so")
MODES = ("walking", "pose", "getting up")
_lib = None


def library():
    """Build (make: only if a source changed) and load the firmware as a shared library."""
    global _lib
    if _lib is not None:
        return _lib
    r = subprocess.run(["make", "-s", "-C", str(FIRMWARE), "lib"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"building the medula firmware failed:\n{r.stdout}\n{r.stderr}")
    lib = ctypes.CDLL(str(LIB))
    P, F, D, I, B = ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double), ctypes.c_int, \
        ctypes.POINTER(ctypes.c_uint8)
    sigs = dict(cogu_nu=([], I), cogu_sensors_size=([], I), cogu_medula_new=([], P), cogu_medula_free=([P], None),
                cogu_medula_begin=([P, P], None), cogu_medula_sense=([P, P], None),
                cogu_medula_receive=([P, B, I, ctypes.c_double], None), cogu_medula_act=([P, F], None),
                cogu_medula_telemetry=([P, B], I), cogu_medula_state=([P], I), cogu_medula_mode=([P], I),
                cogu_medula_info=([P, D], None), cogu_medula_stats=([P, D], None),
                cogu_policy_eval=([I, F, F], None), cogu_imu_new=([ctypes.c_float], P), cogu_imu_free=([P], None),
                cogu_imu_update=([P, F, F, ctypes.c_float, F], None))
    for name, (args, res) in sigs.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, res
    nu = lib.cogu_nu()

    class Sensors(ctypes.Structure):
        _fields_ = [("gyro", ctypes.c_float * 3), ("gravity", ctypes.c_float * 3), ("q", ctypes.c_float * nu),
                    ("qd", ctypes.c_float * nu), ("volts", ctypes.c_float), ("truth", ctypes.c_bool),
                    ("height", ctypes.c_float), ("lin_vel", ctypes.c_float * 3)]
    assert ctypes.sizeof(Sensors) == lib.cogu_sensors_size(), "Sensors layout differs from body.h"
    lib.Sensors, lib.nu = Sensors, nu
    _lib = lib
    return lib


def fptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def policy(which: str, obs: np.ndarray) -> np.ndarray:
    """The firmware's MLP alone: 'walk' or 'getup'."""
    lib = library()
    obs = np.ascontiguousarray(obs, np.float32)
    out = np.zeros(lib.nu, np.float32)
    lib.cogu_policy_eval(0 if which == "walk" else 1, fptr(obs), fptr(out))
    return out


class GravityFilter:
    """The firmware's IMU filter (imu.h): gravity from the gyro and the accelerometer."""

    def __init__(self, gain: float = 2.0):
        self.lib = library()
        self.h = self.lib.cogu_imu_new(gain)
        self.out = np.zeros(3, np.float32)

    def update(self, gyro, accel, dt: float) -> np.ndarray:
        g, a = np.ascontiguousarray(gyro, np.float32), np.ascontiguousarray(accel, np.float32)
        self.lib.cogu_imu_update(self.h, fptr(g), fptr(a), dt, fptr(self.out))
        return self.out.astype(float)

    def __del__(self):
        if getattr(self, "h", None):
            self.lib.cogu_imu_free(self.h)


class BodyReader:
    """The sensor readings the firmware gets, from a MjData: the IMU gyro, the gravity (perfect, or from the
    firmware's filter on the simulated gyro + accelerometer), the joints, the battery; `truth` adds the pelvis
    height and speed the robot cannot measure."""

    def __init__(self, sim, truth: bool = True, imu: str = "perfect"):
        self.sim, self.truth, self.imu = sim, truth, imu
        m = sim.m
        self.acc_adr = m.sensor_adr[m.sensor("accelerometer").id]
        self.filter = GravityFilter() if imu == "filter" else None

    def read(self, volts: float):
        lib, s, d = library(), self.sim, self.sim.d
        out = lib.Sensors()
        gyro = d.sensordata[s.gyro_adr:s.gyro_adr + 3]
        if self.filter is not None:
            g = self.filter.update(gyro, d.sensordata[self.acc_adr:self.acc_adr + 3], s.dt)
        else:
            g = d.site_xmat[s.imu].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        out.gyro[:], out.gravity[:] = gyro.tolist(), g.tolist()
        out.q[:], out.qd[:] = d.qpos[s.qj].tolist(), d.qvel[s.vj].tolist()
        out.volts, out.truth = volts, self.truth
        out.height, out.lin_vel[:] = float(d.qpos[2]), d.qvel[:3].tolist()
        return out


class MedulaC:
    """medula.Medula's interface on the firmware: `receive(packet, t)`, `step()` (50 Hz), `telemetry()`, and the
    same read-outs (`falls`, `stats`, `cmd`, `t`, `state_name`, `battery`). `sim` is a play.Sim2Sim whose MjData is
    the body (its own JAX policy is not used); `getup` and `pose` are accepted for the same signature and ignored."""

    def __init__(self, sim, getup=None, pose=None, battery: Battery | None = None, truth: bool = True,
                 imu: str = "perfect"):
        self.lib = library()
        self.sim, self.dt = sim, sim.dt
        self.battery = battery or Battery()
        self.reader = BodyReader(sim, truth, imu)
        self.h = self.lib.cogu_medula_new()
        self.target = np.zeros(self.lib.nu, np.float32)
        self.history = [sim.d.ctrl.copy() for _ in range(sim.delay + 1)]   # the actuator delay, as in play.py
        self.lib.cogu_medula_begin(self.h, ctypes.byref(self.reader.read(self.battery.volts)))
        self._info = np.zeros(7)
        self._stats = np.zeros(5)

    def __del__(self):
        if getattr(self, "h", None):
            self.lib.cogu_medula_free(self.h)

    def receive(self, data: bytes, t: float):
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        self.lib.cogu_medula_receive(self.h, buf, len(data), t)

    def act(self) -> np.ndarray:
        self.lib.cogu_medula_act(self.h, fptr(self.target))
        return self.target.astype(float)

    def sense(self):
        self.lib.cogu_medula_sense(self.h, ctypes.byref(self.reader.read(self.battery.volts)))

    def step(self):
        s = self.sim
        self.history = [self.act()] + self.history[:-1]
        s.d.ctrl[:] = self.history[s.delay]
        for _ in range(s.n_substeps):
            mujoco.mj_step(s.m, s.d)
        self.sense()
        name, mode = self.state_name, self.mode
        self.battery.step(self.dt, name if mode == "pose" and name in ("sitting", "lying") else mode)

    def telemetry(self) -> bytes:
        out = (ctypes.c_uint8 * proto.TEL_SIZE)()
        n = self.lib.cogu_medula_telemetry(self.h, out)
        return bytes(out[:n])

    def _read(self):
        self.lib.cogu_medula_info(self.h, self._info.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return self._info

    @property
    def state_name(self) -> str:
        return proto.STATES[self.lib.cogu_medula_state(self.h)]

    @property
    def mode(self) -> str:
        return MODES[self.lib.cogu_medula_mode(self.h)]

    @property
    def cmd(self) -> np.ndarray:
        return self._read()[:3].copy()

    @property
    def t(self) -> float:
        return float(self._read()[3])

    @property
    def falls(self) -> int:
        return int(self._read()[4])

    @property
    def cutoff(self) -> bool:
        return bool(self._read()[6])

    @property
    def stats(self) -> dict:
        self.lib.cogu_medula_stats(self.h, self._stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        c, r, s, w, g = self._stats
        return dict(commands=int(c), rejected=int(r), stale=int(s), watchdog=int(w), longest_gap=float(g))


# -- the checks ----------------------------------------------------------------------------------------------------

def check_policies() -> bool:
    """The C++ MLPs against JAX on observations from a real run (the walker walking, the get-up getting up)."""
    import jax
    import jax.numpy as jp

    import pet as petlib
    import play
    from rollout import latest_checkpoint, load_policy

    sim = play.Sim2Sim(petlib.APARTMENT_XML, latest_checkpoint(petlib.WALK), seed=0)
    getup = play.GetupPolicy(sim, latest_checkpoint(petlib.GETUP))
    walk_fn, getup_fn = sim.policy, getup.policy
    obs_w, obs_g = [], []
    sim.command[:] = [0.2, 0.0, 0.3]
    for _ in range(150):
        obs_w.append(sim.obs.copy())
        sim.step()
    sim.d.qpos[3:7] = [math.cos(0.8), 0.0, math.sin(0.8), 0.0]          # tipped over backwards: get up
    mujoco.mj_forward(sim.m, sim.d)
    getup.start()
    for _ in range(250):
        obs_g.append(getup._obs())
        getup.step()
    ok = True
    key = jax.random.PRNGKey(0)
    for name, fn, obs, priv in (("walk", walk_fn, obs_w, 100), ("getup", getup_fn, obs_g, 65)):
        err = 0.0
        for o in obs:
            ref = np.asarray(fn({"state": jp.asarray(o), "privileged_state": jp.zeros(priv)}, key)[0])
            err = max(err, float(np.max(np.abs(policy(name, o) - ref))))
        good = err < 1e-4
        ok &= good
        print(f"  {name:5s} policy, C++ vs JAX on {len(obs)} observations of a run: max |Δaction| {err:.1e}"
              f" {'ok' if good else 'TOO FAR'}")
    return ok


def check_shadow() -> bool:
    """The firmware in the shadow of the Python medula: same readings, same packets, every step."""
    import pet as petlib
    import play
    from medula import Medula
    from rollout import latest_checkpoint

    sim = play.Sim2Sim(petlib.APARTMENT_XML, latest_checkpoint(petlib.WALK), seed=0)
    py = Medula(sim, play.GetupPolicy(sim, latest_checkpoint(petlib.GETUP)), play.PosePolicy(sim, None))
    c = MedulaC(sim, battery=py.battery)        # the same battery object: the same volts
    # (t from, vx, vy, wz, posture, stop): what the phone asks, 10 times a second; push = a shove at that time
    plan = [(0.0, 0.25, 0.0, 0.0, proto.STAND, False), (4.0, 0.15, 0.0, 0.8, proto.STAND, False),
            (7.0, 0.0, 0.0, 0.0, proto.STAND, True), (8.0, 0.0, 0.0, 0.0, proto.SIT, False),
            (15.0, 0.0, 0.0, 0.0, proto.LIE, False), (25.0, 0.0, 0.0, 0.0, proto.STAND, False),
            (40.0, 0.2, 0.05, 0.0, proto.STAND, False), (43.0, 0.0, 0.0, 0.0, proto.STAND, False),
            (60.0, 0.0, 0.0, 0.0, proto.STAND, False)]
    push_at, seconds = 44.0, 60.0
    seq, pid, last_posture = 0, 0, proto.STAND
    worst, worst_t, state_bad, tel_bad, steps = 0.0, 0.0, 0, 0, 0
    states = []
    while py.t < seconds - 1e-9:
        t = py.t
        row = [p for p in plan if p[0] <= t + 1e-9][-1]
        if row[4] != last_posture:
            pid, last_posture = (pid + 1) & 0xFF, row[4]
        if abs(t * 10 - round(t * 10)) < 1e-6:
            data = proto.Command(seq, row[1], row[2], row[3], row[4], pid, row[5]).encode()
            seq = (seq + 1) & 0xFF
            py.receive(data, t)
            c.receive(data, t)
        if abs(t - push_at) < 1e-6:
            sim.d.qvel[0] += 1.2                 # a hard shove forwards: it falls and gets up
        target = c.act()
        py.step()                                # the Python medula drives the body
        err = float(np.max(np.abs(target - sim.d.ctrl)))
        if err > worst:
            worst, worst_t = err, t
        c.sense()
        steps += 1
        state_bad += py.state_name != c.state_name
        a, b = proto.Telemetry.decode(py.telemetry()), proto.Telemetry.decode(c.telemetry())
        tel_bad += (a.state, a.flags, a.ack, a.posture_id, a.falls, round(a.vx, 3), round(a.wz, 3)) != \
            (b.state, b.flags, b.ack, b.posture_id, b.falls, round(b.vx, 3), round(b.wz, 3))
        tilt_err = max(abs(a.roll - b.roll), abs(a.pitch - b.pitch))
        if not states or states[-1] != py.state_name:
            states.append(py.state_name)
    good = worst < 2e-3 and state_bad == 0 and tel_bad == 0 and py.falls == c.falls >= 1
    print(f"  shadow, {steps} steps ({seconds:.0f} s): {' -> '.join(states)}")
    print(f"    servo targets: max |C++ - Python| {worst:.1e} rad (at {worst_t:.1f} s); state differs in {state_bad} "
          f"steps, telemetry in {tel_bad}; roll / pitch off by {tilt_err:.2f} deg at the end; falls {py.falls} / "
          f"{c.falls} {'ok' if good else 'NOT THE SAME'}")
    return good


def check_scenarios(truth: bool = True, imu: str = "perfect") -> bool:
    import medula

    label = "simulator truth" if truth else "robot sensing" + (", IMU filter" if imu == "filter" else "")
    print(f"  medula.py's scenarios on the firmware ({label}):")
    return medula.scenarios(lambda sim, getup, pose, battery: MedulaC(sim, battery=battery, truth=truth, imu=imu)) == 0


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if "--test" not in args:
        print(__doc__)
        return 0
    r = subprocess.run(["make", "-s", "-C", str(FIRMWARE), "test"], capture_output=True, text=True)
    print(r.stdout.rstrip())
    ok = r.returncode == 0
    ok &= check_policies()
    ok &= check_shadow()
    ok &= check_scenarios()
    ok &= check_scenarios(truth=False)
    ok &= check_scenarios(truth=False, imu="filter")
    print("medula firmware: ok" if ok else "medula firmware: NOT AS EXPECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
