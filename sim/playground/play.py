"""Drive a trained Cogu policy in the interactive MuJoCo viewer with the keyboard (sim2sim).

    uv run mjpython sim/playground/play.py                             # latest checkpoint, viewer (macOS: mjpython)
    uv run mjpython sim/playground/play.py --checkpoint sim/playground/checkpoints/cogu_walk_v2
    uv run python sim/playground/play.py --headless --vx 0.15 --seconds 8   # no window: prints the result
    uv run mjpython sim/playground/play.py --follow build/checkpoints/cogu_walk_v2  # watch a training run learn
    uv run mjpython sim/playground/play.py --follow build/checkpoints/cogu_walk_v2 \
        --checkpoint sim/playground/checkpoints/cogu_walk_v1      # ... starting with v1 until the first one lands

`--follow DIR` keeps the viewer open while train.py is running: every time the trainer saves a new
checkpoint in DIR (once per evaluation), the policy is swapped in and the robot restarts from the
rest pose. The overlay shows which checkpoint is driving.

Keys (in the viewer window):
    ↑ / ↓        forward speed  +/− 0.05 m/s
    ← / →        turn left / right  +/− 0.2 rad/s
    PgUp / PgDn  sideways left / right  +/− 0.05 m/s
    End          stop (zero command)
    Home         reset the robot to the rest pose
    , / .        previous / next checkpoint of the same training run (the policy at each evaluation)
    double-click a body to select it, then Ctrl + right-drag: push it (Ctrl + left-drag: twist it)

The env config each checkpoint was trained with (action filter, scales) is read from the
`env_config.json` train.py saves next to it.

This is a *sim2sim* check: the policy was trained on MJX (JAX), and here it runs on plain MuJoCo
(the C engine, CPU), the same engine the real-robot tooling will use. The observation, action
scaling, gait phase and their exact timing are rebuilt from `cogu_env.py` without JAX state, so if
the robot walks here, nothing in the policy depends on MJX-specific behaviour. No observation noise,
no pushes and no domain randomisation unless asked (`--delay`, `--noise`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax  # noqa: E402
import jax_compat  # noqa: E402,F401  (must precede brax)
import jax.numpy as jp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import cogu_env  # noqa: E402
import cogu_getup  # noqa: E402
import getup_starts  # noqa: E402
from rollout import latest_checkpoint, load_policy  # noqa: E402

REPO_ROOT = HERE.parents[1]

# GLFW key codes (the passive viewer forwards raw GLFW keys to key_callback)
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265
KEY_PAGE_UP, KEY_PAGE_DOWN, KEY_HOME, KEY_END = 266, 267, 268, 269
KEY_COMMA, KEY_PERIOD = 44, 46


class Sim2Sim:
    """Runs the policy on plain MuJoCo, reproducing CoguJoystick's observation and action exactly."""

    def __init__(self, xml: Path, ckpt: Path, gait_freq: float = 1.75, delay_steps: int = 0,
                 noise: float = 0.0, seed: int = 0):
        cfg = cogu_env.config_for_checkpoint(ckpt)   # same dynamics (action filter, scales) as in training
        self.cfg = cfg
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.m.opt.timestep = cfg.sim_dt
        self.d = mujoco.MjData(self.m)
        self.dt = cfg.ctrl_dt
        self.n_substeps = int(round(cfg.ctrl_dt / cfg.sim_dt))
        key = self.m.keyframe("rest")
        self.init_q = np.array(key.qpos)
        self.default_pose = np.array(key.qpos[7:])
        self.lowers, self.uppers = self.m.jnt_range[1:].T
        self.imu = self.m.site("imu").id
        self.gyro_adr = self.m.sensor_adr[self.m.sensor("gyro").id]
        self.phase_dt = 2 * np.pi * self.dt * gait_freq
        self.delay = delay_steps
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.key = jax.random.PRNGKey(seed)
        self.load(ckpt)
        self.priv = jp.zeros(100)      # the actor reads only "state"; the critic input is not used here
        self.command = np.zeros(3)
        self.reset()

    def load(self, ckpt: Path):
        self.ckpt = ckpt
        self.cfg = cogu_env.config_for_checkpoint(ckpt)
        self.policy = jax.jit(load_policy(ckpt))

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.init_q
        self.d.ctrl[:] = self.default_pose
        mujoco.mj_forward(self.m, self.d)
        self.last_act = np.zeros(self.m.nu)
        self.filtered = np.zeros(self.m.nu)
        self.phase = np.array([0.0, np.pi])
        self.recover_time = 0.0
        self.recover_cooldown = 0.0
        self.history = [self.default_pose.copy() for _ in range(self.delay + 1)]
        self.obs = self._obs()

    def _noisy(self, x, scale):
        return x + (2 * self.rng.random(x.shape) - 1) * self.noise * scale if self.noise else x

    def _obs(self) -> np.ndarray:
        """Same order and units as CoguJoystick._get_obs()['state'] (43 floats)."""
        s = self.cfg.noise_config.scales
        gyro = self.d.sensordata[self.gyro_adr:self.gyro_adr + 3].copy()
        gravity = self.d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        q = self.d.qpos[7:].copy()
        qd = self.d.qvel[6:].copy()
        phase = np.concatenate([np.cos(self.phase), np.sin(self.phase)])
        return np.concatenate([self._noisy(gyro, s.gyro), self._noisy(gravity, s.gravity), self.command,
                               self._noisy(q, s.joint_pos) - self.default_pose, self._noisy(qd, s.joint_vel),
                               self.last_act, phase]).astype(np.float32)

    def step(self):
        """One 50 Hz control step. Ordering follows CoguJoystick.step(): the observation built after
        the physics uses the *previous* action and the *previous* phase, then both are advanced."""
        self.key, k = jax.random.split(self.key)
        action, _ = self.policy({"state": jp.asarray(self.obs), "privileged_state": self.priv}, k)
        action = np.asarray(action)
        a = self.cfg.action_filter_alpha
        self.filtered = a * action + (1.0 - a) * self.filtered
        target = np.clip(self.default_pose + self.filtered * self.cfg.action_scale, self.lowers, self.uppers)
        self.history = [target] + self.history[:-1]
        self.d.ctrl[:] = self.history[self.delay]
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.m, self.d)
        self.obs = self._obs()
        self.last_act = action
        self.phase = np.fmod(self.phase + self.phase_dt + np.pi, 2 * np.pi) - np.pi
        # a push the IMU feels runs the gait clock for a while (v7), exactly as cogu_env.step does
        rc = self.cfg.recover
        gyro = self.d.sensordata[self.gyro_adr:self.gyro_adr + 2]
        up_z = self.d.site_xmat[self.imu].reshape(3, 3)[2, 2]
        disturbed = rc.enable and (np.linalg.norm(gyro) > rc.gyro or up_z < np.cos(rc.tilt))
        rt, cd = self.recover_time, self.recover_cooldown
        if rc.get("latch", False):                       # v10: one window per push, then the detector rests
            start = disturbed and rt == 0.0 and cd == 0.0
            self.recover_cooldown = rc.cooldown if 0.0 < rt <= self.dt else max(cd - self.dt, 0.0)
        else:
            start = disturbed
        self.recover_time = rc.hold if start else max(rt - self.dt, 0.0)
        stepping = np.linalg.norm(self.command) > 0.05 or self.recover_time > 0.0
        if self.cfg.get("stand_phase_reset", False) and not stepping:
            self.phase = np.array([0.0, np.pi])          # standing: hold the start phase, as in training

    def fallen(self) -> bool:
        up = self.d.site_xmat[self.imu].reshape(3, 3)[:, 2]
        return up[2] < 0.5 or self.d.qpos[2] < 0.10

    def forward_speed(self) -> float:
        """Pelvis speed along its own heading (m/s)."""
        heading = self.d.xmat[1].reshape(3, 3)[:, 0]
        return float(np.dot(self.d.qvel[:3], heading))


    def handover(self):
        """Take over a robot that is already standing (from the get-up policy): keep the physics, restart the
        policy state so the first target is where the joints are now (no jump)."""
        q = self.d.qpos[7:].copy()
        self.last_act = np.zeros(self.m.nu)
        self.filtered = (q - self.default_pose) / self.cfg.action_scale
        self.phase = np.array([0.0, np.pi])
        self.recover_time = 0.0
        self.recover_cooldown = 0.0
        self.history = [q.copy() for _ in range(self.delay + 1)]
        self.obs = self._obs()


def getup_config(ckpt: Path):
    """The get-up env config a checkpoint was trained with; `full_range_actions` marks the first design
    (actions spanning each joint's range), from before `action_scale` existed."""
    import json
    cfg = cogu_getup.default_config()
    saved = {}
    for d in (ckpt, ckpt.parent):
        f = d / cogu_env.ENV_CONFIG_FILE
        if f.exists():
            saved = json.loads(f.read_text())
            break
    cfg.update(saved)
    cfg.full_range_actions = "action_scale" not in saved
    if "reference" not in saved:                     # v1–v2: no scripted catch under the policy
        cfg.reference.enable = False
    elif "throw" not in saved["reference"]:          # v3: catch only, no scripted throw
        cfg.reference.throw = False
    if "reference" in saved and "roll" not in saved["reference"]:   # v1–v4: no relax + roll (v5)
        cfg.reference.roll = False
    if "reference" in saved and "scripted_residual" not in saved["reference"]:   # v5–v6: corrected the scripts too
        cfg.reference.scripted_residual = True
    return cfg


class GetupPolicy:
    """The get-up policy (cogu_getup) driving the same MjData as a Sim2Sim walker: sim2sim of CoguGetup."""

    def __init__(self, sim: Sim2Sim, ckpt: Path, seed: int = 1):
        self.sim, self.ckpt = sim, ckpt
        self.cfg = getup_config(ckpt)
        self.policy = jax.jit(load_policy(ckpt))
        self.key = jax.random.PRNGKey(seed)
        # v1 checkpoints of the first design mapped actions to the whole joint range; from then on: rest + offset
        full = self.cfg.full_range_actions
        self.centre = (sim.lowers + sim.uppers) / 2 if full else sim.default_pose
        self.half = (sim.uppers - sim.lowers) / 2 if full else self.cfg.action_scale
        ref = self.cfg.reference
        if ref.enable:                                   # v3: the policy corrects a scripted catch (cogu_getup)
            self.ref_tilts = np.array(ref.tilts, float)
            legs = np.radians(np.array(ref.legs, float))
            self.ref_legs = np.column_stack([np.zeros((len(legs), 2)), legs])
        self.throw = ref.enable and ref.get("throw", False)
        self.roll = ref.enable and ref.get("roll", False)
        if self.roll:                                    # v5: relax, roll onto the back, catch (cogu_getup)
            specs = {}
            for name in ("front", "side"):
                t = ref[f"roll_{name}"]
                path = getup_starts.ROLLS[name]
                specs[name] = (t.to_dict() if t.durations else json.loads(path.read_text()) if path.exists()
                               else dict(targets=[], durations=[], hold=0.0))
            self.roll_keys, self.roll_ends = getup_starts.roll_table(sim.default_pose, specs)
        if self.throw:                                   # v4: and a scripted throw from lying (cogu_getup)
            self.throws = []
            for side in ("back", "front"):
                t = ref[f"throw_{side}"]
                if not t.durations:
                    path = getup_starts.THROWS[side]
                    t = json.loads(path.read_text()) if path.exists() else dict(targets=[], durations=[], trigger=0.0)
                self.throws.append(dict(keys=np.asarray(t["targets"], float), ends=np.cumsum(t["durations"]),
                                        trigger=float(t["trigger"])) if t["durations"] else None)
        self.calm = 0.0

    def start(self, throw_t: float = -1.0, throw_side: int = 0):
        """`throw_t` >= 0: the reference is that far into throw `throw_side` (a getup_starts `throw` state)."""
        q = self.sim.d.qpos[7:].copy()
        self.ref = q.copy()
        if self.roll and throw_t >= 0.0:                 # the table counts time into the roll; the env relax + roll
            throw_t += self.cfg.reference.relax_time
        self.throw_t, self.throw_side, self.lying_t, self.rolled = throw_t, throw_side, 0.0, False
        held = np.clip((q - self.centre) / self.half, -1.0, 1.0)
        self.filtered = np.zeros_like(q) if self.cfg.reference.enable else held   # the target starts where q is
        self.last_act = np.zeros(self.sim.m.nu)
        self.history = [q.copy() for _ in range(self.sim.delay + 1)]
        self.calm = 0.0

    def _obs(self) -> np.ndarray:
        s, d = self.sim, self.sim.d
        gyro = d.sensordata[s.gyro_adr:s.gyro_adr + 3].copy()
        gravity = d.site_xmat[s.imu].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        ref = [self.ref - s.default_pose] if self.cfg.reference.enable else []
        if self.throw:
            throwing = self.throw_t >= 0.0
            ref.append([float(throwing), self.throw_t / self.throws[self.throw_side]["ends"][-1] if throwing else 0.0])
        elif self.roll:
            throwing = self.throw_t >= 0.0
            total = max(self.roll_ends[self.throw_side, -1] + self.cfg.reference.relax_time, 1e-3)
            ref.append([float(throwing), self.throw_t / total if throwing else 0.0])
        return np.concatenate([gyro, gravity, d.qpos[7:] - s.default_pose, d.qvel[6:], self.last_act, *ref]
                              ).astype(np.float32)

    def _reference_step(self):
        """cogu_getup.CoguGetup.reference_step: the scripted catch's next reference, from the IMU's tilt."""
        s, ref = self.sim, self.cfg.reference
        g = s.d.site_xmat[s.imu].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        tilt = np.degrees(np.arccos(np.clip(-g[2], -1.0, 1.0))) * (1.0 if g[0] <= 0.0 else -1.0)
        want = np.sign(tilt) * max(abs(tilt) - ref.lead, 0.0)
        leg = np.array([np.interp(want, self.ref_tilts, self.ref_legs[:, j]) for j in range(5)])
        step = ref.rate * s.dt
        catch = self.ref + np.clip(np.concatenate([leg, leg]) - self.ref, -step, step)
        if self.roll:
            self._roll_step(-g[0], -g[1], tilt, catch, step)
            return
        if not self.throw:
            self.ref = catch
            return
        guess = 0 if tilt >= 0.0 else 1
        throws = self.throws
        lying = (throws[guess] is not None and self.throw_t < 0.0
                 and abs(tilt) > max(ref.lie_tilt, throws[guess]["trigger"] + 5.0))
        self.lying_t = self.lying_t + s.dt if lying else 0.0
        if self.lying_t >= ref.lie_time:
            self.throw_t, self.throw_side, self.lying_t = 0.0, guess, 0.0
        if self.throw_t >= 0.0:
            t = throws[self.throw_side]
            if abs(tilt) <= t["trigger"] or self.throw_t >= t["ends"][-1]:
                self.throw_t = -1.0
            else:
                self.ref = t["keys"][min(int(np.searchsorted(t["ends"], self.throw_t, side="right")), len(t["ends"]) - 1)]
                self.throw_t += s.dt
                return
        self.ref = catch

    def _roll_step(self, xz: float, yz: float, tilt: float, catch: np.ndarray, step: float):
        """cogu_getup.CoguGetup._roll_step: relax, roll onto the back (front / side / other side), catch."""
        ref, dt = self.cfg.reference, self.sim.dt
        on_back = xz > max(abs(yz), 0.5)
        lay = 0 if on_back else 1 if -xz > max(abs(yz), 0.5) else 2 if yz > 0.5 else 3 if yz < -0.5 else 0
        lying = abs(tilt) > ref.lie_tilt and self.throw_t < 0.0 and (not on_back or not self.rolled)
        self.lying_t = self.lying_t + dt if lying else 0.0
        if self.lying_t >= ref.lie_time:
            self.throw_t, self.lying_t = 0.0, 0.0
        tt = self.throw_t
        relaxing = 0.0 <= tt < ref.relax_time
        if relaxing:
            self.throw_side = lay
        ends = self.roll_ends[self.throw_side]
        tr = tt - ref.relax_time
        rolling = 0.0 <= tr < ends[-1]
        if relaxing:
            self.ref = self.ref + np.clip(self.sim.default_pose - self.ref, -step, step)
        elif rolling:
            self.ref = self.roll_keys[self.throw_side, min(int(np.searchsorted(ends, tr, side="right")), len(ends) - 1)]
        else:
            self.ref = catch
        if tt >= 0.0 and not (relaxing or rolling):
            self.rolled = True
        self.throw_t = tt + dt if (relaxing or rolling) else -1.0

    def step(self):
        s = self.sim
        self.key, k = jax.random.split(self.key)
        obs = self._obs()
        priv = jp.zeros(obs.size + 17)                 # the critic's input: unused by the policy
        action, _ = self.policy({"state": jp.asarray(obs), "privileged_state": priv}, k)
        action = np.clip(np.asarray(action), -1.0, 1.0)   # as CoguGetup.step (the policy may not be squashed)
        a = self.cfg.action_filter_alpha
        self.filtered = a * action + (1.0 - a) * self.filtered
        if self.cfg.reference.enable:
            self._reference_step()
            if self.roll and not self.cfg.reference.scripted_residual and self.throw_t >= 0.0:
                self.filtered[:] = 0.0                     # v7: no correction while relaxing / rolling
            target = np.clip(self.ref + self.filtered * self.cfg.reference.residual_scale, s.lowers, s.uppers)
        else:
            target = np.clip(self.centre + self.filtered * self.half, s.lowers, s.uppers)
        self.history = [target] + self.history[:-1]
        s.d.ctrl[:] = self.history[s.delay]
        for _ in range(s.n_substeps):
            mujoco.mj_step(s.m, s.d)
        self.last_act = action
        # ready to walk: upright, at height, near `rest`, slow — for 0.5 s in a row
        up = s.d.site_xmat[s.imu].reshape(3, 3)[2, 2]
        rested = np.sum(np.square(s.d.qpos[7:] - s.default_pose)) < 0.05
        slow = np.linalg.norm(s.d.qvel[:3]) < 0.1 and np.linalg.norm(s.d.qvel[6:]) < 1.5
        self.calm = self.calm + s.dt if (up > 0.95 and s.d.qpos[2] > 0.18 and rested and slow) else 0.0

    def ready(self) -> bool:
        return self.calm >= 0.5


def run_checkpoints(ckpt: Path) -> list[Path]:
    """All step checkpoints of the training run `ckpt` belongs to (empty for a published folder)."""
    if not ckpt.name.isdigit():
        return []
    return sorted(p for p in ckpt.parent.iterdir()
                  if p.is_dir() and p.name.isdigit() and (p / "ppo_network_config.json").exists())


def checkpoint_label(ckpt: Path) -> str:
    """'cogu_walk_v2 · 4,505,600 steps' for a training step dir, the folder name for a published one."""
    if ckpt.name.isdigit():
        return f"{ckpt.parent.name} · {int(ckpt.name):,} steps"
    return ckpt.name


def run_headless(sim: Sim2Sim, seconds: float) -> int:
    n = int(round(seconds / sim.dt))
    x0 = sim.d.qpos[:2].copy()
    for i in range(n):
        sim.step()
        if sim.fallen():
            print(f"FELL at {(i + 1) * sim.dt:.2f} s")
            return 1
    dx, dy = sim.d.qpos[:2] - x0
    print(f"command vx={sim.command[0]} vy={sim.command[1]} wz={sim.command[2]}  ran {seconds:.1f} s  "
          f"moved dx={dx:+.3f} m dy={dy:+.3f} m  mean vx={dx / seconds:+.3f} m/s  stayed up")
    return 0


def run_viewer(sim: Sim2Sim, speed: float, follow: Path | None = None, getup: GetupPolicy | None = None) -> int:
    import mujoco.viewer

    lims = sim.cfg
    pending = {"reset": False, "browse": 0}

    def on_key(key: int):
        c = sim.command
        if key == KEY_UP:
            c[0] = min(c[0] + 0.05, lims.lin_vel_x[1])
        elif key == KEY_DOWN:
            c[0] = max(c[0] - 0.05, lims.lin_vel_x[0])
        elif key == KEY_LEFT:
            c[2] = min(c[2] + 0.2, lims.ang_vel_yaw[1])
        elif key == KEY_RIGHT:
            c[2] = max(c[2] - 0.2, lims.ang_vel_yaw[0])
        elif key == KEY_PAGE_UP:
            c[1] = min(c[1] + 0.05, lims.lin_vel_y[1])
        elif key == KEY_PAGE_DOWN:
            c[1] = max(c[1] - 0.05, lims.lin_vel_y[0])
        elif key == KEY_END:
            c[:] = 0.0
        elif key == KEY_HOME:
            pending["reset"] = True
        elif key == KEY_COMMA:
            pending["browse"] = -1
        elif key == KEY_PERIOD:
            pending["browse"] = +1
        c[:] = np.round(c, 3)

    print(__doc__.split("Keys (in the viewer window):")[1].split("This is")[0])
    try:
        handle = mujoco.viewer.launch_passive(sim.m, sim.d, key_callback=on_key)
    except RuntimeError as e:                       # macOS: the passive viewer needs mjpython
        print(f"{e}\n\nOn macOS run the viewer with mjpython:\n"
              "  uv run mjpython sim/playground/play.py ...\n"
              "If mjpython fails with 'Library not loaded: @rpath/libpython3.12.dylib' (uv-managed Python), "
              "link the library next to the venv once:\n"
              "  ln -sf \"$(uv run python -c 'import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))')"
              "/libpython3.12.dylib\" .venv/libpython3.12.dylib", file=sys.stderr)
        return 2
    with handle as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = sim.m.body("pelvis").id
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.0, 135.0, -15.0
        falls = 0
        mode = "walking"
        last_poll = time.perf_counter()
        last_newest = sim.ckpt                       # follow only jumps when a NEW checkpoint appears
        while v.is_running():
            t0 = time.perf_counter()
            if follow is not None and t0 - last_poll > 5.0:          # a new checkpoint from train.py?
                last_poll = t0
                try:
                    newest = latest_checkpoint(follow)
                except FileNotFoundError:
                    newest = sim.ckpt
                if newest != last_newest and (newest / "ppo_network_config.json").exists():
                    last_newest = newest
                    time.sleep(2.0)                                   # let orbax finish writing
                    with v.lock():
                        sim.load(newest)
                        sim.reset()
                        falls = 0
                    print(f"now driving: {checkpoint_label(newest)}")
            if pending["browse"]:                                     # , / . : walk through the run
                ckpts = run_checkpoints(sim.ckpt)
                if sim.ckpt in ckpts:
                    i = min(max(ckpts.index(sim.ckpt) + pending["browse"], 0), len(ckpts) - 1)
                    if ckpts[i] != sim.ckpt:
                        with v.lock():
                            sim.load(ckpts[i])
                            sim.reset()
                            falls = 0
                        print(f"now driving: {checkpoint_label(ckpts[i])}")
                pending["browse"] = 0
            with v.lock():
                if pending["reset"]:
                    sim.reset()
                    mode = "walking"
                    pending["reset"] = False
                # mouse pushes: the passive viewer only records them; the physics loop applies them
                sim.d.xfrc_applied[:] = 0.0
                mujoco.mjv_applyPerturbForce(sim.m, sim.d, v.perturb)
                if mode == "getting up":
                    getup.step()
                    if getup.ready():                   # standing in `rest`: the walker takes over
                        sim.handover()
                        mode = "walking"
                else:
                    sim.step()
                    if sim.fallen():
                        falls += 1
                        if getup is not None:           # lying on the floor: the get-up policy takes over
                            getup.start()
                            mode = "getting up"
                        else:
                            sim.reset()
            vx, vy, wz = sim.command
            v.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                         "command\nspeed\nfalls\npolicy\nmode",
                         f"vx {vx:+.2f}  vy {vy:+.2f}  wz {wz:+.1f}\n{sim.forward_speed():+.2f} m/s\n{falls}\n"
                         f"{checkpoint_label(sim.ckpt)}\n{mode}"))
            v.sync()
            sleep = sim.dt / speed - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint dir (default: newest of build/checkpoints/cogu_walk_v2, sim/playground/checkpoints)")
    ap.add_argument("--xml", default=None,
                    help="MJCF (default: the MJX walking model, or the get-up one with --getup; "
                         "try build/mjcf/cogu_lower_body.xml for full mesh contacts)")
    ap.add_argument("--getup", default=None, metavar="CKPT",
                    help="get-up policy: after a fall it takes over, then hands back to the walker")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--gait-freq", type=float, default=1.75, help="Hz (training samples 1.5–2.0)")
    ap.add_argument("--delay", type=int, default=0, help="actuator latency in 20 ms steps (training: 0–2)")
    ap.add_argument("--noise", type=float, default=0.0, help="observation noise level (training: 1.0)")
    ap.add_argument("--speed", type=float, default=1.0, help="viewer playback speed")
    ap.add_argument("--latch", action="store_true",
                    help="v7–v9 checkpoints: use the v10 push detector (one 1.5 s window per push, then 1 s of rest)")
    ap.add_argument("--follow", default=None, metavar="DIR",
                    help="watch a training run: reload the newest checkpoint in DIR as it appears")
    return ap.parse_args(argv)


def default_checkpoint() -> Path:
    for c in (REPO_ROOT / "build" / "checkpoints" / "cogu_walk_v2", HERE / "checkpoints" / "cogu_walk_v2"):
        if c.exists() and any(p.name.isdigit() for p in c.iterdir()):
            return c
    raise FileNotFoundError("no cogu_walk_v2 checkpoint yet: pass --checkpoint, or train one with train.py")


def main(argv=None) -> int:
    args = parse_args(argv)
    follow = Path(args.follow).resolve() if args.follow else None
    if follow is not None:
        has_ckpt = lambda: follow.exists() and any(p.name.isdigit() for p in follow.iterdir())
        if has_ckpt():
            ckpt = latest_checkpoint(follow)
        elif args.checkpoint:                       # drive with another policy until the first one lands
            ckpt = latest_checkpoint(Path(args.checkpoint).resolve())
        else:
            while not has_ckpt():
                print(f"waiting for the first checkpoint in {follow} …")
                time.sleep(10)
            ckpt = latest_checkpoint(follow)
    else:
        ckpt = latest_checkpoint(Path(args.checkpoint).resolve() if args.checkpoint else default_checkpoint())
    xml = args.xml or str(cogu_getup.GETUP_XML if args.getup else cogu_env.DEFAULT_XML)
    print(f"checkpoint: {ckpt}\nmodel:      {xml}")
    sim = Sim2Sim(Path(xml), ckpt, gait_freq=args.gait_freq, delay_steps=args.delay, noise=args.noise)
    if args.latch:
        rc, new = sim.cfg.recover, cogu_env.default_config().recover
        rc.latch, rc.hold, rc.cooldown = True, new.hold, new.cooldown
    sim.command[:] = (args.vx, args.vy, args.wz)
    getup = GetupPolicy(sim, latest_checkpoint(Path(args.getup).resolve())) if args.getup else None
    if args.headless:
        return run_headless(sim, args.seconds)
    return run_viewer(sim, args.speed, follow, getup)


if __name__ == "__main__":
    sys.exit(main())
