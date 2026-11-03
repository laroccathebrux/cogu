"""Cogu sit / lie-down environment for MuJoCo Playground / MJX (2026-11-02, ADR-004).

A third policy, next to walking and getting up: on a command it goes from standing in `rest` to sitting and back,
and from standing or sitting to lying on its back. Getting up from lying is the get-up policy's job (v8 rises from
the back), so `lie` is a one-way command here.

Both moves are scripted references the policy corrects (as the get-up's catch), found in plain MuJoCo
(README, "Sentar e deitar"; `pose_eval.py script` plays them alone):
- **sit**: the pelvis leans back `sit.tilt` degrees while the hips fold to `sit.hip` and the knees to `sit.knee`,
  the ankles keeping the soles flat (ankle = -(hip + knee) - lean); over `sit.time` s with a cosine profile. The
  pelvis drops from 205 to ~165 mm and the robot leans back 35°, looking up as a sitting pet does; the centre of
  mass stays over the soles all the way and ends over the ankles (it holds with ~0.02 N·m). v1 sat leaning 20°:
  a forward push of 0.3 m/s tipped it over the toes.
- **lie**: the get-up's catch path played backwards: each step the pelvis tilt (from the IMU) picks the catch pose
  `lie.lead` degrees further towards lying, reached at `lie.rate` rad/s. Every pose on that path holds by itself, so
  the robot lies down on its back in ~2 s, never faster than 0.1 m/s.

The command (stand / sit / lie) changes every `switch_time` s (random); once `lie` starts it stays. Episodes start
standing in `rest` or sitting (settled), with random heading, pushes while standing or sitting.

Observation (51): IMU gyro and gravity, joint positions (from rest) and velocities, last action, the command
(one-hot, 3), the reference (from rest, 10), how far into sitting the reference is (0..1) and lying (0/1).
Action: correction of the reference, × `residual_scale` rad, low-pass filtered, with the walking env's latency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
from mujoco_playground._src import mjx_env

import cogu_env
import getup_starts

POSE_XML = cogu_env.REPO_ROOT / "build" / "mjcf" / "variants" / "hip_-120.xml"
MODES = ("stand", "sit", "lie")


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        env="pose",
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=600,            # 12 s
        action_repeat=1,
        action_filter_alpha=0.4,
        action_scale=1.0,              # (unused: the target is the reference + the correction)
        max_action_delay_steps=2,
        soft_joint_pos_limit_factor=0.95,
        residual_scale=0.5,            # rad: target = reference + action × this
        start_mix=(0.5, 0.5),          # standing, sitting
        switch_time=(1.5, 4.0),        # s between commands
        mode_probs=(0.4, 0.4, 0.2),    # stand, sit, lie at each switch
        settle_time=0.5,               # s holding the sit pose for the sitting starts
        # m/s, standing or sitting. Leaning 20° (v1) the reference alone held 0.25 m/s pushes every time and fell from
        # a quarter to 40 % of those up to 0.4-0.6; leaning 35° it holds 0.6 every time (pose_eval.py)
        push=config_dict.create(enable=True, interval=(2.0, 5.0), magnitude=(0.05, 0.5)),
        sit=config_dict.create(tilt=35.0, hip=-120.0, knee=90.0, time=2.0,      # deg, deg, deg, s
                               pelvis_tilt=35.0, height=0.165),                 # what it reaches (deg, m)
        lie=config_dict.create(lead=10.0, rate=1.0, end=105.0, tilt=112.0,      # deg, rad/s, deg, deg
                               tilts=[], legs=[]),                              # catch path, filled from the table
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(joint_pos=0.02, joint_vel=1.0, gravity=0.05, gyro=0.2),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                pose=1.0,              # joints close to the reference
                tilt=1.0,              # pelvis lean as the reference expects (sit / stand) or lying on the back (lie)
                height=1.0,            # pelvis height as expected (sit / stand)
                still=-0.5,            # pelvis linear speed² + 0.05 × angular speed²: gentle, no bouncing
                feet_slip=-0.5,
                action_rate=-0.01,
                torques=-0.002,
                dof_pos_limits=-1.0,
                residual=-0.5,         # squared correction (filtered action), kept small
            ),
            pose_sigma=0.3,
            tilt_sigma=10.0,           # deg
            height_sigma=0.02,         # m
            fall_tilt=40.0,            # deg off the expected lean, standing or sitting: fallen (episode ends)
        ),
        impl="jax",
        naconmax=None,
        njmax=None,
    )


def save_config(cfg: config_dict.ConfigDict, out_dir: Union[str, Path]) -> Path:
    path = Path(out_dir) / cogu_env.ENV_CONFIG_FILE
    path.write_text(cfg.to_json(indent=2))
    return path


def is_pose_checkpoint(ckpt: Union[str, Path]) -> bool:
    for d in (Path(ckpt), Path(ckpt).parent):
        f = d / cogu_env.ENV_CONFIG_FILE
        if f.exists():
            return json.loads(f.read_text()).get("env") == "pose"
    return False


def sit_legs(c, cfg) -> tuple:
    """Hip pitch, knee, ankle (rad) at `c` of the way into sitting (0 = rest, 1 = sitting), soles flat."""
    s = cfg.sit
    hip = jp.radians(-35.0 + (s.hip + 35.0) * c)
    knee = jp.radians(70.0 + (s.knee - 70.0) * c)
    lean = jp.radians(s.tilt * c)
    return hip, knee, -(hip + knee) - lean


def ease(s):
    return 0.5 - 0.5 * jp.cos(jp.pi * s)


class CoguPose(cogu_env.CoguJoystick):
    """Stand, sit and lie down on command."""

    def __init__(self, xml_path: Union[str, Path] = POSE_XML, config: config_dict.ConfigDict = default_config()):
        super().__init__(xml_path=xml_path, config=config)
        lie = self._config.lie
        if not lie.tilts:
            path = getup_starts.default_path(xml_path)
            if not path.exists():
                raise FileNotFoundError(f"{path} not found (the catch path). Make it with `uv run python "
                                        f"sim/playground/getup_starts.py --xml {xml_path} --mode roll`.")
            table = np.load(path)
            lie.tilts, lie.legs = table["catch_tilt"].tolist(), table["catch_q"].tolist()
        self._lie_tilts = jp.array(lie.tilts)
        self._lie_legs = jp.radians(jp.array(lie.legs))
        self._settle_steps = int(round(self._config.settle_time / self.sim_dt))

    def _legs(self, hip, knee, ankle) -> jax.Array:
        leg = jp.stack([jp.zeros_like(hip), jp.zeros_like(hip), hip, knee, ankle])
        return jp.concatenate([leg, leg])

    def sit_pose(self, c) -> jax.Array:
        return self._legs(*sit_legs(c, self._config))

    def tilt(self, data: mjx.Data) -> jax.Array:
        """Pelvis lean in the sagittal plane from the IMU, degrees: + back (face up), - forward."""
        g = self.get_gravity(data)
        return jp.degrees(jp.arccos(jp.clip(-g[2], -1.0, 1.0))) * jp.where(g[0] <= 0.0, 1.0, -1.0)

    def reference_step(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        cfg = self._config
        mode = info["mode"]
        lying = (info["lying"] > 0.5) | (mode == 2)
        sit = jp.clip(info["sit"] + jp.where(mode == 1, 1.0, -1.0) * self.dt / cfg.sit.time, 0.0, 1.0)
        info["sit"] = jp.where(lying, info["sit"], sit)
        info["lying"] = lying.astype(jp.float32)
        want = jp.minimum(self.tilt(data) + cfg.lie.lead, cfg.lie.end)
        leg = jax.vmap(lambda col: jp.interp(want, self._lie_tilts, col), in_axes=1)(self._lie_legs)
        step = cfg.lie.rate * self.dt
        lie = info["ref"] + jp.clip(self._legs(leg[0], leg[1], leg[2]) - info["ref"], -step, step)
        return jp.where(lying, lie, self.sit_pose(ease(info["sit"])))

    # ----------------------------------------------------------------- reset
    def reset(self, rng: jax.Array) -> mjx_env.State:
        cfg = self._config
        rng, k_start, k_yaw, k_xy, k_delay, k_switch, k_push = jax.random.split(rng, 7)
        sitting = jax.random.uniform(k_start) >= cfg.start_mix[0]
        yaw = jax.random.uniform(k_yaw, minval=-jp.pi, maxval=jp.pi)
        heading = mjx_math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
        lean = mjx_math.axis_angle_to_quat(jp.array([0.0, -1.0, 0.0]), jp.radians(cfg.sit.pelvis_tilt))
        sit_q = self.sit_pose(1.0)
        qpos = self._init_q.at[:2].set(jax.random.uniform(k_xy, (2,), minval=-0.3, maxval=0.3))
        qpos = jp.where(sitting, qpos.at[2].set(cfg.sit.height + 0.005).at[3:7].set(lean).at[7:].set(sit_q), qpos)
        qpos = qpos.at[3:7].set(mjx_math.quat_mul(heading, qpos[3:7]))
        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=jp.zeros(self.mjx_model.nv), ctrl=qpos[7:],
                                 impl=self.mjx_model.impl.value, naconmax=self._config.naconmax,
                                 njmax=self._config.njmax)
        data = mjx.forward(self.mjx_model, data)
        settled = mjx_env.step(self.mjx_model, data, qpos[7:], self._settle_steps)
        settled = settled.replace(qvel=jp.zeros(self.mjx_model.nv))
        data = jax.tree_util.tree_map(lambda a, b: jp.where(sitting, a, b), settled, data)

        n_delay = cfg.max_action_delay_steps
        start = data.qpos[7:]
        lo, hi = cfg.switch_time
        info = {
            "rng": rng,
            "last_act": jp.zeros(self._nu),
            "action_filtered": jp.zeros(self._nu),
            "ref": start,
            "ctrl_history": jp.tile(start, (n_delay + 1, 1)),
            "action_delay": jax.random.randint(k_delay, (), 0, n_delay + 1),
            "mode": sitting.astype(jp.int32),
            "start_mode": sitting.astype(jp.int32),
            "sit": sitting.astype(jp.float32),
            "start_sit": sitting.astype(jp.float32),
            "lying": jp.zeros(()),
            "switch_t": jax.random.uniform(k_switch, minval=lo, maxval=hi),
            "push_t": jax.random.uniform(k_push, minval=cfg.push.interval[0], maxval=cfg.push.interval[1]),
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in cfg.reward_config.scales.keys()}
        for name in MODES:     # ok_<mode> / in_<mode>: share of the time asked for <mode> that it is there
            metrics[f"ok_{name}"] = jp.zeros(())
            metrics[f"in_{name}"] = jp.zeros(())
        metrics["fell"] = jp.zeros(())
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    # ------------------------------------------------------------------ step
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        cfg = self._config
        info = state.info
        action = jp.clip(action, -1.0, 1.0)
        if "steps" in info:
            # first step of an episode: Playground's auto-reset keeps the old info (see cogu_getup.step)
            fresh = info["steps"] == 0
            q = state.data.qpos[7:]
            info["action_filtered"] = jp.where(fresh, 0.0, info["action_filtered"])
            info["ref"] = jp.where(fresh, q, info["ref"])
            info["mode"] = jp.where(fresh, info["start_mode"], info["mode"])
            info["sit"] = jp.where(fresh, info["start_sit"], info["sit"])
            info["lying"] = jp.where(fresh, 0.0, info["lying"])
            info["ctrl_history"] = jp.where(fresh, jp.tile(q, (cfg.max_action_delay_steps + 1, 1)),
                                            info["ctrl_history"])
            info["last_act"] = jp.where(fresh, 0.0, info["last_act"])

        # the command: a new one every switch_time s; lie is final
        info["rng"], k_mode, k_t, k_push, k_dir = jax.random.split(info["rng"], 5)
        info["switch_t"] = info["switch_t"] - self.dt
        switch = (info["switch_t"] <= 0.0) & (info["mode"] != 2)
        new = jax.random.choice(k_mode, 3, p=jp.array(cfg.mode_probs))
        info["mode"] = jp.where(switch, new, info["mode"])
        info["switch_t"] = jp.where(switch, jax.random.uniform(k_t, minval=cfg.switch_time[0],
                                                               maxval=cfg.switch_time[1]), info["switch_t"])

        # small pushes while standing or sitting
        info["push_t"] = info["push_t"] - self.dt
        push = (info["push_t"] <= 0.0) & (info["lying"] < 0.5) & cfg.push.enable
        theta = jax.random.uniform(k_dir, maxval=2 * jp.pi)
        mag = jax.random.uniform(k_push, minval=cfg.push.magnitude[0], maxval=cfg.push.magnitude[1])
        qvel = state.data.qvel.at[:2].add(jp.where(push, mag, 0.0) * jp.array([jp.cos(theta), jp.sin(theta)]))
        info["push_t"] = jp.where(push, jax.random.uniform(k_t, minval=cfg.push.interval[0],
                                                           maxval=cfg.push.interval[1]), info["push_t"])
        data = state.data.replace(qvel=qvel)

        a = cfg.action_filter_alpha
        filtered = a * action + (1.0 - a) * info["action_filtered"]
        info["action_filtered"] = filtered
        info["ref"] = self.reference_step(data, info)
        targets = jp.clip(info["ref"] + filtered * cfg.residual_scale, self._lowers, self._uppers)
        hist = jp.roll(info["ctrl_history"], 1, axis=0).at[0].set(targets)
        data = mjx_env.step(self.mjx_model, data, hist[info["action_delay"]], self.n_substeps)
        info["ctrl_history"] = hist

        rewards, ok, fell = self._get_reward(data, action, info)
        rewards = {k: v * cfg.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
        info["last_act"] = action
        obs = self._get_obs(data, info)
        done = (fell | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()).astype(reward.dtype)
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        for n, name in enumerate(MODES):
            asked = (info["mode"] == n).astype(jp.float32)
            state.metrics[f"ok_{name}"] = ok * asked
            state.metrics[f"in_{name}"] = asked
        state.metrics["fell"] = fell.astype(jp.float32)
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    # ---------------------------------------------------------- observation
    def _get_obs(self, data: mjx.Data, info: dict[str, Any], contact=None) -> mjx_env.Observation:
        noise = self._config.noise_config
        info["rng"], k1, k2, k3, k4 = jax.random.split(info["rng"], 5)

        def noisy(x, k, scale):
            return x + (2 * jax.random.uniform(k, shape=x.shape) - 1) * noise.level * scale

        gyro, gravity = self.get_gyro(data), self.get_gravity(data)
        q, qd = data.qpos[7:], data.qvel[6:]
        command = [jax.nn.one_hot(info["mode"], 3), info["ref"] - self._default_pose,
                   jp.stack([info["sit"], info["lying"]])]
        state = jp.hstack([
            noisy(gyro, k1, noise.scales.gyro),
            noisy(gravity, k2, noise.scales.gravity),
            noisy(q, k3, noise.scales.joint_pos) - self._default_pose,
            noisy(qd, k4, noise.scales.joint_vel),
            info["last_act"],
            *command,
        ])
        privileged = jp.hstack([
            gyro, gravity, q - self._default_pose, qd, info["last_act"], *command,
            data.qpos[2:3], self.get_local_linvel(data), self.get_global_angvel(data), data.actuator_force,
        ])
        return {"state": state, "privileged_state": privileged}

    # --------------------------------------------------------------- reward
    def expected(self, info: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
        """Pelvis lean (deg) and height (m) of the reference while standing or sitting."""
        s = self._config.sit
        c = ease(info["sit"])
        return s.pelvis_tilt * c, self._init_q[2] + (s.height - self._init_q[2]) * c

    def _get_reward(self, data: mjx.Data, action: jax.Array, info: dict[str, Any]):
        rc = self._config.reward_config
        q = data.qpos[7:]
        tilt, height = self.tilt(data), data.qpos[2]
        lying = info["lying"] > 0.5
        tilt_ref, height_ref = self.expected(info)
        g = self.get_gravity(data)
        on_back = -g[0] > jp.maximum(jp.abs(g[1]), 0.5)
        tilt_err = jp.where(lying, jp.abs(tilt - self._config.lie.tilt), jp.abs(tilt - tilt_ref))
        upright_ok = (tilt_err < 8.0) & (jp.abs(height - height_ref) < 0.02)
        at_end = (info["sit"] < 1e-3) | (info["sit"] > 1.0 - 1e-3)
        ok = jp.where(lying, (tilt > 90.0) & on_back, upright_ok & at_end).astype(jp.float32)
        fell = ~lying & (jp.abs(tilt - tilt_ref) > rc.fall_tilt)
        speed = jp.sum(jp.square(self.get_global_linvel(data))) + 0.05 * jp.sum(jp.square(self.get_gyro(data)))
        contact = self._feet_contact(data)
        rewards = {
            "pose": jp.exp(-jp.sum(jp.square(q - info["ref"])) / rc.pose_sigma),
            "tilt": jp.exp(-jp.square(tilt_err / rc.tilt_sigma)) * jp.where(lying, on_back, 1.0),
            "height": jp.where(lying, 0.0, jp.exp(-jp.square((height - height_ref) / rc.height_sigma))),
            "still": speed,
            "feet_slip": jp.where(lying, 0.0, self._cost_feet_slip(data, contact)),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "torques": jp.sum(jp.abs(data.actuator_force)),
            "dof_pos_limits": self._cost_joint_pos_limits(q),
            "residual": jp.sum(jp.square(info["action_filtered"])),
        }
        return rewards, ok, fell
