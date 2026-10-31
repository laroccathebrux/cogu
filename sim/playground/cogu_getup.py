"""Cogu get-up environment for MuJoCo Playground / MJX (2026-10-30).

A second policy, next to the walking one: it starts with the robot lying on the floor (dropped from a random
orientation and joint pose, then settled on its bubble and legs) and is rewarded for getting upright, at the
standing height, and then into the `rest` pose the walking policy starts from. On the robot, the IMU tells
which one runs: fallen -> get-up; upright and still in `rest` -> walking.

Modelled on MuJoCo Playground's Go1 get-up env. Differences: the action is an offset from the `rest` pose of up to ±1 rad (getting up needs big excursions; zero action
stands), with the same low-pass filter and actuator latency as the walking env;
the model is the get-up MJX variant (`cogu_lower_body_mjx_getup.xml`), where the bubble, the drum and the
leg links touch the floor.

Observation (36): IMU gyro and gravity direction, joint positions (from rest) and velocities, last action.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
from mujoco_playground._src import mjx_env

import cogu_env

GETUP_XML = cogu_env.REPO_ROOT / "build" / "mjcf" / "cogu_lower_body_mjx_getup.xml"


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        env="getup",
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=500,            # 10 s
        action_repeat=1,
        action_filter_alpha=0.4,
        # target = rest pose + action × action_scale (rad), clipped to the joint range: zero action stands. The
        # first version mapped the action to each joint's whole range (zero = mid-range, a crouch); with the early
        # policy's exploration noise even the episodes that started standing fell (0.7 % upright after 3 M steps)
        action_scale=1.0,
        max_action_delay_steps=2,
        soft_joint_pos_limit_factor=0.95,
        # how episodes start (a curriculum by mixture): standing in `rest`; in `rest` but tilted (released mid-fall:
        # learn to catch it and come back); dropped from a random orientation and joint pose and settled (lying).
        # Only "standing" and "dropped" at first: the policy learned to thrash on the floor and did the same when
        # it started standing (1 % upright after 1.5 M steps locally)
        start_mix=(0.3, 0.3, 0.4),     # standing, tilted, dropped
        tilt_range=(0.17, 1.22),       # rad (10°-70°) about a random horizontal axis
        drop_height=0.35,              # m, pelvis
        settle_time=1.0,               # s of physics after the drop, holding the random joint pose
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(joint_pos=0.02, joint_vel=1.0, gravity=0.05, gyro=0.2),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                orientation=1.0,       # ((1 + up_z) / 2)²: dense from upside down to upright
                torso_height=1.0,      # pelvis height / target, capped at 1
                posture=1.0,           # once up: close to `rest` (where the walking policy takes over)
                stand_still=-0.02,     # once up: squared action changes. Was -0.5 on |changes|: with the early,
                                       # noisy policy that cost ~5 per step, more than standing earned (~3), so
                                       # lying down paid better and even the standing starts fell (Colab, 11 M)
                action_rate=-0.01,
                torques=-0.002,
                dof_pos_limits=-1.0,
            ),
            height_target=0.20,        # m, pelvis (rest: 0.205)
            upright_cos=0.95,          # up_z above this counts as upright (~18°)
            posture_sigma=0.5,
        ),
        impl="jax",
        naconmax=None,
        njmax=None,
    )


def save_config(cfg: config_dict.ConfigDict, out_dir: Union[str, Path]) -> Path:
    path = Path(out_dir) / cogu_env.ENV_CONFIG_FILE
    path.write_text(cfg.to_json(indent=2))
    return path


def is_getup_checkpoint(ckpt: Union[str, Path]) -> bool:
    for d in (Path(ckpt), Path(ckpt).parent):
        f = d / cogu_env.ENV_CONFIG_FILE
        if f.exists():
            return json.loads(f.read_text()).get("env") == "getup"
    return False


class CoguGetup(cogu_env.CoguJoystick):
    """Get up from any fallen pose and settle in `rest`."""

    def __init__(self, xml_path: Union[str, Path] = GETUP_XML, config: config_dict.ConfigDict = default_config()):
        super().__init__(xml_path=xml_path, config=config)
        self._settle_steps = int(round(self._config.settle_time / self.sim_dt))

    # ----------------------------------------------------------------- reset
    def _random_fallen_qpos(self, rng: jax.Array) -> jax.Array:
        k_quat, k_q, k_xy = jax.random.split(rng, 3)
        quat = jax.random.normal(k_quat, (4,))
        quat = quat / jp.linalg.norm(quat)
        q = jax.random.uniform(k_q, (self._nu,), minval=self._lowers, maxval=self._uppers)
        xy = jax.random.uniform(k_xy, (2,), minval=-0.3, maxval=0.3)
        return jp.concatenate([xy, jp.array([self._config.drop_height]), quat, q])

    def _tilted_qpos(self, rng: jax.Array) -> jax.Array:
        k_axis, k_angle = jax.random.split(rng)
        heading = jax.random.uniform(k_axis, minval=-jp.pi, maxval=jp.pi)
        axis = jp.array([jp.cos(heading), jp.sin(heading), 0.0])
        lo, hi = self._config.tilt_range
        quat = mjx_math.axis_angle_to_quat(axis, jax.random.uniform(k_angle, minval=lo, maxval=hi))
        return self._init_q.at[3:7].set(mjx_math.quat_mul(quat, self._init_q[3:7]))

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k_start, k_q, k_tilt, k_delay = jax.random.split(rng, 5)
        start_kind = jax.random.choice(k_start, 3, p=jp.array(self._config.start_mix))   # 0 stand, 1 tilt, 2 drop
        qpos = jp.select([start_kind == 0, start_kind == 1],
                         [self._init_q, self._tilted_qpos(k_tilt)], self._random_fallen_qpos(k_q))
        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=jp.zeros(self.mjx_model.nv), ctrl=qpos[7:],
                                 impl=self.mjx_model.impl.value, naconmax=self._config.naconmax,
                                 njmax=self._config.njmax)
        data = mjx.forward(self.mjx_model, data)
        settled = mjx_env.step(self.mjx_model, data, qpos[7:], self._settle_steps)   # fall and come to rest
        settled = settled.replace(qvel=jp.zeros(self.mjx_model.nv))
        data = jax.tree_util.tree_map(lambda a, b: jp.where(start_kind == 2, a, b), settled, data)

        n_delay = self._config.max_action_delay_steps
        start = data.qpos[7:]
        info = {
            "rng": rng,
            "last_act": jp.zeros(self._nu),
            "action_filtered": jp.clip((start - self._default_pose) / self._config.action_scale, -1.0, 1.0),
            "ctrl_history": jp.tile(start, (n_delay + 1, 1)),
            "action_delay": jax.random.randint(k_delay, (), 0, n_delay + 1),
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
        metrics["upright"] = jp.zeros(())
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    # ------------------------------------------------------------------ step
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        cfg = self._config
        a = cfg.action_filter_alpha
        filtered = a * action + (1.0 - a) * state.info["action_filtered"]
        state.info["action_filtered"] = filtered
        targets = jp.clip(self._default_pose + filtered * cfg.action_scale, self._lowers, self._uppers)
        hist = jp.roll(state.info["ctrl_history"], 1, axis=0).at[0].set(targets)
        data = mjx_env.step(self.mjx_model, state.data, hist[state.info["action_delay"]], self.n_substeps)
        state.info["ctrl_history"] = hist

        rewards, upright = self._get_reward(data, action, state.info)
        rewards = {k: v * cfg.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
        state.info["last_act"] = action
        obs = self._get_obs(data, state.info)
        done = (jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()).astype(reward.dtype)
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["upright"] = upright
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    # ---------------------------------------------------------- observation
    def _get_obs(self, data: mjx.Data, info: dict[str, Any], contact=None) -> mjx_env.Observation:
        noise = self._config.noise_config
        info["rng"], k1, k2, k3, k4 = jax.random.split(info["rng"], 5)

        def noisy(x, k, scale):
            return x + (2 * jax.random.uniform(k, shape=x.shape) - 1) * noise.level * scale

        gyro, gravity = self.get_gyro(data), self.get_gravity(data)
        q, qd = data.qpos[7:], data.qvel[6:]
        state = jp.hstack([
            noisy(gyro, k1, noise.scales.gyro),
            noisy(gravity, k2, noise.scales.gravity),
            noisy(q, k3, noise.scales.joint_pos) - self._default_pose,
            noisy(qd, k4, noise.scales.joint_vel),
            info["last_act"],
        ])
        privileged = jp.hstack([
            gyro, gravity, q - self._default_pose, qd, info["last_act"],
            data.qpos[2:3], self.get_local_linvel(data), self.get_global_angvel(data), data.actuator_force,
        ])
        return {"state": state, "privileged_state": privileged}

    # --------------------------------------------------------------- reward
    def _get_reward(self, data: mjx.Data, action: jax.Array, info: dict[str, Any]):
        rc = self._config.reward_config
        up = self.get_upvector(data)[2]
        height = data.qpos[2]
        q = data.qpos[7:]
        upright = ((up > rc.upright_cos) & (height > 0.9 * rc.height_target)).astype(jp.float32)
        rewards = {
            "orientation": jp.square((1.0 + up) / 2.0),
            "torso_height": jp.clip(height / rc.height_target, 0.0, 1.0),
            "posture": jp.exp(-jp.sum(jp.square(q - self._default_pose)) / rc.posture_sigma) * upright,
            "stand_still": jp.sum(jp.square(action - info["last_act"])) * upright,
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "torques": jp.sum(jp.abs(data.actuator_force)),
            "dof_pos_limits": self._cost_joint_pos_limits(q),
        }
        return rewards, upright
