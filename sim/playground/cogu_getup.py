"""Cogu get-up environment for MuJoCo Playground / MJX (2026-10-30).

A second policy, next to the walking one: it starts with the robot lying on the floor (dropped from a random
orientation and joint pose, then settled on its bubble and legs) and is rewarded for getting upright, at the
standing height, and then into the `rest` pose the walking policy starts from. On the robot, the IMU tells
which one runs: fallen -> get-up; upright and still in `rest` -> walking.

Getting up from lying is a throw onto the feet and then a catch (README, "Levantar"): besides drops, episodes
start from states on the feet but tilted and from the middle of open-loop throws (`getup_starts.py`), so the policy
sees landings early and learns the catch before the throw.

Modelled on MuJoCo Playground's Go1 get-up env. Differences: the action is an offset from the `rest` pose of up to ±1 rad (getting up needs big excursions; zero action
stands), with the same low-pass filter and actuator latency as the walking env;
the model is the get-up MJX variant (`cogu_lower_body_mjx_getup.xml`), where the bubble, the drum and the
leg links touch the floor.

Observation (36): IMU gyro and gravity direction, joint positions (from rest) and velocities, last action.

v3 (2026-11-01), `reference`: the policy corrects a scripted catch instead of starting from nothing. Each step the
pelvis tilt (from the IMU's gravity) picks a pose on the catch path of `getup_starts.py` (the slow path from the
edge of the catch window to `rest`, face up and face down), `lead` degrees towards upright; the reference walks to
it at no more than `rate` rad/s per joint, and the target is the reference plus the action × `residual_scale` (1 rad).
Zero action is `getup_reference.py`'s catch (36 % upright from the landings of the foot study, 23 % for `rest`
alone). The reference minus `rest` is added to the observation (46).

v4 (2026-11-02), `reference.throw`: the reference also throws. Lying (|tilt| above `lie_tilt` for `lie_time`), it
plays the scripted throw for that side (`getup_sequences/throw_back.json` / `throw_front.json`, found by
`getup_reference.py --search --full` together with this catch) keyframe by keyframe, and hands over to the catch once
|tilt| is below the throw's trigger or the throw ends; if the robot ends up lying again, it throws again. The `throw`
starts are snapshots of these throws, and start the reference at that moment of the throw. Observation (48): + throwing
(0/1) and how far into the throw.

v5 (2026-11-02), `reference.roll`, on a model whose hip pitch reaches −120° (`variants.py --hip-min -120`; the CAD
stops at −85°): with that hip the robot rises from its back slowly, with no throw (README, "Estudo do quadril"), so the
reference stops throwing. Lying (|tilt| above `lie_tilt` for `lie_time`), the reference first relaxes the legs to
`rest` at `rate` for `relax_time` (the egg then rolls flat), then reads how the robot lies: on the back it goes
straight to the catch (whose path now starts at 105°, still lying); on the front or a side it plays the searched roll
onto the back (`getup_sequences/roll_front.json`, `roll_side.json` or its mirror, `getup_starts.roll_table`), then
the catch. Lying on the back again after a sequence, the catch keeps going; lying any other way, it starts over. The
`throw` starts are snapshots of the rolls (`getup_starts.py --mode roll`). Observation (48): the same two values
(in a sequence, share of it done). Episodes are 15 s (relax + roll + rise take ~10 s).
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

GETUP_XML = cogu_env.REPO_ROOT / "build" / "mjcf" / "cogu_lower_body_mjx_getup.xml"
START_KINDS = ("stand", "tilt", "feet", "throw", "drop")


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        env="getup",
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=750,            # 15 s (v5: relax + roll + rise take ~10 s; was 10 s)
        action_repeat=1,
        action_filter_alpha=0.4,
        # target = rest pose + action × action_scale (rad), clipped to the joint range: zero action stands. The
        # first version mapped the action to each joint's whole range (zero = mid-range, a crouch); with the early
        # policy's exploration noise even the episodes that started standing fell (0.7 % upright after 3 M steps)
        action_scale=1.0,
        max_action_delay_steps=2,
        soft_joint_pos_limit_factor=0.95,
        # how episodes start (a curriculum by mixture): standing in `rest`; in `rest` but tilted (released mid-fall:
        # learn to catch it and come back); on the feet, tilted, in a pose that can hold (a landing); mid-throw (from
        # the open-loop throws of getup_search); dropped from a random orientation and joint pose and settled
        # (lying). Only "standing" and "dropped" at first: the policy learned to thrash on the floor and did the
        # same when it started standing (1 % upright after 1.5 M steps locally); standing / tilted / dropped
        # (0.3 / 0.3 / 0.4) stood 32 % early and unlearned it (0.9 % after 11 M on Colab): it never saw a landing
        start_mix=(0.1, 0.1, 0.35, 0.2, 0.25),   # standing, tilted, feet, throw, dropped
        tilt_range=(0.17, 1.22),       # rad (10°-70°) about a random horizontal axis
        starts_file="",                # getup_starts.py table ("" = build/getup_starts/<xml stem>.npz)
        start_vel_noise=(0.2, 0.5),    # uniform ± on the feet / throw starts: body (m/s, rad/s), joints (rad/s)
        drop_height=0.35,              # m, pelvis
        settle_time=1.0,               # s of physics after the drop, holding the random joint pose
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(joint_pos=0.02, joint_vel=1.0, gravity=0.05, gyro=0.2),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                orientation=1.0,       # ((1 + up_z) / 2)²: dense from upside down to upright
                torso_height=2.0,      # ((pelvis height - lying) / (target - lying))², capped at 1. Was linear in
                                       # height / target: a crouch (pelvis 130 mm) kept 65 % of it, and the policy
                                       # crouched with the joints on their limits, even from standing (local, 0.4 M)
                posture=1.0,           # once up: close to `rest` (where the walking policy takes over)
                stand_still=-0.02,     # once up: squared action changes. Was -0.5 on |changes|: with the early,
                                       # noisy policy that cost ~5 per step, more than standing earned (~3), so
                                       # lying down paid better and even the standing starts fell (Colab, 11 M)
                action_rate=-0.01,
                torques=-0.002,
                dof_pos_limits=-1.0,
            ),
            height_target=0.20,        # m, pelvis (rest: 0.205)
            height_lying=0.07,         # m, pelvis lying on the bubble (0.057-0.074)
            upright_cos=0.95,          # up_z above this counts as upright (~18°)
            posture_sigma=0.5,
        ),
        reference=config_dict.create(
            enable=True,
            lead=10.0,                 # deg towards upright (getup_reference / foot_study: lead 10, rate 1)
            rate=1.0,                  # rad/s per joint
            residual_scale=1.0,        # rad: target = reference + action × this (a throw needs big ones)
            tilts=[],                  # catch path (deg), filled from the starts file and saved with the config
            legs=[],                   # hip pitch / knee / ankle (deg) at each tilt
            throw=False,               # v4: scripted throws from lying (see above)
            roll=True,                 # v5: relax, roll onto the back, rise (see above; needs the hip at -120°)
            relax_time=2.5,            # s of legs going back to `rest` before reading how it lies
            lie_tilt=80.0,             # deg: lying = |tilt| above this and above the throw's trigger + 5
            lie_time=0.3,              # s lying before a throw starts
            throw_back=config_dict.create(targets=[], durations=[], trigger=0.0),   # rad, s, deg: filled from
            throw_front=config_dict.create(targets=[], durations=[], trigger=0.0),  # getup_sequences, saved too
            roll_front=config_dict.create(targets=[], durations=[], hold=0.0),     # rad, s, s: filled from
            roll_side=config_dict.create(targets=[], durations=[], hold=0.0),      # getup_sequences, saved too
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
        path = Path(self._config.starts_file or getup_starts.default_path(xml_path))
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Make it with `uv run python sim/playground/getup_starts.py "
                                    f"--xml {xml_path}`.")
        table = np.load(path)
        self._starts_qpos, self._starts_qvel = jp.array(table["qpos"]), jp.array(table["qvel"])
        self._starts_feet = jp.array(np.flatnonzero(table["kind"] == 0))
        self._starts_throw = jp.array(np.flatnonzero(table["kind"] == 1))
        assert self._starts_feet.size and self._starts_throw.size, f"{path}: no feet or no throw states"
        ref = self._config.reference
        if ref.enable:
            if not ref.tilts:
                assert "catch_tilt" in table, f"{path}: no catch path; rebuild it with getup_starts.py"
                ref.tilts, ref.legs = table["catch_tilt"].tolist(), table["catch_q"].tolist()
            self._ref_tilts = jp.array(ref.tilts)
            legs = np.radians(np.array(ref.legs))
            self._ref_legs = jp.array(np.column_stack([np.zeros((len(ref.tilts), 2)), legs]))   # yaw, roll = 0
            assert not (ref.throw and ref.get("roll", False)), "reference: throw or roll, not both"
            if ref.throw:
                self._init_throws()
            if ref.get("roll", False):
                assert str(table.get("mode", "throw")) == "roll", f"{path}: throw starts; rebuild it (--mode roll)"
                self._init_rolls()
        self._start_throw_t = jp.array(table["throw_t"]) if "throw_t" in table else jp.full(len(table["kind"]), -1.0)
        if ref.enable and ref.get("roll", False):     # the table counts time into the roll; the env, relax + roll
            self._start_throw_t = jp.where(self._start_throw_t >= 0.0, self._start_throw_t + ref.relax_time, -1.0)
        self._start_throw_side = (jp.array(table["throw_side"]) if "throw_side" in table
                                  else jp.zeros(len(table["kind"]), jp.int32))

    def _init_throws(self):
        """Throw tables, back (0) and front (1): keyframes (2, K, nu), their end times (2, K), triggers, present."""
        ref = self._config.reference
        for side in ("back", "front"):
            cfg = ref[f"throw_{side}"]
            path = getup_starts.THROWS[side]
            if not cfg.targets and path.exists():
                spec = json.loads(path.read_text())
                cfg.targets, cfg.durations, cfg.trigger = spec["targets"], spec["durations"], spec["trigger"]
        throws = [ref[f"throw_{side}"] for side in ("back", "front")]
        k = max(max(len(t.durations) for t in throws), 1)
        keys, ends = np.tile(np.asarray(self._default_pose), (2, k, 1)), np.full((2, k), 1e3)
        for i, t in enumerate(throws):
            if t.durations:
                n = len(t.durations)
                keys[i, :n] = np.asarray(t.targets)
                keys[i, n:] = keys[i, n - 1]
                ends[i] = np.concatenate([np.cumsum(t.durations), np.full(k - n, np.sum(t.durations))])
        self._throw_keys, self._throw_ends = jp.array(keys), jp.array(ends)
        self._throw_has = jp.array([bool(t.durations) for t in throws])
        self._throw_trigger = jp.array([t.trigger for t in throws], dtype=jp.float32)

    def _init_rolls(self):
        """Roll tables, back (none) / front / side / other side: keyframes (4, K, nu) and end times (4, K)."""
        ref = self._config.reference
        for name in ("front", "side"):
            cfg = ref[f"roll_{name}"]
            path = getup_starts.ROLLS[name]
            if not cfg.targets and path.exists():
                spec = json.loads(path.read_text())
                cfg.targets, cfg.durations, cfg.hold = spec["targets"], spec["durations"], spec["hold"]
        keys, ends = getup_starts.roll_table(np.asarray(self._default_pose),
                                             {name: ref[f"roll_{name}"].to_dict() for name in ("front", "side")})
        self._roll_keys, self._roll_ends = jp.array(keys), jp.array(ends)
        self._throw_ends = self._roll_ends                 # for _throw_obs
        # which side of the pelvis is up in the searched side roll (getup_search's "left" pose: +y up)
        self._roll_side_up = 1.0

    def _roll_step(self, data: mjx.Data, info: dict[str, Any], tilt: jax.Array, catch: jax.Array) -> jax.Array:
        """v5: relax, roll onto the back, catch. info's throw_t: s into relax + roll (-1 = catching); throw_side: the
        roll (0 back = none, 1 front, 2 side, 3 other side), read when the relax ends; rolled: a sequence ended."""
        ref = self._config.reference
        g = self.get_gravity(data)
        xz, yz = -g[0], -g[1]                             # pelvis x / y axes, world z component
        on_back = xz > jp.maximum(jp.abs(yz), 0.5)
        lay = jp.select([on_back, -xz > jp.maximum(jp.abs(yz), 0.5), yz * self._roll_side_up > 0.5,
                         yz * self._roll_side_up < -0.5], [0, 1, 2, 3], 0)
        tt, side = info["throw_t"], info["throw_side"]
        lying = (jp.abs(tilt) > ref.lie_tilt) & (tt < 0.0) & (~on_back | (info["rolled"] < 0.5))
        lying_t = jp.where(lying, info["lying_t"] + self.dt, 0.0)
        start = lying_t >= ref.lie_time
        tt = jp.where(start, 0.0, tt)
        relaxing = (tt >= 0.0) & (tt < ref.relax_time)
        side = jp.where(relaxing, lay, side)             # read how it lies until the relax ends
        tr = tt - ref.relax_time
        rolling = (tr >= 0.0) & (tr < self._roll_ends[side, -1])
        k = jp.minimum(jp.searchsorted(self._roll_ends[side], tr, side="right"), self._roll_ends.shape[1] - 1)
        step = ref.rate * self.dt
        relax = info["ref"] + jp.clip(self._default_pose - info["ref"], -step, step)
        active = relaxing | rolling
        info["rolled"] = jp.where((tt >= 0.0) & ~active, 1.0, info["rolled"])
        info["throw_t"] = jp.where(active, tt + self.dt, -1.0)
        info["throw_side"] = side
        info["lying_t"] = jp.where(start, 0.0, lying_t)
        return jp.where(relaxing, relax, jp.where(rolling, self._roll_keys[side, k], catch))

    def reference_step(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """The scripted throw and catch: from the IMU's tilt, the next reference (as getup_reference.Reference).
        Updates info's throw_t (s into the throw, -1 = catching), throw_side and lying_t."""
        ref = self._config.reference
        g = self.get_gravity(data)                     # body frame; up_z = -g_z, the pelvis x axis rises when g_x < 0
        tilt = jp.degrees(jp.arccos(jp.clip(-g[2], -1.0, 1.0))) * jp.where(g[0] <= 0.0, 1.0, -1.0)
        want = jp.sign(tilt) * jp.maximum(jp.abs(tilt) - ref.lead, 0.0)
        leg = jax.vmap(lambda col: jp.interp(want, self._ref_tilts, col), in_axes=1)(self._ref_legs)
        goal = jp.concatenate([leg, leg])
        step = ref.rate * self.dt
        catch = info["ref"] + jp.clip(goal - info["ref"], -step, step)
        if ref.get("roll", False):
            return self._roll_step(data, info, tilt, catch)
        if not ref.throw:
            return catch
        tt, side = info["throw_t"], info["throw_side"]
        guess = jp.where(tilt >= 0.0, 0, 1)           # face up: back throw; face down: front throw
        lying = ((jp.abs(tilt) > jp.maximum(ref.lie_tilt, self._throw_trigger[guess] + 5.0))
                 & self._throw_has[guess] & (tt < 0.0))
        lying_t = jp.where(lying, info["lying_t"] + self.dt, 0.0)
        start = lying_t >= ref.lie_time
        tt, side = jp.where(start, 0.0, tt), jp.where(start, guess, side)
        over = (jp.abs(tilt) <= self._throw_trigger[side]) | (tt >= self._throw_ends[side, -1])
        throwing = (tt >= 0.0) & ~over
        k = jp.minimum(jp.searchsorted(self._throw_ends[side], tt, side="right"), self._throw_ends.shape[1] - 1)
        info["throw_t"] = jp.where(throwing, tt + self.dt, -1.0)
        info["throw_side"] = side
        info["lying_t"] = jp.where(start, 0.0, lying_t)
        return jp.where(throwing, self._throw_keys[side, k], catch)

    def _throw_obs(self, info: dict[str, Any]) -> list:
        """Throwing (0/1) and the share of the throw done (v5: of relax + roll)."""
        ref = self._config.reference
        roll = ref.get("roll", False)
        if not (ref.enable and (ref.throw or roll)):
            return []
        throwing = info["throw_t"] >= 0.0
        total = self._throw_ends[info["throw_side"], -1] + (ref.relax_time if roll else 0.0)
        done = jp.where(throwing, info["throw_t"] / jp.maximum(total, 1e-3), 0.0)
        return [jp.stack([throwing.astype(jp.float32), done])]

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

    def _table_state(self, rng: jax.Array, indices: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """A getup_starts state at a random place and heading, with noisy velocities; and its row."""
        k_i, k_yaw, k_xy, k_v = jax.random.split(rng, 4)
        i = jax.random.choice(k_i, indices)
        qpos, qvel = self._starts_qpos[i], self._starts_qvel[i]
        yaw = jax.random.uniform(k_yaw, minval=-jp.pi, maxval=jp.pi)
        heading = mjx_math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
        c, s = jp.cos(yaw), jp.sin(yaw)
        rot = jp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        xy = jax.random.uniform(k_xy, (2,), minval=-0.3, maxval=0.3)
        qpos = qpos.at[:2].set(xy).at[3:7].set(mjx_math.quat_mul(heading, qpos[3:7]))
        body, joints = self._config.start_vel_noise
        scale = jp.concatenate([jp.full(6, body), jp.full(self._nu, joints)])
        qvel = qvel.at[:3].set(rot @ qvel[:3])          # world-frame linear velocity turns with the heading
        qvel = qvel + jax.random.uniform(k_v, qvel.shape, minval=-1.0, maxval=1.0) * scale
        return qpos, qvel, i

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k_start, k_q, k_tilt, k_feet, k_throw, k_delay = jax.random.split(rng, 7)
        # 0 stand, 1 tilt, 2 feet, 3 throw, 4 drop
        start_kind = jax.random.choice(k_start, len(START_KINDS), p=jp.array(self._config.start_mix))
        feet_q, feet_v, _ = self._table_state(k_feet, self._starts_feet)
        throw_q, throw_v, row = self._table_state(k_throw, self._starts_throw)
        qpos = jp.select([start_kind == 0, start_kind == 1, start_kind == 2, start_kind == 3],
                         [self._init_q, self._tilted_qpos(k_tilt), feet_q, throw_q], self._random_fallen_qpos(k_q))
        qvel = jp.select([start_kind == 2, start_kind == 3], [feet_v, throw_v], jp.zeros(self.mjx_model.nv))
        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:],
                                 impl=self.mjx_model.impl.value, naconmax=self._config.naconmax,
                                 njmax=self._config.njmax)
        data = mjx.forward(self.mjx_model, data)
        settled = mjx_env.step(self.mjx_model, data, qpos[7:], self._settle_steps)   # fall and come to rest
        settled = settled.replace(qvel=jp.zeros(self.mjx_model.nv))
        data = jax.tree_util.tree_map(lambda a, b: jp.where(start_kind == 4, a, b), settled, data)

        n_delay = self._config.max_action_delay_steps
        start = data.qpos[7:]
        info = {
            "rng": rng,
            "last_act": jp.zeros(self._nu),
            "action_filtered": self._start_filtered(start),
            "ref": start,
            "ctrl_history": jp.tile(start, (n_delay + 1, 1)),
            "action_delay": jax.random.randint(k_delay, (), 0, n_delay + 1),
            "start_kind": start_kind,
            # the throw the reference is in (`throw` starts: that snapshot's moment); kept for the auto-resets
            "start_throw_t": jp.where(start_kind == 3, self._start_throw_t[row], -1.0),
            "start_throw_side": jp.where(start_kind == 3, self._start_throw_side[row], 0),
            "lying_t": jp.zeros(()),
            "rolled": jp.zeros(()),
        }
        info["throw_t"], info["throw_side"] = info["start_throw_t"], info["start_throw_side"]
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
        metrics["upright"] = jp.zeros(())
        for name in START_KINDS:     # per start kind: upright_<kind> / start_<kind> = upright share of those episodes
            metrics[f"upright_{name}"] = jp.zeros(())
            metrics[f"start_{name}"] = jp.zeros(())
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _start_filtered(self, q: jax.Array) -> jax.Array:
        """Filter state that holds the joints where they are (with the reference: zero, the reference starts there)."""
        if self._config.reference.enable:
            return jp.zeros(self._nu)
        return jp.clip((q - self._default_pose) / self._config.action_scale, -1.0, 1.0)

    # ------------------------------------------------------------------ step
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        cfg = self._config
        action = jp.clip(action, -1.0, 1.0)            # the "normal" policy (train.py) is not squashed
        if "steps" in state.info:
            # first step of an episode: Playground's auto-reset puts back the first data and obs but keeps the old
            # info, so the filter and the delay line still held the end of the previous episode (often lying) and
            # dragged the new start to the floor. The policy learned to lie down even from standing (local, 0.2 M)
            fresh = state.info["steps"] == 0
            q = state.data.qpos[7:]
            state.info["action_filtered"] = jp.where(fresh, self._start_filtered(q), state.info["action_filtered"])
            state.info["ref"] = jp.where(fresh, q, state.info["ref"])
            state.info["throw_t"] = jp.where(fresh, state.info["start_throw_t"], state.info["throw_t"])
            state.info["throw_side"] = jp.where(fresh, state.info["start_throw_side"], state.info["throw_side"])
            state.info["lying_t"] = jp.where(fresh, 0.0, state.info["lying_t"])
            state.info["rolled"] = jp.where(fresh, 0.0, state.info["rolled"])
            state.info["ctrl_history"] = jp.where(fresh, jp.tile(q, (cfg.max_action_delay_steps + 1, 1)),
                                                  state.info["ctrl_history"])
            state.info["last_act"] = jp.where(fresh, jp.zeros(self._nu), state.info["last_act"])
        a = cfg.action_filter_alpha
        filtered = a * action + (1.0 - a) * state.info["action_filtered"]
        state.info["action_filtered"] = filtered
        if cfg.reference.enable:
            state.info["ref"] = self.reference_step(state.data, state.info)
            targets = jp.clip(state.info["ref"] + filtered * cfg.reference.residual_scale, self._lowers, self._uppers)
        else:
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
        for n, name in enumerate(START_KINDS):
            is_kind = (state.info["start_kind"] == n).astype(jp.float32)
            state.metrics[f"upright_{name}"] = upright * is_kind
            state.metrics[f"start_{name}"] = is_kind
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    # ---------------------------------------------------------- observation
    def _get_obs(self, data: mjx.Data, info: dict[str, Any], contact=None) -> mjx_env.Observation:
        noise = self._config.noise_config
        info["rng"], k1, k2, k3, k4 = jax.random.split(info["rng"], 5)

        def noisy(x, k, scale):
            return x + (2 * jax.random.uniform(k, shape=x.shape) - 1) * noise.level * scale

        gyro, gravity = self.get_gyro(data), self.get_gravity(data)
        q, qd = data.qpos[7:], data.qvel[6:]
        ref = [info["ref"] - self._default_pose] if self._config.reference.enable else []
        ref += self._throw_obs(info)
        state = jp.hstack([
            noisy(gyro, k1, noise.scales.gyro),
            noisy(gravity, k2, noise.scales.gravity),
            noisy(q, k3, noise.scales.joint_pos) - self._default_pose,
            noisy(qd, k4, noise.scales.joint_vel),
            info["last_act"],
            *ref,
        ])
        privileged = jp.hstack([
            gyro, gravity, q - self._default_pose, qd, info["last_act"], *ref,
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
            "torso_height": jp.square(jp.clip((height - rc.height_lying) / (rc.height_target - rc.height_lying),
                                              0.0, 1.0)),
            "posture": jp.exp(-jp.sum(jp.square(q - self._default_pose)) / rc.posture_sigma) * upright,
            "stand_still": jp.sum(jp.square(action - info["last_act"])) * upright,
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "torques": jp.sum(jp.abs(data.actuator_force)),
            "dof_pos_limits": self._cost_joint_pos_limits(q),
        }
        return rewards, upright
