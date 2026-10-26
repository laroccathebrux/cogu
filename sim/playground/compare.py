"""Compare walking checkpoints on the current model: distance in 8 s and falls under pushes.

    uv run python sim/playground/compare.py sim/playground/checkpoints/cogu_walk_v0 build/checkpoints/cogu_walk_v1

For each checkpoint: (a) deterministic 8 s rollout from `rest` at vx = 0.15 m/s with no pushes
(distance, fell?), and (b) N stochastic-start episodes of 10 s with pushes and random commands,
like the training evaluation (fall rate, mean episode length). Same seeds for every checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax  # noqa: E402
import jax_compat  # noqa: E402,F401
import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import mjx  # noqa: E402

import cogu_env  # noqa: E402
from rollout import latest_checkpoint, load_policy  # noqa: E402


def straight_walk(env, policy, vx=0.15, seconds=8.0, seed=0):
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    rng = jax.random.PRNGKey(seed)
    s = reset(rng)
    data = s.data.replace(qpos=env._init_q, qvel=jp.zeros_like(s.data.qvel))
    s = s.replace(data=mjx.forward(env.mjx_model, data))
    cmd = jp.array([vx, 0.0, 0.0])
    n = int(round(seconds / env.dt))
    for i in range(n):
        s.info["command"] = cmd
        rng, k = jax.random.split(rng)
        a, _ = policy(s.obs, k)
        s = step(s, a)
        if float(s.done) > 0:
            return float(s.data.qpos[0]), (i + 1) * env.dt
    return float(s.data.qpos[0]), None


def pushed_episodes(env, policy, n_envs=32, seconds=10.0, seed=1):
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    vpolicy = jax.jit(jax.vmap(policy))
    rng = jax.random.PRNGKey(seed)
    rng, k = jax.random.split(rng)
    s = reset(jax.random.split(k, n_envs))
    alive = np.ones(n_envs, dtype=bool)
    length = np.zeros(n_envs)
    n = int(round(seconds / env.dt))
    for i in range(n):
        rng, k = jax.random.split(rng)
        a, _ = vpolicy(s.obs, jax.random.split(k, n_envs))
        s = step(s, a)
        done = np.asarray(s.done) > 0
        length[alive] = (i + 1) * env.dt
        alive &= ~done
        if not alive.any():
            break
    return 1.0 - alive.mean(), length.mean()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--xml", default=str(cogu_env.DEFAULT_XML))
    ap.add_argument("--n-envs", type=int, default=32)
    args = ap.parse_args(argv)

    cfg = cogu_env.default_config()
    cfg.push_config.enable = False
    env_quiet = cogu_env.CoguJoystick(xml_path=args.xml, config=cfg)
    env_push = cogu_env.CoguJoystick(xml_path=args.xml, config=cogu_env.default_config())
    print(f"model: {args.xml}  mass {env_quiet.mj_model.body_subtreemass[1]:.3f} kg")
    print(f"{'checkpoint':40s} {'8 s @0.15 m/s':>16s} {'fell':>6s} | {'pushes: fall rate':>18s} {'mean len':>9s}")
    for c in args.checkpoints:
        ckpt = latest_checkpoint(Path(c).resolve())
        policy = jax.jit(load_policy(ckpt))
        dist, fell_at = straight_walk(env_quiet, policy)
        rate, length = pushed_episodes(env_push, policy, n_envs=args.n_envs)
        fell = f"{fell_at:.1f}s" if fell_at else "no"
        print(f"{ckpt.parent.name + '/' + ckpt.name:40s} {dist:+13.2f} m {fell:>6s} | {rate:17.0%} {length:8.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
