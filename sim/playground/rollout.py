"""Roll out a trained Cogu policy from the `rest` keyframe and record a video.

    uv run python sim/playground/rollout.py                       # latest checkpoint, walk forward
    uv run python sim/playground/rollout.py --vx 0.2 --wz 0.5 --seconds 10 --out build/videos/turn.mp4

Prints how far the robot got and whether it fell. No domain randomisation and no pushes here;
observation noise stays on (it is part of the policy's training distribution).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import imageio.v2 as imageio  # noqa: E402
import jax  # noqa: E402
import jax_compat  # noqa: E402,F401  (must precede brax)
import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402
from brax.training import checkpoint as brax_checkpoint  # noqa: E402
from brax.training import networks as brax_networks  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from ml_collections import config_dict  # noqa: E402
from mujoco import mjx  # noqa: E402

import cogu_env  # noqa: E402

REPO_ROOT = HERE.parents[1]


def latest_checkpoint(root: Path) -> Path:
    if (root / "ppo_network_config.json").exists():
        return root                                    # a step dir was given
    steps = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    if not steps:
        raise FileNotFoundError(f"no checkpoint under {root}; run train.py first")
    return steps[-1]


def load_policy(ckpt: Path, deterministic: bool = True):
    """Rebuild the PPO policy from a brax checkpoint dir.

    Same as `brax.training.agents.ppo.checkpoint.load_policy`, except that brax 0.14.2's
    `load_config` crashes on the `null` kernel-init entries its own `save` writes."""
    cfg = json.loads((ckpt / "ppo_network_config.json").read_text())
    kw = cfg["network_factory_kwargs"]
    for k in list(kw):
        if kw[k] is None and ("kernel_init" in k):
            del kw[k]
    if "activation" in kw:
        kw["activation"] = brax_networks.ACTIVATION[kw["activation"]]
    for k in list(kw):
        if k.endswith("_kernel_init_fn"):
            kw[k] = brax_networks.KERNEL_INITIALIZER[kw[k]]
    config = config_dict.create(**cfg)
    network = brax_checkpoint.get_network(config, ppo_networks.make_ppo_networks)
    params = brax_checkpoint.load(ckpt)
    return ppo_networks.make_inference_fn(network)(params, deterministic=deterministic)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(REPO_ROOT / "build" / "checkpoints" / "cogu_walk"))
    ap.add_argument("--xml", default=str(cogu_env.DEFAULT_XML))
    ap.add_argument("--out", default=str(REPO_ROOT / "build" / "videos" / "walk.mp4"))
    ap.add_argument("--vx", type=float, default=0.15, help="forward command (m/s)")
    ap.add_argument("--vy", type=float, default=0.0, help="lateral command (m/s)")
    ap.add_argument("--wz", type=float, default=0.0, help="yaw-rate command (rad/s)")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--camera", default="track")
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ckpt = latest_checkpoint(Path(args.checkpoint).resolve())
    print(f"checkpoint: {ckpt}")
    policy = load_policy(ckpt, deterministic=not args.stochastic)

    cfg = cogu_env.default_config()
    cfg.push_config.enable = False
    env = cogu_env.CoguJoystick(xml_path=args.xml, config=cfg)
    cmd = jp.array([args.vx, args.vy, args.wz])

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    policy = jax.jit(policy)

    rng = jax.random.PRNGKey(args.seed)
    rng, k = jax.random.split(rng)
    state = reset(k)
    # start exactly at the rest keyframe, facing +x
    data = state.data.replace(qpos=env._init_q, qvel=jp.zeros_like(state.data.qvel))
    data = mjx.forward(env.mjx_model, data)
    state = state.replace(data=data)
    state.info["command"] = cmd

    n_steps = int(round(args.seconds / env.dt))
    trajectory = [state]
    fell_at = None
    for i in range(n_steps):
        state.info["command"] = cmd
        rng, k = jax.random.split(rng)
        action, _ = policy(state.obs, k)
        state = step(state, action)
        trajectory.append(state)
        if fell_at is None and float(state.done) > 0:
            fell_at = (i + 1) * env.dt
            break

    xy0 = np.asarray(trajectory[0].data.qpos[:2])
    xy1 = np.asarray(trajectory[-1].data.qpos[:2])
    duration = (len(trajectory) - 1) * env.dt
    dist = xy1 - xy0
    print(f"command vx={args.vx} vy={args.vy} wz={args.wz}  ran {duration:.2f} s  "
          f"moved dx={dist[0]:+.3f} m dy={dist[1]:+.3f} m  mean vx={dist[0]/max(duration,1e-6):+.3f} m/s  "
          f"final pelvis z={float(trajectory[-1].data.qpos[2]):.3f} m")
    print("FELL at %.2f s" % fell_at if fell_at is not None else "stayed up")

    every = max(1, int(round(1 / (env.dt * args.fps))))
    frames = env.render(trajectory[::every], height=args.height, width=args.width, camera=args.camera)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=args.fps, codec="libx264", quality=8, macro_block_size=None)
    print(f"video: {out} ({len(frames)} frames @ {args.fps} fps)")
    return 0 if fell_at is None else 1


if __name__ == "__main__":
    sys.exit(main())
