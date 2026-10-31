"""Train the Cogu velocity-tracking policy with Brax PPO on MJX (MuJoCo Playground).

    uv run python sim/playground/train.py --steps 200000            # smoke test (minutes on a laptop CPU)
    uv run python sim/playground/train.py --steps 30000000 --num-envs 2048   # real run (GPU / HF Jobs)

Checkpoints go to `--out` (default build/checkpoints/cogu_walk/<step>/), one per evaluation plus
the final one; `rollout.py` picks the latest. The seed is fixed (--seed) so runs are reproducible
on the same hardware. Training metrics are appended to <out>/progress.jsonl.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _early_device_setup() -> int:
    """On CPU, split the host into N virtual XLA devices so brax's pmap uses N cores.
    Must run before jax is imported; a no-op for GPU/TPU backends."""
    import os
    n = 1
    if "--devices" in sys.argv:
        n = int(sys.argv[sys.argv.index("--devices") + 1])
    if n > 1 and "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                                   f" --xla_force_host_platform_device_count={n}").strip()
    return n


_early_device_setup()

import jax  # noqa: E402
import jax_compat  # noqa: E402,F401  (must precede brax)
from brax.training.agents.ppo import checkpoint as ppo_checkpoint  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from mujoco_playground import wrapper  # noqa: E402

import cogu_env  # noqa: E402
import cogu_getup  # noqa: E402
from randomize import make_domain_randomizer  # noqa: E402

REPO_ROOT = HERE.parents[1]


def network_factory(policy_hidden=(128, 128, 128), value_hidden=(256, 256, 256)):
    return functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(policy_hidden),
        value_hidden_layer_sizes=tuple(value_hidden),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=2_000_000, help="total environment steps")
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "build" / "checkpoints" / "cogu_walk"))
    ap.add_argument("--env", choices=["walk", "getup"], default="walk",
                    help="walk: velocity tracking (cogu_env); getup: stand up from a fall (cogu_getup)")
    ap.add_argument("--xml", default=None, help="default: the MJX model of the chosen env")
    ap.add_argument("--num-evals", type=int, default=4)
    ap.add_argument("--episode-length", type=int, default=500)
    ap.add_argument("--unroll-length", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-minibatches", type=int, default=4)
    ap.add_argument("--num-updates-per-batch", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--entropy-cost", type=float, default=5e-3)
    ap.add_argument("--discounting", type=float, default=0.97)
    ap.add_argument("--clipping-epsilon", type=float, default=0.2)
    ap.add_argument("--no-domain-randomization", action="store_true")
    ap.add_argument("--restore", default=None, help="checkpoint step dir to warm-start from")
    ap.add_argument("--devices", type=int, default=1,
                    help="virtual CPU devices for brax's pmap (experimental: >1 segfaults on macOS with "
                         "jax 0.11 / MJX; ignored on GPU). num_envs must be a multiple of it")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert args.batch_size * args.num_minibatches % args.num_envs == 0, \
        "batch_size * num_minibatches must be a multiple of num_envs"
    assert args.num_envs % args.devices == 0, "num_envs must be a multiple of --devices"

    print(f"jax backend: {jax.default_backend()} devices: {jax.devices()}")
    if args.env == "getup":
        env_cfg = cogu_getup.default_config()
        env_cfg.episode_length = args.episode_length
        xml = args.xml or str(cogu_getup.GETUP_XML)
        env = cogu_getup.CoguGetup(xml_path=xml, config=env_cfg)
        cogu_getup.save_config(env_cfg, out)
        eval_env = cogu_getup.CoguGetup(xml_path=xml, config=env_cfg)
    else:
        env_cfg = cogu_env.default_config()
        env_cfg.episode_length = args.episode_length
        xml = args.xml or str(cogu_env.DEFAULT_XML)
        env = cogu_env.CoguJoystick(xml_path=xml, config=env_cfg)
        cogu_env.save_config(env_cfg, out)       # play.py / rollout.py rebuild the same dynamics from it
        eval_env = cogu_env.CoguJoystick(xml_path=xml, config=env_cfg)
    print(f"env: obs {env.observation_size} act {env.action_size} xml {env.xml_path}")

    randomizer = None if args.no_domain_randomization else make_domain_randomizer(env.mj_model)
    progress_path = out / "progress.jsonl"
    t0 = time.time()
    times = [t0]

    def progress(num_steps, metrics):
        times.append(time.time())
        rec = {"step": int(num_steps), "wall_s": round(times[-1] - t0, 1)}
        rec.update({k: float(v) for k, v in metrics.items()})
        with progress_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        reward = metrics.get("eval/episode_reward", float("nan"))
        length = metrics.get("eval/avg_episode_length", float("nan"))
        # brax sums per-step metrics over the episode: divide by the length for a mean
        if args.env == "getup":
            upright = metrics.get("eval/episode_upright", float("nan")) / max(length, 1.0)
            tail = f"upright {100 * upright:5.1f} % of the time"
        else:
            speed = metrics.get("eval/episode_forward_speed", float("nan")) / max(length, 1.0)
            tail = f"mean fwd speed {speed:+.3f} m/s"
        print(f"[{rec['wall_s']:7.1f}s] step {num_steps:>10,d}  eval reward {reward:8.3f}  "
              f"ep len {length:6.1f}  {tail}", flush=True)

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=args.steps,
        num_evals=args.num_evals,
        reward_scaling=1.0,
        episode_length=args.episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=args.unroll_length,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        discounting=args.discounting,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        clipping_epsilon=args.clipping_epsilon,
        max_grad_norm=1.0,
        num_resets_per_eval=1,
        seed=args.seed,
        network_factory=network_factory(),
        randomization_fn=randomizer,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress,
        save_checkpoint_path=str(out),
        restore_checkpoint_path=args.restore,
    )
    make_inference_fn, params, metrics = train_fn(environment=env, eval_env=eval_env)
    del make_inference_fn

    # final checkpoint (brax also saved one per eval under <out>/<step>/)
    ckpt_config = ppo_checkpoint.network_config(
        observation_size=env.observation_size, action_size=env.action_size,
        normalize_observations=True, network_factory=network_factory())
    ppo_checkpoint.save(out, args.steps, params, ckpt_config)
    (out / "LATEST").write_text(f"{args.steps:012d}\n")
    total = time.time() - t0
    print(f"done in {total/60:.1f} min ({args.steps/total:,.0f} env steps/s). "
          f"checkpoint: {out / f'{args.steps:012d}'}")
    print("final eval:", {k: round(float(v), 3) for k, v in metrics.items() if k.startswith("eval/episode_")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
