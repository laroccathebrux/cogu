"""Quick health check of the Cogu MJX environment (used by CI; no training).

    uv run python sim/playground/check_env.py

Builds the env from `build/mjcf/cogu_lower_body_mjx.xml`, resets it, holds the rest pose for
2 s through the actuator-delay path and checks that the robot is still standing, that the
observations are finite and that the domain randomiser produces a batched model.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402

import cogu_env  # noqa: E402
from randomize import make_domain_randomizer  # noqa: E402


def main() -> int:
    cfg = cogu_env.default_config()
    cfg.push_config.enable = False
    env = cogu_env.CoguJoystick(config=cfg)
    print(f"obs {env.observation_size} act {env.action_size} dt {env.dt} substeps {env.n_substeps}")
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    t0 = time.time()
    state = reset(jax.random.PRNGKey(0))
    state.info["command"] = jp.zeros(3)
    for _ in range(100):
        state.info["command"] = jp.zeros(3)
        state = step(state, jp.zeros(env.action_size))
    jax.block_until_ready(state)
    z = float(state.data.qpos[2])
    up = float(env.get_upvector(state.data)[2])
    print(f"after 2 s at rest: pelvis z {z:.3f} m, up {up:.3f}, done {float(state.done):.0f} ({time.time()-t0:.1f} s)")
    ok = (float(state.done) == 0.0 and z > 0.17 and up > 0.9
          and bool(jp.isfinite(state.obs["state"]).all())
          and bool(jp.isfinite(state.obs["privileged_state"]).all()))

    rand = make_domain_randomizer(env.mj_model)
    model_v, in_axes = rand(env.mjx_model, jax.random.split(jax.random.PRNGKey(1), 4))
    ok &= model_v.body_mass.shape[0] == 4 and in_axes.body_mass == 0
    print("domain randomizer: batched masses", model_v.body_mass.shape)
    print("ENV: OK" if ok else "ENV: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
