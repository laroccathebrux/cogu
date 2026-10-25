"""Domain randomisation for the Cogu env (per-environment model parameters).

Randomised, per episode batch: floor friction, link masses, extra pelvis payload (the body,
battery and cap are not in the CAD yet), joint friction/armature, servo stiffness and damping
(kp / kv of the position actuators) and the rest pose offset (qpos0). The actuator *latency*
is randomised inside the env (`action_delay`), not here.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx


def make_domain_randomizer(mj_model: mujoco.MjModel):
    floor_id = mj_model.geom("floor").id
    pelvis_id = mj_model.body("pelvis").id
    nu = mj_model.nu

    def domain_randomize(model: mjx.Model, rng: jax.Array):
        @jax.vmap
        def rand(rng):
            keys = jax.random.split(rng, 8)
            geom_friction = model.geom_friction.at[floor_id, 0].set(
                jax.random.uniform(keys[0], minval=0.4, maxval=1.0))
            frictionloss = model.dof_frictionloss.at[6:].set(
                model.dof_frictionloss[6:] * jax.random.uniform(keys[1], (nu,), minval=0.8, maxval=1.2))
            armature = model.dof_armature.at[6:].set(
                model.dof_armature[6:] * jax.random.uniform(keys[2], (nu,), minval=1.0, maxval=1.05))
            body_mass = model.body_mass * jax.random.uniform(keys[3], (model.nbody,), minval=0.9, maxval=1.1)
            body_mass = body_mass.at[pelvis_id].add(jax.random.uniform(keys[4], minval=-0.02, maxval=0.10))
            qpos0 = model.qpos0.at[7:].add(jax.random.uniform(keys[5], (nu,), minval=-0.03, maxval=0.03))
            # position actuator: gainprm[0] = kp, biasprm[1] = -kp, biasprm[2] = -kv
            kp_scale = jax.random.uniform(keys[6], (nu,), minval=0.8, maxval=1.2)
            kv_scale = jax.random.uniform(keys[7], (nu,), minval=0.8, maxval=1.2)
            gainprm = model.actuator_gainprm.at[:, 0].multiply(kp_scale)
            biasprm = model.actuator_biasprm.at[:, 1].multiply(kp_scale)
            biasprm = biasprm.at[:, 2].multiply(kv_scale)
            return geom_friction, frictionloss, armature, body_mass, qpos0, gainprm, biasprm

        fields = ("geom_friction", "dof_frictionloss", "dof_armature", "body_mass", "qpos0",
                  "actuator_gainprm", "actuator_biasprm")
        values = rand(rng)
        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace({f: 0 for f in fields})
        model = model.tree_replace(dict(zip(fields, values)))
        return model, in_axes

    return domain_randomize
