"""Compatibility shim: brax 0.14.2 still calls `jax.device_put_replicated`, which jax removed
from the public namespace in 0.10 (the implementation is still in `jax._src.api`). Import this
module before brax. Drop it once brax ships the `jax.device_put` port."""

import jax

for _name in ("device_put_replicated", "device_put_sharded"):
    try:
        getattr(jax, _name)
    except AttributeError:
        from jax._src import api as _api
        setattr(jax, _name, getattr(_api, _name))
