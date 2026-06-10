"""Compile-time benchmark: old (closure-based) vs new (jacfwd-based) jit_step.

Usage: python tests/_bench_compile.py {old|new}
Runs one variant per process so each gets a cold JIT cache.
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "pipe_2d_komega"))
sys.path.insert(0, os.path.join(_ROOT, "pipe_2d_komega", "configs"))

import jax
import jax.numpy as jnp
import ml_collections
from default import get_config

variant = sys.argv[1]  # "old" | "new"

if variant == "old":
    # materialise the pre-refactor models.py from git history
    old_path = os.path.join(_HERE, "_models_old_tmp.py")
    if not os.path.exists(old_path):
        import subprocess
        src = subprocess.check_output(
            ["git", "-C", _ROOT, "show", "cc05134^:pipe_2d_komega/models.py"])
        with open(old_path, "wb") as fh:
            fh.write(src)
    import importlib.util
    spec = importlib.util.spec_from_file_location("models_old", old_path)
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
else:
    import models

config = get_config().unlock()
config.arch.num_layers = 2
config.arch.hidden_dim = 64
config.arch.fourier_emb = ml_collections.ConfigDict(
    {"embed_scale": 5.0, "embed_dim": 64})
config = config.lock()

model = models.Pipe2DKOmega(config)
key = jax.random.PRNGKey(0)
batch = jnp.stack([
    jax.random.uniform(key, (256,), minval=0.0, maxval=1.0),
    jax.random.uniform(key, (256,), minval=2e-3, maxval=0.995),
], axis=1)

t0 = time.time()
state = model.step(model.state, batch)
jax.block_until_ready(state.params)
t_compile = time.time() - t0

t0 = time.time()
state = model.step(state, batch)
jax.block_until_ready(state.params)
t_run = time.time() - t0

print(f"{variant}: first step (JIT compile) = {t_compile:.1f}s | second step = {t_run*1000:.0f}ms")
