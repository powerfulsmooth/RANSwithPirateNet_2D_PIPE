"""Entry point for the 2D developing pipe flow example.

    python main.py --config=configs/default.py --workdir=./out
    python main.py --config=configs/default.py --config.physics.re_tau=395
    python main.py --config=configs/default.py --config.mode=eval
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# pinncore lives one level up in both repo layouts:
#   RANSwithPirateNet_2D/examples/pipe_2d_komega  -> ../.. has pinncore  (original)
#   RANSwithPirateNet_2D_PIPE/pipe_2d_komega      -> ..   has pinncore  (PIPE repo)
_PARENT = os.path.abspath(os.path.join(_HERE, ".."))
_GRANDPARENT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _GRANDPARENT)
sys.path.insert(0, _PARENT)
sys.path.insert(0, _HERE)

# Reduce XLA autotuning cost on slow GPUs (Colab T4) — must be set before jax import.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=1")

import jax
jax.config.update("jax_default_matmul_precision", "highest")

# Persistent compilation cache: the first session compiles, later sessions reuse.
# Override the location with JAX_CACHE_DIR (e.g. a mounted Drive path on Colab).
_CACHE_DIR = os.environ.get("JAX_CACHE_DIR", os.path.join(_HERE, ".jax_cache"))
try:
    jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)
except Exception:
    pass  # older jax without the persistent-cache options

from absl import app, flags
from ml_collections import config_flags

import train
import eval as eval_mod

FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", ".", "Directory to store model data.")
config_flags.DEFINE_config_file("config", "./configs/default.py", "Training configuration.", lock_config=True)


def main(argv):
    if FLAGS.config.mode == "train":
        train.train_and_evaluate(FLAGS.config, FLAGS.workdir)
    elif FLAGS.config.mode == "eval":
        eval_mod.evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
