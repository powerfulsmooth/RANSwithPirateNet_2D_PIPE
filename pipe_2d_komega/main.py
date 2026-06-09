"""Entry point for the 2D developing pipe flow example.

    python main.py --config=configs/default.py --workdir=./out
    python main.py --config=configs/default.py --config.physics.re_tau=395
    python main.py --config=configs/default.py --config.mode=eval
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import jax
jax.config.update("jax_default_matmul_precision", "highest")

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
