"""Evaluation driver for 2D developing pipe flow."""
import os
import ml_collections

from pinncore.utils import load_params

import models
import utils


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    model = models.Pipe2DKOmega(config)
    params = load_params(model.state.params, os.path.join(workdir, config.wandb.name, "params.msgpack"))
    fig = os.path.join(workdir, config.wandb.name, "results_eval.png")
    utils.plot_results(model, params, config.physics.re_tau, fig)
    print(f"U_bulk_out+ = {model.bulk_velocity_plus(params, xi=1.0):.3f}")
    print(f"U_center_out+ = {float(model._U(params, 1.0, 0.0)):.3f}")
    print(f"saved {fig}")
