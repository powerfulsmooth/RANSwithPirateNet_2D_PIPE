"""Training driver for the 2D developing pipe flow."""
import math
import os
import time

import jax
import ml_collections

from pinncore.samplers import UniformSampler2D
from pinncore.utils import save_params

import models
import utils


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    use_wandb = config.wandb.mode not in ("disabled",)
    if use_wandb:
        import wandb
        wandb.init(project=config.wandb.project, name=config.wandb.name, mode=config.wandb.mode)

    model = models.Pipe2DKOmega(config)
    evaluator = models.Pipe2DEvaluator(config, model)
    sampler = iter(
        UniformSampler2D(
            (config.training.xi_min, config.training.xi_max),
            (config.training.eta_min, config.training.eta_max),
            config.training.batch_size,
            rng_key=jax.random.PRNGKey(config.seed),
        )
    )

    print("Waiting for JIT...")
    t0 = time.time()
    for step in range(config.training.max_steps):
        batch = next(sampler)
        model.state = model.step(model.state, batch)
        if config.weighting.scheme in ("grad_norm", "ntk"):
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        if step % config.logging.log_every_steps == 0 or step == config.training.max_steps - 1:
            log_dict = evaluator(model.state, batch)
            # jnp scalars -> python floats (wandb can render jnp values as empty charts)
            log_dict = {k: float(v) for k, v in log_dict.items()}
            dt = time.time() - t0; t0 = time.time()
            msg = " | ".join(f"{k}={v:.2e}" for k, v in log_dict.items())
            print(f"step {step:6d} ({dt:5.1f}s) | {msg}")
            if use_wandb:
                import wandb
                wandb.log(log_dict, step=step)

            # NaN guard
            total_loss = sum(v for k, v in log_dict.items() if k.startswith("res_"))
            if not math.isfinite(total_loss):
                print(f"NaN/Inf detected at step {step}, aborting.")
                break

    params = model.state.params
    out_dir = os.path.join(workdir, config.wandb.name)
    os.makedirs(out_dir, exist_ok=True)
    save_params(params, os.path.join(out_dir, "params.msgpack"))
    fig = os.path.join(out_dir, "results.png")
    utils.plot_results(model, params, config.physics.re_tau, fig)
    print(f"\nU_bulk_out+ = {model.bulk_velocity_plus(params, xi=1.0):.3f}")
    print(f"U_center_out+ = {float(model._U(params, 1.0, 0.0)):.3f}")
    print(f"saved {fig}")
    return model
