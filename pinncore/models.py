"""Base PINN model (mirrors jaxpi/models.py, trimmed to single-device).

TrainState carrying adaptive loss weights, grad-norm loss balancing, jitted
`step` and `update_weights`, and an abstract `losses(params, batch) -> dict`.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit
import optax
from flax.training import train_state
from flax import struct

from . import archs


class TrainState(train_state.TrainState):
    weights: dict = struct.field(pytree_node=True)


def _as_periodicity(cfg):
    if not cfg:
        return None
    return {"period": tuple(cfg.period), "axis": tuple(cfg.axis), "trainable": tuple(cfg.trainable)}


def _create_arch(cfg):
    if cfg.arch_name == "PirateNet":
        return archs.PirateNet(
            num_layers=cfg.num_layers,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.out_dim,
            activation=cfg.activation,
            nonlinearity=cfg.get("nonlinearity", 0.0),
            periodicity=_as_periodicity(cfg.get("periodicity", None)),
            fourier_emb=dict(cfg.fourier_emb) if cfg.fourier_emb else None,
            reparam=dict(cfg.reparam) if cfg.reparam else None,
        )
    raise NotImplementedError(f"arch '{cfg.arch_name}' not wired in _create_arch")


def _create_optimizer(cfg):
    lr = optax.exponential_decay(cfg.learning_rate, cfg.decay_steps, cfg.decay_rate)
    return optax.adam(lr, b1=cfg.beta1, b2=cfg.beta2, eps=cfg.eps)


class PINN:
    def __init__(self, config):
        self.config = config
        self.arch = _create_arch(config.arch)
        params = self.arch.init(jax.random.PRNGKey(config.seed), jnp.ones(config.input_dim))
        tx = _create_optimizer(config.optim)
        self.loss_keys = tuple(config.weighting.init_weights.keys())
        weights = {k: float(v) for k, v in config.weighting.init_weights.items()}
        self.state = TrainState.create(apply_fn=self.arch.apply, params=params, tx=tx, weights=weights)

    def losses(self, params, batch):
        raise NotImplementedError

    def weighted_loss(self, params, weights, batch):
        ld = self.losses(params, batch)
        total = sum(weights[k] * ld[k] for k in self.loss_keys)
        return total, ld

    @partial(jit, static_argnums=(0,))
    def compute_weights(self, params, batch):
        # Compute per-loss gradient norms in a single backward pass via jacobian.
        # Previously this ran one grad() per key (N full backward passes); now one
        # jax.jacobian call over a stacked scalar output covers all N terms at once.
        def stacked_losses(p):
            ld = self.losses(p, batch)
            return jnp.stack([ld[k] for k in self.loss_keys])

        J = jax.jacobian(stacked_losses)(params)
        # J is a pytree with the same structure as params but with an extra leading
        # axis of size len(loss_keys).  Slice each row to get per-term grad norm.
        norms = {
            k: optax.global_norm(jax.tree.map(lambda x: x[i], J)) + 1e-8
            for i, k in enumerate(self.loss_keys)
        }
        total = sum(norms.values())
        return {k: total / norms[k] for k in self.loss_keys}

    @partial(jit, static_argnums=(0,))
    def update_weights(self, state, batch):
        w_new = self.compute_weights(state.params, batch)
        m = self.config.weighting.momentum
        w = {k: m * state.weights[k] + (1 - m) * w_new[k] for k in self.loss_keys}
        return state.replace(weights=w)

    @partial(jit, static_argnums=(0,))
    def step(self, state, batch):
        g = grad(lambda p: self.weighted_loss(p, state.weights, batch)[0])(state.params)
        return state.apply_gradients(grads=g)
