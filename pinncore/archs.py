"""Network architectures (mirrors jaxpi/archs.py, 'pirate' branch).

PirateNet: optional periodic embedding + Fourier embedding + random weight
factorization + PI-modified bottleneck blocks with an adaptive residual gate
(alpha, init 0 -> identity mapping at initialisation).
"""
from typing import Callable, Union, Dict

import flax.linen as nn
from flax.core.frozen_dict import freeze
from jax import random
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal, normal, zeros, constant

ACTIVATIONS = {"tanh": jnp.tanh, "gelu": nn.gelu, "swish": nn.swish, "sin": jnp.sin}


def get_activation(name):
    if name not in ACTIVATIONS:
        raise NotImplementedError(f"activation '{name}' not supported")
    return ACTIVATIONS[name]


def _weight_fact(init_fn, mean, stddev):
    def init(key, shape):
        k1, k2 = random.split(key)
        w = init_fn(k1, shape)
        g = jnp.exp(mean + normal(stddev)(k2, (shape[-1],)))
        return g, w / g
    return init


class PeriodEmbs(nn.Module):
    period: tuple
    axis: tuple
    trainable: tuple

    def setup(self):
        params = {}
        for i, tr in enumerate(self.trainable):
            params[f"period_{i}"] = (
                self.param(f"period_{i}", constant(self.period[i]), ()) if tr else self.period[i]
            )
        self.period_params = freeze(params)

    def __call__(self, x):
        # Note: @nn.compact is intentionally absent here.  Parameters are declared
        # in setup(); mixing setup() with @nn.compact on __call__ violates Flax
        # conventions and may silently duplicate or ignore parameters depending on
        # the Flax version (B4 from review-round-1).
        y = []
        for i, xi in enumerate(x):
            if i in self.axis:
                p = self.period_params[f"period_{self.axis.index(i)}"]
                y.extend([jnp.cos(p * xi), jnp.sin(p * xi)])
            else:
                y.append(xi)
        return jnp.hstack(y)


class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2))
        return jnp.concatenate([jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1)


class Dense(nn.Module):
    features: int
    reparam: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        if self.reparam is None:
            kernel = self.param("kernel", glorot_normal(), (x.shape[-1], self.features))
        elif self.reparam["type"] == "weight_fact":
            g, v = self.param(
                "kernel",
                _weight_fact(glorot_normal(), self.reparam["mean"], self.reparam["stddev"]),
                (x.shape[-1], self.features),
            )
            kernel = g * v
        else:
            raise NotImplementedError(self.reparam)
        bias = self.param("bias", zeros, (self.features,))
        return jnp.dot(x, kernel) + bias


class PIModifiedBottleneck(nn.Module):
    hidden_dim: int
    out_features: int
    activation: str
    nonlinearity: float
    reparam: Union[None, Dict]

    @nn.compact
    def __call__(self, x, u, v):
        act = get_activation(self.activation)
        identity = x
        x = act(Dense(self.hidden_dim, self.reparam)(x)); x = x * u + (1 - x) * v
        x = act(Dense(self.hidden_dim, self.reparam)(x)); x = x * u + (1 - x) * v
        x = act(Dense(self.out_features, self.reparam)(x))
        alpha = self.param("alpha", constant(self.nonlinearity), (1,))
        return alpha * x + (1 - alpha) * identity


class PirateNet(nn.Module):
    num_layers: int = 3
    hidden_dim: int = 64
    out_dim: int = 1
    activation: str = "tanh"
    nonlinearity: float = 0.0
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        act = get_activation(self.activation)
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)
        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)
        u = act(Dense(self.hidden_dim, self.reparam)(x))
        v = act(Dense(self.hidden_dim, self.reparam)(x))
        for _ in range(self.num_layers):
            x = PIModifiedBottleneck(
                self.hidden_dim, x.shape[-1], self.activation, self.nonlinearity, self.reparam
            )(x, u, v)
        return Dense(self.out_dim, self.reparam)(x)
