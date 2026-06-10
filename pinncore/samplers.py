"""Collocation samplers (mirrors jaxpi/samplers.py)."""
import jax.numpy as jnp
from jax import random


class UniformSampler1D:
    """Interior collocation points for a 1D problem (endpoints excluded)."""

    def __init__(self, lo, hi, batch_size, rng_key=random.PRNGKey(1234)):
        self.lo, self.hi = lo, hi
        self.batch_size = batch_size
        self.key = rng_key

    def __iter__(self):
        return self

    def __next__(self):
        self.key, sub = random.split(self.key)
        return random.uniform(sub, (self.batch_size,), minval=self.lo, maxval=self.hi)


class UniformSampler2D:
    """Draws (s, eta) collocation points in [s_lo,s_hi] x [eta_lo,eta_hi]."""

    def __init__(self, s_dom, eta_dom, batch_size, rng_key=random.PRNGKey(1234)):
        self.s_dom, self.eta_dom = s_dom, eta_dom
        self.batch_size = batch_size
        self.key = rng_key

    def __iter__(self):
        return self

    def __next__(self):
        self.key, k1, k2 = random.split(self.key, 3)
        s = random.uniform(k1, (self.batch_size,), minval=self.s_dom[0], maxval=self.s_dom[1])
        e = random.uniform(k2, (self.batch_size,), minval=self.eta_dom[0], maxval=self.eta_dom[1])
        return jnp.stack([s, e], axis=1)
