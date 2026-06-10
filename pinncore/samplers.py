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


class EntranceClusteredSampler2D:
    """Uniform sampling mixed with an entrance/wall-clustered fraction.

    A `cluster_frac` share of each batch is drawn with
      s   = s_lo + (s_hi-s_lo) * u^s_power      (clusters toward the inlet s_lo)
      eta = eta_hi - (eta_hi-eta_lo) * u^e_power (clusters toward the wall eta_hi)
    which concentrates collocation points in the entrance shear layer where the
    residuals of a developing-flow problem are largest.  cluster_frac=0 reduces
    to UniformSampler2D.  The biased density acts as deliberate importance
    weighting of the residual loss (no reweighting applied).
    """

    def __init__(self, s_dom, eta_dom, batch_size, cluster_frac=0.4,
                 s_power=2.0, e_power=1.5, rng_key=random.PRNGKey(1234)):
        self.s_dom, self.eta_dom = s_dom, eta_dom
        self.batch_size = batch_size
        self.n_cluster = int(round(batch_size * cluster_frac))
        self.s_power, self.e_power = s_power, e_power
        self.key = rng_key

    def __iter__(self):
        return self

    def __next__(self):
        self.key, k1, k2, k3, k4 = random.split(self.key, 5)
        (s_lo, s_hi), (e_lo, e_hi) = self.s_dom, self.eta_dom
        n_u = self.batch_size - self.n_cluster
        s_u = random.uniform(k1, (n_u,), minval=s_lo, maxval=s_hi)
        e_u = random.uniform(k2, (n_u,), minval=e_lo, maxval=e_hi)
        u_s = random.uniform(k3, (self.n_cluster,))
        u_e = random.uniform(k4, (self.n_cluster,))
        s_c = s_lo + (s_hi - s_lo) * u_s ** self.s_power
        e_c = e_hi - (e_hi - e_lo) * u_e ** self.e_power
        s = jnp.concatenate([s_u, s_c])
        e = jnp.concatenate([e_u, e_c])
        return jnp.stack([s, e], axis=1)
