"""Post-training diagnostics for the 2D pipe PINN.

    python diagnostics.py --config=configs/default.py --workdir=./out

Produces <workdir>/<name>/diagnostics.png with four panels and prints
per-block PirateNet alpha values:

  1. log10|r_c|(xi, eta) heatmap        — WHERE continuity fails
       (corner spike at xi=0, eta=1 -> entrance/BC issue;
        global floor -> weighting or capacity issue)
  2. radial FFT of r_c at xi slices     — spectral-bias evidence
       (power trapped at low k -> network cannot represent the scales)
  3. U+ profiles vs Reichardt target    — physical accuracy at a glance
  4. per-loss-term gradient cosines     — gradient-conflict diagnosis
       (entries near -1 -> terms actively fight each other)

PirateNet alpha per block: alpha ~ 0 means the block is still an identity
map (no nonlinear depth used) — a structural learning-failure signal.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))
sys.path.insert(0, _HERE)

import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flax.traverse_util import flatten_dict

from pinncore.utils import load_params
from pinncore.samplers import UniformSampler2D

import models


def alpha_values(params):
    """Per-block PirateNet residual-gate alpha (init 0 = identity block)."""
    p = params.get("params", params)
    flat = flatten_dict(dict(p))
    return {"/".join(k[:-1]): float(np.ravel(v)[0])
            for k, v in flat.items() if k[-1] == "alpha"}


def residual_grid(model, params, nx=96, ne=120, chunk=4096):
    print(f"[1/4] continuity-residual heatmap ({nx}x{ne} pts, 2nd-order AD; "
          "first chunk includes JIT compile, please wait)...", flush=True)
    xi = jnp.linspace(1e-3, 1.0 - 1e-3, nx)
    eta = jnp.linspace(2e-3, 0.995, ne)
    XI, ETA = jnp.meshgrid(xi, eta)
    r_fn = jax.jit(vmap(model.r_point, in_axes=(None, 0, 0)))
    xs, es = XI.ravel(), ETA.ravel()
    rc_parts = []
    for i in range(0, xs.shape[0], chunk):
        rc, *_ = r_fn(params, xs[i:i + chunk], es[i:i + chunk])
        rc_parts.append(np.array(rc))
        print(f"      {min(i + chunk, xs.shape[0])}/{xs.shape[0]} points done", flush=True)
    rc_all = np.concatenate(rc_parts)
    return np.array(XI), np.array(ETA), rc_all.reshape(ne, nx)


def radial_fft(model, params, xi_slices=(0.05, 0.25, 0.5, 0.9), ne=256):
    print("[2/4] radial FFT of r_c at xi slices...", flush=True)
    eta = jnp.linspace(2e-3, 0.995, ne)
    r_fn = jax.jit(vmap(model.r_point, in_axes=(None, None, 0)))
    out = {}
    for xs in xi_slices:
        rc, *_ = r_fn(params, jnp.asarray(xs), eta)
        spec = np.abs(np.fft.rfft(np.array(rc))) ** 2
        out[xs] = spec / (spec.sum() + 1e-30)
    return out


def grad_cosine_matrix(model, params, batch):
    """Pairwise cosine similarity between per-loss-term parameter gradients."""
    print("[4/4] per-term gradient cosine matrix (jacobian over params)...", flush=True)
    keys = list(model.loss_keys)

    def stacked(p):
        ld = model.losses(p, batch)
        return jnp.stack([ld[k] for k in keys])

    J = jax.jacobian(stacked)(params)
    leaves = jax.tree.leaves(J)
    rows = np.concatenate(
        [np.array(x).reshape(len(keys), -1) for x in leaves], axis=1)
    norms = np.linalg.norm(rows, axis=1, keepdims=True) + 1e-30
    C = (rows / norms) @ (rows / norms).T
    return keys, C


def run(config, workdir):
    model = models.Pipe2DKOmega(config)
    out_dir = os.path.join(workdir, config.wandb.name)
    params = load_params(model.state.params, os.path.join(out_dir, "params.msgpack"))

    # ---- alpha readout ----
    alphas = alpha_values(params)
    print("PirateNet alpha per block (0 = identity / unused depth):")
    for name, a in sorted(alphas.items()):
        print(f"  {name}: {a:+.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ---- 1. continuity residual heatmap ----
    XI, ETA, RC = residual_grid(model, params)
    c = axes[0, 0].pcolormesh(XI, 1 - ETA, np.log10(np.abs(RC) + 1e-12),
                              cmap="magma", shading="auto")
    fig.colorbar(c, ax=axes[0, 0], label="log10|r_c|")
    axes[0, 0].set_xlabel("xi = x/L"); axes[0, 0].set_ylabel("1-eta (wall distance)")
    axes[0, 0].set_title("continuity residual (raw, wall units)")

    # ---- 2. radial FFT of r_c ----
    for xs, spec in radial_fft(model, params).items():
        axes[0, 1].semilogy(spec[:80], label=f"xi={xs:g}")
    axes[0, 1].set_xlabel("radial wavenumber index k")
    axes[0, 1].set_ylabel("normalised power")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_title("r_c radial spectrum (low-k pile-up = spectral bias)")

    # ---- 3. U+ profiles vs Reichardt ----
    print("[3/4] U+ profiles vs Reichardt target...", flush=True)
    eta_p = np.linspace(1e-4, 1 - 1e-5, 300)
    U_target = model._reichardt((1.0 - eta_p) * model.Re_tau)
    for xs, col in [(0.0, "g"), (0.5, "b"), (1.0, "r")]:
        _, yp, Up, _, _ = model.profile_at(params, xi=xs, n=300)
        axes[1, 0].plot(eta_p, Up, col, lw=1.5, label=f"PINN xi={xs:g}")
    axes[1, 0].plot(eta_p, U_target, "k--", lw=1.5, label="Reichardt target")
    axes[1, 0].set_xlabel("eta = r/R"); axes[1, 0].set_ylabel("U+")
    axes[1, 0].legend(fontsize=8); axes[1, 0].set_title("U+ vs fully-developed target")

    # ---- 4. gradient cosine similarity ----
    sampler = iter(UniformSampler2D((0.0, 1.0), (2e-3, 0.995), 256,
                                    rng_key=jax.random.PRNGKey(0)))
    keys, C = grad_cosine_matrix(model, params, next(sampler))
    im = axes[1, 1].imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[1, 1].set_xticks(range(len(keys))); axes[1, 1].set_yticks(range(len(keys)))
    axes[1, 1].set_xticklabels(keys, rotation=60, fontsize=7)
    axes[1, 1].set_yticklabels(keys, fontsize=7)
    fig.colorbar(im, ax=axes[1, 1], label="grad cosine")
    axes[1, 1].set_title("per-term gradient cosine (-1 = conflict)")

    fig.tight_layout()
    path = os.path.join(out_dir, "diagnostics.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved {path}")
    return path


if __name__ == "__main__":
    from absl import app, flags
    from ml_collections import config_flags

    FLAGS = flags.FLAGS
    flags.DEFINE_string("workdir", ".", "Directory with trained params.")
    config_flags.DEFINE_config_file(
        "config", "./configs/default.py", "Training configuration.", lock_config=True)

    def main(argv):
        run(FLAGS.config, FLAGS.workdir)

    app.run(main)
