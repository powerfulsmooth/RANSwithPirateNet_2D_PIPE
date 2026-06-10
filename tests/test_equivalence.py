"""Numerical equivalence: refactored (jacfwd-based) r_point/losses vs the
original closure-based implementation.

The refactor in pipe_2d_komega/models.py replaced ~40 independent
grad(closure) sub-graphs per collocation point with 3 shared traces
(fields, jacfwd(fields), jacfwd(flux_vec)) to fix the XLA jit_step
compile-time explosion.  Physics must be bit-for-bit-level identical;
this test pins that down at rtol=1e-5 in float64.

Run:  python tests/test_equivalence.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)                                   # pinncore
sys.path.insert(0, os.path.join(_ROOT, "pipe_2d_komega"))   # models

import jax
jax.config.update("jax_enable_x64", True)   # compare in float64: fwd- vs rev-mode
                                            # AD round-off would mask real bugs in f32

import jax.numpy as jnp
from jax import grad, vmap
import numpy as np
import ml_collections

import models
from models import ALPHA, BETA, BETASTAR, SIGMA, SIGMASTAR, DELTA


# ---------------------------------------------------------------------------
# Reference implementation: verbatim copy of the ORIGINAL closure-based
# r_point / losses (pre-refactor, commit cc05134^).  Do not "improve" this.
# ---------------------------------------------------------------------------
def _raw(model, params, xi, eta):
    return model.state.apply_fn(params, jnp.array([xi, eta]))


def _U(model, params, xi, eta):
    return (1.0 - eta) * _raw(model, params, xi, eta)[0]


def _V(model, params, xi, eta):
    return eta * (1.0 - eta) * _raw(model, params, xi, eta)[1]


def _P(model, params, xi, eta):
    return _raw(model, params, xi, eta)[2]


def _K(model, params, xi, eta):
    return (1.0 - eta) * jax.nn.softplus(_raw(model, params, xi, eta)[3])


def _W(model, params, xi, eta):
    yp = (1.0 - eta) * model.Re_tau
    return jax.nn.softplus(_raw(model, params, xi, eta)[4]) + 6.0 / (BETA * (yp + DELTA) ** 2)


def _cyl_div_r(f_func, xi, eta_safe, Rt):
    d = grad(lambda e: e * f_func(xi, e))(eta_safe)
    return d / (eta_safe * Rt)


def r_point_ref(model, params, xi, eta):
    cx, cr, Rt = model.cx, model.cr, model.Re_tau

    U = lambda a, b: _U(model, params, a, b)
    V = lambda a, b: _V(model, params, a, b)
    P = lambda a, b: _P(model, params, a, b)
    K = lambda a, b: _K(model, params, a, b)
    W = lambda a, b: _W(model, params, a, b)
    nut = lambda a, b: K(a, b) / W(a, b)

    Ux = cx * grad(U, 0)(xi, eta); Ur = cr * grad(U, 1)(xi, eta)
    Vx = cx * grad(V, 0)(xi, eta); Vr = cr * grad(V, 1)(xi, eta)
    Px = cx * grad(P, 0)(xi, eta); Pr = cr * grad(P, 1)(xi, eta)
    Kx = cx * grad(K, 0)(xi, eta); Kr = cr * grad(K, 1)(xi, eta)
    Wx = cx * grad(W, 0)(xi, eta); Wr = cr * grad(W, 1)(xi, eta)

    Uc = U(xi, eta); Vc = V(xi, eta)
    Kc = K(xi, eta); Wc = W(xi, eta)
    nutc = Kc / Wc
    eta_safe = jnp.maximum(eta, 1e-8)
    Vr_plus = Vc / (eta_safe * Rt)

    S2 = 2.0 * Ux**2 + 2.0 * Vr**2 + (Ur + Vx)**2 + 2.0 * Vr_plus**2
    Pk = nutc * S2

    r_c = Ux + Vr + Vr_plus

    tau_xx = lambda a, b: 2.0 * (1.0 + nut(a, b)) * cx * grad(U, 0)(a, b)
    tau_rx = lambda a, b: (1.0 + nut(a, b)) * (cr * grad(U, 1)(a, b) + cx * grad(V, 0)(a, b))
    DiffU = (cx * grad(tau_xx, 0)(xi, eta)
             + _cyl_div_r(tau_rx, xi, eta_safe, Rt))
    r_x = Uc * Ux + Vc * Ur + Px - DiffU

    tau_rr = lambda a, b: 2.0 * (1.0 + nut(a, b)) * cr * grad(V, 1)(a, b)
    DiffV = (cx * grad(tau_rx, 0)(xi, eta)
             + _cyl_div_r(tau_rr, xi, eta_safe, Rt)
             - 2.0 * (1.0 + nutc) * Vc / (eta_safe * Rt) ** 2)
    r_r = Uc * Vx + Vc * Vr + Pr - DiffV

    gkx = lambda a, b: (1.0 + SIGMASTAR * nut(a, b)) * cx * grad(K, 0)(a, b)
    gkr = lambda a, b: (1.0 + SIGMASTAR * nut(a, b)) * cr * grad(K, 1)(a, b)
    DiffK = cx * grad(gkx, 0)(xi, eta) + _cyl_div_r(gkr, xi, eta_safe, Rt)
    r_k = Uc * Kx + Vc * Kr - Pk + BETASTAR * Kc * Wc - DiffK

    gwx = lambda a, b: (1.0 + SIGMA * nut(a, b)) * cx * grad(W, 0)(a, b)
    gwr = lambda a, b: (1.0 + SIGMA * nut(a, b)) * cr * grad(W, 1)(a, b)
    DiffW = cx * grad(gwx, 0)(xi, eta) + _cyl_div_r(gwr, xi, eta_safe, Rt)
    r_w = Uc * Wx + Vc * Wr - ALPHA * S2 + BETA * Wc**2 - DiffW

    return r_c, r_x, r_r, r_k, r_w


def losses_ref(model, params, batch):
    rc, rx, rr, rk, rw = vmap(
        lambda a, b: r_point_ref(model, params, a, b)
    )(batch[:, 0], batch[:, 1])

    Rt = model.Re_tau
    ARRt = model.AR * Rt
    rc_hat = rc * ARRt
    rx_hat = rx
    rr_hat = rr
    batch_xi, batch_eta = batch[:, 0], batch[:, 1]
    Kc_b = vmap(lambda a, b: _K(model, params, a, b))(batch_xi, batch_eta)
    Wc_b = vmap(lambda a, b: _W(model, params, a, b))(batch_xi, batch_eta)
    denom_k = jnp.mean(BETASTAR * Kc_b * Wc_b) + 1e-8
    denom_w = jnp.mean(BETA * Wc_b ** 2) + 1e-8
    rk_hat = rk / denom_k
    rw_hat = rw / denom_w

    ld = {
        "res_c": jnp.mean(rc_hat**2),
        "res_x": jnp.mean(rx_hat**2),
        "res_r": jnp.mean(rr_hat**2),
        "res_k": jnp.mean(rk_hat**2),
        "res_w": jnp.mean(rw_hat**2),
    }

    ec = 1.0e-3
    dU_c = grad(lambda e: _U(model, params, 0.5, e))(ec)
    dK_c = grad(lambda e: _K(model, params, 0.5, e))(ec)
    dW_c = grad(lambda e: _W(model, params, 0.5, e))(ec)
    ld["sym"] = dU_c**2 + dK_c**2 + dW_c**2

    U_in_hat = vmap(lambda e: _U(model, params, 0.0, e))(model.inlet_eta)
    V_in_hat = vmap(lambda e: _V(model, params, 0.0, e))(model.inlet_eta)
    K_in_hat = vmap(lambda e: _K(model, params, 0.0, e))(model.inlet_eta)
    ld["bc_inlet"] = (jnp.mean((U_in_hat - model.inlet_U) ** 2) / model.U_bulk_fd**2
                      + jnp.mean(V_in_hat**2)
                      + jnp.mean((K_in_hat - model.inlet_K) ** 2) / (model.k_in**2 + 1e-8))

    dU_out = vmap(lambda e: grad(lambda a: _U(model, params, a, e))(1.0))(model.inlet_eta)
    dV_out = vmap(lambda e: grad(lambda a: _V(model, params, a, e))(1.0))(model.inlet_eta)
    dK_out = vmap(lambda e: grad(lambda a: _K(model, params, a, e))(1.0))(model.inlet_eta)
    dW_out = vmap(lambda e: grad(lambda a: _W(model, params, a, e))(1.0))(model.inlet_eta)
    ld["bc_outlet"] = (jnp.mean(dU_out**2) + jnp.mean(dV_out**2)
                       + jnp.mean(dK_out**2) + jnp.mean(dW_out**2))

    ld["pgauge"] = _P(model, params, 1.0, 0.0) ** 2

    if "mass" in model.loss_keys:
        mass_xis = jnp.array([0.25, 0.5, 0.75, 1.0])
        eta_qm = jnp.linspace(1e-3, 0.999, 32)
        def _bulk(xi_a):
            Uq = vmap(lambda e: _U(model, params, xi_a, e))(eta_qm)
            return 2.0 * jnp.trapezoid(Uq * eta_qm, eta_qm)
        bulks = vmap(_bulk)(mass_xis)
        ld["mass"] = jnp.mean((bulks - model.U_bulk_fd) ** 2) / model.U_bulk_fd**2

    if "anchor" in model.loss_keys:
        def _anchor_err(xi_a):
            Ua = vmap(lambda e: _U(model, params, xi_a, e))(model.anchor_eta)
            return jnp.mean((Ua - model.anchor_U) ** 2)
        ld["anchor"] = jnp.mean(vmap(_anchor_err)(model.anchor_xis))

    if "anchor_k" in model.loss_keys:
        def _kanchor_err(xi_a):
            Ka = vmap(lambda e: _K(model, params, xi_a, e))(model.kanchor_eta)
            return jnp.mean((Ka - model.kanchor_K) ** 2)
        ld["anchor_k"] = jnp.mean(vmap(_kanchor_err)(model.anchor_xis)) / model.k_scale**2

    if "wall_shear" in model.loss_keys:
        def _tau_w(xi_a):
            dU = grad(lambda e: _U(model, params, xi_a, e))(1.0)
            return -(1.0 / model.Re_tau) * dU
        tw = vmap(_tau_w)(model.anchor_xis)
        ld["wall_shear"] = jnp.mean((tw - 1.0) ** 2)

    return ld


# ---------------------------------------------------------------------------
def small_config():
    sys.path.insert(0, os.path.join(_ROOT, "pipe_2d_komega", "configs"))
    from default import get_config
    config = get_config().unlock()
    config.arch.num_layers = 2
    config.arch.hidden_dim = 64
    config.arch.fourier_emb = ml_collections.ConfigDict(
        {"embed_scale": 5.0, "embed_dim": 64})
    return config.lock()


def main():
    config = small_config()
    model = models.Pipe2DKOmega(config)
    params = model.state.params

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    n = 32
    xi = jax.random.uniform(k1, (n,), minval=0.0, maxval=1.0)
    eta = jax.random.uniform(k2, (n,), minval=2e-3, maxval=0.995)
    batch = jnp.stack([xi, eta], axis=1)

    rtol = 1e-5
    failed = []

    # 1) per-point residual equivalence
    r_new = vmap(model.r_point, in_axes=(None, 0, 0))(params, xi, eta)
    r_old = vmap(lambda a, b: r_point_ref(model, params, a, b))(xi, eta)
    names = ["r_c", "r_x", "r_r", "r_k", "r_w"]
    for name, a, b in zip(names, r_new, r_old):
        ok = np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=1e-10)
        diff = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        rel = float(np.max(np.abs(np.asarray(a) - np.asarray(b))
                           / (np.abs(np.asarray(b)) + 1e-30)))
        print(f"{'PASS' if ok else 'FAIL'}  {name:6s}  max_abs_diff={diff:.3e}  max_rel={rel:.3e}")
        if not ok:
            failed.append(name)

    # 2) loss-dict equivalence
    ld_new = model.losses(params, batch)
    ld_old = losses_ref(model, params, batch)
    assert set(ld_new.keys()) == set(ld_old.keys()), (
        f"loss keys differ: {set(ld_new)} vs {set(ld_old)}")
    for k in sorted(ld_old.keys()):
        a, b = float(ld_new[k]), float(ld_old[k])
        ok = np.isclose(a, b, rtol=rtol, atol=1e-12)
        print(f"{'PASS' if ok else 'FAIL'}  losses[{k}]: new={a:.10e}  ref={b:.10e}")
        if not ok:
            failed.append(f"losses[{k}]")

    if failed:
        print(f"\nFAILED: {failed}")
        sys.exit(1)
    print("\nALL EQUIVALENCE CHECKS PASSED (rtol=1e-5, float64)")


if __name__ == "__main__":
    main()
