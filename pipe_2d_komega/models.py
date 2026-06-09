"""2D developing turbulent pipe flow, standard Wilcox k-omega (1988).

Network input (xi, eta): xi = x/L in [0,1] (streamwise, NOT periodic),
eta = r/R in [0,1] (eta=0 axis, eta=1 wall).
Outputs (wall units): U+(xi,eta), V+(xi,eta), p+(xi,eta), k+(xi,eta), omega+(xi,eta).

The flow is NOT fully developed: it enters as plug flow and develops downstream.
Converging to fully developed k-omega validates the 2D machinery against the 1D case.

Coordinate mapping (wall units, ν→1, u_τ→1):
  r+ = eta * Re_tau,   x+ = xi * AR * Re_tau   (AR = L/R, aspect ratio)
  cx = 1 / (AR * Re_tau)  [∂/∂x+ = cx ∂/∂xi]
  cr = 1 / Re_tau         [∂/∂r+ = cr ∂/∂eta]

Residuals (cylindrical, axisymmetric, wall units):
  continuity:
    cx Uxi + cr Veta + V/(eta*Re_tau) = 0

  x-momentum:
    U(cx Uxi) + V(cr Ueta) + cx Pxi
      - cx d[2(1+nut)(cx Uxi)]/dxi
      - (1/r+) d[r+(1+nut)(cr Ueta + cx Vxi)]/dr+  = 0

  r-momentum:
    U(cx Vxi) + V(cr Veta) + cr Peta
      - cx d[(1+nut)(cx Vxi + cr Ueta)]/dxi
      - (1/r+) d[r+*2(1+nut)(cr Veta)]/dr+
      + 2(1+nut)*V / r+^2                          = 0

  k:
    U(cx Kxi) + V(cr Keta) - Pk + b* K W
      - cx d[(1+s*nut)(cx Kxi)]/dxi
      - (1/r+) d[r+(1+s*nut)(cr Keta)]/dr+         = 0

  omega:
    U(cx Wxi) + V(cr Weta) - alpha*S2 + beta W^2
      - cx d[(1+s*nut)(cx Wxi)]/dxi
      - (1/r+) d[r+(1+s*nut)(cr Weta)]/dr+         = 0

where:
  nut = K/W
  S2  = 2(cx Uxi)^2 + 2(cr Veta)^2 + (cr Ueta + cx Vxi)^2 + 2*(V/r+)^2
  Pk  = nut * S2
  alpha*(W/K)*Pk = alpha*S2   [B1-type fix: (W/K)*(K/W)*S2 = S2, avoids 0/0 at K->0]

Cylindrical divergence: (1/r+) d(r+ f)/dr+ = (1/(eta*Rt)) * d(eta*f)/d(eta)
  Implemented via a single JAX grad of the product (eta*f), which avoids the
  numerical split f/r + df/dr (each term O(1/eta), their difference is O(1)).

Hard BCs:
  U+(wall)=0  via (1-eta) factor
  V+(wall)=0  via (1-eta) factor; V+(axis)=0 via eta factor  =>  eta*(1-eta)
  k+(wall)=0  via (1-eta) factor + softplus for positivity
  omega wall singularity structural: softplus(net) + 6/(beta*(y++delta)^2)

Soft BCs (as loss terms):
  Inlet xi=0  : U+=U_in, V+=0, k+=k_in, dk/dr=0, domega/dr=0
  Outlet xi=1 : dU/dxi=dV/dxi=dk/dxi=domega/dxi=0  (Neumann)
  Axis eta=0  : dU/dr=0, dK/dr=0, dW/dr=0  (already implied by hard V BC)
  Pressure    : p=0 at (xi=1, eta=0)  (gauge)
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np

from pinncore.models import PINN
from pinncore.evaluator import BaseEvaluator

ALPHA, BETA, BETASTAR = 5.0 / 9.0, 3.0 / 40.0, 9.0 / 100.0
SIGMA, SIGMASTAR = 0.5, 0.5
KAPPA, B_LOG = 0.41, 5.0
DELTA = 1.0e-6


class Pipe2DKOmega(PINN):
    def __init__(self, config):
        super().__init__(config)
        self.Re_tau = float(config.physics.re_tau)
        self.AR = float(config.physics.aspect_ratio)   # L/R
        self.cx = 1.0 / (self.AR * self.Re_tau)
        self.cr = 1.0 / self.Re_tau
        self.U_in = float(config.physics.u_inlet)      # plug inlet velocity in wall units
        self.k_in = float(config.physics.k_inlet)      # inlet k+ (e.g. 0.01*U_in^2)
        # log-law anchor points for branch selection (optional)
        yps = np.array([yp for yp in (30.0, 60.0, 120.0, 250.0) if yp < 0.4 * self.Re_tau])
        self.anchor_eta = jnp.array(1.0 - yps / self.Re_tau)
        self.anchor_U = jnp.array((1.0 / KAPPA) * np.log(yps) + B_LOG)
        self.anchor_xi = 1.0  # anchor at downstream end (should approach fully-developed)

    # ---- reparametrised fields (scalar xi, eta -> scalars) ----
    def _raw(self, params, xi, eta):
        return self.state.apply_fn(params, jnp.array([xi, eta]))

    def _U(self, params, xi, eta):
        return (1.0 - eta) * self._raw(params, xi, eta)[0]

    def _V(self, params, xi, eta):
        # zero at axis (eta=0) and wall (eta=1)
        return eta * (1.0 - eta) * self._raw(params, xi, eta)[1]

    def _P(self, params, xi, eta):
        return self._raw(params, xi, eta)[2]

    def _K(self, params, xi, eta):
        return (1.0 - eta) * jax.nn.softplus(self._raw(params, xi, eta)[3])

    def _W(self, params, xi, eta):
        yp = (1.0 - eta) * self.Re_tau
        return jax.nn.softplus(self._raw(params, xi, eta)[4]) + 6.0 / (BETA * (yp + DELTA) ** 2)

    # ---- cylindrical divergence helper ----
    # (1/r+) d(r+ f) / dr+ = (1/(eta*Rt)) * d(eta * f(xi,eta)) / d(eta)
    # Implemented as a single JAX grad of (eta*f) to avoid the 1/r split.
    # eta_safe is used for BOTH the gradient evaluation point and the denominator
    # so that the autodiff graph is consistent (JAX BUG fix: eta vs eta_safe).
    @staticmethod
    def _cyl_div_r(f_func, xi, eta_safe, Rt):
        d = grad(lambda e: e * f_func(xi, e))(eta_safe)
        return d / (eta_safe * Rt)

    # ---- PDE residuals at one collocation point ----
    def r_point(self, params, xi, eta):
        cx, cr, Rt = self.cx, self.cr, self.Re_tau

        U = lambda a, b: self._U(params, a, b)
        V = lambda a, b: self._V(params, a, b)
        P = lambda a, b: self._P(params, a, b)
        K = lambda a, b: self._K(params, a, b)
        W = lambda a, b: self._W(params, a, b)
        nut = lambda a, b: K(a, b) / W(a, b)

        # first-order derivatives (wall units)
        Ux = cx * grad(U, 0)(xi, eta); Ur = cr * grad(U, 1)(xi, eta)
        Vx = cx * grad(V, 0)(xi, eta); Vr = cr * grad(V, 1)(xi, eta)
        Px = cx * grad(P, 0)(xi, eta); Pr = cr * grad(P, 1)(xi, eta)
        Kx = cx * grad(K, 0)(xi, eta); Kr = cr * grad(K, 1)(xi, eta)
        Wx = cx * grad(W, 0)(xi, eta); Wr = cr * grad(W, 1)(xi, eta)

        Uc = U(xi, eta); Vc = V(xi, eta)
        Kc = K(xi, eta); Wc = W(xi, eta)
        nutc = Kc / Wc
        eta_safe = jnp.maximum(eta, 1e-8)
        Vr_plus = Vc / (eta_safe * Rt)   # V / r+

        # strain rate invariant S^2 = 2 SijSij (axisymmetric cylindrical)
        S2 = 2.0 * Ux**2 + 2.0 * Vr**2 + (Ur + Vx)**2 + 2.0 * Vr_plus**2
        Pk = nutc * S2

        # ---- continuity ----
        r_c = Ux + Vr + Vr_plus  # = cx Uxi + cr Veta + V/r+

        # ---- x-momentum diffusion ----
        # tau_xx = 2(1+nut)*Ux,  tau_rx = (1+nut)*(Ur+Vx)
        tau_xx = lambda a, b: 2.0 * (1.0 + nut(a, b)) * cx * grad(U, 0)(a, b)
        tau_rx = lambda a, b: (1.0 + nut(a, b)) * (cr * grad(U, 1)(a, b) + cx * grad(V, 0)(a, b))
        DiffU = (cx * grad(tau_xx, 0)(xi, eta)
                 + self._cyl_div_r(tau_rx, xi, eta_safe, Rt))
        r_x = Uc * Ux + Vc * Ur + Px - DiffU

        # ---- r-momentum diffusion ----
        # tau_rr = 2(1+nut)*Vr,  tau_xr = tau_rx  (symmetric)
        # hoop stress: 2(1+nut)*V/r+ enters as -tau_theta_theta/r+
        tau_rr = lambda a, b: 2.0 * (1.0 + nut(a, b)) * cr * grad(V, 1)(a, b)
        DiffV = (cx * grad(tau_rx, 0)(xi, eta)
                 + self._cyl_div_r(tau_rr, xi, eta_safe, Rt)
                 - 2.0 * (1.0 + nutc) * Vc / (eta_safe * Rt) ** 2)
        r_r = Uc * Vx + Vc * Vr + Pr - DiffV

        # ---- k diffusion ----
        gkx = lambda a, b: (1.0 + SIGMASTAR * nut(a, b)) * cx * grad(K, 0)(a, b)
        gkr = lambda a, b: (1.0 + SIGMASTAR * nut(a, b)) * cr * grad(K, 1)(a, b)
        DiffK = cx * grad(gkx, 0)(xi, eta) + self._cyl_div_r(gkr, xi, eta_safe, Rt)
        r_k = Uc * Kx + Vc * Kr - Pk + BETASTAR * Kc * Wc - DiffK

        # ---- omega diffusion ----
        gwx = lambda a, b: (1.0 + SIGMA * nut(a, b)) * cx * grad(W, 0)(a, b)
        gwr = lambda a, b: (1.0 + SIGMA * nut(a, b)) * cr * grad(W, 1)(a, b)
        DiffW = cx * grad(gwx, 0)(xi, eta) + self._cyl_div_r(gwr, xi, eta_safe, Rt)
        # B1-type fix: alpha*(W/K)*Pk = alpha*S2  (avoids 0/0 when K->0)
        r_w = Uc * Wx + Vc * Wr - ALPHA * S2 + BETA * Wc**2 - DiffW

        return r_c, r_x, r_r, r_k, r_w

    # ---- loss dict ----
    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        rc, rx, rr, rk, rw = vmap(self.r_point, in_axes=(None, 0, 0))(
            params, batch[:, 0], batch[:, 1]
        )
        # Residual scale normalisation (B2-type fix for 2D cylindrical):
        #   r_c is O(cx * U_in) ~ O(1.6e-3); multiply by AR*Rt to bring to O(U_in) ~ O(18)
        #   r_x, r_r: left as-is (O(1) when nut O(Re_tau))
        #   r_k, r_w: relative residuals (divide by destruction scales)
        Rt = self.Re_tau
        ARRt = self.AR * Rt
        rc_hat = rc * ARRt
        rx_hat = rx
        rr_hat = rr
        # destruction scales broadcast-safe (use batch-mean for stability)
        batch_xi, batch_eta = batch[:, 0], batch[:, 1]
        Kc_b = vmap(lambda a, b: self._K(params, a, b))(batch_xi, batch_eta)
        Wc_b = vmap(lambda a, b: self._W(params, a, b))(batch_xi, batch_eta)
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

        # ---- axis symmetry at eta->0 ----
        ec = 1.0e-3
        dU_c = grad(lambda e: self._U(params, 0.5, e))(ec)
        dK_c = grad(lambda e: self._K(params, 0.5, e))(ec)
        dW_c = grad(lambda e: self._W(params, 0.5, e))(ec)
        ld["sym"] = dU_c**2 + dK_c**2 + dW_c**2

        # ---- inlet BC (xi=0): plug flow U+=U_in, V+=0, k+=k_in ----
        # eta limited to [0.05, 0.80] to avoid the near-wall region where the hard BC
        # U=(1-eta)*raw requires raw=U_in/(1-eta) -> large as eta->1.
        eta_pts = jnp.linspace(0.05, 0.80, 16)
        U_in_hat = vmap(lambda e: self._U(params, 0.0, e))(eta_pts)
        V_in_hat = vmap(lambda e: self._V(params, 0.0, e))(eta_pts)
        K_in_hat = vmap(lambda e: self._K(params, 0.0, e))(eta_pts)
        ld["bc_inlet"] = (jnp.mean((U_in_hat - self.U_in) ** 2) / self.U_in**2
                          + jnp.mean(V_in_hat**2)
                          + jnp.mean((K_in_hat - self.k_in) ** 2) / (self.k_in**2 + 1e-8))

        # ---- outlet BC (xi=1): Neumann dU/dxi=dV/dxi=dK/dxi=dW/dxi=0 ----
        dU_out = vmap(lambda e: grad(lambda a: self._U(params, a, e))(1.0))(eta_pts)
        dV_out = vmap(lambda e: grad(lambda a: self._V(params, a, e))(1.0))(eta_pts)
        dK_out = vmap(lambda e: grad(lambda a: self._K(params, a, e))(1.0))(eta_pts)
        dW_out = vmap(lambda e: grad(lambda a: self._W(params, a, e))(1.0))(eta_pts)
        ld["bc_outlet"] = (jnp.mean(dU_out**2) + jnp.mean(dV_out**2)
                           + jnp.mean(dK_out**2) + jnp.mean(dW_out**2))

        # ---- pressure gauge: p=0 at (xi=1, eta=0) ----
        ld["pgauge"] = self._P(params, 1.0, 0.0) ** 2

        # ---- optional log-law anchor (multiple xi to prevent laminar collapse) ----
        if "anchor" in self.loss_keys:
            anchor_xis = jnp.array([0.5, 0.75, 1.0])
            def _anchor_err(xi_a):
                Ua = vmap(lambda e: self._U(params, xi_a, e))(self.anchor_eta)
                return jnp.mean((Ua - self.anchor_U) ** 2)
            ld["anchor"] = jnp.mean(vmap(_anchor_err)(anchor_xis))

        return ld

    # ---- diagnostics ----
    def profile_at(self, params, xi=1.0, n=300):
        """Radial profile at a given xi (default: outlet)."""
        eta = jnp.linspace(1e-4, 1 - 1e-5, n)
        yp = (1.0 - eta) * self.Re_tau
        Up = vmap(lambda e: self._U(params, xi, e))(eta)
        Kp = vmap(lambda e: self._K(params, xi, e))(eta)
        Wp = vmap(lambda e: self._W(params, xi, e))(eta)
        return np.array(eta), np.array(yp), np.array(Up), np.array(Kp), np.array(Wp)

    def field_grid(self, params, nx=60, ne=80):
        """(xi, eta) -> U+ field for contour plot."""
        xi = jnp.linspace(0.0, 1.0, nx)
        eta = jnp.linspace(1e-4, 1 - 1e-5, ne)
        XI, ETA = jnp.meshgrid(xi, eta)
        U = vmap(lambda a, b: self._U(params, a, b))(XI.ravel(), ETA.ravel()).reshape(ne, nx)
        return np.array(XI), np.array(ETA), np.array(U)

    def bulk_velocity_plus(self, params, xi=1.0, n=300):
        """Bulk (area-averaged) velocity at a given xi, ∫₀¹ U+(r) 2r dr."""
        eta = jnp.linspace(1e-4, 1 - 1e-4, n)
        Up = vmap(lambda e: self._U(params, xi, e))(eta)
        return 2.0 * float(jnp.trapezoid(Up * eta, eta))


class Pipe2DEvaluator(BaseEvaluator):
    def __call__(self, state, batch):
        self.log_dict = super().__call__(state, batch)
        if self.config.logging.log_errors:
            self.log_dict["U_bulk_out"] = self.model.bulk_velocity_plus(state.params, xi=1.0)
            self.log_dict["U_bulk_in"] = self.model.bulk_velocity_plus(state.params, xi=0.0)
            self.log_dict["U_center_out"] = float(self.model._U(state.params, 1.0, 0.0))
        return self.log_dict
