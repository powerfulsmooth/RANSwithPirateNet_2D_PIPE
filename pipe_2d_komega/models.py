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
  Inlet xi=0  : blunted plug U+ = U_plug * tanh(y+/delta_in), V+=0,
                k+ = k_in * (U/U_plug)^2.  The tanh shear layer removes the
                plug/no-slip corner singularity at (xi=0, eta=1) and is
                representable by the hard BC U=(1-eta)*raw (raw stays finite
                at the wall), so the inlet is enforced over the FULL radius —
                no unconstrained annular band, no mass-flux hole.  U_plug is
                set so the inlet bulk equals the fully-developed (Reichardt)
                bulk: in wall units the bulk velocity is fixed by the friction
                law, an independent u_inlet would over-specify the problem.
  Outlet xi=1 : dU/dxi=dV/dxi=dk/dxi=domega/dxi=0  (Neumann)
  Axis eta=0  : dU/dr=0, dK/dr=0, dW/dr=0  (already implied by hard V BC)
  Pressure    : p=0 at (xi=1, eta=0)  (gauge)
  Mass        : bulk velocity 2*int U eta d(eta) == U_bulk_fd at several xi
                (integral mass-conservation constraint)
  Anchor      : Reichardt profile (smooth viscous->log blend) at several xi,
                with eta points spanning wall layer AND centreline, so the
                outer profile / U_center is pinned (the old 2-point log-law
                anchor left the centreline unconstrained at low Re_tau).
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
        self.k_in = float(config.physics.k_inlet)      # inlet peak k+ scale
        self.delta_in = float(config.physics.delta_inlet)  # inlet shear layer (wall units)

        # ---- fully-developed target: Reichardt's wall law ----
        eta_q = np.linspace(0.0, 1.0, 401)
        yp_q = (1.0 - eta_q) * self.Re_tau
        U_fd = self._reichardt(yp_q)
        # bulk velocity is FIXED by Re_tau in wall units (friction law);
        # both the inlet mass flux and the mass-conservation loss target it
        self.U_bulk_fd = 2.0 * float(np.trapezoid(U_fd * eta_q, eta_q))

        # ---- inlet: blunted plug, mass-matched to the FD bulk ----
        shape = np.tanh(yp_q / self.delta_in)
        bulk_shape = 2.0 * float(np.trapezoid(shape * eta_q, eta_q))
        self.U_plug = self.U_bulk_fd / bulk_shape
        inlet_eta = np.linspace(0.02, 0.98, 24)            # full radius
        U_in_prof = self.U_plug * np.tanh((1.0 - inlet_eta) * self.Re_tau / self.delta_in)
        self.inlet_eta = jnp.array(inlet_eta)
        self.inlet_U = jnp.array(U_in_prof)
        # k decays to 0 at the wall like U^2 (constant turbulence intensity)
        self.inlet_K = jnp.array(self.k_in * (U_in_prof / self.U_plug) ** 2)

        # ---- anchor: Reichardt profile across the radius incl. centreline ----
        # Stations retreated to xi={0.75, 1.0}: with stations from xi=0.5 the
        # V+ field showed development artificially truncated at xi=0.5.
        self.anchor_xis = jnp.array([0.75, 1.0])
        anchor_eta = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.70, 0.85])
        self.anchor_eta = jnp.array(anchor_eta)
        self.anchor_U = jnp.array(self._reichardt((1.0 - anchor_eta) * self.Re_tau))

        # ---- k anchor: approximate fully-developed k-omega k+ profile ----
        # Without it the U anchor alone admits the trivial branch k->0 (observed:
        # downstream k+ ~ 0.04-0.1 vs physical O(1-4), nu_t+ ~ 1/20 of physical).
        # Log-layer equilibrium of the k-omega model: -u'v'+ ~ tau+ = eta (linear
        # total stress in a pipe) and k+ = tau+/sqrt(beta*), with near-wall
        # damping (1 - exp(-y+/8))^2 reproducing k+ ~ y+^2 as y+ -> 0.
        k_eta = np.array([0.30, 0.50, 0.70, 0.85, 0.925, 0.95])
        k_yp = (1.0 - k_eta) * self.Re_tau
        k_t = (1.0 / np.sqrt(BETASTAR)) * k_eta * (1.0 - np.exp(-k_yp / 8.0)) ** 2
        self.kanchor_eta = jnp.array(k_eta)
        self.kanchor_K = jnp.array(k_t)
        self.k_scale = 1.0 / np.sqrt(BETASTAR)   # log-layer k+ scale (~3.33)

    @staticmethod
    def _reichardt(yp):
        """Reichardt's smooth wall law (viscous sublayer -> log layer blend)."""
        return ((1.0 / KAPPA) * np.log(1.0 + KAPPA * yp)
                + 7.8 * (1.0 - np.exp(-yp / 11.0) - (yp / 11.0) * np.exp(-yp / 3.0)))

    # ---- reparametrised fields: ONE network forward -> all 5 outputs ----
    # Hard BCs baked in: U,K zero at wall via (1-eta); V zero at axis+wall via
    # eta*(1-eta); k>=0 via softplus; omega wall singularity structural.
    def fields(self, params, xi, eta):
        out = self.state.apply_fn(params, jnp.array([xi, eta]))
        yp = (1.0 - eta) * self.Re_tau
        U = (1.0 - eta) * out[0]
        V = eta * (1.0 - eta) * out[1]
        P = out[2]
        K = (1.0 - eta) * jax.nn.softplus(out[3])
        W = jax.nn.softplus(out[4]) + 6.0 / (BETA * (yp + DELTA) ** 2)
        return jnp.array([U, V, P, K, W])

    # scalar accessors (diagnostics / evaluator API; avoid in hot loops)
    def _U(self, params, xi, eta):
        return self.fields(params, xi, eta)[0]

    def _V(self, params, xi, eta):
        return self.fields(params, xi, eta)[1]

    def _P(self, params, xi, eta):
        return self.fields(params, xi, eta)[2]

    def _K(self, params, xi, eta):
        return self.fields(params, xi, eta)[3]

    def _W(self, params, xi, eta):
        return self.fields(params, xi, eta)[4]

    # ---- PDE residuals at one collocation point ----
    # Compile-friendly formulation: instead of ~40 independent grad(closure)
    # sub-graphs (one per derivative, each re-running the network), the network
    # is traced through THREE shared computations:
    #   1. fields(z)            -> values of all 5 fields
    #   2. jacfwd(fields)(z)    -> all 10 first derivatives at once
    #   3. jacfwd(flux_vec)(z)  -> all 8 second-order flux derivatives at once
    # The physics (residual definitions, B1 fix alpha*S2, S2, cylindrical
    # divergence (1/r+) d(r+ f)/dr+ = d(eta*f)/deta / (eta_safe*Rt)) is
    # numerically identical to the closure-based version (see
    # tests/test_equivalence.py).  eta_safe is used for BOTH the evaluation
    # point and the denominator (original "JAX BUG fix" intent preserved);
    # for eta >= 1e-8 (always true: sampler min is 2e-3) eta_safe == eta.
    def r_point(self, params, xi, eta):
        cx, cr, Rt = self.cx, self.cr, self.Re_tau
        eta_safe = jnp.maximum(eta, 1e-8)
        z = jnp.array([xi, eta_safe])

        f = lambda zz: self.fields(params, zz[0], zz[1])

        F = f(z)                 # (5,)  field values
        J = jax.jacfwd(f)(z)     # (5,2) d/dxi (col 0), d/deta (col 1)

        Uc, Vc, Kc, Wc = F[0], F[1], F[3], F[4]
        Ux, Ur = cx * J[0, 0], cr * J[0, 1]
        Vx, Vr = cx * J[1, 0], cr * J[1, 1]
        Px, Pr = cx * J[2, 0], cr * J[2, 1]
        Kx, Kr = cx * J[3, 0], cr * J[3, 1]
        Wx, Wr = cx * J[4, 0], cr * J[4, 1]

        nutc = Kc / Wc
        Vr_plus = Vc / (eta_safe * Rt)   # V / r+

        # strain rate invariant S^2 = 2 SijSij (axisymmetric cylindrical)
        S2 = 2.0 * Ux**2 + 2.0 * Vr**2 + (Ur + Vx)**2 + 2.0 * Vr_plus**2
        Pk = nutc * S2

        # ---- continuity ----
        r_c = Ux + Vr + Vr_plus  # = cx Uxi + cr Veta + V/r+

        # ---- all 8 second-order fluxes in ONE traced function ----
        # entries 0-3: x-direction fluxes (need d/dxi)
        # entries 4-7: eta * radial fluxes (need d/deta; cylindrical form)
        def flux_vec(zz):
            Fv = f(zz)
            Jv = jax.jacfwd(f)(zz)
            nut_v = Fv[3] / Fv[4]
            Ux_, Ur_ = cx * Jv[0, 0], cr * Jv[0, 1]
            Vx_, Vr_ = cx * Jv[1, 0], cr * Jv[1, 1]
            Kx_, Kr_ = cx * Jv[3, 0], cr * Jv[3, 1]
            Wx_, Wr_ = cx * Jv[4, 0], cr * Jv[4, 1]
            e = zz[1]
            tau_xx = 2.0 * (1.0 + nut_v) * Ux_
            tau_rx = (1.0 + nut_v) * (Ur_ + Vx_)
            tau_rr = 2.0 * (1.0 + nut_v) * Vr_
            gkx = (1.0 + SIGMASTAR * nut_v) * Kx_
            gkr = (1.0 + SIGMASTAR * nut_v) * Kr_
            gwx = (1.0 + SIGMA * nut_v) * Wx_
            gwr = (1.0 + SIGMA * nut_v) * Wr_
            return jnp.array([tau_xx, tau_rx, gkx, gwx,
                              e * tau_rx, e * tau_rr, e * gkr, e * gwr])

        G = jax.jacfwd(flux_vec)(z)   # (8,2)
        inv_rp = 1.0 / (eta_safe * Rt)

        DiffU = cx * G[0, 0] + G[4, 1] * inv_rp
        # hoop stress: 2(1+nut)*V/r+ enters as -tau_theta_theta/r+
        DiffV = cx * G[1, 0] + G[5, 1] * inv_rp - 2.0 * (1.0 + nutc) * Vc * inv_rp**2
        DiffK = cx * G[2, 0] + G[6, 1] * inv_rp
        DiffW = cx * G[3, 0] + G[7, 1] * inv_rp

        r_x = Uc * Ux + Vc * Ur + Px - DiffU
        r_r = Uc * Vx + Vc * Vr + Pr - DiffV
        r_k = Uc * Kx + Vc * Kr - Pk + BETASTAR * Kc * Wc - DiffK
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
        # destruction scales broadcast-safe (use batch-mean for stability);
        # one vmapped fields() forward gives both K and W
        F_b = vmap(lambda a, b: self.fields(params, a, b))(batch[:, 0], batch[:, 1])
        Kc_b, Wc_b = F_b[:, 3], F_b[:, 4]
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

        # ---- axis symmetry at eta->0: single jvp in the eta direction ----
        ec = 1.0e-3
        _, dF_c = jax.jvp(lambda e: self.fields(params, 0.5, e), (ec,), (1.0,))
        ld["sym"] = dF_c[0] ** 2 + dF_c[3] ** 2 + dF_c[4] ** 2   # dU, dK, dW

        # ---- inlet BC (xi=0): blunted plug over the FULL radius ----
        # U_in = U_plug * tanh(y+/delta_in) is wall-compatible (raw stays finite
        # under the hard (1-eta) factor), so no unconstrained annular band.
        F_in = vmap(lambda e: self.fields(params, 0.0, e))(self.inlet_eta)   # (24,5)
        ld["bc_inlet"] = (jnp.mean((F_in[:, 0] - self.inlet_U) ** 2) / self.U_bulk_fd**2
                          + jnp.mean(F_in[:, 1] ** 2)
                          + jnp.mean((F_in[:, 3] - self.inlet_K) ** 2) / (self.k_in**2 + 1e-8))

        # ---- outlet BC (xi=1): Neumann d/dxi of U,V,K,W = 0 ----
        # one jvp in the xi direction per eta point gives all 5 derivatives
        def _dxi_fields(e):
            _, d = jax.jvp(lambda a: self.fields(params, a, e), (1.0,), (1.0,))
            return d
        D_out = vmap(_dxi_fields)(self.inlet_eta)   # (24,5)
        ld["bc_outlet"] = (jnp.mean(D_out[:, 0] ** 2) + jnp.mean(D_out[:, 1] ** 2)
                           + jnp.mean(D_out[:, 3] ** 2) + jnp.mean(D_out[:, 4] ** 2))

        # ---- pressure gauge: p=0 at (xi=1, eta=0) ----
        ld["pgauge"] = self.fields(params, 1.0, 0.0)[2] ** 2

        # ---- integral mass conservation: bulk velocity == U_bulk_fd at several xi ----
        if "mass" in self.loss_keys:
            mass_xis = jnp.array([0.25, 0.5, 0.75, 1.0])
            eta_qm = jnp.linspace(1e-3, 0.999, 32)
            def _bulk(xi_a):
                Uq = vmap(lambda e: self.fields(params, xi_a, e)[0])(eta_qm)
                return 2.0 * jnp.trapezoid(Uq * eta_qm, eta_qm)
            bulks = vmap(_bulk)(mass_xis)
            ld["mass"] = jnp.mean((bulks - self.U_bulk_fd) ** 2) / self.U_bulk_fd**2

        # ---- optional anchor: Reichardt profile at several xi (branch selection
        #      + pins the centreline, unlike the old 2-point log-law anchor) ----
        if "anchor" in self.loss_keys:
            def _anchor_err(xi_a):
                Ua = vmap(lambda e: self.fields(params, xi_a, e)[0])(self.anchor_eta)
                return jnp.mean((Ua - self.anchor_U) ** 2)
            ld["anchor"] = jnp.mean(vmap(_anchor_err)(self.anchor_xis))

        # ---- k anchor: blocks the trivial laminar branch (k -> 0) that the
        #      U-only anchor admits ----
        if "anchor_k" in self.loss_keys:
            def _kanchor_err(xi_a):
                Ka = vmap(lambda e: self.fields(params, xi_a, e)[3])(self.kanchor_eta)
                return jnp.mean((Ka - self.kanchor_K) ** 2)
            ld["anchor_k"] = jnp.mean(vmap(_kanchor_err)(self.anchor_xis)) / self.k_scale**2

        # ---- wall shear consistency: dU+/dy+ = 1 at the wall (definition of
        #      u_tau; exact where the flow is fully developed -> anchor xis).
        #      The collapsed run had dU+/dy+|wall < 1 (wall-unit inconsistency).
        if "wall_shear" in self.loss_keys:
            def _tau_w(xi_a):
                _, d = jax.jvp(lambda e: self.fields(params, xi_a, e), (1.0,), (1.0,))
                return -self.cr * d[0]   # dU+/dy+ = -(1/Rt) dU/deta at eta=1
            tw = vmap(_tau_w)(self.anchor_xis)
            ld["wall_shear"] = jnp.mean((tw - 1.0) ** 2)

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
