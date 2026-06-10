"""Plotting helpers for the 2D developing pipe flow."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KAPPA, B_LOG = 0.41, 5.0


def plot_results(model, params, re_tau, path):
    # profiles at inlet, mid, outlet + entrance-flow development diagnostics
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for xi, label, col in [(0.0, "inlet", "g"), (0.5, "mid", "b"), (1.0, "outlet", "r")]:
        eta, yp, Up, Kp, Wp = model.profile_at(params, xi=xi)
        axes[0, 0].semilogx(yp, Up, color=col, lw=1.5, label=f"PINN {label}")
        axes[0, 1].semilogy(eta, np.maximum(Kp, 1e-12), color=col, lw=1.5, label=f"k+ {label}")
        axes[0, 2].semilogy(eta, np.maximum(Wp, 1e-12), color=col, lw=1.5, label=f"ω+ {label}")

    # reference log law
    yv = np.logspace(0, np.log10(max(5.0, 0.4 * re_tau)), 200)
    axes[0, 0].semilogx(yv[yv < 11], yv[yv < 11], "k:", label="U+=y+")
    axes[0, 0].semilogx(yv[yv > 11], (1 / KAPPA) * np.log(yv[yv > 11]) + B_LOG, "k--", label="log law")
    axes[0, 0].set_xlabel("y+"); axes[0, 0].set_ylabel("U+"); axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_title(f"U+ profiles, Re_tau={re_tau:g}")
    axes[0, 1].set_xlabel("eta = r/R"); axes[0, 1].legend(fontsize=7); axes[0, 1].set_title("k+")
    axes[0, 2].set_xlabel("eta = r/R"); axes[0, 2].legend(fontsize=7); axes[0, 2].set_title("omega+")

    # 2D field
    XI, ETA, U = model.field_grid(params)
    c = axes[1, 0].contourf(XI, 1 - ETA, U, levels=40, cmap="jet")
    fig.colorbar(c, ax=axes[1, 0], label="U+")
    axes[1, 0].set_xlabel("xi = x/L"); axes[1, 0].set_ylabel("1-eta (wall distance)")
    axes[1, 0].set_title("U+ field (x-development)")

    # bulk velocity along x
    xi_arr = np.linspace(0.0, 1.0, 30)
    Ub = [model.bulk_velocity_plus(params, xi=float(x)) for x in xi_arr]
    axes[1, 1].plot(xi_arr, Ub, "b-o", ms=3)
    axes[1, 1].set_xlabel("xi = x/L"); axes[1, 1].set_ylabel("U_bulk+")
    axes[1, 1].set_title("Bulk velocity development")

    # V+ field
    from jax import vmap
    import jax.numpy as jnp
    xi = jnp.linspace(0.0, 1.0, 60)
    eta_v = jnp.linspace(1e-4, 1 - 1e-5, 80)
    XI2, ETA2 = jnp.meshgrid(xi, eta_v)
    Vf = vmap(lambda a, b: model._V(params, a, b))(XI2.ravel(), ETA2.ravel())
    Vf = np.array(Vf).reshape(80, 60)
    c2 = axes[1, 2].contourf(np.array(XI2), 1 - np.array(ETA2), Vf, levels=40, cmap="RdBu_r")
    fig.colorbar(c2, ax=axes[1, 2], label="V+")
    axes[1, 2].set_xlabel("xi = x/L"); axes[1, 2].set_title("V+ (radial velocity)")

    # ---- row 3: development diagnostics (raw U contours hide ~5% changes) ----
    # centreline acceleration
    xi_c, Uc = model.centerline_profile(params)
    axes[2, 0].plot(xi_c, Uc, "b-", lw=2)
    axes[2, 0].axhline(model.anchor_U[0], color="k", ls="--", lw=1, label="FD target")
    axes[2, 0].axhline(model.U_plug, color="g", ls=":", lw=1, label="inlet plug")
    axes[2, 0].set_xlabel("xi = x/L"); axes[2, 0].set_ylabel("U_center+")
    axes[2, 0].legend(fontsize=7); axes[2, 0].set_title("centreline development")

    # wall shear decay (entrance signature: tau_w+ > 1 at inlet, -> 1)
    xi_t, tw = model.wall_shear_profile(params)
    axes[2, 1].plot(xi_t, tw, "r-", lw=2)
    axes[2, 1].axhline(1.0, color="k", ls="--", lw=1, label="FD tau_w+=1")
    tw_in = model.U_plug / model.delta_in
    axes[2, 1].axhline(tw_in, color="g", ls=":", lw=1, label=f"inlet {tw_in:.2f}")
    axes[2, 1].set_xlabel("xi = x/L"); axes[2, 1].set_ylabel("tau_w+")
    axes[2, 1].legend(fontsize=7); axes[2, 1].set_title("wall shear development")

    # deviation from the inlet profile: shows WHERE the flow develops
    U_in_grid = model.inlet_profile(np.array(ETA[:, 0]))[:, None]   # (ne,1)
    dU = U - U_in_grid
    vmax = max(abs(dU.min()), abs(dU.max()), 1e-6)
    c3 = axes[2, 2].contourf(XI, 1 - ETA, dU, levels=40, cmap="RdBu_r",
                             vmin=-vmax, vmax=vmax)
    fig.colorbar(c3, ax=axes[2, 2], label="U+ - U_inlet+")
    axes[2, 2].set_xlabel("xi = x/L"); axes[2, 2].set_ylabel("1-eta (wall distance)")
    axes[2, 2].set_title("development field (U - U_inlet)")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
