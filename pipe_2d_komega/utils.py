"""Plotting helpers for the 2D developing pipe flow."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KAPPA, B_LOG = 0.41, 5.0


def plot_results(model, params, re_tau, path):
    # profiles at inlet, mid, outlet
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

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

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
