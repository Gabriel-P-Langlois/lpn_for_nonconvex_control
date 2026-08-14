"""Cross-section plots comparing the learned and true potential/prior.

Family-agnostic: takes a Problem (for the true curves) and the trained
network(s). Saves to a file so it runs headless.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plotstyle import apply as _apply_style
_apply_style()          # one font size for every figure
import numpy as np

from .recovery import cvx, recover_prior_route1, evaluate_learned_prior_G


def cross_section_points(a, spacing, dim):
    xi = np.linspace(-a, a, spacing)
    p1 = np.zeros((xi.size, dim)); p1[:, 0] = xi
    p2 = np.zeros((xi.size, dim))
    if dim > 1:
        p2[:, 1] = xi
    return xi, p1, p2


def plot_cross_sections(
    problem, model, a, spacing, dim, out_path, model_G=None, inv_alg="cvx_gd",
    alpha=0.0, title=None,
):
    """Overlay learned vs true convex potential and prior along two axes.

    The prior panel shows BOTH recoveries against the reference when ``model_G`` is
    given. Showing only One-shot recovery (the old behaviour) made the plots useless for
    the one comparison they exist to support.
    """
    xi, p1, p2 = cross_section_points(a, spacing, dim)
    axes_pts = [p1] + ([p2] if dim > 1 else [])

    ncol = len(axes_pts)
    fig, axs = plt.subplots(2, ncol, figsize=(6 * ncol, 9), squeeze=False)
    for j, pts in enumerate(axes_pts):
        # convex potential psi
        axs[0][j].plot(xi, cvx(pts, model), "-", lw=2, label=r"$\psi_\theta$ (LPN)")
        axs[0][j].plot(xi, problem.cvx_true(pts), "--", lw=1.5, label=r"$\psi$ (exact)")
        axs[0][j].set_title(f"Convex potential, axis {j + 1}, dim {dim}")
        axs[0][j].set_xlabel(f"$y_{j + 1}$")
        axs[0][j].legend(); axs[0][j].grid(True, alpha=0.3)

        # recovered prior: both recoveries vs the reference
        true_prior = problem.prior_true(pts)
        axs[1][j].plot(xi, true_prior, "k--", lw=1.5, label="$J$ (exact)")
        r1 = recover_prior_route1(pts, model, inv_alg, alpha=alpha)
        axs[1][j].plot(xi, r1, "-", lw=1.5, alpha=0.85,
                       label=rf"Iterative (inversion, $\alpha$={alpha:g})")
        if model_G is not None:
            r2 = evaluate_learned_prior_G(pts, model_G)
            axs[1][j].plot(xi, r2, "-", lw=1.5, alpha=0.85, label="One-shot ($G$)")
        axs[1][j].set_title(f"Recovered prior, axis {j + 1}, dim {dim}")
        axs[1][j].set_xlabel(f"$x_{j + 1}$")
        axs[1][j].legend(); axs[1][j].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
