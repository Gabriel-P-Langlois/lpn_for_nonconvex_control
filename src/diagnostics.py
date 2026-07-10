"""In-distribution diagnostics, for use where cross-sections through the origin
become meaningless.

A cross-section along axis 1 fixes the other d-1 coordinates at EXACTLY zero.
Under the uniform measure on [-A,A]^d that slice has volume fraction
(eps/A)^{d-1} -- about 1e-15 at d=16 and 1e-63 at d=64 -- so no training point
lies anywhere near it. The picture is pure extrapolation while the reported RMSE
is measured on typical points. Both are correct; they describe different regions.

Everything below stays on the data manifold, or says plainly that it does not.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .recovery import (
    cvx,
    prox,
    evaluate_learned_prior_G,
    recover_prior_route1,
)


def exact_grad_psi(problem, y, h=1e-5):
    """Central-difference grad of problem.cvx_true; 2d evaluations, exact enough."""
    y = np.asarray(y, dtype=float)
    g = np.zeros_like(y)
    for j in range(y.shape[1]):
        e = np.zeros(y.shape[1])
        e[j] = h
        g[:, j] = (problem.cvx_true(y + e) - problem.cvx_true(y - e)) / (2 * h)
    return g


def _finish(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- 1 ----------
def plot_conditional_cross_section(
    problem, model, model_G, a, train_a, dim, out_path, n_draws=6, seed=7,
    spacing=120, alpha=0.0, with_route1=True, invert_iters=20000
):
    """Vary x_1; draw the OTHER coordinates from the training distribution.

    The closest in-distribution analogue of the usual cross-section. For a
    separable prior each draw differs from the exact 1D profile by the additive
    constant contributed by the frozen coordinates, so we CENTRE every curve by
    subtracting its value at x_1 = 0. What remains is the x_1-dependence, on the
    data manifold. Compare with the dashed exact curves, centred the same way.

    For a SEPARABLE prior (QuadraticL1, NegL1) the centred exact curves collapse
    onto one line, so any spread among the learned curves is the network's error.
    For a non-separable prior (Minplus, ConcaveQuad, MaxPlus) the exact curves do
    NOT collapse -- the background couples to x_1 -- and the figure should be read
    as learned-vs-exact PER DRAW, not as a single reference.

    Both routes are drawn. Route 1's inversions for every background are batched
    into ONE solve, so the figure costs a single Route-1 solve rather than
    ``n_draws`` of them.
    """
    rng = np.random.default_rng(seed)
    xi = np.linspace(-a, a, spacing)

    pts_all, zero_all = [], []
    for _ in range(n_draws):
        bg = rng.uniform(-train_a, train_a, dim - 1) if dim > 1 else np.zeros(0)
        pts = np.tile(np.concatenate([[0.0], bg]), (spacing, 1))
        pts[:, 0] = xi
        zero = pts.copy()
        zero[:, 0] = 0.0
        pts_all.append(pts)
        zero_all.append(zero)
    P = np.concatenate(pts_all)
    Z = np.concatenate(zero_all)

    psi_l = (cvx(P, model) - cvx(Z, model)).reshape(n_draws, spacing)
    psi_e = (problem.cvx_true(P) - problem.cvx_true(Z)).reshape(n_draws, spacing)
    j_e = (problem.prior_true(P) - problem.prior_true(Z)).reshape(n_draws, spacing)
    j2 = (evaluate_learned_prior_G(P, model_G)
          - evaluate_learned_prior_G(Z, model_G)).reshape(n_draws, spacing)
    if with_route1:
        j1 = (recover_prior_route1(P, model, "cvx_gd", alpha=alpha, max_iters=invert_iters)
              - recover_prior_route1(Z, model, "cvx_gd", alpha=alpha,
                                     max_iters=invert_iters)).reshape(n_draws, spacing)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for k in range(n_draws):
        first = k == 0
        axs[0].plot(xi, psi_l[k], "-", lw=1.2, color="C0", alpha=0.75,
                    label=r"$\psi_\theta$" if first else None)
        axs[0].plot(xi, psi_e[k], "--", lw=1.0, color="k", alpha=0.6,
                    label="exact" if first else None)
        axs[1].plot(xi, j_e[k], "--", lw=1.0, color="k", alpha=0.6,
                    label="exact" if first else None)
        if with_route1:
            axs[1].plot(xi, j1[k], "-", lw=1.2, color="C0", alpha=0.75,
                        label=rf"Route 1 ($\alpha$={alpha:g})" if first else None)
        axs[1].plot(xi, j2[k], "-", lw=1.2, color="C1", alpha=0.75,
                    label="Route 2" if first else None)

    axs[0].set_title(rf"$\psi$, centred, {n_draws} typical backgrounds (dim {dim})")
    axs[0].set_xlabel("$y_1$")
    axs[1].set_title(rf"Recovered prior, centred (dim {dim})")
    axs[1].set_xlabel("$x_1$")
    for ax in axs:
        ax.legend(); ax.grid(True, alpha=0.3)
    return _finish(fig, out_path)


# ---------------------------------------------------------------- 2 ----------
def plot_typical_ray(problem, model, model_G, a, dim, out_path, alpha=0.0,
                     spacing=120, invert_iters=20000):
    """Profile along the bulk direction u = 1/sqrt(d), clipped to the box.

    Coordinates are all equal to t/sqrt(d), so |t| <= a*sqrt(d) keeps the ray in
    [-a,a]^d. Most of the ray lies where the samples are; only the neighbourhood
    of the origin is atypical.
    """
    u = np.ones(dim) / np.sqrt(dim)
    t = np.linspace(-a * np.sqrt(dim), a * np.sqrt(dim), spacing)
    pts = t[:, None] * u[None, :]

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    axs[0].plot(t, cvx(pts, model), "-", lw=1.8, label=r"$\psi_\theta$")
    axs[0].plot(t, problem.cvx_true(pts), "--", lw=1.4, label=r"$\psi$ exact")
    axs[0].set_title(rf"$\psi$ along $t\,\mathbf{{1}}/\sqrt{{d}}$ (dim {dim})")

    axs[1].plot(t, problem.prior_true(pts), "k--", lw=1.4, label="$J$ exact")
    axs[1].plot(t, recover_prior_route1(pts, model, "cvx_gd", alpha=alpha,
                                       max_iters=invert_iters), "-",
                lw=1.4, label=rf"Route 1 ($\alpha$={alpha:g})")
    axs[1].plot(t, evaluate_learned_prior_G(pts, model_G), "-", lw=1.4, label="Route 2")
    axs[1].set_title(rf"Prior along $t\,\mathbf{{1}}/\sqrt{{d}}$ (dim {dim})")
    for ax in axs:
        ax.set_xlabel("$t$"); ax.legend(); ax.grid(True, alpha=0.3)
    return _finish(fig, out_path)


# ---------------------------------------------------------------- 3 ----------
def plot_pred_vs_true(problem, x_test, r1, r2, out_path, alpha=0.0):
    """Recovered J against exact J over the test set, with the identity line.

    Dimension-agnostic; shows bias and spread together; cannot be gamed by the
    choice of a slice. ``r1``/``r2`` are the recoveries, passed in so the
    expensive Route-1 inversion is done ONCE per run, not once per figure.
    """
    j_true = problem.prior_true(x_test)
    lo, hi = float(min(j_true.min(), r1.min(), r2.min())), float(max(j_true.max(), r1.max(), r2.max()))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True)
    for ax, est, nm in ((axs[0], r1, rf"Route 1 ($\alpha$={alpha:g})"), (axs[1], r2, "Route 2")):
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
        ax.scatter(j_true, est, s=6, alpha=0.35)
        rmse = float(np.sqrt(np.mean((est - j_true) ** 2)))
        ax.set_title(f"{nm}   RMSE {rmse:.4f}")
        ax.set_xlabel("$J(x)$ exact"); ax.legend(); ax.grid(True, alpha=0.3)
    axs[0].set_ylabel(r"$\widehat{J}(x)$")
    return _finish(fig, out_path)


# ---------------------------------------------------------------- 5 ----------
def plot_prox_scatter(problem, model, y_train_like, out_path, max_pts=4000):
    """Learned grad psi against the exact prox, coordinatewise, in-distribution.

    A diagnostic of the SHARED first network, not of Route 1. Route 2 does not
    bypass psi_theta: G is trained on the conjugate samples y_k = grad psi(x_k),
    G_k = <y_k,x_k> - psi(x_k), so every error in psi_theta propagates into G's
    targets. This plot therefore bounds BOTH routes -- Route 1 because it inverts
    grad psi, Route 2 because it is fitted to grad psi's image.
    Coordinates are pooled: one point per (sample, coordinate).
    """
    y = np.asarray(y_train_like)[:max_pts]
    g_learned = prox(y, model).ravel()
    g_exact = exact_grad_psi(problem, y).ravel()
    lo, hi = float(min(g_exact.min(), g_learned.min())), float(max(g_exact.max(), g_learned.max()))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
    ax.scatter(g_exact, g_learned, s=4, alpha=0.2)
    rmse = float(np.sqrt(np.mean((g_learned - g_exact) ** 2)))
    ax.set_xlabel(r"exact $\nabla\psi(y)=\mathrm{prox}_J(y)$")
    ax.set_ylabel(r"learned $\nabla\psi_\theta(y)$")
    ax.set_title("Prox map of the SHARED first network, coordinatewise\n"
                 f"(bounds both routes)   RMSE {rmse:.4f}".format(rmse=rmse))
    ax.legend(); ax.grid(True, alpha=0.3)
    return _finish(fig, out_path)


# ---------------------------------------------------------------- 5b ---------
def plot_preimage_scatter(problem, model_G, x_test, out_path):
    """Route 2's own check: grad G(x) against the EXACT preimage y*(x).

    Route 2 rests on the identity grad G = (grad psi)^{-1}, since G ~= psi*. The
    exact preimage is available analytically (problem.preimage), so scattering
    grad G(x) against y*(x) tests the identity Route 2 actually relies on. It is
    the direct counterpart of the prox scatter, and costs one forward pass.
    Coordinates are pooled: one point per (sample, coordinate).
    """
    y_learned = prox(x_test, model_G).ravel()        # grad G(x)
    y_exact = np.asarray(problem.preimage(x_test)).ravel()
    lo = float(min(y_exact.min(), y_learned.min()))
    hi = float(max(y_exact.max(), y_learned.max()))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
    ax.scatter(y_exact, y_learned, s=4, alpha=0.2, color="C1")
    rmse = float(np.sqrt(np.mean((y_learned - y_exact) ** 2)))
    ax.set_xlabel(r"exact preimage $y^\star(x)=(\nabla\psi)^{-1}(x)$")
    ax.set_ylabel(r"learned $\nabla G(x)$")
    ax.set_title("Route 2: does $\\nabla G$ invert $\\nabla\\psi$?\n"
                 f"coordinatewise, on the query box   RMSE {rmse:.4f}")
    ax.legend(); ax.grid(True, alpha=0.3)
    return _finish(fig, out_path)
