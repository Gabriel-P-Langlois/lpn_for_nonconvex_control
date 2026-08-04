"""The Experiment-1 figure: one panel per sigma, four curves each.

    t*J        what the prior is (scaled to the recoverable object's units)
    f_reg      what a denoiser at this sigma can reveal at best (= t*J_BVS)
    fit (a)    the semiconvex readout G_theta - |y|^2/2  -- lands on f_reg
    fit (b)    the convex class                          -- bridges the well

All curves are de-meaned on |y| <= 4 (the constant is unidentifiable:
prox_{f+c} = prox_f).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import mixture as mx
from . import paths, readout
from src.gradfit import net_value


def _demean(f, m):
    return f - f[m].mean()


def figure_experiment1(models, tag=""):
    """models: {sigma: (G, Jb, window)}. Writes PNG + PDF, returns the PNG path.
    `tag` distinguishes variants ("", "_smoke", "_uniform").

    Each panel is drawn on that run's data window — the sampled range of D,
    the region on which the regularizer is pinned (thm:prior_optimality).
    """
    g, _ = mx.grid()
    sigmas = sorted(models, reverse=True)

    fig, axes = plt.subplots(1, len(sigmas), figsize=(6.4 * len(sigmas), 4.6),
                             squeeze=False)
    for ax, sigma in zip(axes[0], sigmas):
        t = sigma ** 2
        G, Jb, (lo, hi) = models[sigma]
        m = (g >= lo) & (g <= hi)
        tJ = _demean(t * mx.J(g), m)
        fr = _demean(mx.freg(g, sigma), m)
        fa = _demean(readout.value(G, g), m)
        fb = _demean(net_value(Jb, g.reshape(-1, 1)), m)

        ax.plot(g[m], tJ[m], color="0.55", lw=1.6, ls="--",
                label=r"$tJ$ (true prior, hidden)")
        ax.plot(g[m], fr[m], color="k", lw=2.2,
                label=r"$f_{\mathrm{reg}} = t\,J_{\mathrm{BVS}}$ (recoverable)")
        ax.plot(g[m], fa[m], color="C0", lw=1.8,
                label=r"fit (a): ICNN $-\,\|y\|^2/2$ (semiconvex)")
        ax.plot(g[m], fb[m], color="C3", lw=1.8, ls="-.",
                label="fit (b): plain ICNN (convex)")
        ax.set_title(rf"$\sigma = {sigma:g}$,  $t = \sigma^2 = {t:g}$"
                     rf"   (curvature of $f_{{\mathrm{{reg}}}}/t \geq -1/t = {-1/t:g}$)")
        ax.set_xlabel(r"$y$")
        ax.legend(loc="upper center", fontsize=8.5, framealpha=0.9)
        ax.set_xlim(lo, hi)

    axes[0][0].set_ylabel("regularizer value (constant removed)")
    fig.tight_layout()

    paths.ensure_dirs()
    png = os.path.join(paths.FIGS, f"experiment1_mixture{tag}.png")
    fig.savefig(png, dpi=180)
    if "smoke" not in tag:
        fig.savefig(os.path.join(paths.FIGS, f"experiment1_mixture{tag}.pdf"))
    plt.close(fig)
    return png
