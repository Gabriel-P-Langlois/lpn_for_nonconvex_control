"""Figures for CLAUDE/theoretical_work/convergence_work.tex.

Three print figures (PDF, one axis each, Okabe-Ito colorblind-safe colors with
distinct linestyles/markers as secondary encoding, direct labels):

  figs/sandwich1d.pdf : the duality sandwich in 1-D, generic vs pushforward
                        sampling of the l1 family (kink chord vs kink atom).
  figs/rates.pdf      : (a) certificate decay in K at n=2, three settings, with
                        reference slopes; (b) fill distances vs N.
  figs/stability.pdf  : argmin-stability shifts vs eps against the 2*sqrt(eps)
                        bound (smooth perturbation ~ eps; dip attains sqrt(eps)).

Run inside numerics/__new_theory_exps with the lpn_env python.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import helpers as H

OUT = Path(__file__).resolve().parents[2] / "CLAUDE" / "theoretical_work" / "figs"
OUT.mkdir(exist_ok=True)

# Okabe-Ito (colorblind-safe): blue, vermillion, bluish green, black.
BLUE, VERM, GREEN, INK = "#0072B2", "#D55E00", "#009E73", "#1a1a1a"
GRAY = "#8a8a8a"

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "lines.linewidth": 1.6, "figure.dpi": 150,
})


def g1(y):
    return np.abs(y) + 0.5 * y**2


def envelope_1d(ys, slopes, y):
    """L_K on a grid: max of tangents g1(ys_k) + slope_k (y - ys_k)."""
    return np.max(g1(ys)[None, :] + slopes[None, :] * (y[:, None] - ys[None, :]),
                  axis=1)


def fig_sandwich():
    y = np.linspace(-1.6, 1.6, 801)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True)

    # (a) generic samples: kink at 0 falls between samples.
    ys_a = np.array([-1.4, -0.75, -0.25, 0.35, 0.9, 1.5])
    sl_a = ys_a + np.sign(ys_a)
    # (b) pushforward samples: atom at 0 carrying several subgradients.
    x_b = np.array([-2.4, -1.7, -0.6, 0.3, 0.8, 1.6, 2.5])
    ys_b = np.sign(x_b) * np.maximum(np.abs(x_b) - 1, 0)
    sl_b = x_b

    for ax, ys, sl, title in [
        (axes[0], ys_a, sl_a, "(a) generic samples: kink between samples"),
        (axes[1], ys_b, sl_b, "(b) pushforward samples: atom on the kink"),
    ]:
        yy = np.unique(ys)
        m = (y >= yy.min()) & (y <= yy.max())
        L = envelope_1d(ys, sl, y)
        U = np.interp(y[m], yy, g1(yy))
        ax.fill_between(y[m], L[m], U, color=GRAY, alpha=0.25, linewidth=0,
                        label="certificate $U_K-L_K$")
        ax.plot(y, g1(y), color=INK, linestyle="-", label="$g$")
        ax.plot(y, L, color=BLUE, linestyle="--", label="$L_K$")
        ax.plot(y[m], U, color=VERM, linestyle=":", label="$U_K$")
        ax.plot(yy, g1(yy), "o", color=INK, markersize=4.5,
                markerfacecolor="white", label="samples")
        ax.set_xlabel("$y$")
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(-0.15, 3.2)
    axes[0].set_ylabel("value")
    axes[0].annotate("$O(h)$ chord gap", xy=(0.0, np.interp(0.0, np.unique(ys_a), g1(np.unique(ys_a)))),
                     xytext=(-1.5, 1.55), fontsize=7.5,
                     arrowprops=dict(arrowstyle="->", lw=0.7, color=INK))
    axes[1].annotate("slopes fill the fan", xy=(0.0, 0.0), xytext=(-1.55, 1.0),
                     fontsize=7.5,
                     arrowprops=dict(arrowstyle="->", lw=0.7, color=INK))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", frameon=False, ncol=5,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "sandwich1d.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_rates():
    Ks = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    curves = [
        ("huber, pushforward", "huber", "pushforward", 1, BLUE, "-", "o"),
        ("$\\ell_1$, pushforward", "l1", "pushforward", 1, GREEN, "--", "s"),
        ("$\\ell_1$, generic", "l1", "generic", 4, VERM, "-.", "^"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))

    ax = axes[0]
    for label, fam, samp, reps, color, ls, mk in curves:
        r = H.exp1_decay(fam, 2, Ks, n_query=150, sampling=samp, replicates=reps)
        ax.loglog([x["K"] for x in r], [x["sup_gap"] for x in r], color=color,
                  linestyle=ls, marker=mk, markersize=4, label=label)
    Kref = np.array([256.0, 4096.0])
    for expo, txt, dy, (tx, ty) in [(-1.0, "$K^{-1}$", 0.42, (900.0, 0.055)),
                                    (-0.5, "$K^{-1/2}$", 2.4, (900.0, 1.6))]:
        ax.loglog(Kref, dy * (Kref / Kref[0]) ** expo, color=GRAY,
                  linestyle=":", linewidth=1.1)
        ax.annotate(txt, xy=(tx, ty), fontsize=7.5, color=GRAY)
    ax.set_xlabel("$K$")
    ax.set_ylabel("$\\sup_{\\mathcal{Y}}(U_K-L_K)$")
    ax.set_title("(a) certificate decay, $n=2$", fontsize=8.5)
    ax.set_xlim(25, 6e3)
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    r2 = H.exp2_fill("l1", 2, [64, 128, 256, 512, 1024, 2048, 4096])
    N = np.array([x["N"] for x in r2], float)
    ax.loglog(N, [x["h_X"] for x in r2], color=BLUE, linestyle="-", marker="o",
              markersize=4, label="$h_X$ (x-side)")
    ax.loglog(N, [x["h_Y"] for x in r2], color=VERM, linestyle="--", marker="s",
              markersize=4, label="$h_Y$ (pushforward)")
    pred = np.array([x["pred"] for x in r2])
    c = r2[-1]["h_X"] / pred[-1]
    ax.loglog(N, c * pred, color=GRAY, linestyle=":", linewidth=1.1,
              label="$c\\,(\\log N/N)^{1/2}$")
    ax.set_xlabel("$N$")
    ax.set_ylabel("fill distance")
    ax.set_title("(b) fill distances, $\\ell_1$, $n=2$", fontsize=8.5)
    ax.legend(frameon=False, loc="lower left")

    fig.tight_layout()
    fig.savefig(OUT / "rates.pdf")
    plt.close(fig)


def fig_stability():
    eps = np.logspace(-6, -1, 6)
    a = H.exp3_smooth_perturbation(eps)
    b = H.exp3_dip_perturbation(eps)
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.loglog(eps, [r["bound"] for r in b], color=GRAY, linestyle=":",
              linewidth=1.2, label="bound $2\\sqrt{\\varepsilon/\\sigma}$")
    ax.loglog(eps, [r["shift"] for r in b], color=VERM, linestyle="-",
              marker="^", markersize=4, label="dip perturbation")
    ax.loglog(eps, [r["max_shift"] for r in a], color=BLUE, linestyle="--",
              marker="o", markersize=4, label="smooth perturbation")
    ax.annotate("slope $1/2$", xy=(1e-3, 4.4e-2), fontsize=7, color=VERM)
    ax.annotate("slope $1$", xy=(1e-3, 2.5e-4), fontsize=7, color=BLUE)
    ax.set_xlabel("model error $\\varepsilon$")
    ax.set_ylabel("minimizer shift")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "stability.pdf")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "cheap"):
        fig_sandwich()
        fig_stability()
        print("sandwich1d.pdf, stability.pdf written")
    if which in ("all", "rates"):
        fig_rates()
        print("rates.pdf written")
