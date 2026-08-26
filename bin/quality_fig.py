"""How good is the trained network, from data only? A dimension study.

Post-hoc consumer of the psi and G checkpoints bin/_run.py writes (a sibling of
bin/plot.py): no retraining, forward passes only, so evaluating the existing
high-dimensional checkpoints is cheap and is not a new run.

For each trained (family, dim) it reports, on the shared test set, two signals a
practitioner can compute WITHOUT the ground-truth prior, next to the actual
recovered-prior error (which we know here, for validation):

  * eps_rel  -- relative L2 error of the first network against the exact
                potential psi = 0.5||x||^2 - S, whose targets are the HJ data
                itself, so this is measurable in practice. By the conjugation
                transfer (convergence_work.tex Prop. on transfer to the prior),
                the recovered prior error is controlled by this quantity.
  * prox residual -- ||grad psi(grad G(x)) - x|| / max(1,||x||), per query: how
                far the two trained networks are from being mutually conjugate
                at x. Zero iff exact; needs no ground truth; flags where to
                distrust the network.
  * prior_rel (validation) -- relative L2 error of the recovered prior
                J = G - 0.5||.||^2 against the true J_BVS. The truth, shown
                dashed, to check the two signals above track it.

    python bin/quality_fig.py                    # dims 2 3 4 8 16 32 64
    python bin/quality_fig.py --dims 2 4 8

Writes numerics/figs/network_quality.{pdf,png} and logs/quality_metrics.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

BIN = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BIN)
sys.path.insert(0, BIN)
sys.path.insert(0, ROOT)

import quadratic_l1
import minplus
from _run import uniform_inputs
from src.network import LPN
from src.recovery import (
    cvx, route2_preimage, prox_residual, fenchel_young_gap,
    evaluate_learned_prior_G,
)

LOGS = os.path.join(ROOT, "logs")
FIGS = os.path.join(ROOT, "figs")

FAMILIES = {"quadratic_l1": quadratic_l1.config, "minplus": minplus.config}
LABELS = {"quadratic_l1": "$\\ell_1$", "minplus": "min-plus"}


def load_ckpt(name, which):
    p = os.path.join(LOGS, "ckpt", f"{name}_{which}.pth")
    if not os.path.exists(p):
        return None
    blob = torch.load(p, map_location="cpu", weights_only=False)
    m = LPN(in_dim=blob["in_dim"], hidden=blob["hidden"],
            layers=blob["layers"], beta=blob["beta"])
    m.load_state_dict(blob["state"])
    m.eval()
    return m


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def exact_g(problem, x):
    """g(x) = psi^*(x) = <z, x> - psi(z) with z = (grad psi)^{-1}(x) = preimage(x).

    Then J_BVS(x) = g(x) - 0.5||x||^2 is the backward viscosity solution the
    pipeline actually recovers -- for a nonconvex prior this is NOT J (Theorem on
    prior optimality: J >= J_BVS), which is why the recovery must be scored
    against J_BVS, not against problem.prior_true = J.
    """
    z = problem.preimage(np.asarray(x))
    return np.sum(z * np.asarray(x), axis=1) - problem.cvx_true(z)


def quality(family, dim, a=4.0, n_eval=1000):
    name = f"{family}_{dim}D"
    psi, G = load_ckpt(name, "psi"), load_ckpt(name, "G")
    if psi is None or G is None:
        return None
    problem = FAMILIES[family](dim)
    x = uniform_inputs(dim, 4000, a, seed=3)[:n_eval]   # the run's test points

    # data-only signals (no ground-truth prior)
    eps_rel = rel_l2(cvx(x, psi), problem.cvx_true(x))  # psi fit vs HJ data
    y2 = route2_preimage(x, G)                          # grad G(x)
    resid = prox_residual(x, y2, psi)                   # per query, in [0, inf)
    fy = fenchel_young_gap(x, y2, psi, G)               # per query, >= 0

    # the recovered prior, and the two references it can be scored against
    Jhat = evaluate_learned_prior_G(x, G)               # G(x) - 0.5||x||^2
    Jtrue = problem.prior_true(x)                       # the true prior J
    Jbvs = exact_g(problem, x) - 0.5 * np.sum(x * x, axis=1)  # J_BVS (recovered)

    return {
        "family": family, "n": dim,
        "eps_rel": eps_rel,
        "recover_rel": rel_l2(Jhat, Jbvs),   # network error vs the recovered object
        "vs_true_rel": rel_l2(Jhat, Jtrue),  # vs J: includes the viscosity gap
        "gap_rel": rel_l2(Jbvs, Jtrue),      # J vs J_BVS: theory gap, ~0 if convex
        "resid_median": float(np.median(resid)),
        "resid_p90": float(np.percentile(resid, 90)),
        "fy_median": float(np.median(fy)),
    }


def make_figure(rows_by_family, dims, out_base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "legend.fontsize": 7.0, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
        "lines.linewidth": 1.6, "figure.dpi": 150,
    })
    INK, BLUE, VERM, GRAY = "#1a1a1a", "#0072B2", "#D55E00", "#8a8a8a"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), squeeze=False)
    for ax, fam in zip(axes[0], FAMILIES):
        rows = [r for r in rows_by_family[fam] if r is not None]
        ns = [r["n"] for r in rows]
        convex = np.median([r["gap_rel"] for r in rows]) < 1e-10
        ax.loglog(ns, [r["recover_rel"] for r in rows], color=INK, linestyle="-",
                  marker="o", markersize=4,
                  label="recovery error $\\|\\hat J-J_{BVS}\\|/\\|J_{BVS}\\|$")
        ax.loglog(ns, [r["eps_rel"] for r in rows], color=BLUE, linestyle="--",
                  marker="s", markersize=4,
                  label="$\\psi$ fit $\\|\\psi_\\theta-\\psi\\|/\\|\\psi\\|$ (data only)")
        ax.loglog(ns, [r["resid_median"] for r in rows], color=VERM, linestyle="--",
                  marker="^", markersize=4, label="prox residual, median (data only)")
        vals = [r[k] for r in rows for k in ("recover_rel", "eps_rel", "resid_median")]
        if convex:
            ax.text(0.5, 0.06, "viscosity gap $\\|J-J_{BVS}\\|=0$ (convex)",
                    transform=ax.transAxes, ha="center", fontsize=7.0, color=GRAY)
        else:
            ax.loglog(ns, [r["gap_rel"] for r in rows], color=GRAY,
                      linestyle=":", marker="D", markersize=3,
                      label="viscosity gap $\\|J-J_{BVS}\\|/\\|J\\|$")
            vals += [r["gap_rel"] for r in rows]
        ax.set_ylim(0.4 * min(vals), 2.2 * max(vals))
        ax.set_xlabel("dimension $n$")
        ax.set_title(LABELS[fam], fontsize=9)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
    axes[0][0].set_ylabel("relative $L^2$ error")
    handles, labels = axes[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.04), fontsize=7.0)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4, 8, 16, 32, 64])
    ap.add_argument("--n-eval", type=int, default=1000)
    args = ap.parse_args()

    rows_by_family, summary = {}, {}
    for fam in FAMILIES:
        rows = []
        for n in args.dims:
            r = quality(fam, n, n_eval=args.n_eval)
            if r is None:
                print(f"[{fam} n={n}] no checkpoint, skipped")
                continue
            rows.append(r)
            print(f"[{fam:12s} n={n:2d}] recover_rel={r['recover_rel']:.3e}  "
                  f"gap_rel={r['gap_rel']:.3e}  eps_rel={r['eps_rel']:.3e}  "
                  f"prox_resid_med={r['resid_median']:.3e}")
        rows_by_family[fam] = rows
        summary[fam] = rows

    out_base = os.path.join(FIGS, "network_quality")
    make_figure(rows_by_family, args.dims, out_base)
    with open(os.path.join(LOGS, "quality_metrics.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out_base}.pdf/.png and logs/quality_metrics.json")


if __name__ == "__main__":
    main()
