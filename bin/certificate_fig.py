"""Practitioner certificate figure: the a-posteriori error bar and its decay.

Post-hoc consumer of the psi and G checkpoints bin/_run.py writes (a sibling of
bin/plot.py): no retraining. For each (family, dim) and a ladder of sample
counts K it builds the duality sandwich L_K <= g <= U_K from the first network's
conjugate samples (src.recovery.conjugate_samples), then reports, at held-out
query points inside the hull, the a-posteriori error bound of
convergence_work.tex Prop. 5.1 for the TRAINED second network G:

    |G(y) - g(y)|  <=  max( |G(y) - L_K(y)|, |G(y) - U_K(y)| ),

computable with NO ground truth. We plot the median of that bound against K
(solid), and overlay the actual error |G(y) - g(y)| (dashed), using the exact
g(y) = <preimage(y), y> - psi(preimage(y)) available for these families, to show
the bound holds and tightens toward the true network error as K grows.

Two things a practitioner wants: a ground-truth-free error bar, and its decay.

    python bin/certificate_fig.py                 # dims 2 4, families l1 + min-plus
    python bin/certificate_fig.py --dims 2 3 4
    python bin/certificate_fig.py --smoke         # short K ladder, quick check

Writes numerics/figs/certificate_decay.{pdf,png} and logs/certificate_metrics.json.
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
from _run import uniform_inputs, run
from src.network import LPN
from src.recovery import conjugate_samples, prox, cvx
from src.certificate import certificate, a_posteriori_bound

LOGS = os.path.join(ROOT, "logs")
FIGS = os.path.join(ROOT, "figs")

# The two families shown, config(dim) -> Problem (exact preimage + cvx_true).
FAMILIES = {"quadratic_l1": quadratic_l1.config, "minplus": minplus.config}
LABELS = {"quadratic_l1": "$\\ell_1$", "minplus": "min-plus"}

# Okabe-Ito (colorblind-safe), matching the theory figures.
COLORS = {"quadratic_l1": "#009E73", "minplus": "#0072B2"}
GRAY = "#8a8a8a"


def load_ckpt(name, which):
    p = os.path.join(LOGS, "ckpt", f"{name}_{which}.pth")
    if not os.path.exists(p):
        sys.exit(f"missing checkpoint {p}\nRun bin/<family>.py --dim <d> first, "
                 f"or pass --train-missing.")
    blob = torch.load(p, map_location="cpu", weights_only=False)
    m = LPN(in_dim=blob["in_dim"], hidden=blob["hidden"],
            layers=blob["layers"], beta=blob["beta"])
    m.load_state_dict(blob["state"])
    m.eval()
    return m


def ensure_checkpoint(family, dim):
    """Train the standard pipeline for (family, dim) if its checkpoint is absent
    (figure-support dims like d=3 not exposed by the per-family CLI). Uses the
    SAME _run.run as every other checkpoint, so the run is protocol-identical."""
    p = os.path.join(LOGS, "ckpt", f"{family}_{dim}D_psi.pth")
    if os.path.exists(p):
        return
    if dim > 8:
        sys.exit(f"refusing to train d={dim} > 8 unprompted (see run_all --allow-high-dim)")
    print(f"[train-missing] {family}_{dim}D: no checkpoint, running full pipeline")
    run(f"{family}_{dim}D", FAMILIES[family](dim), dim=dim, a=4.0)


def exact_g(problem, y):
    """Exact g(y) = psi^*(y) = <x, y> - psi(x) with x = (grad psi)^{-1}(y).

    Both the preimage map and psi = cvx_true are closed form for these families,
    so this is the true target the second network estimates -- no network in it.
    """
    x = problem.preimage(np.asarray(y))
    return np.sum(x * np.asarray(y), axis=1) - problem.cvx_true(x)


def query_points(psi, dim, train_a, n_query, seed=99):
    """Fixed y-queries inside conv{y_k}: images under grad psi of x-points on a
    shrunk training box, reused across the whole K ladder."""
    xq = uniform_inputs(dim, n_query, 0.85 * train_a, seed=seed)
    return prox(xq, psi)  # y_q = grad psi(x_q)


def sweep_family(family, dim, Ks, n_query=120, a=4.0, min_inhull=10):
    """For one trained (family, dim): the median a-posteriori bar and the median
    true error of the second network, at each K.

    G and the true g at the queries are FIXED (they do not depend on K); only the
    sandwich L_K, U_K tightens, so the bar decreases toward the true error.
    """
    name = f"{family}_{dim}D"
    problem = FAMILIES[family](dim)
    train_a = problem.train_halfwidth(a)
    psi, G = load_ckpt(name, "psi"), load_ckpt(name, "G")

    n_train = 15_000 * dim
    x_full = uniform_inputs(dim, n_train, train_a, seed=1)
    yq = query_points(psi, dim, train_a, n_query)
    Ghat = cvx(yq, G)               # G(y_q), the network estimate of g (fixed)
    gtrue = exact_g(problem, yq)    # exact g(y_q) (fixed)

    rows = []
    for K in Ks:
        x_k = x_full[:K]
        y_k, G_k = conjugate_samples(x_k, psi)  # (y_k, g(y_k)); slopes are x_k
        G_k = np.asarray(G_k).ravel()
        L, U, ok = certificate(yq, y_k, G_k, x_k)
        n_ok = int(ok.sum())
        if n_ok < min_inhull:
            rows.append({"family": family, "n": dim, "K": int(K), "n_inhull": n_ok,
                         "bar_med": float("nan"), "err_med": float("nan"),
                         "gap_med": float("nan")})
            continue
        bar = a_posteriori_bound(Ghat[ok], L[ok], U[ok])
        err = np.abs(Ghat[ok] - gtrue[ok])
        rows.append({
            "family": family, "n": dim, "K": int(K), "n_inhull": n_ok,
            "bar_med": float(np.median(bar)),      # certified, no ground truth
            "err_med": float(np.median(err)),      # actual error (fixed in K)
            "gap_med": float(np.median(U[ok] - L[ok])),
        })
    return rows


def make_figure(results, dims, out_base):
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
    fig, axes = plt.subplots(1, len(dims), figsize=(3.3 * len(dims), 2.9),
                             squeeze=False)
    for ax, n in zip(axes[0], dims):
        for fam in FAMILIES:
            rows = [r for r in results[(fam, n)] if np.isfinite(r["bar_med"])]
            if not rows:
                continue
            Kx = [r["K"] for r in rows]
            ax.loglog(Kx, [r["bar_med"] for r in rows], color=COLORS[fam],
                      linestyle="-", marker="o", markersize=4,
                      label=f"{LABELS[fam]}: certified bound")
            ax.loglog(Kx, [r["err_med"] for r in rows], color=COLORS[fam],
                      linestyle="--", marker=None, alpha=0.9,
                      label=f"{LABELS[fam]}: actual error")
        ax.set_xlabel("$K$")
        ax.set_title(f"$n = {n}$", fontsize=9)
    axes[0][0].set_ylabel("median error at held-out queries")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 0.02), fontsize=7.5)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--n-query", type=int, default=120)
    ap.add_argument("--train-missing", action="store_true",
                    help="train (full pipeline) any figure-support checkpoint "
                         "that is absent, e.g. d=3; d>8 still refused")
    ap.add_argument("--smoke", action="store_true",
                    help="short K ladder for a quick end-to-end check")
    args = ap.parse_args()

    Ks = ([64, 128, 256, 512] if args.smoke
          else [64, 128, 256, 512, 1024, 2048, 4096])

    if args.train_missing:
        for n in args.dims:
            for fam in FAMILIES:
                ensure_checkpoint(fam, n)

    results, summary = {}, {}
    for n in args.dims:
        for fam in FAMILIES:
            rows = sweep_family(fam, n, Ks, n_query=args.n_query)
            results[(fam, n)] = rows
            summary[f"{fam}_n{n}"] = rows
            last = next((r for r in reversed(rows) if np.isfinite(r["bar_med"])), None)
            if last:
                print(f"[{fam:12s} n={n}] K={last['K']}: certified bound "
                      f"{last['bar_med']:.3e}, actual error {last['err_med']:.3e}")

    out_base = os.path.join(FIGS, "certificate_decay")
    make_figure(results, args.dims, out_base)
    with open(os.path.join(LOGS, "certificate_metrics.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out_base}.pdf/.png and logs/certificate_metrics.json")


if __name__ == "__main__":
    main()
