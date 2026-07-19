"""E.2 / referee R2.3: decay of the max-plus approximant Gamma_K, and the
computable duality sandwich Gamma_K <= S <= U_M.

Runs on J = ||x||_1, t = 1, where the Hopf solution is the separable Huber and
every ingredient is closed-form: no network, no training, no ground-truth proxy.
See src/maxplus_bounds.py for the mathematics.

Outputs (logs/):
    gamma_decay.csv            error of Gamma_K vs K, per dimension, per sampler
    gamma_decay.png            log-log decay with the predicted K^{-2/d} slopes
    gamma_sandwich_1d.png      Gamma_K <= S <= U_M in 1D, tangency at samples
    gamma_certificate.png      certificate U_M - Gamma_K vs the true error

    python bin/gamma_decay.py
    python bin/gamma_decay.py --quick
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.maxplus_bounds import (
    convex_upper_bound,
    decay_curve,
    fit_loglog_slope,
    gamma_K,
    huber_S,
    slopes_grid_l1,
    slopes_random_l1,
    slopes_tangent_l1,
)

# The K^{-2/d} law is asymptotic in the fill distance. Fit only the largest-K
# points: on a tensor grid delta = 2/(m-1) while K = m^d, so the smallest grids
# read too steep. See src.maxplus_bounds.fit_loglog_slope.
FIT_TAIL = 4

LOGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

# Query box matches the experiments' [-4,4]^d, and the query seed matches the
# protocol's test seed so these points are the ones the networks are judged on.
A = 4.0
N_QUERY = 2000
QUERY_SEED = 3

# Grid sampler: K = m^d, so the usable m shrinks fast with d. Random sampler
# decouples K from d and is the only option once m^d explodes.
GRID_M = {1: [2, 4, 8, 16, 32, 64, 128], 2: [2, 3, 4, 6, 8, 12, 16, 24],
          3: [2, 3, 4, 5, 6, 8, 10], 4: [2, 3, 4, 5, 6, 7]}
RANDOM_K = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
RANDOM_DIMS = [2, 4, 8, 16]

# The sampler determines the exponent, not just the constant. See the "two
# regimes" section of src/maxplus_bounds.py: uniform slopes miss the boundary of
# dom J* and pay K^{-1/d}; a grid resolves every face and gets K^{-2/d}.
#
# The TANGENT sampler has no predicted exponent, at any d. Its slopes lie on the
# (d-1)-dimensional boundary of dom J* and split into a small interior
# population and a large boundary one; the two serve disjoint query regions and
# decay at different rates, so the aggregate error is a MIXTURE, not a power
# law. Fitting it produces a number that moves with the query draw. We still
# plot the curve -- it is the sampler the theory recommends -- but we neither
# predict nor fit a slope for it.
PRED_EXP = {"grid": lambda d: -2.0 / d,
            "random": lambda d: -1.0 / d,
            "tangent": lambda d: None}


def _grid_sampler(d, m, rng):
    return slopes_grid_l1(d, m)


def _random_sampler(d, K, rng):
    return slopes_random_l1(d, K, rng)


def _tangent_sampler(d, K, rng):
    """p_k = grad S(y_k) = clip(y_k), with y_k drawn from the QUERY law."""
    return slopes_tangent_l1(rng.uniform(-A, A, (K, d)))


def run_decay(quick=False):
    rows = []
    grid_m = {d: (ms[:4] if quick else ms) for d, ms in GRID_M.items()}
    rand_k = RANDOM_K[:5] if quick else RANDOM_K
    rand_d = RANDOM_DIMS[:2] if quick else RANDOM_DIMS

    for d, ms in grid_m.items():
        rng = np.random.default_rng(QUERY_SEED)
        y = rng.uniform(-A, A, (N_QUERY, d))
        for r in decay_curve(d, ms, y, _grid_sampler, rng=rng):
            r["sampler"] = "grid"
            rows.append(r)

    for sampler, fn in (("random", _random_sampler), ("tangent", _tangent_sampler)):
        for d in rand_d:
            rng = np.random.default_rng(QUERY_SEED)
            y = rng.uniform(-A, A, (N_QUERY, d))
            for r in decay_curve(d, rand_k, y, fn, rng=np.random.default_rng(11)):
                r["sampler"] = sampler
                rows.append(r)
    return rows


def write_csv(rows, path):
    fields = ["sampler", "d", "K", "sup_err", "rms_err", "mean_err"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in rows)


TITLES = {
    "grid": "grid slopes (resolves $\\partial\\,$dom$\\,J^*$)",
    "random": "uniform slopes (misses $\\partial\\,$dom$\\,J^*$)",
    "tangent": "tangent slopes $p_k=\\nabla S(y_k)$",
}


def plot_decay(rows, path):
    samplers = [s for s in ("grid", "random", "tangent")
                if any(r["sampler"] == s for r in rows)]
    fig, axes = plt.subplots(1, len(samplers), figsize=(5.2 * len(samplers), 4.5))
    axes = np.atleast_1d(axes)
    for ax, sampler in zip(axes, samplers):
        sub = [r for r in rows if r["sampler"] == sampler]
        for d in sorted({r["d"] for r in sub}):
            rr = sorted((r for r in sub if r["d"] == d), key=lambda r: r["K"])
            K = [r["K"] for r in rr]
            e = [r["rms_err"] for r in rr]
            pred = PRED_EXP[sampler](d)
            if pred is None:                      # tangent: no rate to claim
                ax.loglog(K, e, "o-", ms=4, label=f"$d={d}$  (no power law)")
                continue
            s = fit_loglog_slope(K, e, tail=FIT_TAIL)
            line, = ax.loglog(K, e, "o-", ms=4,
                              label=f"$d={d}$  fit {s:.2f}, pred {pred:.2f}")
            # reference slope, anchored at the LARGEST K, where the law applies
            Kp = np.array(K, float)
            ax.loglog(Kp, e[-1] * (Kp / Kp[-1]) ** pred, "--", lw=1,
                      color=line.get_color(), alpha=0.45)
        ax.set_xlabel("$K$ (number of slopes)")
        ax.set_ylabel(r"RMS error  $\|S-\Gamma_K\|$")
        exp = {"random": "$K^{-1/d}$", "grid": "$K^{-2/d}$",
               "tangent": "no rate (mixture)"}[sampler]
        ax.set_title(f"{TITLES[sampler]}\ndashed $=$ {exp}", fontsize=9)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle(r"Decay of the max-plus lower bound $\Gamma_K \leq S$  "
                 r"($J=\|x\|_1$, $t=1$): the sampler sets the exponent, not just "
                 r"the constant", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sandwich_1d(path, M_in=5, M_out=8):
    """The sandwich in 1D, where U_M is a plain PL interpolant and everything is
    visible: Gamma_K is tangent from below at the samples, U_M chords from above.

    Samples are STRATIFIED, M_in inside [-1,1] and M_out outside. Uniform
    sampling on [-3.5,3.5] puts only ~2 points in [-1,1], which is where all the
    curvature -- and hence all the error -- lives. The stratification is for
    legibility of the figure only; every rate in this file uses unstratified
    draws, and the point the figure makes (both gaps vanish identically outside
    [-1,1]) is independent of it.
    """
    rng = np.random.default_rng(0)
    ym = np.sort(np.concatenate([
        rng.uniform(-1.0, 1.0, M_in),
        rng.uniform(1.05, 3.5, M_out) * rng.choice([-1.0, 1.0], M_out),
    ]))[:, None]
    M = len(ym)
    sm = huber_S(ym)
    P = slopes_tangent_l1(ym)                       # exact supporting slopes
    yq = np.linspace(-3.9, 3.9, 600)[:, None]
    S = huber_S(yq)
    lo = gamma_K(yq, P)
    hi = convex_upper_bound(yq, ym, sm)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.4]})
    ax.axvspan(-1, 1, color="0.85", alpha=0.5, zorder=0)
    ax.plot(yq, S, "k", lw=2.2, label=r"$S(y,1)$  (exact Huber)")
    ax.plot(yq, lo, "C0", lw=1.4, label=rf"$\Gamma_K$, $K={M}$  (lower, max-plus)")
    fin = np.isfinite(hi)
    ax.plot(yq[fin], hi[fin], "C3", lw=1.4, label=r"$U_M$  (upper, convex interpolant)")
    ax.plot(ym, sm, "ko", ms=5, zorder=5, label="samples $y_m$")
    ax.set_ylabel("value")
    ax.legend(fontsize=8, loc="upper center")
    ax.grid(alpha=0.25)
    ax.set_title(r"Computable duality sandwich  $\Gamma_K \leq S \leq U_M$   "
                 r"(both bounds ground-truth-free)")

    # Floor at a visible epsilon: the gaps are EXACTLY 0 outside [-1,1] (the
    # tangent slopes clip to +-1 there, attaining p*), and a log axis cannot
    # show zero. Ticks every decade -- at 4-decade spacing the plateau reads an
    # order of magnitude wrong.
    FLOOR = 1e-6
    ax2.semilogy(yq[fin], np.maximum(hi[fin] - lo[fin], FLOOR), "C2",
                 label=r"certificate  $U_M-\Gamma_K$  (free)")
    ax2.semilogy(yq, np.maximum(S - lo, FLOOR), "C0", ls="--",
                 label=r"true error  $S-\Gamma_K$")
    ax2.plot(ym, np.full(M, FLOOR), "ko", ms=4, clip_on=False, zorder=5)
    ax2.axvspan(-1, 1, color="0.85", alpha=0.5, zorder=0)
    ax2.text(0, FLOOR * 3, r"$\|y\|_\infty\leq 1$: $p^*$ interior",
             ha="center", fontsize=7, color="0.3")
    ax2.set_ylim(FLOOR / 2, None)
    ax2.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=8))
    ax2.set_xlabel("$y$")
    ax2.set_ylabel("gap")
    ax2.legend(fontsize=8, loc="center right")
    ax2.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_certificate(path, d=2, M=80, n_q=300):
    """Does the free certificate track the true error? Scatter, in-hull queries."""
    rng = np.random.default_rng(1)
    ym = rng.uniform(-3.5, 3.5, (M, d))
    sm = huber_S(ym)
    P = slopes_tangent_l1(ym)
    yq = rng.uniform(-2.0, 2.0, (n_q, d))
    S = huber_S(yq)
    lo = gamma_K(yq, P)
    hi = convex_upper_bound(yq, ym, sm)
    fin = np.isfinite(hi)
    cert, true_err = hi[fin] - lo[fin], S[fin] - lo[fin]

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.loglog(np.maximum(true_err, 1e-12), np.maximum(cert, 1e-12), "o", ms=4, alpha=0.6)
    lim = [1e-6, max(cert.max(), true_err.max()) * 2]
    ax.plot(lim, lim, "k--", lw=1, label=r"$y=x$ (certificate $=$ error)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"true error  $S-\Gamma_K$")
    ax.set_ylabel(r"certificate  $U_M-\Gamma_K$")
    ax.set_title(f"Certificate bounds the error ($d={d}$, $M={M}$)\n"
                 f"median slack {np.median(cert/np.maximum(true_err,1e-12)):.1f}x")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return float(np.median(cert / np.maximum(true_err, 1e-12))), int(fin.sum()), n_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short K lists, fewer dims")
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)

    rows = run_decay(quick=args.quick)
    write_csv(rows, os.path.join(LOGS, "gamma_decay.csv"))
    plot_decay(rows, os.path.join(LOGS, "gamma_decay.png"))
    plot_sandwich_1d(os.path.join(LOGS, "gamma_sandwich_1d.png"))
    slack, n_in, n_q = plot_certificate(os.path.join(LOGS, "gamma_certificate.png"))

    print(f"tail fit on the {FIT_TAIL} largest K.  Predicted exponent depends on "
          f"the sampler:\n  grid -> -2/d (resolves d(dom J*));  "
          f"uniform -> -1/d (misses it);  tangent -> no power law.\n")
    print(f"{'sampler':>8} {'d':>3} {'K_max':>7} {'slope':>8} {'pred':>8} "
          f"{'ratio':>7}  note")
    for sampler in ("grid", "random", "tangent"):
        for d in sorted({r["d"] for r in rows if r["sampler"] == sampler}):
            rr = sorted((r for r in rows if r["sampler"] == sampler and r["d"] == d),
                        key=lambda r: r["K"])
            K = [r["K"] for r in rr]
            e = [r["rms_err"] for r in rr]
            s = fit_loglog_slope(K, e, tail=FIT_TAIL)
            pred = PRED_EXP[sampler](d)
            if pred is None:
                print(f"{sampler:>8} {d:>3} {max(K):>7} {s:>8.3f} {'--':>8} "
                      f"{'--':>7}  mixture of two decays; this slope is NOT a rate")
            else:
                print(f"{sampler:>8} {d:>3} {max(K):>7} {s:>8.3f} {pred:>8.3f} "
                      f"{s/pred:>7.2f}")
    print(f"\ncertificate median slack {slack:.1f}x over the true error "
          f"({n_in}/{n_q} queries inside conv(samples))")
    print(f"-> {LOGS}/gamma_decay.{{csv,png}}, gamma_sandwich_1d.png, gamma_certificate.png")


if __name__ == "__main__":
    main()
