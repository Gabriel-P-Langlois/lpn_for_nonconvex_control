"""Experiment 1: the mixture figure (task document, Section 6.1).

    ~/miniforge3/envs/lpn_env/bin/python -m pnpreg.experiment            # full, ~2 min
    ~/miniforge3/envs/lpn_env/bin/python -m pnpreg.experiment --smoke    # ~10 s

Two fits from the SAME data and the SAME loop (src.gradfit.train_grad):

    (a) semiconvex: G_theta an ICNN trained on grad G(y_k) = x_k, read out as
        J_a = G_theta - |y|^2/2. The class is exactly the 1-semiconvex
        functions, which contains the target f_reg (Proposition prop:tight).
    (b) convex control: J_b an ICNN trained on grad J(y_k) = x_k - y_k, the
        tv_pm design. The target has curvature -1 between the modes, which a
        convex network cannot represent; (b) must fail there.

The outcome is guaranteed by theory. This run is an implementation test and
produces the paper figure; the falsifiable experiment is the PIRATE diagnosis
(task document, Section 6.2).
"""
import argparse
import json
import os

import numpy as np
import torch

from . import mixture as mx
from . import paths, readout
from src.gradfit import net_grad, net_value, train_grad
from src.network import LPN

CFG = dict(hidden=64, layers=2, beta=20, batch=512, steps=20_000, seed=1)
SIGMAS = (0.5, 0.25)
QUERY_A = 2.5   # query half-width for the uniform-coverage sampling


def make_data(sigma, n_train, n_val, sampling="pz"):
    """(x, y, xv, yv) with y = D(x) exact; seeds 1 (train) and 2 (val).

    sampling="pz":      x_k ~ p_z, the denoiser's operating distribution.
    sampling="uniform": x_k uniform on [-A, A] with A = D^{-1}(QUERY_A) — the
        manuscript's training-box construction, so that y = D(x) covers the
        query window [-QUERY_A, QUERY_A] by design and f_reg receives data
        everywhere on it (no unsampled gap).
    """
    if sampling == "pz":
        x = mx.sample_pz(n_train, sigma, seed=1)
        xv = mx.sample_pz(n_val, sigma, seed=2)
    elif sampling == "uniform":
        A = float(mx.D_inverse(np.array([QUERY_A]), sigma)[0])
        x = np.random.default_rng(1).uniform(-A, A, n_train)
        xv = np.random.default_rng(2).uniform(-A, A, n_val)
    else:
        raise ValueError(f"unknown sampling {sampling!r}")
    return x, mx.D(x, sigma), xv, mx.D(xv, sigma)


def fit_pair(sigma, smoke=False, quiet=True, sampling="pz"):
    """Both fits at one sigma. Returns (G, Jb, info)."""
    n, nv = (4000, 1000) if smoke else (20_000, 4000)
    steps = 500 if smoke else CFG["steps"]
    x, y, xv, yv = make_data(sigma, n, nv, sampling=sampling)
    col = lambda a: a.reshape(-1, 1)

    torch.manual_seed(CFG["seed"])
    G = LPN(in_dim=1, hidden=CFG["hidden"], layers=CFG["layers"], beta=CFG["beta"])
    ha = train_grad(G, col(y), col(x), col(yv), col(xv),
                    batch_size=CFG["batch"], steps=steps, quiet=quiet)

    torch.manual_seed(CFG["seed"])
    Jb = LPN(in_dim=1, hidden=CFG["hidden"], layers=CFG["layers"], beta=CFG["beta"])
    hb = train_grad(Jb, col(y), col(x - y), col(yv), col(xv - yv),
                    batch_size=CFG["batch"], steps=steps, quiet=quiet)

    # The evaluation window is the SAMPLED range of D: recovery is pinned on
    # range(D) and merely bounded below off it (thm:prior_optimality). Under
    # p_z sampling that range is data-dependent (about [-2.4, 2.4] at
    # sigma = 0.5, with an unsampled gap between the modes); under uniform
    # coverage it is the query window by construction.
    if sampling == "uniform":
        window = (-QUERY_A, QUERY_A)
    else:
        window = tuple(np.quantile(y, [0.005, 0.995]))

    return G, Jb, dict(sigma=sigma, t=sigma ** 2, n=n, steps=steps,
                       sampling=sampling,
                       best_val_a=ha["best_val"], best_val_b=hb["best_val"],
                       xv=xv, yv=yv, window=window)


def _demean(f, m):
    return f - f[m].mean()


def metrics(G, Jb, info):
    """Score both fits against the exact f_reg on the grid, plus the held-out
    prox residual (the certificate available when no ground truth exists)."""
    sigma = info["sigma"]
    g, _ = mx.grid()
    lo, hi = info["window"]
    m = (g >= lo) & (g <= hi)
    dx = g[1] - g[0]
    fr = mx.freg(g, sigma)

    fa = readout.value(G, g)                      # fit (a), semiconvex readout
    fb = net_value(Jb, g.reshape(-1, 1))          # fit (b), convex class

    out = dict(sigma=sigma, t=info["t"], n=info["n"], steps=info["steps"],
               window=[float(lo), float(hi)],
               best_val_a=info["best_val_a"], best_val_b=info["best_val_b"])
    exact = _demean(fr, m)
    for tag, f in (("a", _demean(fa, m)), ("b", _demean(fb, m))):
        out[f"rel_l2_{tag}"] = float(np.linalg.norm((f - exact)[m])
                                     / np.linalg.norm(exact[m]))
    # Curvature on a DECIMATED stencil: the networks evaluate in float32, and a
    # second difference amplifies value noise by 1/dx^2, so the native grid
    # spacing (~5e-4) would read pure quantization noise. Spacing ~0.02 keeps
    # the exact dip (curvature -1, width ~0.4) fully resolved.
    k = max(1, int(round(0.02 / dx)))
    for tag, f in (("exact", fr), ("a", fa), ("b", fb)):
        fc, mc = f[::k], m[::k]
        c2 = (fc[2:] - 2 * fc[1:-1] + fc[:-2]) / (k * dx) ** 2
        out[f"min_curv_{tag}"] = float(c2[mc[1:-1]].min())

    xv, yv = info["xv"], info["yv"]
    tgt = xv - yv
    ra = np.abs(readout.grad(G, yv) - tgt)
    rb = np.abs(net_grad(Jb, yv.reshape(-1, 1)).ravel() - tgt)
    out["resid_a"] = float(np.median(ra / np.abs(tgt)))
    out["resid_b"] = float(np.median(rb / np.abs(tgt)))
    return out


def run(smoke=False, quiet=True, sampling="pz"):
    paths.ensure_dirs()
    sigmas = SIGMAS[:1] if smoke else SIGMAS
    results, models = [], {}
    for sigma in sigmas:
        G, Jb, info = fit_pair(sigma, smoke=smoke, quiet=quiet, sampling=sampling)
        met = metrics(G, Jb, info)
        results.append(met)
        models[sigma] = (G, Jb, info["window"])
        print(f"sigma={sigma:.2f}  rel_l2 a/b = {met['rel_l2_a']:.3f}/{met['rel_l2_b']:.3f}"
              f"   min curv a/b/exact = {met['min_curv_a']:+.2f}/{met['min_curv_b']:+.2f}/"
              f"{met['min_curv_exact']:+.2f}   resid a/b = {met['resid_a']:.3f}/{met['resid_b']:.3f}")

    from .figures import figure_experiment1
    tag = ("_smoke" if smoke else "") + ("_uniform" if sampling == "uniform" else "")
    fig_path = figure_experiment1(models, tag=tag)

    mpath = os.path.join(paths.RESULTS, f"metrics{tag}.json")
    with open(mpath, "w") as f:
        json.dump({"config": {**CFG, "sampling": sampling}, "runs": results}, f, indent=2)
    print(f"figure  -> {fig_path}\nmetrics -> {mpath}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="500 steps, 4000 samples, sigma=0.5 only (~10 s)")
    ap.add_argument("--sampling", choices=("pz", "uniform"), default="pz",
                    help="x_k ~ p_z (operating distribution) or uniform on "
                         "[-A, A], A = D^{-1}(QUERY_A) (coverage by design)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run(smoke=args.smoke, quiet=not args.verbose, sampling=args.sampling)
