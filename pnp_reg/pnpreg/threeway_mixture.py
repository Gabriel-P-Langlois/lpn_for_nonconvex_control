"""Experiment 1, extended to the SAME three methods benchmarked in tv_pm/prior_routes.

Experiment 1 as shipped compares two fits. Both are direct gradient fits with no
inversion; they differ ONLY in the function class. That confounds two independent
axes -- "is there a recovery step?" and "which class is J in?" -- and leaves the
folder unable to say where the semiconvexity comes from. Adding the third method
separates them:

  Iterative     psi_theta fitted (grad psi(x_k) = y_k), then
                J_1(y) = <y,w> - psi(w) - 0.5 y^2 with grad psi(w) = y.
                An INVERSION per query point (1-D: monotone bisection).
                Class: psi* is convex, so J_1 is 1-SEMICONVEX for free.
  Two-network   G_theta ~ psi_theta* on conjugate pairs made from
                psi_theta, J_2 = G - 0.5 y^2. One forward pass.
                Class: 1-SEMICONVEX, same reason.
  Direct fit    J_theta a plain ICNN, grad J(y_k) = x_k - y_k.
                Class: CONVEX. Does not contain a nonconvex f_reg.

THE POINT. The -0.5||.||^2 in the first two is not a trick of the one-shot route:
it falls out of the Fenchel formula that BOTH LPN routes share. So semiconvexity is
a property of the conjugate structure, not of skipping the inversion -- and the
direct fit's convexity is an undocumented ASSUMPTION about the prior, invisible on
TV (where f_reg is convex and it wins) and fatal here (where f_reg has curvature
-0.91). Only a three-way run can distinguish those.

Budgets, network and data are Experiment 1's, unchanged, so the (a)/(b) numbers
reproduce; psi_theta gets the same budget as the other two.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from . import mixture as mx
from . import readout
from src.gradfit import net_grad, net_value, train_grad
from src.network import LPN

CFG = dict(hidden=64, layers=2, beta=20, batch=512, steps=20_000, seed=1)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")


def col(a):
    return np.asarray(a, float).reshape(-1, 1)


def fit(inputs, targets, vin, vtar, steps, seed=1):
    torch.manual_seed(seed)
    net = LPN(in_dim=1, hidden=CFG["hidden"], layers=CFG["layers"], beta=CFG["beta"])
    hist = train_grad(net, col(inputs), col(targets), col(vin), col(vtar),
                      batch_size=CFG["batch"], steps=steps, quiet=True)
    return net, hist["best_val"]


def invert_psi(psi_net, y, lo=-40.0, hi=40.0, iters=80):
    """w with grad psi_theta(w) = y, by bisection.

    In 1-D the inversion is EXACT to machine precision and costs nothing, which is
    deliberate: it removes the optimizer from the comparison, so any gap between
    Iterative and Two-network here is the recovery MATHEMATICS,
    not a solver artifact. (At field dimension the same step is the expensive,
    alpha-dependent solve that tv_pm/prior_routes measures.)
    """
    y = np.asarray(y, float)
    a = np.full_like(y, lo)
    b = np.full_like(y, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        gm = net_grad(psi_net, col(m)).ravel()
        left = gm < y
        a = np.where(left, m, a)
        b = np.where(left, b, m)
    return 0.5 * (a + b)


def J_iterative(psi_net, y):
    """J_1(y) = <y, w> - psi_theta(w) - 0.5 y^2,  grad psi_theta(w) = y."""
    w = invert_psi(psi_net, y)
    return y * w - net_value(psi_net, col(w)) - 0.5 * y ** 2


def demean(f, m):
    return f - f[m].mean()


def curvature(f, g, m, spacing=0.02):
    """Second difference on a DECIMATED stencil (Experiment 1, decision 2)."""
    dx = g[1] - g[0]
    k = max(1, int(round(spacing / dx)))
    fc, gc, mc = f[::k], g[::k], m[::k]
    d2 = (fc[2:] - 2 * fc[1:-1] + fc[:-2]) / (gc[1] - gc[0]) ** 2
    return float(d2[mc[1:-1]].min())


def run(sigma=0.5, steps=None, smoke=False):
    steps = steps or (500 if smoke else CFG["steps"])
    n, nv = (4000, 1000) if smoke else (20_000, 4000)
    x = mx.sample_pz(n, sigma, seed=1)
    xv = mx.sample_pz(nv, sigma, seed=2)
    y, yv = mx.D(x, sigma), mx.D(xv, sigma)
    print(f"sigma={sigma}  t={sigma**2}  n={n}  steps={steps}", flush=True)

    # --- psi_theta: the LPN denoiser.  grad psi(x_k) = y_k = D(x_k) ----------
    t0 = time.time()
    psi_net, val_psi = fit(x, y, xv, yv, steps)
    t_psi = time.time() - t0

    # --- Two-network: G ~ psi_theta*, on pairs MADE FROM psi_theta -------------
    # inputs are psi_theta's own outputs, not the exact denoiser's, so G is the
    # conjugate of the LEARNED psi -- the definition tv_pm/prior_routes uses.
    yh = net_grad(psi_net, col(x)).ravel()
    yhv = net_grad(psi_net, col(xv)).ravel()
    t0 = time.time()
    G, val_G = fit(yh, x, yhv, xv, steps)
    t_G = time.time() - t0

    # --- Direct fit: plain ICNN on the regularizer itself -------------------
    t0 = time.time()
    Jd, val_Jd = fit(y, x - y, yv, xv - yv, steps)
    t_Jd = time.time() - t0

    # --- score on the sampled range of D (Experiment 1, decision 1) ---------
    g, _ = mx.grid()
    lo, hi = np.quantile(y, [0.005, 0.995])
    m = (g >= lo) & (g <= hi)
    exact = mx.freg(g, sigma)

    t0 = time.time(); f_it = J_iterative(psi_net, g); t_eval_it = time.time() - t0
    t0 = time.time(); f_os = readout.value(G, g);     t_eval_os = time.time() - t0
    t0 = time.time(); f_dr = net_value(Jd, col(g));   t_eval_dr = time.time() - t0

    # certificate: ||grad J(y) - (x - y)|| / ||x - y|| on held-out points
    xh, yh2 = xv, yv
    tgt = xh - yh2
    def resid(gr):
        return float(np.median(np.abs(gr - tgt) / np.abs(tgt)))
    w_h = invert_psi(psi_net, yh2)
    r_it = resid(w_h - yh2)                      # grad J_1(y) = w(y) - y
    r_os = resid(readout.grad(G, yh2))
    r_dr = resid(net_grad(Jd, col(yh2)).ravel())

    ex_c = demean(exact, m)
    rows = []
    for name, f, r, t_tr, t_ev, cls in [
        ("Iterative", f_it, r_it, t_psi, t_eval_it, "1-semiconvex"),
        ("Two-network", f_os, r_os, t_psi + t_G, t_eval_os, "1-semiconvex"),
        ("Direct fit", f_dr, r_dr, t_Jd, t_eval_dr, "convex"),
    ]:
        fc = demean(f, m)
        rows.append(dict(
            method=name, cls=cls,
            rel_l2=float(np.linalg.norm((fc - ex_c)[m]) / np.linalg.norm(ex_c[m])),
            min_curv=curvature(f, g, m),
            resid_median=r,
            train_s=t_tr, eval_s=t_ev))

    ex_curv = curvature(exact, g, m)
    print(f"\nexact min curvature on the window: {ex_curv:.4f}   (floor -1/t = {-1/sigma**2:.1f}"
          f", f_reg units -1)")
    print(f"window [{lo:.3f}, {hi:.3f}]\n")
    print(f"{'method':26s} {'class':14s} {'rel-L2':>9s} {'min curv':>10s} "
          f"{'resid':>9s} {'train s':>9s} {'eval s':>9s}")
    print("-" * 92)
    for r in rows:
        print(f"{r['method']:26s} {r['cls']:14s} {100*r['rel_l2']:8.2f}% "
              f"{r['min_curv']:10.4f} {100*r['resid_median']:8.2f}% "
              f"{r['train_s']:9.1f} {r['eval_s']:9.3f}")
    print("-" * 92)
    print(f"{'exact':26s} {'1-semiconvex':14s} {'--':>9s} {ex_curv:10.4f}")

    # ---- figure: the class failure, displayed ------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:                                  # one font size, repo-wide
        from src.plotstyle import apply as _apply_style
        _apply_style()
    except Exception as e:                # never let styling kill a long run
        print(f"  [plotstyle unavailable: {e}; using matplotlib defaults]")

    gm = g[m]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].plot(gm, demean(exact, m)[m], "k-", lw=2.5, label="exact  $f_{reg}=t\\,J_{BVS}$")
    for (name, f), c, ls in zip([("Iterative", f_it),
                                 ("Two-network", f_os),
                                 ("Direct fit", f_dr)],
                                ["#4C6EF5", "#F59F00", "#2F9E44"], ["--", "-.", ":"]):
        ax[0].plot(gm, demean(f, m)[m], ls, lw=2, color=c, label=name)
    ax[0].set_xlabel("$y$"); ax[0].set_ylabel("$f_{reg}$ (centred)")
    ax[0].set_title("The recovered regularizer")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    dx = g[1] - g[0]; k = max(1, int(round(0.02 / dx)))
    gc, mc = g[::k], m[::k]
    d2 = lambda f: (f[::k][2:] - 2 * f[::k][1:-1] + f[::k][:-2]) / (gc[1] - gc[0]) ** 2
    xx = gc[1:-1][mc[1:-1]]
    ax[1].plot(xx, d2(exact)[mc[1:-1]], "k-", lw=2.5, label="exact")
    for (name, f), c, ls in zip([("Iterative", f_it),
                                 ("Two-network", f_os),
                                 ("Direct fit", f_dr)],
                                ["#4C6EF5", "#F59F00", "#2F9E44"], ["--", "-.", ":"]):
        ax[1].plot(xx, d2(f)[mc[1:-1]], ls, lw=2, color=c, label=name)
    ax[1].axhline(0, color="#868E96", lw=1)
    ax[1].axhline(-1, color="#E03131", ls="--", lw=1.5)
    ax[1].text(gm[0], -1 + 0.06, "semiconvexity floor $-1$", color="#E03131")
    ax[1].set_xlabel("$y$"); ax[1].set_ylabel("curvature  $f_{reg}''$")
    ax[1].set_title("Curvature")
    ax[1].set_ylim(-1.35, 1.2); ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.suptitle(f"Three recoveries of a NONCONVEX prior  —  1-D mixture, "
                 f"$\\sigma={sigma}$, all nets {steps} steps\n"
                 f"both LPN routes are 1-semiconvex by construction and find the dip; "
                 f"the convex class flattens it")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    stem = os.path.join(OUT, "figs", f"threeway_mixture_sigma{sigma}")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {stem}.png / .pdf")

    out = dict(sigma=sigma, t=sigma ** 2, steps=steps, n=n,
               window=[float(lo), float(hi)], exact_min_curv=ex_curv,
               best_val=dict(psi=val_psi, G=val_G, direct=val_Jd), rows=rows)
    os.makedirs(os.path.join(OUT, "ckpt"), exist_ok=True)
    for nm, net in (("psi", psi_net), ("G", G), ("direct", Jd)):
        torch.save({"state": net.state_dict(), "hidden": CFG["hidden"],
                    "layers": CFG["layers"], "beta": CFG["beta"], "steps": steps},
                   os.path.join(OUT, "ckpt", f"threeway_mixture_{nm}_s{sigma}.pth"))
    np.savez_compressed(os.path.join(OUT, f"threeway_mixture_curves_s{sigma}.npz"),
                        g=g, m=m, exact=exact, it=f_it, os=f_os, dr=f_dr)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"threeway_mixture_sigma{sigma}.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\n-> {p}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(sigma=a.sigma, steps=a.steps, smoke=a.smoke)
