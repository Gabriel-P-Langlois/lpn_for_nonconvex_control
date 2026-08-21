"""Experiment 3.2, four methods: reuses the checkpoints from threeway_bimodal.

WHY A FOURTH ROW. `tv_pm/prior_routes` defines One-shot recovery as the conjugate
of the LEARNED psi: G is trained on pairs manufactured from psi_theta, so its
inputs are psi_theta's outputs. Experiment 3.2's own fit (a) trains the same
network on the SAME targets but anchored on the TRUE denoiser outputs y_k = D(z_k).

In 1-D those coincide (psi_theta is near-exact, so its outputs are the data). At
n = 64 they do not, and the gap is large. Both belong in the table:

  One-shot (psi-anchored)   inputs grad psi_theta(z_k)  -- Route 2 as defined
  One-shot (data-anchored)  inputs y_k = D(z_k)         -- Exp 3.2's fit (a)

The second is the better estimator and is available whenever the denoiser's
outputs are the training data -- which is the entire premise of the task. The
first inherits psi_theta's error through the conjugate.
"""
import json
import os
import time

import numpy as np
import torch

from . import bimodal as bm
from . import bimodal_run as br
from . import paths
from . import threeway_bimodal as tw
from src.gradfit import net_grad, Units
from src.network import LPN

OUT, CKPT = tw.OUT, tw.CKPT


def load(name, sigma):
    ck = torch.load(os.path.join(CKPT, f"threeway_bimodal_{name}_s{sigma}.pth"),
                    weights_only=False)
    m = LPN(in_dim=bm.N, hidden=br.CFG["hidden"], layers=br.CFG["layers"],
            beta=br.CFG["beta"])
    m.load_state_dict(ck["state"]); m.eval()
    return m


def run(sigma=0.5, steps=None, batch=None):
    steps = steps or br.CFG["steps"]
    batch = batch or br.CFG["batch"]
    (ytr, ztr), (yva, zva), (yev, zev) = br.make_data(
        sigma, br.CFG["n_train"], br.CFG["n_val"], br.CFG["n_eval"])

    psi_net = load("psi", sigma)
    psi_u = Units(ztr, standardize=True)
    G_psi = load("G", sigma)
    yh = net_grad(psi_net, psi_u.z(ztr)) / np.asarray(psi_u.s)
    G_psi_u = Units(yh, standardize=True)
    Jd = load("direct", sigma)
    Jd_u = Units(ytr, standardize=True)

    # the fourth row: Experiment 3.2's fit (a), anchored on the true y_k
    p4 = os.path.join(CKPT, f"threeway_bimodal_Gdata_s{sigma}.pth")
    if os.path.exists(p4):
        G_dat = load("Gdata", sigma); G_dat_u = Units(ytr, standardize=True)
        print("G_data: cached")
    else:
        t0 = time.time()
        G_dat, G_dat_u, h = br.fit("a", ytr, ztr, yva, zva, steps, batch)
        torch.save({"state": G_dat.state_dict()}, p4)
        print(f"G_data  {(time.time()-t0)/60:.1f} min  best val {h['best_val']:.3e}",
              flush=True)

    psi_w = tw.PsiInY(psi_net, psi_u)

    def val_it(y):
        w = tw.preimage(psi_w, y)
        with torch.no_grad():
            pw = psi_w.scalar(torch.tensor(w).float()).numpy().ravel()
        y = np.asarray(y)
        return (y * w).sum(axis=1) - pw - 0.5 * (y ** 2).sum(axis=1)

    def grad_it(y):
        return tw.preimage(psi_w, y) - np.asarray(y)

    methods = [
        ("LPN Iterative recovery", val_it, grad_it, "1-semiconvex", "inversion/query"),
        ("One-shot (psi-anchored)", lambda y: br.value_J(G_psi, G_psi_u, y, "a"),
         lambda y: br.grad_G(G_psi, G_psi_u, y) - np.asarray(y), "1-semiconvex", "forward"),
        ("One-shot (data-anchored)", lambda y: br.value_J(G_dat, G_dat_u, y, "a"),
         lambda y: br.grad_G(G_dat, G_dat_u, y) - np.asarray(y), "1-semiconvex", "forward"),
        ("Direct fit", lambda y: br.value_J(Jd, Jd_u, y, "b"),
         lambda y: br.grad_G(Jd, Jd_u, y), "convex", "forward"),
    ]

    exact_val = bm.freg(yev, sigma)
    exact_perp = bm.freg_perp_coeffs(sigma)
    anchors = yev[:6]
    su, s_grid, ex_uv = tw.u_slice_curvature(lambda Y: bm.freg(Y, sigma), sigma)
    tgt = zev - yev
    tgt_n = np.linalg.norm(tgt, axis=1)

    rows, curves = [], {"s": s_grid, "exact": ex_uv}
    for name, valf, gradf, cls, cost in methods:
        t0 = time.time()
        rel = br.rel_l2_centered(valf(yev), exact_val)
        r = np.linalg.norm(gradf(yev) - tgt, axis=1) / tgt_n
        cmin, _, uv = tw.u_slice_curvature(valf, sigma)
        sp = tw.spectrum(gradf, sigma, anchors)
        rows.append(dict(
            method=name, cls=cls, cost=cost, rel_l2=rel, u_min_curv=cmin,
            perp_rel_l2=float(np.linalg.norm(sp - exact_perp) / np.linalg.norm(exact_perp)),
            perp_worst=float(np.max(np.abs(sp - exact_perp) / np.abs(exact_perp))),
            resid_median=float(np.median(r)), resid_p90=float(np.percentile(r, 90)),
            score_s=time.time() - t0))
        curves[name] = uv
        print(f"  scored {name:26s} {(time.time()-t0)/60:.1f} min", flush=True)

    # the iterative route's own certificate: did the inversion converge?
    w_ev = tw.preimage(psi_w, yev)
    cert = tw.inversion_certificate(psi_w, yev, w_ev)
    pre_max = float(np.abs(w_ev).max())

    print(f"\nexact u-slice min curvature {su:.4f}   (f_reg floor -1)")
    print(f"inversion certificate  median {np.median(cert):.2e}  max {cert.max():.2e}"
          f"   max|w|_inf {pre_max:.2f}")
    print(f"\n{'method':26s} {'class':14s} {'cost':16s} {'rel-L2':>9s} {'u curv':>9s} "
          f"{'perp L2':>9s} {'resid med':>10s} {'resid p90':>10s}")
    print("-" * 118)
    for r in rows:
        print(f"{r['method']:26s} {r['cls']:14s} {r['cost']:16s} {100*r['rel_l2']:8.2f}% "
              f"{r['u_min_curv']:9.4f} {100*r['perp_rel_l2']:8.2f}% "
              f"{100*r['resid_median']:9.2f}% {100*r['resid_p90']:9.2f}%")
    print("-" * 118)
    print(f"{'exact':26s} {'1-semiconvex':14s} {'':16s} {'--':>9s} {su:9.4f}")

    out = dict(sigma=sigma, t=sigma ** 2, steps=steps, exact_u_min_curv=su,
               inversion=dict(cert_median=float(np.median(cert)),
                              cert_max=float(cert.max()), preimage_linf=pre_max,
                              iters=tw.INVERT_ITERS, lr=tw.INVERT_LR, alpha=tw.ALPHA),
               rows=rows)
    with open(os.path.join(OUT, f"threeway_bimodal_final_s{sigma}.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    np.savez_compressed(os.path.join(OUT, f"threeway_bimodal_curves_s{sigma}.npz"), **curves)

    # ---- figure -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from src.plotstyle import apply as _apply_style
        _apply_style()
    except Exception:
        pass
    colors = {"LPN Iterative recovery": "#4C6EF5", "One-shot (psi-anchored)": "#F59F00",
              "One-shot (data-anchored)": "#E8590C", "Direct fit": "#2F9E44"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    dm = lambda v: v - v.mean()
    ax[0].plot(s_grid, dm(ex_uv), "k-", lw=2.5, label="exact  $t\\,J_{BVS}$")
    for name in colors:
        ax[0].plot(s_grid, dm(curves[name]), "--", lw=1.8, color=colors[name], label=name)
    ax[0].set_xlabel("$s=\\langle u,y\\rangle$"); ax[0].set_ylabel("$f_{reg}$ (centred)")
    ax[0].set_title("Along the NONCONVEX direction $u$"); ax[0].legend(); ax[0].grid(alpha=0.3)

    d2 = lambda v: br.curvature_on_slice(v, s_grid[1] - s_grid[0], 4)
    xs = s_grid[::4][1:-1]
    ax[1].plot(xs, d2(ex_uv), "k-", lw=2.5, label="exact")
    for name in colors:
        ax[1].plot(xs, d2(curves[name]), "--", lw=1.8, color=colors[name], label=name)
    ax[1].axhline(0, color="#868E96", lw=1)
    ax[1].axhline(-1, color="#E03131", ls="--", lw=1.5)
    ax[1].set_ylim(-1.4, 1.4); ax[1].set_xlabel("$s$"); ax[1].set_ylabel("curvature")
    ax[1].set_title("Only the convex class cannot go below 0"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle(f"Bimodal field prior, $n=64$, $\\sigma={sigma}$, all nets {steps} steps — "
                 f"a NONCONVEX prior at field dimension")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    stem = os.path.join(OUT, "figs", f"threeway_bimodal_s{sigma}")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {stem}.png / .pdf")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=0.5)
    a = ap.parse_args()
    run(sigma=a.sigma)
