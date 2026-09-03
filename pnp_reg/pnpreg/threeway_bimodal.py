"""Experiment 3.2, extended to the three methods of tv_pm/prior_routes.

THE MISSING CELL. Across the repository the recovery methods have been compared
in three of four regimes; this run supplies the fourth -- a NONCONVEX prior at
FIELD DIMENSION, where the direct fit is mathematically excluded and the choice
between the two admissible routes is decided by the inversion's conditioning:

                      | f_reg CONVEX            | f_reg NONCONVEX
    ------------------+-------------------------+--------------------------
    1-D, exact        |  --                     | Exp 1: Iterative 0.02%
    inversion         |                         | beats Two-network 1.22%
    ------------------+-------------------------+--------------------------
    64-D, iterative   | TV: Direct 8.07% wins,  | THIS RUN
    solver            | Iterative 14.48% worst  |

METHODS (Experiment 3.2's data, network, budget and scoring, unchanged):

  Iterative     psi_theta fitted (grad psi(z_k) = y_k), then
                J_1(y) = <y,w> - psi(w) - 0.5||y||^2, grad psi(w) = y,
                w by src.invert.invert_cvx_gd -- the SAME Adam
                inverter, lr and alpha tv_pm uses. An inversion per
                query point. Class: 1-SEMICONVEX for free, because
                psi* is convex and the -0.5||.||^2 is in the Fenchel
                formula.
  Two-network   G_theta ~ psi*, trained on (y_k, z_k) with y_k the
                DENOISER's own output, J_2 = G - 0.5||y||^2. One
                forward pass. Class: 1-SEMICONVEX, same reason.
                This is Experiment 3.2's fit (a).
  Direct fit    plain ICNN, grad J(y_k) = z_k - y_k. Experiment 3.2's
                fit (b), the convex control. Class: CONVEX -- it does
                not contain a nonconvex f_reg.

ANCHORING. G is trained on the denoiser's own outputs y_k = D(z_k), which ARE
the training data. Manufacturing G's inputs from psi_theta instead -- the literal
reading of bin/_run.py's conjugate-sampling step, which had no denoiser data to
anchor on -- costs a factor of 19 in value error at n = 64 (measured 2026-08):
psi_theta's error enters through the conjugate and accumulates as a systematic
value offset while the gradients stay fine. Data-anchoring is free and strictly
better, so it is what "Two-network" means here and in tv_pm/prior_routes.

Everything is scored against the closed-form backward viscosity solution in
bimodal.py, on the operating support, exactly as bimodal_run.py does.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from . import bimodal as bm
from . import bimodal_run as br
from . import paths
from src.gradfit import train_grad, net_grad, net_value, Units
from src.network import LPN
from src.invert import invert_cvx_gd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")
CKPT = os.path.join(OUT, "ckpt")


INVERT_ITERS = 20_000
INVERT_LR = 1e-2          # pinned by the inversion's own certificate, as in tv_pm
ALPHA = 0.0               # the unregularized inversion; alpha>0 biases the prior


class PsiInY:
    """psi_theta in Y-UNITS, so src/invert.py can be used verbatim.

    psi_theta is fitted in standardized units; invert_cvx_gd optimizes
    model.scalar(v) - <y, v> and needs grad psi in the same units as the query.
    """

    def __init__(self, net, units):
        self.net = net
        self.mu = torch.tensor(np.asarray(units.mu)).float()
        self.s = torch.tensor(np.asarray(units.s)).float()

    def scalar(self, v):
        return self.net.scalar((v - self.mu) / self.s)

    def parameters(self):
        return self.net.parameters()


def fit_psi(ztr, ytr, zva, yva, steps, batch):
    """grad psi_theta(z_k) = y_k -- the LPN denoiser. Inputs z, targets y."""
    units = Units(ztr, standardize=True)
    model = LPN(in_dim=bm.N, hidden=br.CFG["hidden"], layers=br.CFG["layers"],
                beta=br.CFG["beta"])
    hist = train_grad(model, units.z(ztr), units.target(ytr),
                      units.z(zva), units.target(yva),
                      batch_size=batch, steps=steps, quiet=True)
    return model, units, hist


def invert(psi_wrapped, y, iters=INVERT_ITERS, alpha=ALPHA):
    return invert_cvx_gd(np.asarray(y, np.float32), psi_wrapped,
                         max_iters=iters, lr=INVERT_LR, alpha=alpha)


_PRE = {}


def preimage(psi_wrapped, y):
    """w(y) with grad psi_theta(w) = y, memoized on the exact query block.

    The value and the gradient of J_1 both need the SAME w, and every metric
    batches its queries into one array, so one Adam solve serves each metric
    instead of one per call. Without this the spectrum alone issued 63 modes x
    anchors separate 20000-iteration solves (measured: 18 min).
    """
    y = np.ascontiguousarray(np.asarray(y, np.float32))
    key = (y.shape, hash(y.tobytes()))
    if key not in _PRE:
        if len(_PRE) > 6:
            _PRE.clear()                  # bounded; blocks are a few MB each
        _PRE[key] = invert(psi_wrapped, y)
    return _PRE[key]


def inversion_certificate(psi_wrapped, y, w):
    """||grad psi(w) - y|| / max(1, ||y||): separates a failed solve from a
    wrong prior, exactly as tv_pm/prior_routes reports it."""
    wt = torch.tensor(np.asarray(w, np.float32)).requires_grad_(True)
    g = torch.autograd.grad(psi_wrapped.scalar(wt).sum(), wt)[0].detach().numpy()
    y = np.asarray(y)
    return np.linalg.norm(g - y, axis=1) / np.maximum(1.0, np.linalg.norm(y, axis=1))


# ---------------------------------------------------------------- scoring ---
def u_slice_curvature(valf, sigma, s_lo=-3.0, s_hi=3.0, n=241, decimate=4):
    """Min curvature along the nonconvex direction u, on bimodal_run's stencil."""
    s = np.linspace(s_lo, s_hi, n)
    v = valf(np.outer(s, bm.V[:, 0]))
    return float(br.curvature_on_slice(v, s[1] - s[0], decimate).min()), s, v


def spectrum(gradf, sigma, anchors, n_w=21):
    """Per-mode curvature by the slope of the gradient slice (bimodal_run's
    instrument iii), with EVERY query point of every mode and anchor stacked
    into ONE call to gradf. The instrument is unchanged; only the evaluation
    order is, because for the iterative route each gradf call is an inversion.
    """
    g = bm.gains(sigma)
    blocks, meta = [], []
    for k in range(1, bm.N):
        wr = 2 * g[k - 1] * np.sqrt(bm.LAM[k - 1] + sigma ** 2)
        wg = np.linspace(-wr, wr, n_w)
        for y0 in anchors:
            blocks.append(y0[None, :] + np.outer(wg, bm.V_PERP[:, k - 1]))
            meta.append((k, wg))
    G = gradf(np.concatenate(blocks, axis=0))
    coefs, acc, cur = np.zeros(bm.N - 1), {}, 0
    for (k, wg), blk in zip(meta, blocks):
        gr = G[cur:cur + len(wg)]
        cur += len(wg)
        acc.setdefault(k, []).append(np.polyfit(wg, gr @ bm.V_PERP[:, k - 1], 1)[0])
    for k, vals in acc.items():
        coefs[k - 1] = float(np.mean(vals))
    return coefs


def _load(name, sigma):
    p = os.path.join(CKPT, f"threeway_bimodal_{name}_s{sigma}.pth")
    if not os.path.exists(p):
        return None
    ck = torch.load(p, weights_only=False)
    m = LPN(in_dim=bm.N, hidden=br.CFG["hidden"], layers=br.CFG["layers"],
            beta=br.CFG["beta"])
    m.load_state_dict(ck["state"]); m.eval()
    return m


def _save(name, net, sigma):
    os.makedirs(CKPT, exist_ok=True)
    torch.save({"state": net.state_dict()},
               os.path.join(CKPT, f"threeway_bimodal_{name}_s{sigma}.pth"))


def run(sigma=0.5, steps=None, batch=None, smoke=False, force=False):
    steps = steps or (400 if smoke else br.CFG["steps"])
    batch = batch or br.CFG["batch"]
    n_tr, n_va, n_ev = ((4000, 1000, 800) if smoke else
                        (br.CFG["n_train"], br.CFG["n_val"], br.CFG["n_eval"]))
    (ytr, ztr), (yva, zva), (yev, zev) = br.make_data(sigma, n_tr, n_va, n_ev)
    print(f"sigma={sigma} t={sigma**2}  n_train={n_tr} steps={steps}", flush=True)

    psi_u = Units(ztr, standardize=True)
    psi_net = None if force else _load("psi", sigma)
    if psi_net is None:
        t0 = time.time(); psi_net, psi_u, h = fit_psi(ztr, ytr, zva, yva, steps, batch)
        _save("psi", psi_net, sigma)
        print(f"  psi_theta   {(time.time()-t0)/60:.1f} min  best val {h['best_val']:.3e}", flush=True)
    else:
        print("  psi_theta   cached", flush=True)

    G_u = Units(ytr, standardize=True)
    G = None if force else _load("G", sigma)
    if G is None:
        t0 = time.time(); G, G_u, h = br.fit("a", ytr, ztr, yva, zva, steps, batch)
        _save("G", G, sigma)
        print(f"  G_theta     {(time.time()-t0)/60:.1f} min  best val {h['best_val']:.3e}", flush=True)
    else:
        print("  G_theta     cached", flush=True)

    Jd_u = Units(ytr, standardize=True)
    Jd = None if force else _load("direct", sigma)
    if Jd is None:
        t0 = time.time(); Jd, Jd_u, h = br.fit("b", ytr, ztr, yva, zva, steps, batch)
        _save("direct", Jd, sigma)
        print(f"  J_direct    {(time.time()-t0)/60:.1f} min  best val {h['best_val']:.3e}", flush=True)
    else:
        print("  J_direct    cached", flush=True)

    psi_w = PsiInY(psi_net, psi_u)

    def val_it(y):
        w = preimage(psi_w, y)
        with torch.no_grad():
            pw = psi_w.scalar(torch.tensor(w).float()).numpy().ravel()
        y = np.asarray(y)
        return (y * w).sum(axis=1) - pw - 0.5 * (y ** 2).sum(axis=1)

    methods = [
        ("Iterative", val_it,
         lambda y: preimage(psi_w, y) - np.asarray(y), "1-semiconvex", "inversion/query"),
        ("Two-network", lambda y: br.value_J(G, G_u, y, "a"),
         lambda y: br.grad_G(G, G_u, y) - np.asarray(y), "1-semiconvex", "forward"),
        ("Direct fit", lambda y: br.value_J(Jd, Jd_u, y, "b"),
         lambda y: br.grad_G(Jd, Jd_u, y), "convex", "forward"),
    ]

    exact_val = bm.freg(yev, sigma)
    exact_perp = bm.freg_perp_coeffs(sigma)
    anchors = yev[:(2 if smoke else 6)]
    su, s_grid, ex_uv = u_slice_curvature(lambda Y: bm.freg(Y, sigma), sigma)
    tgt = zev - yev
    tgt_n = np.linalg.norm(tgt, axis=1)

    rows, curves = [], {"s": s_grid, "exact": ex_uv}
    for name, valf, gradf, cls, cost in methods:
        t0 = time.time()
        rel = br.rel_l2_centered(valf(yev), exact_val)
        r = np.linalg.norm(gradf(yev) - tgt, axis=1) / tgt_n
        cmin, _, uv = u_slice_curvature(valf, sigma)
        sp = spectrum(gradf, sigma, anchors)
        rows.append(dict(
            method=name, cls=cls, cost=cost, rel_l2=rel, u_min_curv=cmin,
            perp_rel_l2=float(np.linalg.norm(sp - exact_perp) / np.linalg.norm(exact_perp)),
            perp_worst=float(np.max(np.abs(sp - exact_perp) / np.abs(exact_perp))),
            resid_median=float(np.median(r)), resid_p90=float(np.percentile(r, 90)),
            score_s=time.time() - t0))
        curves[name] = uv
        print(f"  scored {name:24s} {(time.time()-t0)/60:.1f} min", flush=True)

    w_ev = preimage(psi_w, yev)
    cert = inversion_certificate(psi_w, yev, w_ev)

    print(f"\nexact u-slice min curvature {su:.4f}   (f_reg floor -1)")
    print(f"inversion certificate  median {np.median(cert):.2e}  max {cert.max():.2e}"
          f"   max|w|_inf {np.abs(w_ev).max():.2f}")
    print(f"\n{'method':24s} {'class':14s} {'cost':16s} {'rel-L2':>9s} {'u curv':>9s} "
          f"{'perp L2':>9s} {'resid med':>10s} {'resid p90':>10s}")
    print("-" * 116)
    for r in rows:
        print(f"{r['method']:24s} {r['cls']:14s} {r['cost']:16s} {100*r['rel_l2']:8.2f}% "
              f"{r['u_min_curv']:9.4f} {100*r['perp_rel_l2']:8.2f}% "
              f"{100*r['resid_median']:9.2f}% {100*r['resid_p90']:9.2f}%")
    print("-" * 116)
    print(f"{'exact':24s} {'1-semiconvex':14s} {'':16s} {'--':>9s} {su:9.4f}")

    out = dict(sigma=sigma, t=sigma ** 2, steps=steps, exact_u_min_curv=su,
               inversion=dict(cert_median=float(np.median(cert)),
                              cert_max=float(cert.max()),
                              preimage_linf=float(np.abs(w_ev).max()),
                              iters=INVERT_ITERS, lr=INVERT_LR, alpha=ALPHA),
               rows=rows)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"threeway_bimodal_s{sigma}.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    np.savez_compressed(os.path.join(OUT, f"threeway_bimodal_curves_s{sigma}.npz"), **curves)

    # ---- figure ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from src.plotstyle import apply as _apply_style
        _apply_style()
    except Exception as e:
        print(f"  [plotstyle unavailable: {e}]")
    colors = {"Iterative": "#4C6EF5", "Two-network": "#F59F00",
              "Direct fit": "#2F9E44"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    dm = lambda v: v - v.mean()
    ax[0].plot(s_grid, dm(ex_uv), "k-", lw=2.5, label="exact  $t\\,J_{BVS}$")
    for nm, c in colors.items():
        ax[0].plot(s_grid, dm(curves[nm]), "--", lw=1.8, color=c, label=nm)
    # clip to the exact curve's own range: the iterative route blows up outside
    # the operating support (|s| > 2.5, where no data lives), and letting that
    # set the scale compresses the region the comparison is about.
    e = dm(ex_uv)
    ax[0].set_ylim(e.min() - 1.5, e.max() + 1.5)
    ax[0].set_xlabel("$s=\\langle u,y\\rangle$"); ax[0].set_ylabel("$f_{reg}$ (centred)")
    ax[0].set_title("Along the NONCONVEX direction $u$")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    d2 = lambda v: br.curvature_on_slice(v, s_grid[1] - s_grid[0], 4)
    xs = s_grid[::4][1:-1]
    ax[1].plot(xs, d2(ex_uv), "k-", lw=2.5, label="exact")
    for nm, c in colors.items():
        ax[1].plot(xs, d2(curves[nm]), "--", lw=1.8, color=c, label=nm)
    ax[1].axhline(0, color="#868E96", lw=1)
    ax[1].axhline(-1, color="#E03131", ls="--", lw=1.5)
    ax[1].set_ylim(-1.4, 1.4); ax[1].set_xlabel("$s$"); ax[1].set_ylabel("curvature")
    ax[1].set_title("Curvature")
    ax[1].legend(); ax[1].grid(alpha=0.3)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain even if cached")
    a = ap.parse_args()
    run(sigma=a.sigma, steps=a.steps, smoke=a.smoke, force=a.force)
