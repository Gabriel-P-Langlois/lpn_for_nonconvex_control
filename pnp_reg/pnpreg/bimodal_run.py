"""Experiment 3.2 driver: recover the backward viscosity solution.

    ~/miniforge3/envs/lpn_env/bin/python -m pnpreg.bimodal_run --smoke   (~2 min)
    ~/miniforge3/envs/lpn_env/bin/python -m pnpreg.bimodal_run           (~45 min, CPU)

The exact model is pnpreg/bimodal.py: a bimodal field prior on 8x8 patches,
nonconvex along one bump direction u, Gaussian with spectrum lambda_k on the
other 63 directions. Training pairs are (y_k, z_k) with y_k = D(z_k) from
the EXACT denoiser; two fits per noise level, both LPN(64, hidden 256,
layers 2, beta 20), both trained by src.gradfit.train_grad:

  fit (a)  semiconvex readout: grad G_theta(y_k) = z_k, J_theta = G_theta - q.
           Estimates f_reg = t * J_BVS -- the network solves the backward
           Hamilton-Jacobi problem from forward-flow samples.
  fit (b)  convex control: plain ICNN, grad J_b(y_k) = z_k - y_k.
           Must fail along u (the target is nonconvex there) and succeed on
           the 63 convex directions -- the class-separation result.

Metrics (all against closed forms; constants aligned by mean difference,
since gradient training leaves the additive constant free):
relative L2 of centered values on held-out y = D(z); the u-slice with its
curvature against the exact values and the -1 floor (in f_reg units);
the perp spectrum from the fitted Hessian in V-coordinates vs t/lambda_k,
with the off-block norm as the fit's separability diagnostic; held-out
residuals (median and 95th percentile). Outputs (tracked):
results/bimodal_metrics.json, figs/experiment32_bimodal.{png,pdf};
checkpoints under results/ckpt/ (gitignored, regenerable).

Budget: 50 000 steps per fit (the Experiment-1-style economy budget,
recorded in DESIGN.md as a deviation from the 250k protocol); CPU only.
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch

from . import bimodal as bm
from . import paths
from src.gradfit import train_grad, net_grad, net_value, Units
from src.network import LPN

SEED_TRAIN, SEED_VAL, SEED_EVAL = 1, 2, 3
CFG = dict(hidden=256, layers=2, beta=20, batch=512, steps=50_000,
           n_train=40_000, n_val=8_000, n_eval=4_000)


def make_data(sigma, n_train, n_val, n_eval):
    zt, _ = bm.sample_data(n_train, sigma, SEED_TRAIN)
    zv, _ = bm.sample_data(n_val, sigma, SEED_VAL)
    ze, _ = bm.sample_data(n_eval, sigma, SEED_EVAL)
    return (bm.D(zt, sigma), zt), (bm.D(zv, sigma), zv), (bm.D(ze, sigma), ze)


def fit(which, ytr, ztr, yva, zva, steps, batch, quiet=True):
    """One gradient-supervised fit. which = 'a' (target z, G = f_reg + q) or
    'b' (target z - y, plain convex J). Returns (model, units, hist)."""
    units = Units(ytr, standardize=True)
    X, Xv = units.z(ytr), units.z(yva)
    if which == "a":
        G, Gv = units.target(ztr), units.target(zva)
    else:
        G, Gv = units.target(ztr - ytr), units.target(zva - yva)
    model = LPN(in_dim=bm.N, hidden=CFG["hidden"], layers=CFG["layers"],
                beta=CFG["beta"])
    hist = train_grad(model, X, G, Xv, Gv, batch_size=batch, steps=steps,
                      quiet=quiet)
    return model, units, hist


def value_J(model, units, y, which):
    """The recovered regularizer's values at y: fit (a) subtracts the
    quadratic at readout; fit (b) is the regularizer directly."""
    v = net_value(model, units.z(y))
    if which == "a":
        return v - 0.5 * (np.atleast_2d(y) ** 2).sum(axis=1)
    return v


def grad_G(model, units, y):
    """grad of the trained potential in y-units (chain rule through the
    standardization)."""
    return net_grad(model, units.z(y)) / np.asarray(units.s)


def rel_l2_centered(a, b):
    """Relative L2 after removing each side's mean: the additive constant is
    not identified by gradient training."""
    a = a - a.mean()
    b = b - b.mean()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def curvature_on_slice(vals, dx, decimate):
    """Second difference on a decimated stencil (the Experiment-1 float32
    lesson: native-spacing stencils read quantization noise)."""
    v = vals[::decimate]
    return (v[2:] - 2 * v[1:-1] + v[:-2]) / (decimate * dx) ** 2


def spectrum_by_slices(model, units, which, sigma, anchors):
    """Per-mode curvature as the slope of the GRADIENT slice: a line fitted
    to <grad J_theta(w v_k), v_k> over the operating range of the mode
    (|w| <= 2 g_k sqrt(lam_k + sigma^2), the std of y_k = g_k z_k).

    Chosen after two failed instruments (measured 2026-07-31, DESIGN.md):
    the pointwise Hessian carries O(0.1) spatial wiggle that swamps the
    soft modes (curvature 0.06-0.25), and the value slice's quadratic
    signal over the operating range is ~2e-6 at the stiff modes, below
    float32 value noise. The gradient slope's signal is curvature x range
    (0.2-1.0 for EVERY mode, far above gradient noise), and the gradient
    is the quantity training supervised. Slices are ANCHORED at held-out
    operating points and averaged: a slice through the origin has u-coordinate
    s = 0, the unsampled inter-mode gap, where the fit is inductive bias
    (measured 2026-07-31, DESIGN.md)."""
    coefs = np.zeros(bm.N - 1)
    g = bm.gains(sigma)
    for k in range(1, bm.N):
        lam = bm.LAM[k - 1]
        wr = 2 * g[k - 1] * np.sqrt(lam + sigma ** 2)
        wg = np.linspace(-wr, wr, 41)
        acc = 0.0
        for y0 in anchors:
            Y = y0[None, :] + np.outer(wg, bm.V_PERP[:, k - 1])
            gr = grad_G(model, units, Y)
            if which == "a":
                gr = gr - Y
            acc += np.polyfit(wg, gr @ bm.V_PERP[:, k - 1], 1)[0]
        coefs[k - 1] = acc / len(anchors)
    return coefs


def hessian_in_V(model, units, y_pts, which):
    """The fitted regularizer's Hessian at y_pts, rotated to V-coordinates.
    Fit (a): hess J = hess G / s_i s_j - I. Returns (mean H_V, per-point)."""
    Hs = []
    s = torch.tensor(np.asarray(units.s), dtype=torch.float64)
    mu = torch.tensor(np.asarray(units.mu), dtype=torch.float64)
    model64 = model.double()
    for y0 in y_pts:
        z0 = (torch.tensor(y0, dtype=torch.float64) - mu) / s
        H = torch.autograd.functional.hessian(
            lambda zz: model64.scalar(zz.unsqueeze(0)).sum(), z0)
        Hy = (H / (s[:, None] * s[None, :])).numpy()
        if which == "a":
            Hy = Hy - np.eye(bm.N)
        Hs.append(bm.V.T @ Hy @ bm.V)
    return np.mean(Hs, axis=0), Hs


def run_sigma(sigma, steps, batch, n_train, n_val, n_eval, quiet=True,
              rescore=False):
    t = sigma ** 2
    print(f"== sigma = {sigma} (t = {t})", flush=True)
    (ytr, ztr), (yva, zva), (yev, zev) = make_data(sigma, n_train, n_val, n_eval)
    out = {"sigma": sigma, "t": t}
    models = {}
    for which in ("a", "b"):
        ckpt_path = os.path.join(paths.RESULTS, "ckpt", f"bimodal_s{sigma:g}_{which}.pth")
        t0 = time.time()
        if rescore:
            # metrics-only pass over an existing checkpoint (no retraining)
            ck = torch.load(ckpt_path, weights_only=False)
            model = LPN(in_dim=bm.N, hidden=ck["cfg"]["hidden"],
                        layers=ck["cfg"]["layers"], beta=ck["cfg"]["beta"])
            model.load_state_dict(ck["state"])
            model.eval()
            units = Units.from_saved(ck["mu"], ck["s"])
            hist = {"best_val": float("nan")}
        else:
            model, units, hist = fit(which, ytr, ztr, yva, zva, steps, batch, quiet)
        secs = time.time() - t0
        models[which] = (model, units)
        if not rescore:
            os.makedirs(os.path.join(paths.RESULTS, "ckpt"), exist_ok=True)
            torch.save({"state": model.state_dict(), "mu": np.asarray(units.mu),
                        "s": np.asarray(units.s), "which": which, "sigma": sigma,
                        "steps": steps, "cfg": CFG}, ckpt_path)
        # values on held-out operating support
        Jhat = value_J(model, units, yev, which)
        rel = rel_l2_centered(Jhat, bm.freg(yev, sigma))
        # residual certificate
        if which == "a":
            resid = np.linalg.norm(grad_G(model, units, yev) - zev, axis=1)
        else:
            resid = np.linalg.norm(grad_G(model, units, yev) - (zev - yev), axis=1)
        resid = resid / np.maximum(np.linalg.norm(zev, axis=1), 1.0)
        out[which] = {
            "best_val": hist.get("best_val"),
            "seconds": round(secs, 1),
            "rel_l2_values": rel,
            "resid_median": float(np.median(resid)),
            "resid_q95": float(np.quantile(resid, 0.95)),
        }
        print(f"  fit ({which}): val {hist.get('best_val'):.3e} | rel L2 {rel:.3%} | "
              f"resid med {out[which]['resid_median']:.2%} q95 {out[which]['resid_q95']:.2%} | "
              f"{secs:.0f}s", flush=True)

    # u-slice: values and curvature (fit vs exact), constants aligned
    sg = np.arange(-2.5, 2.5 + 1e-9, 0.01)
    y_slice = np.outer(sg, bm.U)
    exact_slice = bm.freg_u_slice(sg, sigma)
    slices = {"s": sg.tolist(), "exact": exact_slice.tolist(),
              "exact_tJ": (t * (bm.J_true(y_slice) - bm.J_true(np.zeros((1, bm.N)))[0]
                                + exact_slice[len(sg) // 2])).tolist()}
    dec = 5   # 0.05 stencil
    curv = {"exact": curvature_on_slice(exact_slice, 0.01, dec)}
    for which in ("a", "b"):
        model, units = models[which]
        v = value_J(model, units, y_slice, which)
        v = v - v.mean() + exact_slice.mean()
        slices[which] = v.tolist()
        curv[which] = curvature_on_slice(np.array(v), 0.01, dec)
    out["u_slice"] = slices
    out["curvature"] = {
        "exact_min": float(curv["exact"].min()),
        "a_min": float(curv["a"].min()),
        "b_min": float(curv["b"].min()),
        "floor": -1.0,
        "stencil": dec * 0.01,
    }
    print(f"  u-slice min curvature: exact {out['curvature']['exact_min']:.3f} | "
          f"fit a {out['curvature']['a_min']:.3f} | fit b {out['curvature']['b_min']:.3f} "
          f"(floor -1)", flush=True)

    # perp spectrum from value slices (the robust readout); the pointwise
    # Hessian serves only the separability diagnostic
    exact_coeffs = bm.freg_perp_coeffs(sigma)
    for which in ("a", "b"):
        model, units = models[which]
        diag = spectrum_by_slices(model, units, which, sigma, yev[:4])
        rel_spec = float(np.linalg.norm(diag - exact_coeffs) / np.linalg.norm(exact_coeffs))
        rel_max = float(np.max(np.abs(diag - exact_coeffs) / exact_coeffs))
        HV, _ = hessian_in_V(model, units, yev[:12], which)
        off = HV - np.diag(np.diag(HV))
        out[which]["spectrum_rel_l2"] = rel_spec
        out[which]["spectrum_rel_max_per_mode"] = rel_max
        out[which]["hessian_offdiag_max"] = float(np.abs(off).max())
        out[which]["hessian_u_curv"] = float(HV[0, 0])
        out[which]["spectrum_fit"] = diag.tolist()
        print(f"  fit ({which}) perp spectrum: rel L2 {rel_spec:.3%}, "
              f"max per-mode rel err {rel_max:.3%} | "
              f"hess offdiag max {out[which]['hessian_offdiag_max']:.3e}", flush=True)
    out["spectrum_exact"] = exact_coeffs.tolist()
    return out


def make_figure(results, out_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = results[0]
    sg = np.array(r["u_slice"]["s"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax = axes[0]
    ax.plot(sg, r["u_slice"]["exact_tJ"], "k:", lw=1.2, label=r"$t\,J$ (true prior)")
    ax.plot(sg, r["u_slice"]["exact"], "k-", lw=1.6, label=r"$t\,J_{BVS}$ (exact target)")
    ax.plot(sg, r["u_slice"]["a"], "C0--", lw=1.4, label="fit (a): semiconvex readout")
    ax.plot(sg, r["u_slice"]["b"], "C3-.", lw=1.4, label="fit (b): convex control")
    ax.set_xlabel(r"$s$ (coordinate along $u$)")
    ax.set_ylabel(r"$f_{\mathrm{reg}}$")
    ax.set_title(f"the nonconvex direction (sigma = {r['sigma']})")
    ax.legend(fontsize=8)

    ax = axes[1]
    k = np.arange(1, bm.N)
    ax.loglog(k, r["spectrum_exact"], "k-", lw=1.6, label=r"exact $t/\lambda_k$")
    ax.loglog(k, r["a"]["spectrum_fit"], "C0--", lw=1.2, label="fit (a)")
    ax.loglog(k, r["b"]["spectrum_fit"], "C3-.", lw=1.2, label="fit (b)")
    ax.set_xlabel(r"perp mode $k$")
    ax.set_ylabel("curvature")
    ax.set_title("the 63 Gaussian directions")
    ax.legend(fontsize=8)

    ax = axes[2]
    labels = ["fit (a)", "fit (b)"]
    rel = [100 * r["a"]["rel_l2_values"], 100 * r["b"]["rel_l2_values"]]
    spec = [100 * r["a"]["spectrum_rel_l2"], 100 * r["b"]["spectrum_rel_l2"]]
    x = np.arange(2)
    ax.bar(x - 0.18, rel, 0.32, label="values rel. L2 (%)")
    ax.bar(x + 0.18, spec, 0.32, label="perp spectrum rel. L2 (%)")
    ax.set_xticks(x, labels)
    ax.set_yscale("log")
    ax.set_title("recovery errors vs the exact backward solution")
    ax.legend(fontsize=8)

    fig.suptitle("Experiment 3.2: the backward viscosity solution recovered by a "
                 "structure-preserving network (bimodal field prior, n = 64)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(paths.FIGS, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.5])
    ap.add_argument("--rescore", action="store_true",
                    help="metrics + figure from saved checkpoints, no retraining")
    ap.add_argument("--steps", type=int, default=CFG["steps"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    steps, n_train, n_val, n_eval = a.steps, CFG["n_train"], CFG["n_val"], CFG["n_eval"]
    if a.smoke:
        steps, n_train, n_val, n_eval = 1500, 6_000, 1_500, 1_000
        a.tag = a.tag or "smoke"

    results = [run_sigma(s, steps, CFG["batch"], n_train, n_val, n_eval,
                         rescore=a.rescore)
               for s in a.sigmas]

    tag = f"_{a.tag}" if a.tag else ""
    paths.ensure_dirs()
    out = {"config": {**CFG, "steps": steps, "n_train": n_train, "n_val": n_val,
                      "n_eval": n_eval, "sigmas": a.sigmas,
                      "seeds": [SEED_TRAIN, SEED_VAL, SEED_EVAL],
                      "date": datetime.datetime.now().isoformat(timespec="seconds"),
                      "torch": torch.__version__},
           "runs": results}
    jpath = os.path.join(paths.RESULTS, f"bimodal_metrics{tag}.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {jpath}", flush=True)
    make_figure(results, os.path.join(paths.FIGS, f"experiment32_bimodal{tag}"))
    print(f"-> {paths.FIGS}/experiment32_bimodal{tag}.png/pdf", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
