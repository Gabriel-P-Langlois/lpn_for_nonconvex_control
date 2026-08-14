"""Compare the three RECOVERED PRIORS against the sampler's u_PM.

    LPN Iterative recovery  J_1(x) = <x, w> - psi(w) - 0.5||x||^2,  grad psi(w) = x
                       -- an INVERSION per query point (src/invert.py, alpha its
                          only knob), exactly as arXiv:2310.14344 specifies.
    One-shot recovery         J_2(x) = G(x) - 0.5||x||^2
                       -- one FORWARD PASS through the conjugate net.
    Direct fit      J_theta(x)
                       -- one FORWARD PASS; nothing was recovered, J was fitted.

HOW A PRIOR IS SCORED AGAINST u_PM, WITH NO GROUND TRUTH. There is none to be
had: evaluating f_reg needs S_eps, the log-partition, and the sampler returns
only the posterior MEAN. But u_PM IS a prox (Gribonval), so its optimality
condition pins any correct prior on range(u_PM):

    u_PM(x) = prox_{f_reg}(x)   <=>   grad f_reg(y_k) = x_k - y_k,   y_k = u_PM(x_k)

so the residual  ||grad J(y_k) - (x_k - y_k)|| / ||x_k - y_k||  is a THEOREM'S
HYPOTHESIS, not a validation loss, and the additive constant -- the only thing
left unidentified -- does not matter because prox_{f+c} = prox_f. Read against
delta, the sampler's own noise floor (tvpm/recover.py):

    ~ delta   the prior extracted everything the data contains
    >> delta  the recovery failed

EACH METHOD'S GRADIENT, and why the comparison is symmetric:

    grad J_1(y) = w(y) - y,  grad psi(w(y)) = y   ->  residual = ||w(y_k) - x_k||
    grad J_2(y) = grad G(y) - y                   ->  residual = ||grad G(y_k) - x_k||
    grad J_theta(y)                               ->  residual = ||grad J(y_k) - (x_k-y_k)||

The first needs an inversion at every point; the other two are forward passes.
That is the cost asymmetry the two recoveries were built to expose, now measured on
a real denoiser instead of a synthetic target.
"""
import json
import os
import sys
import time

# This package lives at tv_pm/prior_routes/. `tvpm` and `src` are resolved from
# tv_pm/ and the repo root, so this directory (for its sibling modules), its
# parent (for the tvpm package) and the repo root (for src/) all go on the path.
# Anchoring on __file__ rather than the working directory is what lets the
# scripts and the notebook run from anywhere without a stray copy shadowing them.
HERE = os.path.dirname(os.path.abspath(__file__))          # tv_pm/prior_routes
TV_PM = os.path.dirname(HERE)                              # tv_pm
ROOT = os.path.dirname(TV_PM)                              # repo root, holds src/
for _p in (HERE, TV_PM, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plotstyle import apply as _apply_style
_apply_style()          # one font size for every figure


import three_way as TW
from tvpm import dataset
from tvpm.recover import atv, estimate_delta, find_checkpoint, load_checkpoint
from src.invert import invert_cvx_gd

# Only alpha = 0, the UNREGULARIZED inversion, is reported. alpha = 0.1 was
# measured and dropped from the comparison: at alpha > 0 the recovered object is
# the prox of a DIFFERENT prior, biased at order alpha (src/invert.py), and it
# scored 99.55 % here -- an alpha-bias readout, not a recovery. Reporting it
# alongside alpha = 0 compared a method against its own documented distortion.
ALPHAS = (0.0,)
# The inversion budget and the eval-set size are env-tunable because LPN Iterative
# recovery's cost does NOT shrink with the training budget: a badly fitted psi makes
# the solve converge LATER, not sooner, so a 3000-step smoke net runs the full 20000
# iterations (~13 min) where the trained net early-stops in ~13 s. That asymmetry is
# the finding, not an accident -- so a smoke run shrinks the EVAL SET instead of
# pretending the inversion is cheap.
INVERT_ITERS = int(os.environ.get("TW_INVERT_ITERS", 20000))
# lr for invert_cvx_gd. NOT a free knob -- it is set by the inversion's OWN
# certificate ||grad psi(w) - y||, measured on 500 eval patches:
#     lr=1e-3, 20000 it -> cert 3.4e-3, prior residual 28.5 %   (under-solved)
#     lr=1e-2, 20000 it -> cert 3.9e-4, prior residual 11.6 %
#     lr=1e-2, 60000 it -> cert 3.9e-4, prior residual 11.6 %   (converged, stops on tol)
# Reporting the 1e-3 number would charge LPN Iterative recovery for an unconverged optimizer
# rather than for its method, so we solve it out and report the converged value.
INVERT_LR = 1e-2
N_EVAL = int(os.environ.get("TW_N_EVAL", 2000))   # shared by all three methods


# ---------------------------------------------------------------- priors ----
def grad_J_direct(model, units, y):
    """grad J_theta(y), the direct fit. One backward pass."""
    return units.grad(model, y)


def value_J_direct(model, units, y):
    from src.gradfit import net_value
    return net_value(model, units.z(y))


def grad_J_route2(G, y):
    """grad J_2(y) = grad G(y) - y. One backward pass, NO inversion."""
    yt = torch.tensor(np.asarray(y)).float().requires_grad_(True)
    g = torch.autograd.grad(G.scalar(yt).sum(), yt)[0].detach().numpy()
    return g - np.asarray(y)


def value_J_route2(G, y):
    """J_2(y) = G(y) - 0.5||y||^2."""
    with torch.no_grad():
        Gv = G.scalar(torch.tensor(np.asarray(y)).float()).numpy().ravel()
    return Gv - 0.5 * np.sum(np.asarray(y) ** 2, axis=1)


class PsiInX:
    """psi_theta expressed in X-UNITS, so src/invert.py can be used verbatim.

    invert_cvx_gd optimizes model.scalar(v) - <z, v> over v and needs grad psi in
    the SAME units as the query. psi_theta was fitted in z-units, so wrap it:
    psi_x(v) = psi_tilde((v - mu)/s) has grad_v psi_x = grad_z psi_tilde / s,
    which is u_PM -- the units the queries y_k live in.
    """

    def __init__(self, psi, units):
        self.psi, self.mu, self.s = psi, torch.tensor(units.mu).float(), \
            torch.tensor(units.s).float()

    def scalar(self, v):
        return self.psi.scalar((v - self.mu) / self.s)

    def parameters(self):
        return self.psi.parameters()


def grad_value_J_route1(psi, units, y, alpha, iters=INVERT_ITERS):
    """LPN Iterative recovery, exactly as Fang et al. specify: invert psi at y, then Fenchel.

    J_1(y) = <y, w> - psi(w) - 0.5||y||^2 with grad psi(w) = y (alpha=0), and
    grad J_1(y) = w - y. Returns (grad, value, preimage w).
    """
    wrapped = PsiInX(psi, units)
    w = invert_cvx_gd(y, wrapped, max_iters=iters, lr=INVERT_LR, alpha=alpha)
    with torch.no_grad():
        psi_w = wrapped.scalar(torch.tensor(w).float()).numpy().ravel()
    val = np.sum(np.asarray(y) * w, axis=1) - 0.5 * np.sum(np.asarray(y) ** 2, axis=1) - psi_w
    # the inversion's own certificate (src/recovery.py::prox_residual): is w
    # really a preimage of y under psi_theta? Separates "the solve failed" from
    # "the recovered prior is wrong".
    wt = torch.tensor(w).float().requires_grad_(True)
    gw = torch.autograd.grad(wrapped.scalar(wt).sum(), wt)[0].detach().numpy()
    cert = np.linalg.norm(gw - np.asarray(y), axis=1) / \
        np.maximum(1.0, np.linalg.norm(np.asarray(y), axis=1))
    return w - np.asarray(y), val, w, cert


# ---------------------------------------------------------------- scoring ---
def resid(g, target):
    """||grad J(y_k) - (x_k - y_k)|| / ||x_k - y_k||, per point. The ONLY
    normalization comparable to delta (tvpm/recover.py)."""
    return np.linalg.norm(g - target, axis=1) / np.linalg.norm(target, axis=1)


def report(name, g, target, delta, cost):
    r = resid(g, target)
    med = float(np.median(r))
    return {"method": name, "resid_rel_median": med, "resid_rel_p90": float(np.percentile(r, 90)),
            "ratio_to_delta": med / delta, "cost": cost, "_r": r}


def main(n_eval=None, invert_iters=None):
    """`n_eval` / `invert_iters` override the module defaults.

    Passed EXPLICITLY by the notebooks rather than read from the environment at
    call time: the module constants are bound at import, so a notebook that
    changed its preset and re-ran without restarting the kernel would silently
    keep the old ones. Arguments cannot go stale that way.
    """
    N_EVAL = int(n_eval) if n_eval else globals()["N_EVAL"]
    INVERT_ITERS = int(invert_iters) if invert_iters else globals()["INVERT_ITERS"]
    psi, units, psi_ck = TW.load_psi()
    # One-shot recovery uses the GRADIENT-supervised G. The value-supervised variant
    # (bin/_run.py's protocol) was measured and dropped: its values are right to
    # 8.4e-4 but grad G(y_k) vs x_k is 13.8 % median, and the one-shot recovery's prior needs
    # the gradient, so it scored 101.6 % -- see three_way.py::train_G_grad.
    G, g_ck = TW.load_G(TW.GG_CK)
    jck = find_checkpoint(arch="fc", sweeps=TW.SWEEPS, beta=TW.BETA, steps=250000,
                          sigma=TW.SIGMA, t=TW.T)
    Jm, Ju, _ = load_checkpoint(jck)

    ev = dataset.load("eval", sweeps=TW.SWEEPS, sigma=TW.SIGMA, t=TW.T)
    x, y = TW.flat(ev["x"])[:N_EVAL], TW.flat(ev["y"])[:N_EVAL]
    target = x - y                                  # what grad f_reg must equal
    print(f"eval split (cameraman patches): {x.shape}", flush=True)

    print("estimating the sampler noise floor delta ...", flush=True)
    delta = estimate_delta(ev["x"], TW.SWEEPS, sigma=TW.SIGMA, t=TW.T)
    print(f"delta = {100*delta:.3f} %", flush=True)

    rows, values = [], {}

    # ---- Direct fit -----------------------------------------------------
    t0 = time.time()
    gd = grad_J_direct(Jm, Ju, y)
    td = time.time() - t0
    rows.append(report("Direct fit", gd, target, delta, f"{td:.2f} s (fwd)"))
    values["Direct fit"] = value_J_direct(Jm, Ju, y)

    # ---- One-shot recovery: J_2 = G - 0.5||.||^2, one forward pass ---------
    t0 = time.time()
    g2 = grad_J_route2(G, y)
    t2 = time.time() - t0
    rows.append(report("One-shot recovery", g2, target, delta, f"{t2:.2f} s (fwd)"))
    values["One-shot recovery"] = value_J_route2(G, y)

    # ---- LPN Iterative recovery: invert psi at each alpha ------------------
    per_alpha = {}
    for al in ALPHAS:
        t0 = time.time()
        g1, v1, w1, c1 = grad_value_J_route1(psi, units, y, alpha=al,
                                             iters=INVERT_ITERS)
        t1 = time.time() - t0
        # divergence tripwire, the analogue of bin/_run.py's preimage bound:
        # the exact preimage of a query in range(u_PM) is the noisy patch, which
        # lives in [0,1]; leaving that box by 50% is a runaway solve.
        pre_max = float(np.abs(w1).max())
        div = pre_max > 1.5
        per_alpha[al] = dict(g=g1, v=v1, w=w1, t=t1, pre_max=pre_max, diverged=div,
                             cert=float(np.median(c1)))
        r = report(f"LPN Iterative recovery, alpha={al}", g1, target, delta,
                   f"{t1:.1f} s (inv)")
        print(f"  alpha={al}: resid {100*r['resid_rel_median']:.2f} %, "
              f"inversion cert {np.median(c1):.2e}, |w|_inf {pre_max:.3f}"
              f"{'  DIVERGED' if div else ''}, {t1:.1f} s", flush=True)

    # BOTH alphas are reported, not just the winner. bin/_run.py collapses to
    # alpha* because the synthetic families have a ground-truth prior to rank
    # against; here the interesting fact is that NEITHER setting works, which a
    # single "best" row would hide. src/invert.py already says why: alpha=0 need
    # not be coercive, alpha>0 returns the prox of a DIFFERENT prior biased at
    # order alpha. This is that sentence, measured.
    for al in ALPHAS:
        d = per_alpha[al]
        tag = "  [tripwire]" if d["diverged"] else ""
        rr = report(f"LPN Iterative recovery{tag}", d["g"], target, delta,
                    f"{d['t']:.1f} s (inversion)")
        rr["inversion_cert"] = d["cert"]
        rr["preimage_linf"] = d["pre_max"]
        rr["diverged"] = d["diverged"]
        rows.append(rr)
        values["LPN Iterative recovery"] = d["v"]

    # ---------------------------------------------------------------- table --
    order = ["LPN Iterative recovery", "One-shot recovery", "Direct fit"]
    rows.sort(key=lambda r: next((i for i, o in enumerate(order)
                                  if r["method"].startswith(o)), 9))
    print("\n" + "=" * 96)
    print(f"PRIOR RECOVERY, scored against u_PM   (sampler noise floor delta = {100*delta:.2f} %)")
    print("-" * 96)
    print(f"{'recovered prior':38s} {'resid/target':>13s} {'p90':>8s} "
          f"{'x delta':>9s} {'cost':>20s}")
    print("-" * 96)
    for r in rows:
        print(f"{r['method']:38s} {100*r['resid_rel_median']:12.2f}% "
              f"{100*r['resid_rel_p90']:7.2f}% {r['ratio_to_delta']:8.1f}x "
              f"{r['cost']:>20s}")
    print("=" * 96, flush=True)
    print("\niterative-recovery inversion diagnostics (alpha is its only knob):")
    for al in ALPHAS:
        d = per_alpha[al]
        frac = float(np.mean(np.abs(d["w"]).max(axis=1) > 1.5))
        print(f"  alpha={al}: certificate ||grad psi(w)-y|| = {d['cert']:.2e}"
              f"   max|w|_inf = {d['pre_max']:.3f}"
              f"   patches outside 1.5x[0,1]: {100*frac:.2f} %"
              f"{'   <- TRIPWIRE' if d['diverged'] else ''}")
    print("  (at alpha>0 the certificate is SUPPOSED to be ~alpha*||w||: it reports "
          "the bias, it does not\n   conceal it -- src/invert.py. So alpha=0.1's "
          "9e-2 is the method's distortion, not a failed solve.)", flush=True)

    # ---- prior VALUES: identified only up to a constant, so centre them -----
    print("\nprior values on the same points (each centred; J is identified only "
          "up to a constant, and prox_{f+c} = prox_f):")
    tv = atv(ev["y"][:N_EVAL])
    cen = {k: v - v.mean() for k, v in values.items()}
    for k, v in cen.items():
        print(f"  {k:32s} corr(J, ATV) = {np.corrcoef(v, tv)[0,1]:+.4f}")
    keys = list(cen)
    if "Direct fit" in cen:
        base = cen["Direct fit"]
        print("\n  agreement with Direct fit, centred RMSE / std(J_direct):")
        for k in keys:
            if k == "Direct fit":
                continue
            print(f"    {k:32s} {np.sqrt(np.mean((cen[k]-base)**2))/base.std():.4f}")

    # ------------------------------------------------------------- figure ----
    lbl = {"LPN Iterative recovery": "Iterative\ninversion per query",
           "One-shot recovery": "One-shot\nforward pass through $G$",
           "Direct fit": "Direct fit\nno recovery step"}
    plot = [(lbl[k], v) for k, v in
            [(kk, next((r for r in rows if r["method"].startswith(kk)), None))
             for kk in lbl] if v is not None and v["resid_rel_median"] is not None]

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    names = [p[0] for p in plot]
    meds = [100 * p[1]["resid_rel_median"] for p in plot]
    p90s = [100 * p[1]["resid_rel_p90"] for p in plot]
    # colour by method, keyed on the SHORT display names above
    col = ["#4C6EF5" if n.startswith("Iterative") else
           "#F59F00" if n.startswith("One-shot") else "#2F9E44" for n in names]
    xi = np.arange(len(names))
    ax[0].bar(xi, meds, color=col, width=0.6)
    ax[0].errorbar(xi, meds, yerr=[np.zeros(len(meds)), np.array(p90s) - np.array(meds)],
                   fmt="none", ecolor="#343A40", capsize=4, lw=1.2)
    ax[0].axhline(100 * delta, ls="--", c="#E03131", lw=1.5)
    ax[0].text(-0.45, 100 * delta * 1.10, f"sampler noise floor $\\delta$ = "
               f"{100*delta:.2f}%", color="#E03131", ha="left")
    ax[0].set_ylim(100 * delta * 0.75, None)
    ax[0].set_yscale("log")
    ax[0].set_xticks(xi); ax[0].set_xticklabels(names)
    ax[0].set_ylabel(r"$\|\nabla J(y_k)-(x_k-y_k)\|\,/\,\|x_k-y_k\|$   (median, bar to p90)")
    ax[0].set_title("Does the recovered prior satisfy $u_{PM}$'s\nprox optimality "
                    "condition?  (lower = better)")
    ax[0].grid(axis="y", alpha=0.3)

    for (n, r), c in zip(plot, col):
        v = np.sort(r["_r"])
        ls = "-"
        ax[1].plot(100 * v, np.linspace(0, 1, v.size), lw=2, color=c, ls=ls,
                   label=n.replace("\n", " "))
    ax[1].axvline(100 * delta, ls="--", c="#E03131", lw=1.5)
    ax[1].set_xscale("log"); ax[1].set_xlabel("relative prox residual (%)")
    ax[1].set_ylabel("fraction of eval patches")
    ax[1].set_title("Distribution over the 2000 held-out\ncameraman patches")
    ax[1].legend(loc="lower right"); ax[1].grid(alpha=0.3)

    for k, c in zip(lbl, ["#4C6EF5", "#F59F00", "#2F9E44"]):
        if k in cen:
            ax[2].scatter(tv, cen[k], s=5, alpha=0.35, color=c,
                          label=f"{k}  (r={np.corrcoef(cen[k],tv)[0,1]:+.3f})")
    ax[2].set_xlabel("anisotropic TV of the patch"); ax[2].set_ylabel("recovered $J$ (centred)")
    ax[2].set_title("The recovered prior vs TV\n(correlated, not equal: a SMOOTHED TV)")
    ax[2].legend(); ax[2].grid(alpha=0.3)

    fig.suptitle("Prior recovery on a real denoiser — cameraman eval split, "
                 f"$\\sigma=t=20/256$, $m={TW.SWEEPS}$, all networks {psi_ck['steps']} steps")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    stem = os.path.join(TW.FIG_DIR, f"prior_recovery{TW.OUT_SUF}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {stem}.png / .pdf")

    out = {"delta": delta, "n_eval": N_EVAL,
           "rows": [{k: v for k, v in r.items() if k != "_r"} for r in rows],
           "alphas": {str(a): {"resid_median": float(np.median(resid(d["g"], target))),
                               "preimage_linf": d["pre_max"], "diverged": d["diverged"],
                               "seconds": d["t"], "inversion_cert_median": d["cert"]} for a, d in per_alpha.items()},
           "corr_J_ATV": {k: float(np.corrcoef(v, tv)[0, 1]) for k, v in cen.items()}}
    with open(os.path.join(TW.RES_DIR, f"prior_recovery{TW.OUT_SUF}.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    import csv
    with open(os.path.join(TW.RES_DIR, f"prior_recovery{TW.OUT_SUF}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["recovered prior", "resid_rel_median_pct", "resid_rel_p90_pct",
                    "ratio_to_delta", "cost"])
        for r in rows:
            w.writerow([r["method"],
                        None if r["resid_rel_median"] is None else 100 * r["resid_rel_median"],
                        None if r["resid_rel_p90"] is None else 100 * r["resid_rel_p90"],
                        r["ratio_to_delta"], r["cost"]])


if __name__ == "__main__":
    main()
