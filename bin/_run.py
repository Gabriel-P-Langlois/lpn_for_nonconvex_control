"""Shared experiment runner. One file per FAMILY in bin/ supplies a config and
calls ``cli``; the whole pipeline (data, both networks, both recovery methods,
plot) lives here so each bug is fixed once. Dimension is a flag, not a file:
the only dimension-dependent choice is the width, and hidden_width(dim) owns it.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network import LPN, hidden_width
from src.train import train_potential
from src.recovery import (
    conjugate_samples,
    recover_prior_route1,
    route2_preimage,
    route_report,
    evaluate_learned_prior_G,
)
from src.plotting import plot_cross_sections


def uniform_inputs(dim, n, a, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-a, a, (n, dim))


def run(
    name,
    problem,
    dim,
    a=4.0,
    n_train=None,  # None -> 15000 * dim (D2 amendment); N grows with dimension
    n_val=4000,
    n_test=4000,
    beta=5,
    layers=2,
    steps=250_000,  # optimizer steps, NOT epochs (D2 amendment)
    lr=1e-3,
    batch_size=512,
    invert_iters=20000,
    # Fang et al.'s strong-convexity regularizer. LPN Iterative recovery is run at each value and
    # its BEST RMSE is reported (see below). alpha=0 is the unregularized method.
    alphas=(0.0, 0.1),
    n_eval=1000,  # prior-recovery points, SHARED by both recoveries (None = all of x_te)
    smoke=False,
    out_dir=None,
):
    """Full two-network pipeline for one experiment. Returns a metrics dict."""
    n_train = n_train or 15_000 * dim
    if smoke:
        steps = min(steps, 2000)
        invert_iters = min(invert_iters, 2000)
        n_eval = min(n_eval or n_test, 200)
        n_train = min(n_train, 20000)
    out_dir = out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
    )
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1)
    device = torch.device("cpu")  # D2: CPU (MPS not worth it once mini-batched)
    hidden = hidden_width(dim)

    # --- boxes (D2 amendment, 2026-07-09) ---
    # psi is TRAINED on [-train_a, train_a]^d, chosen so that grad psi maps it
    # onto the query box [-a, a]^d. Both recoveries need this: LPN Iterative recovery evaluates psi
    # at preimages y* (which can leave the query box), and One-shot recovery trains G at
    # inputs y_k = grad psi(x_k), whose support is range(grad psi). Queries and
    # all reported errors stay on [-a, a]^d.
    train_a = problem.train_halfwidth(a)

    # --- data (train/val/test with distinct seeds; D2) ---
    x_tr = uniform_inputs(dim, n_train, train_a, seed=1)
    x_va = uniform_inputs(dim, n_val, train_a, seed=2)
    x_te = uniform_inputs(dim, n_test, a, seed=3)  # queries: the ORIGINAL box
    psi_tr, psi_va = problem.cvx_true(x_tr), problem.cvx_true(x_va)
    print(f"[{name}] query box +-{a:g}, psi train box +-{train_a:g}, "
          f"N={n_train} (={n_train/dim:g}*d), {steps} steps "
          f"(~{steps*batch_size/n_train:.0f} epochs)")

    # --- first network: the convex potential psi ---
    print(f"[{name}] training psi network (dim {dim}, hidden {hidden})")
    model = LPN(in_dim=dim, hidden=hidden, layers=layers, beta=beta).to(device)
    hist_psi = train_potential(
        model, x_tr, psi_tr, x_va, psi_va, batch_size=batch_size, steps=steps, lr=lr
    )

    # --- second network: one-shot-recovery conjugate G ---
    print(f"[{name}] training G network (One-shot recovery)")
    yk_tr, Gk_tr = conjugate_samples(x_tr, model)
    yk_va, Gk_va = conjugate_samples(x_va, model)
    model_G = LPN(in_dim=dim, hidden=hidden, layers=layers, beta=beta).to(device)
    hist_G = train_potential(
        model_G, yk_tr, Gk_tr, yk_va, Gk_va, batch_size=batch_size, steps=steps, lr=lr
    )

    # --- one-shot-recovery coverage: does G's training support contain the query box? ---
    # y_k are G's inputs; if the queries reach past them, One-shot recovery extrapolates.
    yk_reach = float(np.max(np.abs(yk_tr)))
    frac_q_uncovered = float(np.mean(np.max(np.abs(x_te), axis=1) > yk_reach))

    # --- recovery on the held-out test set: SAME points, SAME certificates ---
    # LPN Iterative recovery runs exactly as Fang et al. specify (alpha its only knob, no box,
    # no projection, no abstention). the one-shot recovery's implied preimage is grad G(x),
    # free, because G ~= psi*. Both are then judged by the same prox residual
    # and Fenchel-Young gap, neither of which touches the ground truth.
    # One evaluation set for BOTH recoveries. LPN Iterative recovery costs an optimization per
    # query, One-shot recovery a forward pass; that asymmetry is the method's, not the
    # protocol's, so we do not compensate for it -- we just measure both on the
    # same points. 1000 is ample for an RMSE at these error magnitudes.
    xs = x_te if n_eval is None else x_te[:n_eval]
    true_s = problem.prior_true(xs)

    # LPN Iterative recovery is run at EVERY alpha in `alphas` and we report the best result
    # among the runs that did NOT diverge. This is deliberately generous to
    # LPN Iterative recovery: alpha = 0 is the unregularized inversion, whose objective is
    # coercive only where x lies in range(grad psi_theta) -- true once psi_theta
    # has converged, false when it is under-trained, in which case the iterate
    # runs away and the "error" is a readout of invert_iters. alpha > 0 makes the
    # objective strongly convex so a minimizer always exists, at the price of an
    # O(alpha) bias, since grad psi(y) = x - alpha*y at the solution.
    # One-shot recovery has no alpha and is untouched, so this doubles only the iterative recovery's cost.
    per_alpha = {}
    for al in alphas:
        r1_a, y1_a = recover_prior_route1(
            xs, model, inv_alg="cvx_gd", return_preimage=True,
            max_iters=invert_iters, alpha=al,
        )
        per_alpha[al] = route_report(r1_a, true_s, xs, y1_a, model, model_G)

    # Zero-cost divergence tripwire. The exact preimage of the query box obeys
    # problem.preimage_bound(a); a solve that leaves it by 50% has run away.
    # We test against that analytic bound, NOT against train_a: the two agree
    # for the expanding families but train_a is 2x too loose for ConcaveQuad
    # (bound a/2 = 2, box 4), which would leave the tripwire 3x too permissive.
    pre_bound = problem.preimage_bound(a)
    diverged = {al: rp["max_preimage_linf"] > 1.5 * pre_bound
                for al, rp in per_alpha.items()}

    # Diverged runs are EXCLUDED, not merely ranked. Taking the plain min over
    # alpha is unsound: divergence does not reliably produce a worse RMSE. At
    # d=16 the alpha=0 solve ran to max|y|=10.94 (bound 5) yet scored 0.667
    # against the regularized run's 0.800, so a plain min would report a
    # meaningless number. If every alpha diverges we report the failure rather
    # than a number: prior_rmse_route1 is None, not the min over the diverged
    # runs -- that fallback would silently reinstate the rule this comment
    # disavows, and the summary table has no column to catch it.
    ok = [al for al in per_alpha if not diverged[al]]
    all_diverged = not ok
    alpha_best = (min(ok, key=lambda k: per_alpha[k]["rmse"]) if ok else
                  min(per_alpha, key=lambda k: per_alpha[k]["rmse"]))
    rep1 = per_alpha[alpha_best]

    r2 = evaluate_learned_prior_G(xs, model_G)
    y2 = route2_preimage(xs, model_G)
    rep2 = route_report(r2, true_s, xs, y2, model, model_G)

    metrics = {
        "name": name,
        "dim": dim,
        "hidden": hidden,
        "query_halfwidth": float(a),
        "train_halfwidth": float(train_a),
        "preimage_bound": float(pre_bound),
        "n_train": int(n_train),
        "steps": int(steps),
        "invert_alphas": list(alphas),
        "invert_alpha_best": float(alpha_best),
        "route1_rmse_per_alpha": {a_: r["rmse"] for a_, r in per_alpha.items()},
        "route1_diverged_per_alpha": diverged,
        "route1_all_alphas_diverged": all_diverged,
        # best_val, not val[-1]: train_potential restores the best-validation
        # weights, so the last epoch's val MSE describes a discarded model.
        "psi_val_mse": hist_psi["best_val"],
        "G_val_mse": hist_G["best_val"],
        # unconditional RMSE, all queries, both recoveries. None when every iterative-recovery
        # solve diverged: there is no trustworthy number to report.
        "prior_rmse_route1": None if all_diverged else rep1["rmse"],
        "prior_rmse_route2": rep2["rmse"],
        # How far the reported iterative-recovery solve pushed psi_theta past the exact
        # preimage bound. >1 means psi was extrapolated at some query, which is
        # not divergence but is worth knowing; the tripwire only fires at 1.5.
        "route1_preimage_excess": rep1["max_preimage_linf"] / pre_bound,
        "route2_preimage_excess": rep2["max_preimage_linf"] / pre_bound,
        # the same certificate for each recovery
        "route1_median_prox_residual": rep1["median_prox_residual"],
        "route2_median_prox_residual": rep2["median_prox_residual"],
        "route1_median_fy_gap": rep1["median_fy_gap"],
        "route2_median_fy_gap": rep2["median_fy_gap"],
        "route1_max_preimage_linf": rep1["max_preimage_linf"],
        "route2_max_preimage_linf": rep2["max_preimage_linf"],
        "route2_yk_reach": yk_reach,
        "route2_frac_query_uncovered": frac_q_uncovered,
        "route1_diagnostic": rep1,
        "route2_diagnostic": rep2,
    }
    if frac_q_uncovered > 0.01:
        print(f"[{name}] WARNING: {frac_q_uncovered:.1%} of queries lie outside "
              f"G's training support (max|y_k|_inf = {yk_reach:.2f} < a = {a:g}); "
              f"one-shot-recovery error will be dominated by extrapolation.")
    if all_diverged:
        print(f"[{name}] WARNING: every alpha diverged; prior_rmse_route1 = None.")
    elif metrics["route1_preimage_excess"] > 1.0:
        # Only meaningful for a solve that did NOT trip the wire: psi_theta is
        # being evaluated past the exact preimage bound, but by less than 1.5x.
        print(f"[{name}] NOTE: iterative-recovery preimage reaches |y|_inf = "
              f"{rep1['max_preimage_linf']:.2f} vs the exact bound {pre_bound:g} "
              f"({metrics['route1_preimage_excess']:.2f}x); psi_theta is "
              f"extrapolated at some query. Not divergence (tripwire is 1.5x).")

    # --- convexity sanity: inner weights must stay non-negative after wclip ---
    min_inner_w = min(
        float(core.weight.data.min()) for core in model.lin[1:]
    )
    metrics["min_inner_weight"] = min_inner_w

    # --- persist the networks and the metrics -------------------------------
    # Every diagnostic we have wanted so far cost a full retrain, and each
    # retrain is a fresh chance for a protocol change to slip in between the
    # numbers and the figure. Checkpoints make plotting free and make any figure
    # traceable to the exact weights that produced it. ~0.5 MB per network.
    ckpt_dir = os.path.join(out_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"state": model.state_dict(), "in_dim": dim, "hidden": hidden,
                "layers": layers, "beta": beta}, os.path.join(ckpt_dir, f"{name}_psi.pth"))
    torch.save({"state": model_G.state_dict(), "in_dim": dim, "hidden": hidden,
                "layers": layers, "beta": beta}, os.path.join(ckpt_dir, f"{name}_G.pth"))

    plot_path = os.path.join(out_dir, f"{name}_cross_sections.png")
    plot_cross_sections(
        problem, model, a, 100, dim, plot_path, model_G=model_G, alpha=alpha_best,
        title=(f"{name}   |   width {hidden}, N={n_train}, S={steps}   |   "
               f"R1 RMSE {rep1['rmse']:.4f} (a={alpha_best:g})   "
               f"R2 RMSE {rep2['rmse']:.4f}"),
    )
    metrics["plot"] = plot_path

    per_a = "  ".join(f"a={a_:g}: {r['rmse']:.4f}{'*' if a_ == alpha_best else ''}"
                      f"{' DIVERGED' if diverged[a_] else ''}"
                      for a_, r in per_alpha.items())
    tail = " (ALL DIVERGED -- iterative-recovery number is NOT trustworthy)" if all_diverged else ""
    print(f"[{name}] iterative-recovery by alpha ({per_a})  -> reporting alpha={alpha_best:g}{tail}")
    # JSON keys must be strings; alpha-keyed dicts and numpy scalars need care.
    def _jsonable(o):
        if isinstance(o, dict):
            return {str(k): _jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_jsonable(v) for v in o]
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    with open(os.path.join(out_dir, f"{name}_metrics.json"), "w") as fh:
        json.dump(_jsonable(metrics), fh, indent=2)

    print(f"[{name}] metrics: {metrics}")
    return metrics


def cli(family, make_problem, dims, argv=None):
    """Standard entry point for a family config: ``python bin/<family>.py --dim d``."""
    ap = argparse.ArgumentParser(description=f"{family} experiment family")
    ap.add_argument("--dim", type=int, default=dims[0], choices=list(dims))
    ap.add_argument("--a", type=float, default=4.0, help="query-box half-width")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.1],
                    help="iterative-recovery regularizer grid (Fang et al.); the BEST RMSE "
                         "over these is reported. 0 = unregularized")
    ap.add_argument("--smoke", action="store_true", help="fast, low-fidelity check")
    args = ap.parse_args(argv)
    return run(
        f"{family}_{args.dim}D",
        make_problem(args.dim),
        dim=args.dim,
        a=args.a,
        alphas=tuple(args.alphas),
        smoke=args.smoke,
    )
