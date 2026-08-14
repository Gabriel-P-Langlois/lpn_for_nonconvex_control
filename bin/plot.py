"""Regenerate figures from a saved run, without retraining.

    python bin/plot.py --family quadratic_l1 --dim 2            # all diagnostics
    python bin/plot.py --family quadratic_l1 --dim 16 --open    # and open them

Reads logs/ckpt/<run>_psi.pth, <run>_G.pth and logs/<run>_metrics.json, which
bin/_run.py writes on every run. Figures land in logs/<run>_*.png.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

BIN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BIN)
sys.path.insert(0, os.path.dirname(BIN))

import concave_quad
import minplus
import negl1
import quadratic_l1
sys.path.insert(0, os.path.join(BIN, "not_in_paper"))
import maxplus_case3
import maxplus_case4
from _run import uniform_inputs
from src.network import LPN
from src.recovery import recover_prior_route1, evaluate_learned_prior_G
from src.plotting import plot_cross_sections
from src.diagnostics import (
    plot_conditional_cross_section,
    plot_typical_ray,
    plot_pred_vs_true,
    plot_prox_scatter,
    plot_preimage_scatter,
)

# Every family exposes config(dim) -> Problem. A Problem must supply cvx_true,
# prior_true and preimage() for the full diagnostic set; all of ours do.
FAMILIES = {
    "quadratic_l1": quadratic_l1.config,
    "negl1": negl1.config,
    "concave_quad": concave_quad.config,
    "minplus": minplus.config,
    "maxplus_case3": maxplus_case3.config,   # not in the paper
    "maxplus_case4": maxplus_case4.config,   # not in the paper
}
LOGS = os.path.join(os.path.dirname(BIN), "logs")


def load(name, which):
    p = os.path.join(LOGS, "ckpt", f"{name}_{which}.pth")
    if not os.path.exists(p):
        sys.exit(f"missing checkpoint {p}\nRun bin/<family>.py --dim <d> first.")
    blob = torch.load(p, map_location="cpu", weights_only=False)
    m = LPN(in_dim=blob["in_dim"], hidden=blob["hidden"],
            layers=blob["layers"], beta=blob["beta"])
    m.load_state_dict(blob["state"])
    m.eval()
    return m


def render(family, dim, name=None, a=4.0, n_eval=1000, invert_iters=20000):
    """Regenerate every diagnostic figure for one saved run; return the paths.

    Importable entry point (the notebooks in ../notebooks/ call this); the CLI
    below is a thin wrapper. Reads logs/ckpt/<name>_{psi,G}.pth and, if present,
    <name>_metrics.json for the run's chosen alpha; writes logs/<name>_*.png.
    """
    name = name or f"{family}_{dim}D"
    problem = FAMILIES[family](dim)
    train_a = problem.train_halfwidth(a)
    psi, G = load(name, "psi"), load(name, "G")

    mpath = os.path.join(LOGS, f"{name}_metrics.json")
    alpha = 0.0
    if os.path.exists(mpath):
        with open(mpath) as fh:
            alpha = float(json.load(fh)["invert_alpha_best"])

    # the SAME test points the run scored, and training-box points for the prox
    x_te = uniform_inputs(dim, 4000, a, seed=3)[:n_eval]
    y_tr = uniform_inputs(dim, 4000, train_a, seed=1)

    # ONE iterative-recovery inversion on the test set, shared by the scatter and profile.
    r1 = recover_prior_route1(x_te, psi, "cvx_gd", alpha=alpha,
                              max_iters=invert_iters)
    r2 = evaluate_learned_prior_G(x_te, G)

    # The axis cross-section (freeze all but one coordinate at 0) is removed: at
    # dim > 2 that slice has volume fraction ~(eps/a)^(dim-1) ~ 0, so no training
    # point is near it and the panel showed pure extrapolation, not accuracy. The
    # conditional cross-section (backgrounds from the query distribution) is the
    # valid high-dimensional view.
    out = []
    out.append(plot_conditional_cross_section(
        problem, psi, G, a, train_a, dim,
        os.path.join(LOGS, f"{name}_conditional.png"), alpha=alpha,
        invert_iters=invert_iters))
    out.append(plot_typical_ray(
        problem, psi, G, a, dim,
        os.path.join(LOGS, f"{name}_typical_ray.png"), alpha=alpha,
        invert_iters=invert_iters))
    out.append(plot_pred_vs_true(
        problem, x_te, r1, r2, os.path.join(LOGS, f"{name}_pred_vs_true.png"), alpha=alpha))
    out.append(plot_prox_scatter(
        problem, psi, y_tr, os.path.join(LOGS, f"{name}_prox_scatter.png")))
    out.append(plot_preimage_scatter(
        problem, G, x_te, os.path.join(LOGS, f"{name}_preimage_scatter.png")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--name", default=None,
                    help="checkpoint/figure basename; defaults to <family>_<dim>D. "
                         "Set it to plot a non-default run (e.g. a depth variant) "
                         "without colliding with the sweep's figures.")
    ap.add_argument("--a", type=float, default=4.0)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--invert-iters", type=int, default=20000,
                    help="iterative-recovery inversion budget for the figures; lower is faster "
                         "but changes the iterative recovery's curve, so keep the run's value")
    ap.add_argument("--open", action="store_true", help="open the figures (macOS)")
    args = ap.parse_args()

    out = render(args.family, args.dim, name=args.name, a=args.a,
                 n_eval=args.n_eval, invert_iters=args.invert_iters)
    for p in out:
        print(p)
    if args.open:
        subprocess.run(["open", *out], check=False)


if __name__ == "__main__":
    main()
