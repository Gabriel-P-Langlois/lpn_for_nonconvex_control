"""Experiment C: prior recovery on held-out test data (experiments_plan.tex).

One network (exact prox) vs two networks (the paper), on IDENTICAL training
samples, an IDENTICAL held-out test set, and an IDENTICAL per-network budget.
The only difference between the methods is where the conjugate network's training
targets come from: the exact closed-form prox (Method 1) or a trained first
network (Method 2).

Both methods produce one conjugate network G, and from it, by the same formulas,
the recovered prior J_hat = G - 0.5||.||^2 and the recovered inverse prox grad G.
Both are scored on the same held-out query-box test points against closed-form
ground truth. See experiments_plan.tex sec:cmp / sub:fair for the fairness
argument.

No reported module is edited. Method 2 reuses the reported pipeline's own
functions (train_potential, conjugate_samples); Method 1 reuses train_potential
on exact pairs built by src/exact_prox.py.

    python bin/compare_recovery.py --smoke                 # one family, n=2, tiny
    python bin/compare_recovery.py --dims 2 4 8 --steps 20000
    python bin/compare_recovery.py --dims 2 4 8            # matched full budget (250k)
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

BIN = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BIN)
sys.path.insert(0, ROOT)
sys.path.insert(0, BIN)

from src.network import LPN, hidden_width
from src.train import train_potential
from src.recovery import conjugate_samples, evaluate_learned_prior_G, prox as net_prox
from src.exact_prox import build_triples, jbvs_exact, rel_l2
from _run import uniform_inputs

import quadratic_l1
import negl1
import concave_quad
import minplus

FAMILIES = {
    "quadratic_l1": quadratic_l1.config,
    "negl1": negl1.config,
    "concave_quad": concave_quad.config,
    "minplus": minplus.config,
}
SAFE_DIMS = (2, 4, 8)


def new_net(dim):
    """Fresh network with a fixed init, so every network in every method starts
    identically -- the comparison is then only about data and budget."""
    torch.manual_seed(0)
    return LPN(in_dim=dim, hidden=hidden_width(dim), layers=2, beta=5).to("cpu")


def score(model_G, z, jbvs_true, invprox_true):
    """Test accuracy of a conjugate network G: recovered prior and inverse prox."""
    Jhat = evaluate_learned_prior_G(z, model_G)      # G(z) - 0.5||z||^2
    gradG = net_prox(z, model_G)                     # grad G(z) ~ (grad psi)^{-1}
    return (rel_l2(Jhat, jbvs_true), rel_l2(gradG, invprox_true))


def run_unit(problem, family, dim, steps, n_train, n_eval, ckpt_dir):
    A = problem.train_halfwidth(4.0)
    x_tr = uniform_inputs(dim, n_train, A, seed=1)
    x_va = uniform_inputs(dim, 4000, A, seed=2)
    z_te = uniform_inputs(dim, 4000, 4.0, seed=3)[:n_eval]

    # closed-form ground truth on the held-out test set
    jbvs_true = jbvs_exact(problem, z_te)
    invprox_true = problem.preimage(z_te)            # (grad psi)^{-1} = grad g

    # ---- Method 1: one network, exact prox ------------------------------------
    y_tr, g_tr, _ = build_triples(problem, x_tr)
    y_va, g_va, _ = build_triples(problem, x_va)
    G1 = new_net(dim)
    h1 = train_potential(G1, y_tr, g_tr, y_va, g_va, steps=steps, verbose=False)
    one_prior, one_prox = score(G1, z_te, jbvs_true, invprox_true)

    # ---- Method 2: two networks, the paper ------------------------------------
    psi = new_net(dim)
    hpsi = train_potential(psi, x_tr, problem.cvx_true(x_tr),
                           x_va, problem.cvx_true(x_va), steps=steps, verbose=False)
    yk_tr, Gk_tr = conjugate_samples(x_tr, psi)
    yk_va, Gk_va = conjugate_samples(x_va, psi)
    G2 = new_net(dim)
    h2 = train_potential(G2, yk_tr, Gk_tr, yk_va, Gk_va, steps=steps, verbose=False)
    two_prior, two_prox = score(G2, z_te, jbvs_true, invprox_true)

    for tag, m in (("1_G", G1), ("2_psi", psi), ("2_G", G2)):
        torch.save({"state": m.state_dict(), "in_dim": dim, "hidden": m.hidden},
                   os.path.join(ckpt_dir, f"{family}_{dim}D_method{tag}.pth"))

    return {
        "name": f"{family}_{dim}D", "family": family, "dim": dim,
        "hidden": hidden_width(dim), "steps": int(steps),
        "n_train": int(n_train), "n_eval": int(n_eval),
        "train_halfwidth": float(A),
        # Method 1 (one network, exact prox)
        "one_net_prior_relL2": one_prior,
        "one_net_invprox_relL2": one_prox,
        "one_net_G_val_mse": h1["best_val"],
        # Method 2 (two networks, paper)
        "two_net_prior_relL2": two_prior,
        "two_net_invprox_relL2": two_prox,
        "two_net_psi_val_mse": hpsi["best_val"],
        "two_net_G_val_mse": h2["best_val"],
    }


def _done(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path) as fh:
            return json.load(fh).get("status") == "done"
    except (json.JSONDecodeError, OSError):
        return False


def _log(progress, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line)
    with open(progress, "a") as fh:
        fh.write(line + "\n")


def main():
    ap = argparse.ArgumentParser(description="one net (exact prox) vs two nets (paper)")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument("--dims", nargs="+", type=int, default=list(SAFE_DIMS))
    ap.add_argument("--steps", type=int, default=250_000, help="budget per network (BOTH methods)")
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-high-dim", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs"))
    args = ap.parse_args()

    if args.smoke:
        args.families, args.dims = ["quadratic_l1"], [2]
        args.steps, args.n_eval = min(args.steps, 2000), 200

    high = [d for d in args.dims if d > max(SAFE_DIMS)]
    if high and not args.allow_high_dim:
        ap.error(f"dims {high} exceed d={max(SAFE_DIMS)}; pass --allow-high-dim only "
                 f"after the user approves.")

    os.makedirs(args.out, exist_ok=True)
    ckpt_dir = os.path.join(args.out, "compare_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    progress = os.path.join(args.out, "compare_progress.log")

    rows = []
    units = [(f, d) for d in sorted(args.dims) for f in args.families]
    _log(progress, f"START {len(units)} units: families={args.families} "
                   f"dims={sorted(args.dims)} steps/network={args.steps}")
    for fam, dim in units:
        path = os.path.join(args.out, f"{fam}_{dim}D_compare_metrics.json")
        if _done(path) and not args.force:
            _log(progress, f"SKIP  {fam}_{dim}D (done)")
            with open(path) as fh:
                rows.append(json.load(fh))
            continue
        n_train = 15_000 * dim if not args.smoke else 20_000
        t0 = time.time()
        try:
            m = run_unit(FAMILIES[fam](dim), fam, dim, args.steps, n_train,
                         args.n_eval, ckpt_dir)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            _log(progress, f"FAIL  {fam}_{dim}D: {exc}")
            continue
        m["seconds"] = round(time.time() - t0, 1)
        m["status"] = "done"
        with open(path, "w") as fh:
            json.dump(m, fh, indent=2)
        rows.append(m)
        _log(progress, f"DONE  {fam}_{dim}D ({m['seconds']}s)  "
                       f"prior 1-net {m['one_net_prior_relL2']:.4g} vs 2-net "
                       f"{m['two_net_prior_relL2']:.4g}  |  invprox 1-net "
                       f"{m['one_net_invprox_relL2']:.4g} vs 2-net "
                       f"{m['two_net_invprox_relL2']:.4g}")

    fields = ["name", "family", "dim", "hidden", "steps", "n_train", "n_eval",
              "seconds", "one_net_prior_relL2", "two_net_prior_relL2",
              "one_net_invprox_relL2", "two_net_invprox_relL2",
              "one_net_G_val_mse", "two_net_psi_val_mse", "two_net_G_val_mse"]
    with open(os.path.join(args.out, "summary_compare.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    _log(progress, f"END {len(rows)} unit(s) on record.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
