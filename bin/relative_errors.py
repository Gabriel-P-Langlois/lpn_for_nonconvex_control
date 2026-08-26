"""Experiment R driver: relative-error renormalization of the reported tables.

Recomputes the four families' reported errors as relative L2 quantities and adds
a proximal-map column, writing logs/summary_relerr.csv. Read-only over the
reported runs: no run is re-trained.

    rel value error = ||J_hat - J||_2 / ||J||_2
                    = (reported RMSE) / rms(J)          [from stored metrics]
    rel prox error  = ||prox_hat - prox_{tJ}||_2 / ||prox_{tJ}||_2

The value columns come purely from the stored ``prior_rmse_route{1,2}`` and the
closed-form ground truth on the shared seed-3 evaluation points -- no network is
touched. The proximal column is not stored anywhere (only the prox RESIDUAL, a
certificate, is), so it requires one FORWARD pass of the saved psi / G networks
against the closed-form maps. That is a reload but not a re-fit; the plan's own
proximal definition forces it.

Reference convention matches the reported tables: the value error is scored
against ``prior_true`` (the nonconvex J for NegL1, whose J != J_BVS gap is the
known ~5% floor), NOT against J_BVS. The exact-prox ablation instead reports
against J_BVS; the two references are deliberately different.

    python bin/relative_errors.py
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

BIN = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BIN)
sys.path.insert(0, ROOT)
sys.path.insert(0, BIN)

from src.network import LPN
from src.recovery import prox as net_prox, evaluate_learned_prior_G
from src.exact_prox import grad_psi, rel_l2
from _run import uniform_inputs

import quadratic_l1
import negl1
import concave_quad
import minplus

FAMILIES = {
    "quadratic_l1": (quadratic_l1.config, quadratic_l1.DIMS),
    "negl1": (negl1.config, negl1.DIMS),
    "concave_quad": (concave_quad.config, concave_quad.DIMS),
    "minplus": (minplus.config, minplus.DIMS),
}


def load_net(path):
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = LPN(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"],
                layers=ckpt.get("layers", 2), beta=ckpt.get("beta", 5))
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model


def rms(a):
    a = np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(a * a)))


def row_for(family, dim, logs):
    mpath = os.path.join(logs, f"{family}_{dim}D_metrics.json")
    if not os.path.exists(mpath):
        return None
    with open(mpath) as fh:
        m = json.load(fh)
    make_problem, _ = FAMILIES[family]
    problem = make_problem(dim)

    n_eval = 1000
    xs = uniform_inputs(dim, 4000, 4.0, seed=3)[:n_eval]
    J = problem.prior_true(xs)
    rms_J = rms(J)

    r1 = m.get("prior_rmse_route1")
    r2 = m.get("prior_rmse_route2")
    row = {
        "name": f"{family}_{dim}D", "family": family, "dim": dim,
        "rms_J": rms_J,
        "prior_rmse_route1": r1, "prior_rmse_route2": r2,
        "rel_value_route1": (r1 / rms_J) if r1 is not None else None,
        "rel_value_route2": (r2 / rms_J) if r2 is not None else None,
        "psi_val_mse": m.get("psi_val_mse"),
        "G_val_mse": m.get("G_val_mse"),
    }

    ckpt = os.path.join(logs, "ckpt")
    model_psi = load_net(os.path.join(ckpt, f"{family}_{dim}D_psi.pth"))
    model_G = load_net(os.path.join(ckpt, f"{family}_{dim}D_G.pth"))
    if model_psi is not None:
        prox_true = grad_psi(problem, xs)                 # forward prox = grad psi
        row["rel_prox_forward"] = rel_l2(net_prox(xs, model_psi), prox_true)
    if model_G is not None:
        pre_true = problem.preimage(xs)                   # inverse prox = grad g
        row["rel_prox_inverse"] = rel_l2(net_prox(xs, model_G), pre_true)
        # cross-check: relative value from the reloaded G must match the stored-RMSE route.
        jhat = evaluate_learned_prior_G(xs, model_G)
        row["rel_value_route2_direct"] = rel_l2(jhat, J)
        if row["rel_value_route2"] is not None:
            row["value_route2_crosscheck_delta"] = abs(
                row["rel_value_route2_direct"] - row["rel_value_route2"])
    return row


def main():
    ap = argparse.ArgumentParser(description="relative-error renormalization (Experiment R)")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument("--dims", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "summary_relerr.csv"))
    args = ap.parse_args()

    rows = []
    for fam in args.families:
        _, dims = FAMILIES[fam]
        for dim in [d for d in dims if d in args.dims]:
            r = row_for(fam, dim, os.path.join(ROOT, "logs"))
            if r is not None:
                rows.append(r)

    fields = ["name", "family", "dim", "rms_J",
              "prior_rmse_route1", "prior_rmse_route2",
              "rel_value_route1", "rel_value_route2", "rel_value_route2_direct",
              "value_route2_crosscheck_delta",
              "rel_prox_forward", "rel_prox_inverse",
              "psi_val_mse", "G_val_mse"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def f(v, p=4, w=9):
        return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'n/a':>{w}}"

    print(f"{'run':>18} | {'relV R1':>9} {'relV R2':>9} | "
          f"{'proxFwd':>9} {'proxInv':>9} | {'xchk d':>8}")
    for r in rows:
        print(f"{r['name']:>18} | {f(r['rel_value_route1'])} {f(r['rel_value_route2'])} "
              f"| {f(r.get('rel_prox_forward'))} {f(r.get('rel_prox_inverse'))} "
              f"| {f(r.get('value_route2_crosscheck_delta'), 2, 8)}")
    print(f"\n{len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
