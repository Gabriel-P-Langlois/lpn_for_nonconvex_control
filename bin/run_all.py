"""Sweep driver: run whole families across dimensions and tabulate the metrics.

    python bin/run_all.py --smoke                      # everything, fast
    python bin/run_all.py --families quadratic_l1 minplus --dims 2 4
    python bin/run_all.py                              # full production sweep

Writes logs/summary.csv. Each row is one (family, dim). A run that raises is
recorded with its error rather than killing the sweep.
"""
import argparse
import csv
import os
import sys
import traceback

BIN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BIN)

import concave_quad
import minplus
import negl1
import quadratic_l1
from _run import run

# Runs at d > 8 are expensive (width 256) and must be approved by the user each
# time; see --allow-high-dim. Sweeps default to these dimensions only.
SAFE_DIMS = (1, 2, 4, 8)

FAMILIES = {
    "quadratic_l1": (quadratic_l1.config, quadratic_l1.DIMS),
    "negl1": (negl1.config, negl1.DIMS),
    "concave_quad": (concave_quad.config, concave_quad.DIMS),
    "minplus": (minplus.config, minplus.DIMS),
}

# route1_all_alphas_diverged and route1_diverged_per_alpha are NOT optional.
# Without them a diverged iterative-recovery RMSE enters the table unmarked and the table
# cannot be audited from itself -- exactly the failure that once reported a
# diverged d=16 solve (0.667) as beating a converged one (0.800).
FIELDS = [
    "name", "dim", "hidden", "n_train", "steps", "query_halfwidth",
    "train_halfwidth", "preimage_bound", "invert_alpha_best",
    "psi_val_mse", "G_val_mse", "prior_rmse_route1", "prior_rmse_route2",
    "route1_median_prox_residual", "route2_median_prox_residual",
    "route1_median_fy_gap", "route2_median_fy_gap",
    "route1_max_preimage_linf", "route2_max_preimage_linf",
    "route1_preimage_excess", "route2_preimage_excess",
    "route1_all_alphas_diverged", "route1_diverged_per_alpha",
    "route2_frac_query_uncovered", "min_inner_weight", "error",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument("--dims", nargs="+", type=int, default=list(SAFE_DIMS),
                    help="restrict to these dims (intersected with each family's DIMS)")
    ap.add_argument("--a", type=float, default=4.0)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.1],
                    help="iterative-recovery regularizer grid; best RMSE over these is reported")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-high-dim", action="store_true",
                    help=f"permit dims > {max(SAFE_DIMS)}; ask the user first")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(BIN), "logs", "summary.csv"))
    args = ap.parse_args()

    high = [d for d in args.dims if d > max(SAFE_DIMS)]
    if high and not args.allow_high_dim:
        ap.error(
            f"dims {high} exceed d={max(SAFE_DIMS)}. High-dimension runs are expensive "
            f"on the CPU-only stack and are opt-in: pass --allow-high-dim only after "
            f"the user has explicitly approved it."
        )

    rows = []
    for fam in args.families:
        make_problem, dims = FAMILIES[fam]
        dims = [d for d in dims if d in args.dims]
        for dim in dims:
            name = f"{fam}_{dim}D"
            try:
                m = run(name, make_problem(dim), dim=dim, a=args.a,
                        alphas=tuple(args.alphas), smoke=args.smoke)
            except Exception:
                traceback.print_exc()
                m = {"name": name, "dim": dim, "error": traceback.format_exc(limit=1).strip()}
            rows.append({k: m.get(k) for k in FIELDS})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    def fmt(v, w=9, p=4):
        return f"{v:{w}.{p}f}" if v is not None else f"{'n/a':>{w}}"

    print("\n  RMSE: unconditional, all queries, both recoveries -- no abstention.")
    print("  prox resid: ||grad psi(y)-x||/max(1,||x||), y from the solve (R1) or")
    print("              y = grad G(x) (R2). Same certificate, no ground truth.")
    print(f"\n{'experiment':>18} | {'R1 RMSE':>9} {'R2 RMSE':>9} | "
          f"{'R1 resid':>9} {'R2 resid':>9} | {'alpha*':>6} {'R1 max|y|':>9}")
    for r in rows:
        if r.get("error"):
            print(f"{r['name']:>18} |   FAILED")
            continue
        mark = "  ALL-DIVERGED" if r.get("route1_all_alphas_diverged") else (
            "  R1 extrap" if (r.get("route1_preimage_excess") or 0) > 1.0 else "")
        print(f"{r['name']:>18} | {fmt(r['prior_rmse_route1'])} {fmt(r['prior_rmse_route2'])} "
              f"| {fmt(r['route1_median_prox_residual'])} "
              f"{fmt(r['route2_median_prox_residual'])} "
              f"| {r['invert_alpha_best']:6.3g} "
              f"{fmt(r['route1_max_preimage_linf'], 9, 2)}{mark}")
    n_fail = sum(1 for r in rows if r.get("error"))
    n_div = sum(1 for r in rows if r.get("route1_all_alphas_diverged"))
    print(f"\n{len(rows) - n_fail}/{len(rows)} succeeded -> {args.out}")
    if n_div:
        print(f"{n_div} run(s) had every alpha diverge; their iterative-recovery RMSE is blank.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
