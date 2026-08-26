"""Render the Experiment C figure (experiments_plan.tex, sub:code).

Two rows, one quantity per axis, four family columns; two lines per panel:
  one network (exact prox)  vs  two networks (the paper).

  Row 1 -- prior accuracy on held-out test:  ||J_hat - J_BVS|| / ||J_BVS||
  Row 2 -- inverse-prox accuracy on test:    ||grad G - (grad psi)^{-1}|| / ||...||

Both methods are trained fresh at the same per-network budget; the reader is
told the budget in the title, and warned while it is below the production 250k.
Thin wrapper: reads logs/<family>_<dim>D_compare_metrics.json only.

    python bin/plot_compare.py [--dims 2 4 8] [--out figs/compare_recovery.png]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIN = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BIN)
sys.path.insert(0, ROOT)
from src import plotstyle

plotstyle.apply()

FAMILY_TITLE = {
    "quadratic_l1": r"$\|\cdot\|_1$",
    "negl1": r"$-\|\cdot\|_1$",
    "concave_quad": r"concave quad.",
    "minplus": r"min-plus",
}
ONE = dict(color="#2ca02c", marker="o", label="one network (exact prox)")
TWO = dict(color="#1f77b4", marker="s", label="two networks (paper)")


def load(logs, family, dim):
    p = os.path.join(logs, f"{family}_{dim}D_compare_metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser(description="Experiment C figure")
    ap.add_argument("--logs", default=os.path.join(ROOT, "logs"))
    ap.add_argument("--dims", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--families", nargs="+", default=list(FAMILY_TITLE),
                    choices=list(FAMILY_TITLE))
    ap.add_argument("--out", default=os.path.join(ROOT, "figs", "compare_recovery.png"))
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    # keep only requested families that actually have data, so partial sweeps
    # render clean (no empty columns).
    fams = [f for f in args.families
            if any(load(args.logs, f, d) for d in args.dims)]
    if not fams:
        print("no compare metrics found for the requested families/dims")
        return 1
    fig, axes = plt.subplots(2, len(fams), figsize=(4.0 * len(fams), 7.2), sharex=True)
    steps_seen = set()

    rows = [("one_net_prior_relL2", "two_net_prior_relL2",
             r"prior: rel. $L^2$ error in $\hat J$"),
            ("one_net_invprox_relL2", "two_net_invprox_relL2",
             r"inverse prox: rel. $L^2$ error in $\nabla G$")]

    for r, (k_one, k_two, ylab) in enumerate(rows):
        for j, fam in enumerate(fams):
            ax = axes[r][j]
            d1, y1, d2, y2 = [], [], [], []
            for d in args.dims:
                m = load(args.logs, fam, d)
                if not m:
                    continue
                steps_seen.add(m["steps"])
                d1.append(d); y1.append(m[k_one])
                d2.append(d); y2.append(m[k_two])
            if d1:
                ax.plot(d1, y1, color=ONE["color"], marker=ONE["marker"], markersize=6)
                ax.plot(d2, y2, color=TWO["color"], marker=TWO["marker"], markersize=6)
            if r == 0:
                ax.set_title(FAMILY_TITLE[fam])
            ax.set_yscale("log"); ax.set_xscale("log", base=2)
            ax.set_xticks(args.dims); ax.set_xticklabels(args.dims)
            ax.grid(True, which="both", alpha=0.25)
            if r == 1:
                ax.set_xlabel("dimension $d$")
            if j == 0:
                ax.set_ylabel(ylab)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=ONE["color"], marker=ONE["marker"], markersize=6),
               Line2D([0], [0], color=TWO["color"], marker=TWO["marker"], markersize=6)]
    fig.legend(handles, [ONE["label"], TWO["label"]], loc="lower center", ncol=2,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.005))

    S = min(steps_seen) if steps_seen else 0
    title = ("Prior recovery on held-out test data: one network (exact prox) vs "
             "two networks (paper)\n"
             f"matched samples, test set, and budget ($S={S:,}$ steps per network)")
    if S and S < 250_000:
        title += "  --  quick check below production budget (250k)"
    fig.suptitle(title, y=0.995)
    fig.text(0.5, 0.055, r"query box $[-4,4]^d$, $t=1$, $N=15000\,d$ training "
             r"samples, relative $L^2$ on 1000 held-out test points (seed 3)",
             ha="center", va="bottom", fontsize=9, color="0.35")
    fig.tight_layout(rect=(0, 0.085, 1, 0.95))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")
    if args.open:
        os.system(f"open '{args.out}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
