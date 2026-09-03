"""Paper-style cross-section along the diagonal ray x = t * 1_d / sqrt(d).

The analog of src/diagnostics.py:plot_typical_ray for Experiment C. Along the
bulk direction u = 1/sqrt(d) (coordinates all equal to t/sqrt(d), so the ray
runs corner to corner of the query box), for each family and dimension:

  Row 1 -- LPN potential psi:  psi exact  vs  the two-network method's psi_theta.
           The one-network method uses the EXACT prox, so its psi is the exact
           curve by construction (nothing learned there).
  Row 2 -- prior:  J_BVS exact  vs  one-network J_hat  vs  two-network J_hat,
           with J_hat = G - 0.5||.||^2.

One figure per family; columns are dimensions. Reads only the overnight
checkpoints in logs/compare_ckpt/. Thin wrapper; all math in src/.

    python bin/plot_ray.py --families quadratic_l1 negl1 --dims 2 4 8 16 32 64
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIN = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BIN)
sys.path.insert(0, ROOT)
sys.path.insert(0, BIN)

from src import plotstyle
from src.network import LPN
from src.recovery import cvx as net_cvx, evaluate_learned_prior_G
from src.exact_prox import jbvs_exact
import quadratic_l1
import negl1
import concave_quad
import minplus

plotstyle.apply()

FAMILIES = {
    "quadratic_l1": (quadratic_l1.config, r"$\|\cdot\|_1$"),
    "negl1": (negl1.config, r"$-\|\cdot\|_1$"),
    "concave_quad": (concave_quad.config, "concave quad."),
    "minplus": (minplus.config, "min-plus"),
}
EXACT = dict(color="k", ls="--", lw=1.4)
REALJ = dict(color="#ff7f0e", ls=":", lw=1.9)
ONE = dict(color="#2ca02c", lw=1.6)
TWO = dict(color="#1f77b4", lw=1.6)


def load_net(path):
    if not os.path.exists(path):
        return None
    c = torch.load(path, map_location="cpu", weights_only=False)
    m = LPN(in_dim=c["in_dim"], hidden=c["hidden"], layers=c.get("layers", 2),
            beta=c.get("beta", 5))
    m.load_state_dict(c["state"]); m.eval()
    return m


def ray_points(dim, a=4.0, spacing=160):
    """x(t) = t * 1_d/sqrt(d), t in [-a sqrt(d), a sqrt(d)] (corner to corner)."""
    u = np.ones(dim) / np.sqrt(dim)
    t = np.linspace(-a * np.sqrt(dim), a * np.sqrt(dim), spacing)
    return t, t[:, None] * u[None, :]


def figure_for(family, dims, ckpt, out):
    make_problem, pretty = FAMILIES[family]
    fig, axes = plt.subplots(2, len(dims), figsize=(12, 5.6))
    if len(dims) == 1:
        axes = axes.reshape(2, 1)

    for j, dim in enumerate(dims):
        problem = make_problem(dim)
        t, pts = ray_points(dim)
        psi2 = load_net(os.path.join(ckpt, f"{family}_{dim}D_method2_psi.pth"))
        G1 = load_net(os.path.join(ckpt, f"{family}_{dim}D_method1_G.pth"))
        G2 = load_net(os.path.join(ckpt, f"{family}_{dim}D_method2_G.pth"))

        # Row 1: LPN potential psi -- a diagnostic of METHOD 2's first network
        # ONLY. Method 1 trains no psi network (it builds G directly from the
        # prox and potential values at the samples), so the dashed curve is
        # ground truth, not a one-net output.
        ax = axes[0][j]
        ax.plot(t, problem.cvx_true(pts), label=r"$\psi$ (ground truth)", **EXACT)
        if psi2 is not None:
            ax.plot(t, net_cvx(pts, psi2), label=r"two-net $\psi_\theta$ (LPN)", **TWO)
        ax.set_title(rf"$d={dim}$")
        if j == 0:
            ax.set_ylabel(r"$\psi$ along $t\,\mathbf{1}/\sqrt{d}$")
            ax.legend(loc="upper center", framealpha=0.9)

        # Row 2: prior. Overlay the REAL prior J (= -||.||_1 for negl1; it
        # coincides with J_BVS for the convex families) alongside J_BVS.
        ax = axes[1][j]
        jb = jbvs_exact(problem, pts)
        ax.plot(t, problem.prior_true(pts), label=r"real prior $J$", **REALJ)
        ax.plot(t, jb, label=r"$J_{\mathrm{BVS}}$ exact", **EXACT)
        if G1 is not None:
            ax.plot(t, evaluate_learned_prior_G(pts, G1), label="one net", **ONE)
        if G2 is not None:
            ax.plot(t, evaluate_learned_prior_G(pts, G2), label="two net", **TWO)
        ax.set_xlabel("$t$")
        if j == 0:
            ax.set_ylabel(r"prior along $t\,\mathbf{1}/\sqrt{d}$")
            # place the legend in the empty lobe: above the vertex for a V
            # (min at centre), below the peak for an inverted V (max at centre).
            loc = "upper center" if jb[len(jb) // 2] < jb[0] else "lower center"
            ax.legend(loc=loc, framealpha=0.9)
        for a_ in (axes[0][j], axes[1][j]):
            a_.grid(True, alpha=0.3)

    fig.suptitle(rf"Cross-section along $t\,\mathbf{{1}}_d/\sqrt{{d}}$ --- "
                 rf"{pretty}: one network (exact prox) vs two networks (paper), "
                 rf"$S=250{{,}}000$", y=0.995)
    fig.text(0.5, 0.005, r"Top row diagnoses Method 2's first network only; "
             r"Method 1 trains no $\psi$ (dashed = ground truth). Both methods' "
             r"trained network is $G$ (approximate) --- its error is the prior row.",
             ha="center", va="bottom", fontsize=9, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="diagonal-ray cross-section (Experiment C)")
    ap.add_argument("--families", nargs="+", default=["quadratic_l1", "negl1"],
                    choices=list(FAMILIES))
    ap.add_argument("--dims", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "logs", "compare_ckpt"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "figs"))
    args = ap.parse_args()
    for fam in args.families:
        figure_for(fam, args.dims, args.ckpt,
                   os.path.join(args.outdir, f"ray_{fam}.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
