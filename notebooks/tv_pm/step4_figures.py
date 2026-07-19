"""Step 4: the qualitative figures -- what the recovered prior f_reg LOOKS like.

    ~/miniforge3/envs/lpn_env/bin/python notebooks/tv_pm/step4_figures.py   (~1 min)

There is no ground truth for f_reg at n=64 (it needs S_eps, the log-partition
function; the sampler returns only the mean). So Step 4 does not plot recovery
error -- it shows the SHAPE of the recovered prior and checks it against what the
theory says it must be: a SMOOTHED total variation. Loads the saved checkpoints
and only evaluates them; nothing is retrained.

Four panels (DESIGN.md, Step 4):
  1. J_theta vs TV on held-out patches -- strong correlation expected but NOT
     equality; f_reg is a smoothed TV and the gap is the point (no staircasing).
  2. J_theta on smooth vs textured patches -- the prior must penalise structure.
  3. A conditional cross-section -- one pixel varies, the other 63 drawn from the
     patch distribution (NEVER fixed at a constant: axis slices lie in high n,
     numerics_audit.tex). Each slice is convex, a visible check of the ICNN.
  4. BONUS (conv-ICNN only): the learned first-layer 3x3 kernels. If they emerge
     as difference stencils [+1,-1], the net rediscovered TV structure from
     denoiser evaluations alone.

Reads the EVAL split (cameraman) -- f_reg is a property of the DENOISER, so the
figures are made on the transfer image, not the training one.
"""
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")                          # headless: save files, no display
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import dataset
from conv_icnn import ConvICNN
from recover import atv, net_value
from src.network import LPN

OUT = os.path.join(HERE, "results", "figs")
# Look in logs/ckpt/ (local training output) first, then results/ckpt/ (the
# tracked copies that travel to a colleague; logs/ and *.pth are gitignored).
CKPT_DIRS = [os.path.join(ROOT, "logs", "ckpt"), os.path.join(HERE, "results", "ckpt")]


def find_ckpt(name):
    for d in CKPT_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(CKPT_DIRS[0], name)     # non-existent -> caller skips


FC = find_ckpt("tv_pm_64D_m8000_w256_b20.0_s250000.pth")            # production
CONV = find_ckpt("tv_pm_64D_conv_m8000_C32_B2_b20.0_s250000.pth")   # for kernels


def load(path):
    """Rebuild the fitter from a checkpoint and return (model, mu, s, meta)."""
    ck = torch.load(path, weights_only=False)
    arch = ck.get("arch", "fc")
    if arch == "conv":
        hw = int(round(ck["in_dim"] ** 0.5))
        m = ConvICNN(hw=(hw, hw), channels=ck["channels"], blocks=ck["blocks"], beta=ck["beta"])
    else:
        m = LPN(in_dim=ck["in_dim"], hidden=ck["hidden"], layers=ck["layers"], beta=ck["beta"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck["mu"], ck["s"], {"arch": arch, "beta": ck["beta"]}


def J_of_y(model, mu, s, y_flat):
    """Recovered prior value J_theta(y), in the standardized units the net saw.
    Constant is arbitrary (the data fixes f_reg only up to +c), so callers center."""
    return net_value(model, (y_flat - mu) / s)


def figure_core(model, mu, s, meta, ev):
    """Panels 1-3 for one model."""
    y = ev["y"]                               # (N, 8, 8)
    n = y.shape[0]
    yf = y.reshape(n, -1)
    tv = atv(y)
    J = J_of_y(model, mu, s, yf)
    J = J - J.mean()                          # center: the constant is unidentified
    corr = np.corrcoef(J, tv)[0, 1]

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    tag = f"{meta['arch']}-ICNN, $\\beta$={meta['beta']:g}"

    # --- Panel 1: J vs TV ---
    ax[0].scatter(tv, J, s=6, alpha=0.25, color="C0", edgecolors="none")
    b, a = np.polyfit(tv, J, 1)               # best line, to show departure from it
    xs = np.array([tv.min(), tv.max()])
    ax[0].plot(xs, a + b * xs, "k--", lw=1.2, label=f"best line (corr {corr:.3f})")
    ax[0].set_xlabel("total variation  TV$(y)$")
    ax[0].set_ylabel(r"recovered $J_\theta(y)$  (centered)")
    ax[0].set_title("1. recovered prior vs TV: correlated, not equal")
    ax[0].legend(fontsize=8, loc="upper left")

    # --- Panel 2: smooth vs textured ---
    q = np.argsort(tv)
    terc = [q[:n // 3], q[n // 3:2 * n // 3], q[2 * n // 3:]]
    parts = ax[1].violinplot([J[t] for t in terc], showmeans=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("C0"); pc.set_alpha(0.5)
    ax[1].set_xticks([1, 2, 3]); ax[1].set_xticklabels(["smooth\n(low TV)", "mid", "textured\n(high TV)"])
    ax[1].set_ylabel(r"$J_\theta(y)$  (centered)")
    ax[1].set_title("2. the prior penalises structure")

    # thumbnails: the smoothest and most textured eval patch
    for k, (idx, xy, lab) in enumerate([(q[0], (0.02, 0.72), "smoothest"),
                                        (q[-1], (0.02, 0.02), "most textured")]):
        iax = ax[1].inset_axes([xy[0], xy[1], 0.26, 0.26])
        iax.imshow(y[idx], cmap="gray", vmin=0, vmax=1); iax.set_xticks([]); iax.set_yticks([])
        iax.set_title(lab, fontsize=6)

    # --- Panel 3: conditional cross-section ---
    rng = np.random.default_rng(11)
    n_bg, n_g, p = 24, 120, 27                 # p = a central pixel (row 3, col 3)
    bg = y.reshape(n, -1)[rng.choice(n, n_bg, replace=False)]
    grid = np.linspace(0.0, 1.0, n_g)
    for row in bg:
        pts = np.tile(row, (n_g, 1)); pts[:, p] = grid
        Js = J_of_y(model, mu, s, pts)
        ax[2].plot(grid, Js - Js.mean(), color="0.6", lw=0.8)
    ax[2].set_xlabel(f"pixel {p} value (others from the patch distribution)")
    ax[2].set_ylabel(r"$J_\theta$  (centered per slice)")
    ax[2].set_title("3. conditional slices: smoothed-kink, convex")

    fig.suptitle(f"Recovered TV regularizer $f_{{reg}}$ — {tag}, eval on cameraman",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, corr


def figure_kernels(model, meta):
    """Panel 4: the conv-ICNN's learned first-layer 3x3 kernels."""
    W = model.in_conv.weight.detach().cpu().numpy()[:, 0]     # (C, 3, 3)
    C = W.shape[0]
    cols = 8
    rows = int(np.ceil(C / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.1))
    vmax = np.abs(W).max()
    for k in range(rows * cols):
        a = ax.flat[k]; a.set_xticks([]); a.set_yticks([])
        if k < C:
            a.imshow(W[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        else:
            a.axis("off")
    fig.suptitle(f"4. learned first-layer 3x3 kernels ({meta['arch']}-ICNN) — "
                 f"look for difference stencils", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    ev = dataset.load("eval")

    made = []
    for name, path in (("fc", FC), ("conv", CONV)):
        if not os.path.exists(path):
            print(f"  [skip] {name}: checkpoint not found ({path})")
            continue
        model, mu, s, meta = load(path)
        fig, corr = figure_core(model, mu, s, meta, ev)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(OUT, f"step4_core_{name}.{ext}"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {name}-ICNN: corr(J,TV) = {corr:.3f}  -> step4_core_{name}.png/pdf")
        made.append(name)

        if meta["arch"] == "conv":
            fk = figure_kernels(model, meta)
            for ext in ("png", "pdf"):
                fk.savefig(os.path.join(OUT, f"step4_kernels_{name}.{ext}"), dpi=150, bbox_inches="tight")
            plt.close(fk)
            print(f"  {name}-ICNN: learned kernels -> step4_kernels_{name}.png/pdf")

    print(f"\n  figures in {OUT}" if made else "  no checkpoints found; nothing made")


if __name__ == "__main__":
    main()
