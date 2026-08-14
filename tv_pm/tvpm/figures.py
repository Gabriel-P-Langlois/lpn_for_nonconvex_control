"""Step 4: the qualitative figures -- what the recovered prior f_reg LOOKS like.

    ~/miniforge3/envs/lpn_env/bin/python -m tvpm.figures                     (~1 min, from tv_pm/)

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

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")                          # headless: save files, no display
import matplotlib.pyplot as plt

from src.plotstyle import apply as _apply_style
_apply_style()          # one font size for every figure

from . import dataset
from src.conv_icnn import ConvICNN
from .paths import FIGS as OUT
from .recover import atv, find_checkpoint, net_value
from src.network import LPN            # noqa: E402  (paths.py puts ROOT on sys.path)
# Look in logs/ckpt/ (local training output) first, then results/ckpt/ (the
# tracked copies that travel to a colleague; logs/ and *.pth are gitignored).


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
    return m, ck["mu"], ck["s"], {"arch": arch, "beta": ck["beta"],
                                  "sweeps": ck.get("sweeps"), "steps": ck.get("steps")}


def J_of_y(model, mu, s, y_flat):
    """Recovered prior value J_theta(y), in the standardized units the net saw.
    Constant is arbitrary (the data fixes f_reg only up to +c), so callers center."""
    return net_value(model, (y_flat - mu) / s)


def _cfg(meta):
    """Config suffix for a figure title, from whatever meta carries.

    'm=8000, FIT_STEPS=250000' -- sweeps behind the data and optimizer steps
    behind the fit, the two knobs a reader needs to place a figure. Silently
    omits either if absent (e.g. a legacy checkpoint that never stored steps).
    """
    bits = []
    if meta.get("sweeps") is not None:
        bits.append(f"PM_SWEEPS={meta['sweeps']}")
    if meta.get("steps") is not None:
        bits.append(f"FIT_STEPS={meta['steps']}")
    return ("  [" + ", ".join(bits) + "]") if bits else ""


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
    ax[0].legend(loc="upper left")

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
        iax.set_title(lab)

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

    fig.suptitle(f"Recovered TV regularizer $f_{{reg}}$ — {tag}, eval on cameraman"
                 f"{_cfg(meta)}")
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
                 f"look for difference stencils{_cfg(meta)}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main(sigma=dataset.SIGMA, t=dataset.T, beta=20, steps=250000, sweeps=8000):
    os.makedirs(OUT, exist_ok=True)
    ev = dataset.load("eval", sweeps=sweeps, sigma=sigma, t=t)
    sfx = dataset.tag(sigma, t)          # keeps a (sigma, t) run off the defaults' figures

    made = []
    for name in ("fc", "conv"):
        # resolve the checkpoint for THIS (sigma, t) -- not a fixed default, so
        # figures for a non-default pair use the right net rather than the shipped one.
        path = find_checkpoint(arch=name, sweeps=sweeps, beta=beta, steps=steps,
                               sigma=sigma, t=t)
        if path is None or not os.path.exists(path):
            print(f"  [skip] {name}: no checkpoint for sigma={sigma:g}, t={t:g}, "
                  f"beta={beta:g}, steps={steps}, m={sweeps}")
            continue
        model, mu, s, meta = load(path)
        fig, corr = figure_core(model, mu, s, meta, ev)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(OUT, f"step4_core_{name}{sfx}.{ext}"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {name}-ICNN: corr(J,TV) = {corr:.3f}  -> step4_core_{name}{sfx}.png/pdf")
        made.append(name)

        if meta["arch"] == "conv":
            fk = figure_kernels(model, meta)
            for ext in ("png", "pdf"):
                fk.savefig(os.path.join(OUT, f"step4_kernels_{name}{sfx}.{ext}"), dpi=150, bbox_inches="tight")
            plt.close(fk)
            print(f"  {name}-ICNN: learned kernels -> step4_kernels_{name}{sfx}.png/pdf")

    print(f"\n  figures in {OUT}" if made else "  no checkpoints found; nothing made")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sigma", type=float, default=dataset.SIGMA)
    ap.add_argument("--t", type=float, default=dataset.T)
    ap.add_argument("--beta", type=float, default=20)
    ap.add_argument("--steps", type=int, default=250000)
    ap.add_argument("--sweeps", type=int, default=8000)
    a = ap.parse_args()
    main(sigma=a.sigma, t=a.t, beta=a.beta, steps=a.steps, sweeps=a.sweeps)
