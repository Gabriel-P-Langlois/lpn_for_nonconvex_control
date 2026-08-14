"""The cameraman comparison: u_PM vs the learned denoisers, with PSNR.

THREE METHODS, and only two of them are denoisers:

  u_PM              MCMC posterior-mean denoiser, the ground truth.
  u_theta           = grad psi_theta(x). DEPLOYED BY BOTH LPN RECOVERIES: G is made
                    from psi, and grad G = (grad psi)^-1, so neither Fang et al.'s
                    inversion nor our conjugate net is used to denoise. One
                    forward pass.
  u_hat             = prox_{J_theta}(x), the direct fit. A convex solve, but on
                    the SAME object it recovered -- there is no second object
                    for it to disagree with.

The prox_{J_2} panel is NOT a denoising method. It asks whether the prior LPN
(Ours) reports has the denoiser as its proximal operator; computing a prox is
what forces an inversion there. the one-shot recovery's own no-inversion claim concerns prior
VALUES and is measured in prior_compare.py, untouched by this.

Figure conventions follow tvpm/denoise.py (grayscale panels at vmin/vmax 0/1,
inferno difference maps at 0/0.1) so this figure sits beside the existing ones.
"""
import csv
import json
import os
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plotstyle import apply as _apply_style
_apply_style()          # one font size for every figure

from scipy.io import loadmat

from tvpm.paths import IMAGES
from tvpm.denoise import tile, untile, psnr, ssim, rel_l2
from tvpm.sampler import from_sigma_t, sample_pm

import three_way as TW


def _cache(path, fn):
    if os.path.exists(path):
        return np.load(path)["a"]
    a = fn()
    np.savez_compressed(path, a=a)
    return a


def diverged(u, bound=1.5):
    """bin/_run.py's preimage-bound rule, transposed. u_PM maps into the OPEN box
    [0,1]^64, so a solve landing outside 1.5x that box has left the region where
    G was ever fitted. Reported, never used to abstain."""
    return np.abs(u.reshape(u.shape[0], -1)).max(axis=1) > bound


def compare(seed=3, sweeps=TW.SWEEPS):
    sigma, t = TW.SIGMA, TW.T
    lam = from_sigma_t(sigma, t)[1]
    img = np.asarray(loadmat(os.path.join(IMAGES, TW.IMAGE + ".mat"))[TW.IMAGE], float)
    H, W = img.shape
    xc = tile(img)
    xn = np.clip(xc + np.random.default_rng(seed).normal(0, sigma, xc.shape), 0, 1)
    print(f"image {TW.IMAGE} {H}x{W}, {xn.shape[0]} patches, sigma={sigma:.5f}, "
          f"t={t:.5f}, lam={lam:.5f}", flush=True)

    # ---- ground truth: the MCMC posterior-mean denoiser --------------------
    t0 = time.time()
    u_pm = _cache(os.path.join(TW.CACHE_DIR, f"threeway_u_pm_m{sweeps}_seed{seed}.npz"),
                  lambda: sample_pm(xn, sigma, lam, sweeps=sweeps, w=1.0,
                                    seed=100)["u_pm"])
    time_pm = time.time() - t0

    # ---- the LPN denoiser: grad psi_theta, one forward pass ----------------
    psi, units, psi_ck = TW.load_psi()
    t0 = time.time()
    u_lpn = TW.denoise_route1(psi, units, xn)
    time_lpn = time.time() - t0

    # ---- consistency check: the prox of One-shot recovery's recovered prior -------
    G, g_ck = TW.load_G(TW.GG_CK)
    t0 = time.time()
    u_chk, resid_chk = TW.denoise_route2(G, units, xn)
    time_chk = time.time() - t0

    # ---- the direct fit: prox of J_theta -----------------------------------
    from tvpm.recover import find_checkpoint
    jck = find_checkpoint(arch="fc", sweeps=sweeps, beta=TW.BETA, steps=250000,
                          sigma=sigma, t=t)
    t0 = time.time()
    u_dir, resid_dir, j_ck = TW.denoise_direct(jck, xn)
    time_dir = time.time() - t0
    print(f"J_theta checkpoint: {os.path.basename(jck)}", flush=True)

    I = {k: untile(v, H, W) for k, v in
         {"clean": xc, "noisy": xn, "u_pm": u_pm, "lpn": u_lpn,
          "chk": u_chk, "dir": u_dir}.items()}

    methods = [
        ("noisy input", "noisy", 0.0, f"sigma = {sigma:.4f}"),
        ("u_PM  (MCMC sampler)", "u_pm", time_pm, "ground truth, m=8000"),
        ("u_theta = grad psi  [LPN, both recoveries]", "lpn", time_lpn,
         "1 forward pass, no inversion"),
        ("prox of J_2, One-shot recovery  [check]", "chk", time_chk,
         "consistency check, not a denoiser"),
        ("u_hat  direct = prox J_theta", "dir", time_dir,
         "recovered prior's prox IS the denoiser"),
    ]

    rows = []
    for name, k, secs, note in methods:
        ref = None if k in ("noisy", "u_pm") else "u_pm"
        rows.append({
            "method": name,
            "PSNR vs clean (dB)": psnr(I[k], I["clean"]),
            "SSIM vs clean": ssim(I[k], I["clean"]),
            "PSNR vs u_PM (dB)": None if ref is None else psnr(I[k], I["u_pm"]),
            "SSIM vs u_PM": None if ref is None else ssim(I[k], I["u_pm"]),
            "rel-L2 vs u_PM (%)": None if ref is None else 100 * rel_l2(I[k], I["u_pm"]),
            "time (s)": secs,
            "note": note,
        })

    print("\n" + "=" * 104)
    print(f"{'method':40s} {'PSNR vs clean':>14s} {'SSIM clean':>11s} "
          f"{'PSNR vs u_PM':>13s} {'rel-L2 %':>9s} {'time s':>9s}")
    print("-" * 104)
    for r in rows:
        f = lambda v, d=2: ("   --  " if v is None else f"{v:.{d}f}")
        print(f"{r['method']:40s} {f(r['PSNR vs clean (dB)']):>14s} "
              f"{f(r['SSIM vs clean'],4):>11s} {f(r['PSNR vs u_PM (dB)']):>13s} "
              f"{f(r['rel-L2 vs u_PM (%)']):>9s} {r['time (s)']:>9.2f}")
    print("=" * 104, flush=True)

    b = diverged(u_chk)
    print(f"\nconsistency-check solve residual : median {resid_chk['median']:.2e}, "
          f"max {resid_chk['max']:.2e}")
    print(f"consistency-check diverged patches: {int(b.sum())} / {b.size} "
          f"({100*b.mean():.2f} %)")
    print(f"direct prox residual (max)       : {resid_dir:.2e}")
    print(f"psi_theta : {psi_ck['steps']} steps, best val {psi_ck['best_val']:.4e}")
    print(f"G_theta   : {g_ck['steps']} steps (grad-fit), best val {g_ck['best_val']:.4e}")

    tr, va, _ = TW.load_splits()
    xv = TW.flat(va["x"])[:2000]
    yk, _ = TW.conjugate_pairs_x(psi, units, xv)
    yt = torch.tensor(yk).float().requires_grad_(True)
    gg = torch.autograd.grad(G.scalar(yt).sum(), yt)[0].detach().numpy()
    rel = np.linalg.norm(gg - xv, axis=1) / np.linalg.norm(xv, axis=1)
    gp = units.grad(psi, xv)
    yv = TW.flat(va["y"])[:2000]
    rp = np.linalg.norm(gp - yv, axis=1) / np.linalg.norm(yv, axis=1)
    print(f"grad G(y_k) vs x_k  [held-out]   : median {100*np.median(rel):.2f} %, "
          f"p90 {100*np.percentile(rel,90):.2f} %")
    print(f"grad psi(x) vs u_PM [held-out]   : median {100*np.median(rp):.2f} %, "
          f"p90 {100*np.percentile(rp,90):.2f} %", flush=True)

    # ---- figure ------------------------------------------------------------
    top = [("clean", "clean"), ("noisy input", "noisy"),
           ("$u_{PM}$  (MCMC sampler)\nGROUND TRUTH", "u_pm"),
           ("$u_\\theta=\\nabla\\psi_\\theta(x)$\nDEPLOYED: Iterative & One-shot", "lpn"),
           ("$\\mathrm{prox}_{J_2}$, One-shot\nCHECK, not a denoiser", "chk"),
           ("$\\hat u=\\mathrm{prox}_{J_\\theta}(x)$\nDEPLOYED, Direct fit", "dir")]
    n = len(top)

    fig, ax = plt.subplots(2, n, figsize=(3.3 * n, 8.4))
    for a, (title, k) in zip(ax[0], top):
        a.imshow(I[k], cmap="gray", vmin=0, vmax=1)
        a.set_xticks([]); a.set_yticks([])
        sub = "" if k == "clean" else f"\n{psnr(I[k], I['clean']):.2f} dB vs clean"
        a.set_title(title + sub)

    bottom = [None, None, ("$|u_{PM}-\\mathrm{clean}|$", "u_pm", "clean"),
              ("$|\\nabla\\psi_\\theta-u_{PM}|$", "lpn", "u_pm"),
              ("$|\\mathrm{prox}_{J_2}-u_{PM}|$", "chk", "u_pm"),
              ("$|\\mathrm{prox}_{J_\\theta}-u_{PM}|$", "dir", "u_pm")]
    for a, spec in zip(ax[1], bottom):
        if spec is None:
            a.axis("off"); continue
        title, k, ref = spec
        h = a.imshow(np.abs(I[k] - I[ref]), cmap="inferno", vmin=0, vmax=0.1)
        a.set_xticks([]); a.set_yticks([])
        sub = "" if ref == "clean" else f"\n{psnr(I[k], I['u_pm']):.2f} dB vs $u_{{PM}}$"
        a.set_title(title + sub)
        fig.colorbar(h, ax=a, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Reproducing the TV posterior-mean denoiser — cameraman, $\\sigma=t=20/256$, "
        f"$m={sweeps}$ sweeps, every network trained for {psi_ck['steps']} steps\n"
        "Neither LPN recovery inverts anything to DENOISE: $\\nabla G=(\\nabla\\psi)^{-1}$, "
        "so both deploy $\\nabla\\psi_\\theta$. The $\\mathrm{prox}_{J_2}$ panel asks a "
        "different question —\ndoes the prior that One-shot recovery REPORTS have the denoiser as "
        "its prox? For the direct fit those two objects are identical by construction.")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.subplots_adjust(hspace=0.30)
    stem = os.path.join(TW.FIG_DIR, f"threeway_cameraman{TW.OUT_SUF}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {stem}.png / .pdf", flush=True)

    out = {"rows": rows, "seed": seed, "sweeps": sweeps, "sigma": sigma, "t": t,
           "grad_G_vs_x_median_pct": 100 * float(np.median(rel)),
           "grad_psi_vs_uPM_median_pct": 100 * float(np.median(rp)),
           "check_solve_residual": resid_chk, "direct_prox_residual": resid_dir,
           "check_diverged": int(b.sum()),
           "psi_steps": psi_ck["steps"], "psi_best_val": psi_ck["best_val"],
           "J_ckpt": os.path.basename(jck)}
    with open(os.path.join(TW.RES_DIR, f"threeway_metrics{TW.OUT_SUF}.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    with open(os.path.join(TW.RES_DIR, f"threeway_psnr{TW.OUT_SUF}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out
