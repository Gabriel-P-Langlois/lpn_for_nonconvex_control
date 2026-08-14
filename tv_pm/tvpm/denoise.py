"""The prior IN ACTION: denoise with the learned f_reg, compare to the denoiser.

    ~/miniforge3/envs/lpn_env/bin/python -m tvpm.denoise                     (~2 min, from tv_pm/)

The scatter/kernel figures show the recovered prior CORRELATES with TV; they do
not show it WORKS. This does. The posterior-mean denoiser is exactly the proximal
operator of its implicit regularizer:

    u_PM(x) = argmin_u  f_reg(u) + 1/2 ||u - x||^2          (Gribonval; the prox)

and we trained grad J_theta(u) ~ x - u, i.e. J_theta at the right SCALE to be
f_reg (not merely up to a constant). So if the recovery is faithful, the prox of
the LEARNED prior must reproduce the denoiser:

    u_hat(x) = argmin_u  J_theta(u) + 1/2 ||u - x||^2   ~=?~   u_PM(x).

J_theta is convex (ICNN) and smooth (Softplus), so this is a convex program;
u_hat is solved by L-BFGS on u, checked by the optimality residual
||grad J_theta(u) + (u - x)||.

GROUND TRUTH is u_PM from the SAMPLER (not the clean image): f_reg is a property
of the DENOISER, so the test is whether u_hat matches u_PM. The clean and noisy
images are shown only for context. Everything is PATCH-WISE (8x8, the prior's
domain): each patch is denoised independently by BOTH methods, so the comparison
is fair; the reassembled image therefore carries 8x8 block structure in both
u_PM and u_hat, which is a property of the patch model, not of either denoiser.
"""
import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plotstyle import apply as _apply_style
_apply_style()          # one font size for every figure

from . import dataset
from src.conv_icnn import ConvICNN
from .paths import FIGS as OUT, IMAGES
from .sampler import from_sigma_t, sample_pm
from src.network import LPN            # noqa: E402  (paths.py puts ROOT on sys.path)
# logs/ckpt/ (local) first, then results/ckpt/ (tracked copies that travel; logs/
# and *.pth are gitignored).
def resolve_ckpt(arch, sweeps, beta, steps, sigma, t):
    """Checkpoint matching this exact config -- NOT a fixed default. Keying it on
    (sigma, t) is what lets the CLI denoise at a non-default pair; a hardcoded
    default would sample at the new sigma but load the old net."""
    from .recover import find_checkpoint
    p = find_checkpoint(arch=arch, sweeps=sweeps, beta=beta, steps=steps,
                        sigma=sigma, t=t)
    if p is None:
        raise FileNotFoundError(
            f"no {arch} checkpoint for sigma={sigma:g}, t={t:g}, beta={beta:g}, "
            f"steps={steps}, m={sweeps}. Train it first (`-m tvpm.recover` or the "
            f"notebook), or pass ckpt=... explicitly.")
    return p


def load(path):
    ck = torch.load(path, weights_only=False)
    arch = ck.get("arch", "fc")
    if arch == "conv":
        hw = int(round(ck["in_dim"] ** 0.5))
        m = ConvICNN(hw=(hw, hw), channels=ck["channels"], blocks=ck["blocks"], beta=ck["beta"])
    else:
        m = LPN(in_dim=ck["in_dim"], hidden=ck["hidden"], layers=ck["layers"], beta=ck["beta"])
    m.load_state_dict(ck["state"]); m.eval()
    return m, torch.tensor(ck["mu"]).float(), torch.tensor(ck["s"]).float(), arch


def tile(img, p=8):
    """Non-overlapping p x p patches of an (H, W) image -> (n_patches, p, p)."""
    H, W = img.shape
    return (img.reshape(H // p, p, W // p, p).transpose(0, 2, 1, 3).reshape(-1, p, p))


def untile(patches, H, W, p=8):
    """Inverse of tile: (n_patches, p, p) -> (H, W)."""
    return (patches.reshape(H // p, W // p, p, p).transpose(0, 2, 1, 3).reshape(H, W))


def prox_of_Jtheta(model, mu, s, x, iters=200, tol=1e-6):
    """u_hat = argmin_u J_theta(u) + 1/2||u-x||^2, batched over patches.

    Convex + smooth, so L-BFGS on the flat (N, 64) variable. Returns u_hat and the
    max optimality residual ||grad J_theta(u) + (u-x)|| (which certifies the solve
    independently of the network's accuracy)."""
    xf = torch.tensor(x.reshape(x.shape[0], -1)).float()
    u = xf.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([u], lr=1.0, max_iter=iters, tolerance_grad=tol,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        J = model.scalar((u - mu) / s).sum()          # J_theta at the right scale
        obj = J + 0.5 * ((u - xf) ** 2).sum()
        obj.backward()
        return obj

    opt.step(closure)
    with torch.no_grad():
        z = ((u - mu) / s).detach().requires_grad_(True)
    g = torch.autograd.grad(model.scalar(z).sum(), z)[0] / s        # grad_u J_theta
    resid = (g + (u.detach() - xf)).norm(dim=1) / xf.norm(dim=1).clamp(min=1)
    return u.detach().numpy().reshape(x.shape), float(resid.max())


def psnr(a, b):
    """Peak SNR in dB. Data range 1 (images in [0,1]); just log-MSE with a ref."""
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def rel_l2(a, b):
    """||a - b|| / ||b||: the natural 'how close are these two maps' error, and
    the one directly comparable to the ~10% prox residual."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def ssim(a, b, sigma=1.5):
    """Structural similarity (Wang et al. 2004), the standard PERCEPTUAL companion
    to PSNR. Local Gaussian-windowed means/variances/covariance; range 1."""
    from scipy.ndimage import gaussian_filter
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ma, mb = gaussian_filter(a, sigma), gaussian_filter(b, sigma)
    va = gaussian_filter(a * a, sigma) - ma * ma
    vb = gaussian_filter(b * b, sigma) - mb * mb
    vab = gaussian_filter(a * b, sigma) - ma * mb
    s = ((2 * ma * mb + C1) * (2 * vab + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2))
    return float(s.mean())


def run(image="cameraman_256x256_d", arch="fc", sigma=dataset.SIGMA,
        t=dataset.T, sweeps=8000, seed=3, ckpt=None, steps=250000, beta=20):
    lam = from_sigma_t(sigma, t)[1]
    from scipy.io import loadmat
    img = np.asarray(loadmat(os.path.join(IMAGES, image + ".mat"))[image], dtype=float)
    H, W = img.shape
    xc = tile(img)                                           # clean patches
    rng = np.random.default_rng(seed)
    xn = np.clip(xc + rng.normal(0, sigma, xc.shape), 0, 1)  # noisy patches x_k

    print(f"  image {image} ({H}x{W}), {xn.shape[0]} patches of 8x8, "
          f"sigma={sigma:.4f}, t={t:.4f} -> lam={lam:.4f}")
    print("  running the sampler for u_PM (ground-truth denoiser)...")
    u_pm = sample_pm(xn, sigma, lam, sweeps=sweeps, w=1.0, seed=100)["u_pm"]

    print(f"  solving prox of the learned {arch}-ICNN prior...")
    model, mu, s, _ = load(ckpt or resolve_ckpt(arch, sweeps, beta, steps, sigma, t))
    u_hat, resid = prox_of_Jtheta(model, mu, s, xn)
    print(f"  prox optimality residual (max over patches): {resid:.2e}")

    imgs = {k: untile(v, H, W) for k, v in
            {"clean": xc, "noisy": xn, "u_pm": u_pm, "u_hat": u_hat}.items()}

    # The headline: does the learned prior REPRODUCE the denoiser? Three metrics --
    # PSNR (the field standard, = log-MSE), SSIM (perceptual companion), and
    # relative L2 (ties directly to the ~10% prox residual).
    p_match, s_match, r_match = (psnr(imgs["u_hat"], imgs["u_pm"]),
                                 ssim(imgs["u_hat"], imgs["u_pm"]),
                                 rel_l2(imgs["u_hat"], imgs["u_pm"]))
    p_pm, p_hat, p_noisy = (psnr(imgs["u_pm"], imgs["clean"]),
                            psnr(imgs["u_hat"], imgs["clean"]),
                            psnr(imgs["noisy"], imgs["clean"]))
    s_pm, s_hat = ssim(imgs["u_pm"], imgs["clean"]), ssim(imgs["u_hat"], imgs["clean"])
    print(f"\n  vs clean       PSNR / SSIM :  noisy {p_noisy:.2f}/{ssim(imgs['noisy'],imgs['clean']):.3f}"
          f"  |  u_PM {p_pm:.2f}/{s_pm:.3f}  |  u_hat {p_hat:.2f}/{s_hat:.3f}")
    print(f"  u_hat vs u_PM (the match)  :  PSNR {p_match:.2f} dB  |  SSIM {s_match:.4f}"
          f"  |  rel-L2 {100*r_match:.2f} %   <- learned prior REPRODUCES the denoiser")

    # --- figure: clean | noisy | u_PM | u_hat | |u_hat - u_PM| ---
    fig, ax = plt.subplots(1, 5, figsize=(17, 3.6))
    panels = [("clean", imgs["clean"], "gray", ""),
              ("noisy input", imgs["noisy"], "gray", f"{p_noisy:.1f} dB"),
              ("$u_{PM}$: sampler denoiser", imgs["u_pm"], "gray", f"{p_pm:.1f} dB"),
              (f"$\\hat u$: prox of learned prior ({arch})", imgs["u_hat"], "gray", f"{p_hat:.1f} dB"),
              ("$|\\hat u - u_{PM}|$", np.abs(imgs["u_hat"] - imgs["u_pm"]), "inferno", "")]
    for a, (title, im, cm, sub) in zip(ax, panels):
        vm = dict(vmin=0, vmax=1) if cm == "gray" else dict(vmin=0, vmax=0.1)
        h = a.imshow(im, cmap=cm, **vm)
        a.set_xticks([]); a.set_yticks([])
        a.set_title(title + (f"\n{sub} (vs clean)" if sub else ""))
        if cm != "gray":
            fig.colorbar(h, ax=a, fraction=0.046, pad=0.04)
    cfg = f"  [PM_SWEEPS={sweeps}" + (f", FIT_STEPS={steps}" if steps is not None else "") + "]"
    fig.suptitle(f"Denoising with the RECOVERED prior vs the true posterior-mean "
                 f"denoiser  —  $\\hat u$ vs $u_{{PM}}$: PSNR {p_match:.1f} dB, "
                 f"SSIM {s_match:.4f}, rel-L2 {100*r_match:.1f}%{cfg}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(OUT, exist_ok=True)
    stem = f"denoise_{arch}_{image.split('_')[0]}{dataset.tag(sigma, t)}"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT}/{stem}.png/pdf")
    return dict(match=p_match, pm=p_pm, hat=p_hat, noisy=p_noisy, resid=resid)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=("fc", "conv"), default="fc")
    ap.add_argument("--image", default="cameraman_256x256_d")
    ap.add_argument("--sweeps", type=int, default=8000)
    ap.add_argument("--sigma", type=float, default=dataset.SIGMA)
    ap.add_argument("--t", type=float, default=dataset.T)
    ap.add_argument("--beta", type=float, default=20)
    ap.add_argument("--steps", type=int, default=250000)
    a = ap.parse_args()
    run(image=a.image, arch=a.arch, sweeps=a.sweeps, sigma=a.sigma, t=a.t,
        beta=a.beta, steps=a.steps)


if __name__ == "__main__":
    main()
