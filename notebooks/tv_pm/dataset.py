"""Step 1: the data. Noisy patches x_k, and the denoiser's answer y_k = u_PM(x_k).

    ~/miniforge3/envs/lpn_env/bin/python notebooks/tv_pm/dataset.py          # ~15 min
    ~/miniforge3/envs/lpn_env/bin/python notebooks/tv_pm/dataset.py --smoke  # ~20 s

Everything the method is allowed to see: a point, one denoiser call, and the
difference between them. Prox optimality makes

    grad f_reg(y_k) = x_k - y_k     at   y_k = u_PM(x_k)

exact, so the pair (y_k, x_k - y_k) is the whole training set. No value function,
no S_eps, no ground truth -- none of it exists here (DESIGN.md).

WHY THIS IS A SEPARATE, CACHING SCRIPT. The sampler is the entire cost of the
experiment (~15 min; training is ~10). Nothing downstream may retrigger it, or a
bias sweep over m turns into an afternoon. One run per (split, m) -> data/*.npz,
and `recover.py` only ever reads. That the sampler is trustworthy at all is
Step 0's result (`test_quadrature.py`): unbiased with TV on, |bias| < 0.055% at
m = 8000, ~40x under a single chain's 2.2% noise.

SCOPE, decided 2026-07-16 and not a caveat to bury: x_k are natural-image patches
plus noise, so f_reg is recovered on the NATURAL-PATCH REGION of (0,1)^n, not on
all of it. The denoiser is probed where images actually live -- what a
plug-and-play deployment needs -- and f_reg is simply unconstrained off that
region. Uniform x_k on [0,1]^64 was rejected: in 64 dimensions every uniform
sample is a noise image.

DELIBERATELY NOT IN DESIGN.md's file list, which routed data through recover.py.
Splitting it keeps the expensive, cached, run-once stage out of the importable
training code, and lets the bias sweep re-enter at one m without touching it.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sampler import params, sample_pm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))       # numerics/
IMAGES = os.path.join(ROOT, "ideas", "old_files")
CACHE = os.path.join(HERE, "data")

SIGMA, LAM = 10 / 256, 32 / 256      # the MATLAB's tabulated pair
PATCH = 8                            # 8x8 -> n = 64  (decided 2026-07-17)
SWEEPS = 8000                        # delta ~ 1.2% at n=64 (noise_diagnostic.ipynb)

# (split, image, count, seed). Train/val from Barbara, eval from cameraman: f_reg
# is a property of the DENOISER, not of any image, so it must transfer. If it does
# not, we learned the patch distribution instead -- which is the point of the split.
SPLITS = (("train", "barbara_256x256_d", 20_000, 1),
          ("val",   "barbara_256x256_d",  4_000, 2),
          ("eval",  "cameraman_256x256_d", 4_000, 3))


def load_image(name):
    """The MATLAB's own 256x256 data, already in [0,1]."""
    from scipy.io import loadmat
    return np.asarray(loadmat(os.path.join(IMAGES, name + ".mat"))[name], dtype=float)


def patches(img, n, size, rng, taken=None):
    """n random size x size patches, at DISTINCT top-left positions.

    "Disjoint" can only mean distinct positions, not disjoint pixels: a 256x256
    image holds just (256/8)^2 = 1024 non-overlapping 8x8 patches and we need
    24000. So train and val patches DO share pixels. This is worth being explicit
    about, because it means val is not held out in the strong sense:

      * what it still tests -- each x_k carries a fresh noise draw and an
        independent chain, so val measures generalization over the noise and the
        sampler, which is where the error actually comes from;
      * what it cannot test -- transfer off the training image's content. That is
        exactly what the cameraman eval split is for, and why the headline number
        is read there.

    `taken` is a set of positions already used, so val cannot reuse a train one.
    """
    h, w = img.shape
    lim = (h - size + 1) * (w - size + 1)
    taken = set() if taken is None else taken
    picked = []
    while len(picked) < n:
        need = n - len(picked)
        for f in rng.choice(lim, size=min(2 * need, lim), replace=False):
            if f not in taken:
                taken.add(int(f))
                picked.append(int(f))
                if len(picked) == n:
                    break
    out = np.empty((n, size, size))
    for k, f in enumerate(picked):
        i, j = divmod(f, w - size + 1)
        out[k] = img[i:i + size, j:j + size]
    return out, taken


def build(split, image, n, seed, sweeps=SWEEPS, size=PATCH, taken=None, quiet=False):
    """Clean patches -> noisy x_k -> one chain each -> y_k = u_PM(x_k)."""
    rng = np.random.default_rng(seed)
    clean, taken = patches(load_image(image), n, size, rng, taken)

    # CLIPPED (decided 2026-07-17): matches the MATLAB's imnoise and the input a
    # deployed denoiser actually receives. It restricts which region of f_reg is
    # seen, which is a scope statement, not a defect.
    x = np.clip(clean + rng.normal(0, SIGMA, clean.shape), 0.0, 1.0)

    t0 = time.time()
    out = sample_pm(x, SIGMA, LAM, sweeps=sweeps, w=1.0, seed=100 + seed)
    y = out["u_pm"]
    dt = time.time() - t0

    if not quiet:
        eps, t = params(SIGMA, LAM)
        # y must be strictly inside: the [0,1] box is in the PRIOR, so u_PM maps
        # into the OPEN box and f_reg is defined only there. y on a face means the
        # sampler, not the model, put it there.
        assert np.all((y > 0) & (y < 1)), "u_PM left the open box"
        print(f"  {split:5s}: n={n:6d}  {dt/60:5.2f} min  accept={out['accept']:.3f}  "
              f"|x-y| mean={np.mean(np.abs(x - y)):.5f} (sigma={SIGMA:.4f})  "
              f"y in [{y.min():.4f}, {y.max():.4f}]")
    return {"x": x, "y": y, "g": x - y, "clean": clean,
            "sweeps": sweeps, "accept": out["accept"], "eps": eps, "t": t}, taken


def path(split, sweeps=SWEEPS, size=PATCH):
    return os.path.join(CACHE, f"{split}_{size}x{size}_m{sweeps}.npz")


def load(split, sweeps=SWEEPS, size=PATCH):
    """Read a cached split. Never samples -- a miss is an error, not a rebuild."""
    p = path(split, sweeps, size)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing; run dataset.py first (it is ~15 min)")
    return dict(np.load(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweeps", type=int, default=SWEEPS,
                    help="MCMC sweeps per chain; the bias sweep re-enters here")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale every split size. The m-sweep needs three m at a "
                         "FIXED N, and m=32000 costs 72 min at full N but 18 at "
                         "0.25 -- affordable because N is not binding (~10%% at 4x)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny N and few sweeps, to prove the wiring, not the data")
    ap.add_argument("--force", action="store_true", help="resample over a cached split")
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    splits = SPLITS
    if args.scale != 1.0:
        splits = tuple((s, im, int(n * args.scale), sd) for s, im, n, sd in SPLITS)
    if args.smoke:
        splits = tuple((s, im, 200, sd) for s, im, _, sd in SPLITS)

    print(f"patch {PATCH}x{PATCH} (n={PATCH**2}), sigma={SIGMA:.4f}, lam={LAM:.4f}, "
          f"sweeps={args.sweeps}, clipped x")
    taken = set()
    for split, image, n, seed in splits:
        p = path(split, args.sweeps)
        if os.path.exists(p) and not args.force and not args.smoke:
            print(f"  {split:5s}: cached, skipping ({p})")
            continue
        # Train and val must not share a position; eval is a different image, so
        # its position pool is its own.
        d, taken = build(split, image, n, seed, sweeps=args.sweeps,
                         taken=taken if image == "barbara_256x256_d" else set())
        if not args.smoke:
            np.savez_compressed(p, **d)
            print(f"         -> {p} ({os.path.getsize(p)/1e6:.1f} MB)")
    print("done" + (" (smoke: nothing written)" if args.smoke else ""))


if __name__ == "__main__":
    main()
