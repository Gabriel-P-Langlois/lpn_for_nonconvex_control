"""Step 1: the data. Noisy patches x_k, and the denoiser's answer y_k = u_PM(x_k).

    ~/miniforge3/envs/lpn_env/bin/python -m tvpm.dataset                     # ~15 min, from notebooks/tv_pm/
    ~/miniforge3/envs/lpn_env/bin/python -m tvpm.dataset --smoke             # ~20 s

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
import time

import numpy as np


from .paths import DATA as CACHE, IMAGES
from .sampler import from_sigma_t, params, sample_pm


# (sigma, t) are the CHOSEN pair; (eps, lam) follow -- see sampler.from_sigma_t.
# The defaults reproduce the MATLAB's tabulated (sigma, lam) = (10/256, 32/256).
SIGMA, T = 10 / 256, 16 / 256        # t = lam/2
EPS, LAM = from_sigma_t(SIGMA, T)
PATCH = 8                            # 8x8 -> n = 64  (decided 2026-07-17)
SWEEPS = 8000                        # delta ~ 1.2% at n=64 (noise_diagnostic.ipynb)


def tag(sigma=SIGMA, t=T):
    """Filename suffix identifying (sigma, t). EMPTY at the defaults.

    Empty by design: the shipped caches, checkpoints and figures were all built
    at the default pair and carry no suffix, so keeping it empty there leaves
    every existing artifact resolvable. Any other pair gets its own names and
    cannot clobber them.
    """
    if (sigma, t) == (SIGMA, T):
        return ""
    return f"_sig{sigma:.5f}_t{t:.5f}"

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


def build(split, image, n, seed, sweeps=SWEEPS, size=PATCH, taken=None, quiet=False,
          sigma=SIGMA, t=T):
    """Clean patches -> noisy x_k -> one chain each -> y_k = u_PM(x_k)."""
    eps, lam = from_sigma_t(sigma, t)
    rng = np.random.default_rng(seed)
    clean, taken = patches(load_image(image), n, size, rng, taken)

    # CLIPPED (decided 2026-07-17): matches the MATLAB's imnoise and the input a
    # deployed denoiser actually receives. It restricts which region of f_reg is
    # seen, which is a scope statement, not a defect.
    x = np.clip(clean + rng.normal(0, sigma, clean.shape), 0.0, 1.0)

    t0 = time.time()
    out = sample_pm(x, sigma, lam, sweeps=sweeps, w=1.0, seed=100 + seed)
    y = out["u_pm"]
    dt = time.time() - t0

    if not quiet:
        # y must be strictly inside: the [0,1] box is in the PRIOR, so u_PM maps
        # into the OPEN box and f_reg is defined only there. y on a face means the
        # sampler, not the model, put it there.
        assert np.all((y > 0) & (y < 1)), "u_PM left the open box"
        print(f"  {split:5s}: n={n:6d}  {dt/60:5.2f} min  accept={out['accept']:.3f}  "
              f"|x-y| mean={np.mean(np.abs(x - y)):.5f} (sigma={sigma:.4f})  "
              f"y in [{y.min():.4f}, {y.max():.4f}]")
    return {"x": x, "y": y, "g": x - y, "clean": clean, "sweeps": sweeps,
            "accept": out["accept"], "eps": eps, "t": t,
            "sigma": sigma, "lam": lam}, taken


def path(split, sweeps=SWEEPS, size=PATCH, sigma=SIGMA, t=T, scale=1.0):
    # `scale` IS in the name. It changes N, and a reduced-N cache sharing a name
    # with the full one would be picked up as if it were the full dataset -- a
    # silent 100x change in training-set size. Empty at scale=1.0 so the
    # existing full-size caches keep their names.
    sc = "" if scale == 1.0 else f"_x{scale:g}"
    return os.path.join(CACHE,
                        f"{split}_{size}x{size}_m{sweeps}{tag(sigma, t)}{sc}.npz")


def load(split, sweeps=SWEEPS, size=PATCH, sigma=SIGMA, t=T, scale=1.0):
    """Read a cached split. Never samples -- a miss is an error, not a rebuild.
    Use ensure() to sample what is missing."""
    p = path(split, sweeps, size, sigma, t, scale)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} missing; build it with dataset.ensure(sweeps={sweeps}, "
            f"sigma={sigma:g}, t={t:g}) or `python -m tvpm.dataset` (~15 min)")
    d = dict(np.load(p))
    # The name alone is not proof: caches built before (sigma, t) entered the
    # filename carry the DEFAULT name whatever they were sampled at. The npz
    # records eps and t, so check rather than trust.
    eps = sigma ** 2 / t
    for key, want in (("t", t), ("eps", eps)):
        if key in d and not np.isclose(float(d[key]), want, rtol=1e-9):
            raise ValueError(
                f"{p} was sampled at {key}={float(d[key]):.6g}, not {want:.6g} "
                f"(requested sigma={sigma:.6g}, t={t:.6g}). Re-run dataset.py "
                f"--sigma {sigma} --t {t} --force.")
    return d


def plan(sweeps=SWEEPS, sigma=SIGMA, t=T, scale=1.0):
    """What ensure() would sample, without sampling it. Returns the split names
    that are missing -- so a caller can price the work before starting it."""
    return [s for s, _, _, _ in _splits(scale)
            if not os.path.exists(path(s, sweeps, PATCH, sigma, t, scale))]


def _splits(scale=1.0):
    if scale == 1.0:
        return SPLITS
    return tuple((s, im, max(1, int(n * scale)), sd) for s, im, n, sd in SPLITS)


def ensure(sweeps=SWEEPS, sigma=SIGMA, t=T, scale=1.0, force=False, quiet=False):
    """Return (train, val, eval) for this configuration, sampling what is missing.

    The one entry point a caller needs: cached splits are read, absent ones are
    built and written, and nothing else has to know which happened.

    REBUILDS BARBARA'S SPLITS TOGETHER. train and val are drawn from the same
    image and are kept at DISTINCT patch positions by threading a `taken` set
    through both draws. That set cannot be reconstructed from a cached file --
    positions are not stored, and the draw depends on `taken` itself -- so
    building val alone against an empty `taken` would silently let it reuse
    train's positions and quietly destroy the only sense in which val is held
    out. If either Barbara split is missing, both are resampled. `eval` is a
    different image with its own position pool, so it is independent.
    """
    os.makedirs(CACHE, exist_ok=True)
    splits = _splits(scale)
    eps, lam = from_sigma_t(sigma, t)

    def missing(s):
        return force or not os.path.exists(path(s, sweeps, PATCH, sigma, t, scale))

    shared = [s for s, im, _, _ in splits if im == "barbara_256x256_d"]
    redo = set(s for s, _, _, _ in splits if missing(s))
    if redo & set(shared):                     # any Barbara split -> all of them
        redo |= set(shared)

    if redo and not quiet:
        print(f"patch {PATCH}x{PATCH} (n={PATCH**2}), sigma={sigma:.4f}, t={t:.4f} "
              f"-> eps={eps:.5f}, lam={lam:.4f}, m={sweeps}, scale={scale:g}")
        print(f"  sampling: {', '.join(sorted(redo))}"
              + (f"   (cached: {', '.join(s for s, _, _, _ in splits if s not in redo)})"
                 if len(redo) < len(splits) else ""))

    taken = set()
    for split, image, n, seed in splits:
        p = path(split, sweeps, PATCH, sigma, t, scale)
        if split not in redo:
            continue
        d, taken = build(split, image, n, seed, sweeps=sweeps,
                         taken=taken if image == "barbara_256x256_d" else set(),
                         sigma=sigma, t=t, quiet=quiet)
        np.savez_compressed(p, **d)
        if not quiet:
            print(f"         -> {os.path.basename(p)} ({os.path.getsize(p)/1e6:.1f} MB)")

    return tuple(load(s, sweeps, PATCH, sigma, t, scale)
                 for s, _, _, _ in splits)


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
    ap.add_argument("--sigma", type=float, default=SIGMA,
                    help="noise level; with --t it fixes eps=sigma^2/t and lam=2t")
    ap.add_argument("--t", type=float, default=T,
                    help="denoising strength (eps->0 gives u_PM = prox_{t*ATV})")
    args = ap.parse_args()
    # A thin wrapper over ensure(): the notebook calls ensure() directly, so the
    # CLI must not carry any sampling logic of its own that could drift from it.
    scale = 0.01 if args.smoke else args.scale
    ensure(sweeps=args.sweeps, sigma=args.sigma, t=args.t, scale=scale,
           force=args.force)
    print("done")


if __name__ == "__main__":
    main()
