"""Batched posterior-mean sampler for anisotropic TV with a quadratic fidelity.

A Python port of `ideas/old_files/proj1_mcmc_alg_pme.m` (Metropolis-within-Gibbs,
single-site random-walk proposals, pixels constrained to [0,1]), with ONE change
that decides whether the whole direction is affordable.

WHY THE PORT EXISTS. The MATLAB script computes u_PM at ONE x -- the noisy
Barbara -- using m = 20000 sweeps x R = 65536 sites x 2 chains ~ 2.6e9 single-site
steps. The prior-recovery experiment needs u_PM(x_k) at N DIFFERENT x_k, so that
cost is multiplied by N and the script's structure cannot deliver it.

THE FIX: the N chains are INDEPENDENT (different x_k), so they batch. Anisotropic
TV on a 4-neighbourhood is a nearest-neighbour MRF, so pixels of one checkerboard
colour are conditionally independent given the other colour and can be updated
SIMULTANEOUSLY AND EXACTLY -- across every chain at once. One sweep becomes two
vectorized ops on an (N, H*W/2) array instead of H*W sequential site visits. The
Markov chain is unchanged; only the scan order is (a valid Gibbs scan, and a
better-mixing one than random-with-replacement).

THE MODEL, exactly as in the MATLAB:

    E(u) = ||u - x||^2 / (2t)  +  w * ATV(u),        u in [0,1]^n
    posterior(u) ~ exp(-E(u) / eps)
    ATV(u) = sum over unordered 4-neighbour PAIRS |u_i - u_j|
    u_PM(x) = E[u]

with eps = 2*sigma^2/lam and t = sigma^2/eps = lam/2, so sqrt(t*eps) = sigma is
the per-pixel marginal std. `w` is a test hook: the MATLAB has w = 1 hardwired,
and w = 0 collapses the posterior to independent truncated Gaussians whose mean
is known in closed form -- the exact oracle `test_sampler.py` checks against.

The [0,1] box is not incidental: it makes the effective prior TV + indicator, so
u_PM maps into the OPEN box and the implicit regularizer is only defined there.
"""
import numpy as np
from scipy.stats import norm

# The four 4-neighbourhood shifts, as (axis, offset) on an (N, H, W) array.
_NEIGHBOURS = ((1, -1), (1, +1), (2, -1), (2, +1))


def params(sigma, lam):
    """(eps, t) from the noise and smoothing levels, as in the MATLAB header."""
    eps = 2.0 * sigma ** 2 / lam
    return eps, sigma ** 2 / eps          # t = lam/2


def from_sigma_t(sigma, t):
    """(eps, lam) from the CHOSEN pair (sigma, t). Inverse of params().

    The model has two degrees of freedom, not four: sqrt(t*eps) = sigma and
    lam = 2t. We take (sigma, t) as the chosen pair because they are the two
    that mean something on their own -- sigma is the noise the denoiser is
    built for, t is the denoising strength (eps -> 0 sends u_PM to the MAP,
    prox_{t*ATV}) -- and let eps = sigma^2/t and lam = 2t follow. eps is then
    the temperature, i.e. exactly the amount by which f_reg is a SMOOTHED TV
    rather than TV.
    """
    return sigma ** 2 / t, 2.0 * t


def _shift(u, axis, off):
    """Neighbour values along `axis`, and a mask marking where one exists.

    Boundaries do NOT wrap: the MATLAB guards each neighbour with i>1 / i<size,
    so edge pixels have 3 neighbours and corners 2. Rolling without the mask
    would silently impose periodic boundaries and change the model.
    """
    nb = np.roll(u, off, axis=axis)
    valid = np.ones(u.shape[1:], dtype=bool)
    idx = 0 if off > 0 else -1           # the row/col that wrapped around
    if axis == 1:
        valid[idx, :] = False
    else:
        valid[:, idx] = False
    return nb, valid


def energy(u, x, t, w=1.0):
    """E(u) per chain, from scratch. Used to test the incremental form."""
    quad = np.sum((u - x) ** 2, axis=(1, 2)) / (2.0 * t)
    tv = np.zeros(u.shape[0])
    for axis, off in ((1, +1), (2, +1)):          # each unordered pair once
        nb, valid = _shift(u, axis, off)
        tv += np.sum(np.abs(u - nb) * valid, axis=(1, 2))
    return quad + w * tv


def _delta_energy(u, x, d, t, w):
    """E(u + d*e_i) - E(u) at every site simultaneously, for a proposal field d.

    Valid site-by-site ONLY for sites of one checkerboard colour, since it holds
    every neighbour fixed. That is exactly the conditional independence the scan
    exploits.
    """
    dE = (d / t) * (0.5 * d + u - x)              # the quadratic part, exact
    for axis, off in _NEIGHBOURS:
        nb, valid = _shift(u, axis, off)
        dE += w * (np.abs(nb - (u + d)) - np.abs(nb - u)) * valid
    return dE


def checkerboard(h, w):
    i, j = np.indices((h, w))
    return (i + j) % 2


def sample_pm(x, sigma, lam, sweeps, alpha=None, burn=None, w=1.0, seed=0,
              u0=None, thin=1, track=None):
    """Posterior-mean estimate u_PM(x_k) for EVERY x_k at once.

    x       : (N, H, W) noisy images -- one independent chain each
    sweeps  : total sweeps; each sweep updates both colours once
    burn    : sweeps discarded before averaging (default sweeps // 5)
    alpha   : uniform proposal half-width; default sqrt(3)*sigma, so the proposal
              std equals sigma. (The MATLAB's alpha = factor*sqrt(3)/scale is the
              same rule -- at its tabulated lam=32, factor=10, scale=256 the
              proposal std is 10/256 = sigma.)
    u0      : chain start; default x itself. The MATLAB starts at u_MAP, which
              needs a TV solve per chain; starting at x costs burn-in instead.
    track   : if an int, also return the running mean every `track` sweeps, for
              convergence diagnostics.

    Returns dict with u_pm (N, H, W), acceptance rate, and optionally the trace.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    N, H, W = x.shape
    eps, t = params(sigma, lam)
    alpha = np.sqrt(3.0) * sigma if alpha is None else alpha
    burn = sweeps // 5 if burn is None else burn
    colour = checkerboard(H, W)

    u = (x.copy() if u0 is None else np.asarray(u0, dtype=float)).clip(0.0, 1.0)
    acc = np.zeros(N)
    prop = 0
    total = np.zeros_like(u)
    n_avg = 0
    trace = []

    for s in range(sweeps):
        for c in (0, 1):
            mask = (colour == c)
            d = rng.uniform(-alpha, alpha, size=u.shape) * mask
            cand = u + d
            inbox = (cand >= 0.0) & (cand <= 1.0)      # the MATLAB's box guard
            dE = _delta_energy(u, x, d, t, w)
            # exp(-dE/eps) > rand, guarded against overflow for dE << 0
            take = mask & inbox & (rng.random(u.shape) < np.exp(np.minimum(-dE / eps, 0.0)))
            u = np.where(take, cand, u)
            acc += take.sum(axis=(1, 2))
            prop += mask.sum()

        if s >= burn and (s - burn) % thin == 0:
            total += u
            n_avg += 1
            if track and (s - burn) % track == 0:
                trace.append(total / n_avg)

    out = {"u_pm": total / n_avg, "accept": float(np.mean(acc) / prop),
           "n_avg": n_avg, "eps": eps, "t": t, "alpha": alpha}
    if track:
        out["trace"] = np.array(trace)
    return out


def pm_truncated_gaussian(x, sigma):
    """EXACT u_PM when w = 0: the posterior factorizes into independent
    N(x_i, sigma^2) truncated to [0,1], whose mean is closed-form.

    This is the sampler's oracle. It exercises the MH kernel, the acceptance
    rule, the box guard, the batching and the burn-in -- everything but the TV
    term, which `test_sampler.py` pins separately against `energy`.
    """
    a, b = (0.0 - x) / sigma, (1.0 - x) / sigma
    Z = norm.cdf(b) - norm.cdf(a)
    return x + sigma * (norm.pdf(a) - norm.pdf(b)) / Z


def load_image_file(path, size):
    """A clean size x size patch in [0,1]; the x_k are this plus noise."""
    from PIL import Image
    im = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=float) / 255.0
