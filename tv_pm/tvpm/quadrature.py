"""Exact u_PM, S_eps, psi and f_reg at n = 2, by quadrature, with TV PRESENT.

WHY THIS EXISTS (DESIGN.md, Step 0). `test_sampler.py` proves the TV arithmetic
is right -- the incremental dE the MH step uses equals `energy()` from scratch to
1e-14 -- and that the kernel is unbiased at w = 0, where the posterior collapses
to truncated Gaussians. Neither statement covers the one case we actually run:
w = 1, TV on. A chain can have exact energies and still sample the wrong
distribution (burn-in bias), and the held-out prox residual CANNOT see it: a
biased y-hat means we fit the prox of the wrong function and the residual, scored
against that same biased y-hat, stays low. So the bias must be ruled out here or
not at all.

At n = 2 (a 1x2 image) u_PM needs no MCMC:

    E(u) = ||u - x||^2/(2t) + w|u1 - u2|,   u in [0,1]^2,   p(u) ~ exp(-E/eps)
    u_PM(x) = int u p(u) du,   Z(x) = int exp(-E(u)/eps) du

THE METHOD. A 2-D grid across the |u1-u2| kink converges like a trapezoid rule on
a nonsmooth integrand -- slowly, and to an error we would then have to defend.
Rotate instead:

    s = (u1+u2)/sqrt2,  r = (u1-u2)/sqrt2      (orthogonal, so du = ds dr)
    ||u-x||^2 = (s-sx)^2 + (r-rx)^2,   |u1-u2| = sqrt2 |r|

The rotation is exactly the one the model asks for: TV now touches r ALONE. The
box [0,1]^2 becomes the diamond |r| <= 1/sqrt2, s in [|r|, sqrt2 - |r|], so for
each fixed r the s-integral is a Gaussian over an interval -- a truncated-Gaussian
moment, closed form via erf. Only r is quadratured, and its integrand is analytic
on each of r < 0 and r > 0, so Gauss-Legendre on each half separately converges
spectrally. The kink sits on the split point and is never integrated across.

`u_pm_grid` is a dumb dense 2-D grid kept as an INDEPENDENT check on the above;
`test_quadrature.py` holds them against each other. Cleverness that agrees with
brute force is worth something; cleverness that is only checked against itself is
not.

THE IDENTITY CHAIN, also verified here (it is quadrature-available at n = 2 and
nowhere else):

    d/dx_i log Z = (u_PM,i - x_i)/(eps t)          [differentiate under the integral]
    S_eps  = -eps log( Z / (2 pi t eps)^{n/2} )    =>  grad S_eps = (x - u_PM)/t
    psi    = ||x||^2/2 - t S_eps                   =>  grad psi   = u_PM
    f_reg  = <x,y> - psi(x) - ||y||^2/2  at y = u_PM(x)   =>  grad f_reg(y) = x - y

The last line is the target the whole experiment regresses on, so being able to
evaluate f_reg itself -- once, at n = 2 -- is what makes Step 0 a check of the
chain and not just of the sampler.
"""
import numpy as np
from scipy.special import ndtr, roots_legendre

SQRT2 = np.sqrt(2.0)
SQRT2PI = np.sqrt(2.0 * np.pi)


def _as_pairs(x):
    """(N,2) view of x, accepting (N,2), (2,), or the sampler's (N,1,2)."""
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, 2) if x.ndim != 2 or x.shape[1] != 2 else x


def _gauss_nodes(n_gl):
    """GL nodes/weights on [-1/sqrt2, 0] and [0, 1/sqrt2], concatenated.

    Splitting at r = 0 is the whole point: |r| is analytic on each half, so the
    rule converges spectrally instead of stalling at the kink.
    """
    xi, wi = roots_legendre(n_gl)
    half = 1.0 / SQRT2
    nodes, weights = [], []
    for lo, hi in ((-half, 0.0), (0.0, half)):
        mid, rad = 0.5 * (lo + hi), 0.5 * (hi - lo)
        nodes.append(mid + rad * xi)
        weights.append(rad * wi)
    return np.concatenate(nodes), np.concatenate(weights)


def _s_moments(r, sx, sd):
    """int exp(-(s-sx)^2/(2 sd^2)) ds and int s exp(...) ds over s in [|r|, sqrt2-|r|].

    Closed form: the interval is where the diamond |r| <= 1/sqrt2 lives.
    """
    lo, hi = np.abs(r), SQRT2 - np.abs(r)
    a, b = (lo - sx) / sd, (hi - sx) / sd
    m0 = sd * SQRT2PI * (ndtr(b) - ndtr(a))
    m1 = sx * m0 + sd ** 2 * (np.exp(-0.5 * a ** 2) - np.exp(-0.5 * b ** 2))
    return m0, m1


def log_z_and_pm(x, sigma, lam, w=1.0, n_gl=200):
    """Exact (log Z, u_PM) at n = 2. Returns (N,) and (N,2).

    n_gl is nodes PER HALF of the r-interval; 200 is far past converged (see
    test_quadrature.py, which pins the value against n_gl = 400).
    """
    x = _as_pairs(x)
    eps, t = 2.0 * sigma ** 2 / lam, lam / 2.0
    sd = np.sqrt(t * eps)                       # = sigma, the marginal std
    sx = (x[:, 0] + x[:, 1]) / SQRT2
    rx = (x[:, 0] - x[:, 1]) / SQRT2

    r, wt = _gauss_nodes(n_gl)                  # (K,)
    r = r[None, :]                              # (1,K) against (N,1)
    m0, m1 = _s_moments(r, sx[:, None], sd)

    # Work in logs and factor out the max: exp(-w sqrt2 |r|/eps) is tiny when
    # eps is small, and Z would otherwise underflow before the ratio is taken.
    with np.errstate(divide="ignore"):
        log_h = (-(r - rx[:, None]) ** 2 / (2.0 * sd ** 2)
                 - w * SQRT2 * np.abs(r) / eps + np.log(m0))
    shift = np.max(log_h, axis=1, keepdims=True)
    h = np.exp(log_h - shift)                   # (N,K), max 1

    z_shifted = np.sum(wt * h, axis=1)
    log_z = np.log(z_shifted) + shift[:, 0]

    # E[s] and E[r]: same weights, integrand reweighted by m1/m0 and by r.
    ratio = np.where(m0 > 0, m1 / np.where(m0 > 0, m0, 1.0), 0.0)
    e_s = np.sum(wt * h * ratio, axis=1) / z_shifted
    e_r = np.sum(wt * h * r, axis=1) / z_shifted

    u_pm = np.stack([(e_s + e_r) / SQRT2, (e_s - e_r) / SQRT2], axis=1)
    return log_z, u_pm


def u_pm_grid(x, sigma, lam, w=1.0, n_grid=2048):
    """u_PM by a dense tensor grid on [0,1]^2 -- the independent check.

    Deliberately naive: midpoint rule, no rotation, straight across the kink.
    Slow and only ~1e-6 accurate, which is why it is the witness and not the
    instrument.
    """
    x = _as_pairs(x)
    eps, t = 2.0 * sigma ** 2 / lam, lam / 2.0
    g = (np.arange(n_grid) + 0.5) / n_grid
    u1, u2 = np.meshgrid(g, g, indexing="ij")
    u1, u2 = u1.ravel(), u2.ravel()
    tv = np.abs(u1 - u2)

    out = np.empty_like(x)
    for k in range(x.shape[0]):                 # loop: the grid is the memory hog
        e = ((u1 - x[k, 0]) ** 2 + (u2 - x[k, 1]) ** 2) / (2.0 * t) + w * tv
        p = np.exp(-(e - e.min()) / eps)
        p /= p.sum()
        out[k] = [np.dot(p, u1), np.dot(p, u2)]
    return out


def s_eps(x, sigma, lam, w=1.0, n_gl=200):
    """S_eps(x) = -eps log( Z / (2 pi t eps)^{n/2} ), n = 2."""
    eps, t = 2.0 * sigma ** 2 / lam, lam / 2.0
    log_z, _ = log_z_and_pm(x, sigma, lam, w, n_gl)
    return -eps * (log_z - np.log(2.0 * np.pi * t * eps))


def psi(x, sigma, lam, w=1.0, n_gl=200):
    """psi(x) = ||x||^2/2 - t S_eps(x). Its gradient is u_PM -- that is the claim."""
    x = _as_pairs(x)
    t = lam / 2.0
    return 0.5 * np.sum(x ** 2, axis=1) - t * s_eps(x, sigma, lam, w, n_gl)


def f_reg(x, sigma, lam, w=1.0, n_gl=200):
    """f_reg at y = u_PM(x), via Fenchel: <x,y> - psi(x) - ||y||^2/2.

    Returns (y, f_reg(y)). Evaluating f_reg needs S_eps, i.e. the log-partition
    function -- which the sampler does NOT return. That is exactly why there is
    no ground truth for f_reg at n = 64, and why this check is only possible here.
    """
    x = _as_pairs(x)
    _, y = log_z_and_pm(x, sigma, lam, w, n_gl)
    val = np.sum(x * y, axis=1) - psi(x, sigma, lam, w, n_gl) - 0.5 * np.sum(y ** 2, axis=1)
    return y, val
