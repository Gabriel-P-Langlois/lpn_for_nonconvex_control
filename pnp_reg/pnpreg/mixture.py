"""The 1-D Gaussian-mixture prior and its exact plug-and-play objects.

Everything Experiment 1 needs, in closed form or on a dense grid, with no MCMC
and no network (task document, Section 6.1). Conventions follow the manuscript:
epsilon = 1, t = sigma^2, and

    J      = -log p_phi                       the true prior (nonconvex)
    p_z    = p_phi * G_sigma                  the same mixture, variance nu + sigma^2
    D(x)   = x + sigma^2 (log p_z)'(x)        the exact MMSE denoiser (Tweedie)
    h      = -sigma^2 log p_z = t S_eps(.,t)  the implicit regularizer
    psi    = x^2/2 - h                        grad psi = D, convex for ANY prior
    f_reg  = psi^* - x^2/2 = t J_BVS          what the denoiser can reveal
                                              (eq:formula_recovery_prior)

The prior is a separated two-component mixture BECAUSE it is not log-concave:
f_reg is then nonconvex (curvature down to -1), which is what the experiment
displays. The derivative (log p_z)' is analytic (mixture posterior-weighted
means), not finite differences.
"""
import numpy as np
from scipy.special import logsumexp

MU = np.array([-2.0, 2.0])
NU = 0.05                      # component variance of the PRIOR
PI = np.array([0.5, 0.5])


def _log_mix(x, var):
    """log sum_i pi_i N(x; MU_i, var) for scalar or array x (one shared var)."""
    x = np.asarray(x, float)
    z = np.stack([np.log(PI[i]) - 0.5 * ((x - MU[i]) ** 2 / var
                                         + np.log(2 * np.pi * var))
                  for i in range(len(PI))])
    return logsumexp(z, axis=0)


def _dlog_mix(x, var):
    """(d/dx) log p for the same mixture, analytically.

    log p = logsumexp_i(l_i), so (log p)' = sum_i w_i * l_i' with posterior
    weights w_i = exp(l_i - logsumexp), and l_i' = -(x - MU_i)/var.
    """
    x = np.asarray(x, float)
    l = np.stack([np.log(PI[i]) - 0.5 * ((x - MU[i]) ** 2 / var
                                         + np.log(2 * np.pi * var))
                  for i in range(len(PI))])
    w = np.exp(l - logsumexp(l, axis=0, keepdims=True))
    dl = np.stack([-(x - MU[i]) / var for i in range(len(PI))])
    return (w * dl).sum(axis=0)


def J(x):
    """The true prior, J = -log p_phi (epsilon = 1)."""
    return -_log_mix(x, NU)


def log_pz(x, sigma):
    """log p_z, p_z = p_phi * G_sigma: the mixture widened to variance NU + sigma^2."""
    return _log_mix(x, NU + sigma ** 2)


def D(x, sigma):
    """The exact MMSE denoiser, D(x) = x + sigma^2 (log p_z)'(x). Closed form."""
    return np.asarray(x, float) + sigma ** 2 * _dlog_mix(x, NU + sigma ** 2)


def _ddlog_mix(x, var):
    """(d^2/dx^2) log p for the same mixture, analytically.

    With l_i the component log-densities and posterior weights
    w_i = exp(l_i - logsumexp): (log p)'' = sum_i w_i (l_i'' + (l_i')^2)
    - ((log p)')^2, where l_i' = -(x - MU_i)/var and l_i'' = -1/var.
    """
    x = np.asarray(x, float)
    l = np.stack([np.log(PI[i]) - 0.5 * ((x - MU[i]) ** 2 / var
                                         + np.log(2 * np.pi * var))
                  for i in range(len(PI))])
    w = np.exp(l - logsumexp(l, axis=0, keepdims=True))
    dl = np.stack([-(x - MU[i]) / var for i in range(len(PI))])
    first = (w * dl).sum(axis=0)
    second = (w * (-1.0 / var + dl ** 2)).sum(axis=0)
    return second - first ** 2


def dD(x, sigma):
    """D'(x) = 1 + sigma^2 (log p_z)''(x), analytically.

    Equal to Var[phi | x] / sigma^2 (the conditional variance identity behind
    Tweedie), hence strictly positive -- D is strictly increasing -- and it
    EXCEEDS 1 wherever the posterior variance exceeds sigma^2, which happens
    between the modes of a separated mixture: the condition-3 failure the
    probe's calibration relies on (dD(0, 0.5) ~ 11.3).
    """
    return 1.0 + sigma ** 2 * _ddlog_mix(x, NU + sigma ** 2)


def h(x, sigma):
    """The implicit regularizer h = -sigma^2 log p_z = t S_eps(., t)."""
    return -sigma ** 2 * log_pz(x, sigma)


def psi(x, sigma):
    """psi = x^2/2 - h; grad psi = D and psi is convex (eq:psi_defn, viscous)."""
    x = np.asarray(x, float)
    return 0.5 * x ** 2 - h(x, sigma)


def grid(a=6.0, n=24001, pad=8.0):
    """(g, gw): the plot grid [-a, a] and a wider grid at the same spacing.

    Kept for callers that want a wide grid for grid-based checks. The conjugate
    itself (`freg`) does NOT use gw: a bounded grid truncates the supremum --
    at sigma = 0.5, psi^*(y) at y = 6 is attained at x = D^{-1}(6) ~ 26, far
    outside any affordable grid -- so it is computed through the exact inverse
    of D instead (found 2026-07-29; the truncated version corrupted f_reg for
    |y| beyond ~4 at sigma = 0.5 and ~2.5 at sigma = 1).
    """
    dx = 2 * a / (n - 1)
    m = int(round(pad / dx))
    gw = np.linspace(-a - m * dx, a + m * dx, n + 2 * m)
    return gw[m:-m], gw


def D_inverse(y, sigma, lo=-1e4, hi=1e4, iters=90):
    """x = D^{-1}(y), vectorized bisection. D = grad psi is strictly increasing
    (its derivative is the posterior variance over sigma^2, positive), so the
    root is unique; 90 halvings of [-1e4, 1e4] give ~1e-13 in x."""
    y = np.asarray(y, float)
    a = np.full_like(y, lo)
    b = np.full_like(y, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        left = D(m, sigma) < y
        a = np.where(left, m, a)
        b = np.where(left, b, m)
    return 0.5 * (a + b)


def freg(g, sigma):
    """f_reg = psi^* - y^2/2, exactly: psi^*(y) = x*y - psi(x*) at the unique
    x* = D^{-1}(y) where the supremum sup_x {xy - psi(x)} is attained."""
    g = np.asarray(g, float)
    xs = D_inverse(g, sigma)
    return xs * g - psi(xs, sigma) - 0.5 * g ** 2


def jbvs(g, sigma):
    """The backward viscosity solution J_BVS = f_reg / t (eq:backward_hj_prior
    at tau = 0 with the viscous terminal data; t = sigma^2)."""
    return freg(g, sigma) / sigma ** 2


def sample_pz(n, sigma, seed):
    """n samples from p_z: pick a component, then N(MU_i, NU + sigma^2)."""
    rng = np.random.default_rng(seed)
    comp = rng.choice(len(PI), size=n, p=PI)
    return rng.normal(MU[comp], np.sqrt(NU + sigma ** 2))
