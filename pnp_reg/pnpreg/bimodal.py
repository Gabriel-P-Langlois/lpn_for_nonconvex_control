"""Experiment 3.2: an exactly solvable bimodal field prior (Route B).

The prior lives on 8x8 field patches (n = 64) and factorizes in a fixed
orthonormal basis V = [u, V_perp]:

  * along u (a normalized centered Gaussian bump -- one coherent
    deformation pattern), the coordinate s follows the Experiment-1
    two-component mixture (modes +-2, component variance NU = 0.05): the
    field either contains the pattern positively or negatively. The true
    prior J is NONCONVEX along u, so its f_reg = t * J_BVS is the
    proximal-hull object of the backward Hamilton-Jacobi problem, with
    curvature floored at -1 in f_reg units;
  * on u-perp, the coordinates w_k follow independent Gaussians with the
    decaying spectrum lambda_k = 4 / k^2 (a smoothness prior: high modes
    are suppressed). The quadratic J is convex there, so J_BVS = J exactly
    -- the backward problem loses nothing on these 63 directions.

With isotropic noise z = y + sigma * xi the posterior factorizes over the
orthogonal subspaces and every object is exact (t = sigma^2, eps = 1):

  D(z)     = D_mix(s_z) u + sum_k g_k w_k v_k,     g_k = lam_k / (lam_k + sigma^2)
  psi(z)   = psi_mix(s_z) + 1/2 sum_k g_k w_k^2    (grad psi = D, convex)
  f_reg(y) = f_mix(s_y)   + t/2 sum_k w_k^2/lam_k  (= t * J_BVS)
  J(y)     = J_mix(s_y)   + 1/2 sum_k w_k^2/lam_k

with the mixture factors from pnpreg/mixture.py (closed form) and the
Gaussian factors elementary (Wiener gains g_k). The recovery experiment
(pnpreg/bimodal_run.py) trains our networks on pairs (D(z), z) and compares
against these references: it is a numerical solver for the backward
viscosity problem, checked against its closed form.

All numpy float64; nothing here touches torch.
"""
import numpy as np

from . import mixture as mx

N = 64
SIDE = 8
K = np.arange(1, N)                  # perp mode indices 1..63
LAM = 4.0 / K.astype(float) ** 2     # perp spectrum lambda_k


def build_basis():
    """V = [u, V_perp], orthonormal. u is a normalized Gaussian bump at the
    patch center (width 1.5 pixels); V_perp completes it via QR against the
    2-D DCT-II basis (deterministic, smoothness-ordered)."""
    ii, jj = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij")
    bump = np.exp(-(((ii - 3.5) ** 2 + (jj - 3.5) ** 2) / (2 * 1.5 ** 2)))
    u = (bump / np.linalg.norm(bump)).reshape(-1)

    # 1-D DCT-II orthonormal matrix, then the 2-D basis by outer products,
    # columns ordered by total frequency p + q
    k = np.arange(SIDE)
    C = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * SIDE))
    C[0] *= np.sqrt(1.0 / SIDE)
    C[1:] *= np.sqrt(2.0 / SIDE)
    freqs = [(p + q, p, q) for p in range(SIDE) for q in range(SIDE)]
    freqs.sort()
    B = np.stack([np.outer(C[p], C[q]).reshape(-1) for (_, p, q) in freqs], axis=1)

    Q, _ = np.linalg.qr(np.concatenate([u[:, None], B], axis=1))
    Q = Q[:, :N]
    if Q[:, 0] @ u < 0:
        Q[:, 0] *= -1
    for j in range(1, N):
        i = np.argmax(np.abs(Q[:, j]))
        if Q[i, j] < 0:
            Q[:, j] *= -1
    return Q


V = build_basis()
U = V[:, 0]
V_PERP = V[:, 1:]


def to_coords(y):
    """(m, 64) fields -> (s, w): s (m,), w (m, 63)."""
    y = np.atleast_2d(y)
    return y @ U, y @ V_PERP


def from_coords(s, w):
    return np.outer(np.atleast_1d(s), U) + np.atleast_2d(w) @ V_PERP.T


def sample_prior(m, seed):
    """y ~ p: s from the two-component mixture (modes +-2, var NU), w_k ~
    N(0, lambda_k)."""
    rng = np.random.default_rng(seed)
    comp = rng.choice(len(mx.PI), size=m, p=mx.PI)
    s = rng.normal(mx.MU[comp], np.sqrt(mx.NU))
    w = rng.normal(0.0, np.sqrt(LAM)[None, :], size=(m, N - 1))
    return from_coords(s, w)


def sample_data(m, sigma, seed):
    """(z, y): y ~ prior, z = y + sigma * xi."""
    y = sample_prior(m, seed)
    rng = np.random.default_rng(seed + 10_000)
    z = y + sigma * rng.standard_normal(y.shape)
    return z, y


def gains(sigma):
    """The Wiener factors on u-perp."""
    return LAM / (LAM + sigma ** 2)


def D(z, sigma):
    """The exact MMSE denoiser."""
    s, w = to_coords(z)
    return from_coords(mx.D(s, sigma), w * gains(sigma)[None, :])


def psi(z, sigma):
    """psi with grad psi = D (convex; the mixture factor from mixture.psi,
    the Gaussian factors quadratic with the Wiener gains)."""
    s, w = to_coords(z)
    return mx.psi(s, sigma) + 0.5 * (gains(sigma)[None, :] * w ** 2).sum(axis=1)


def freg(y, sigma):
    """The exact implicit regularizer f_reg = t * J_BVS at t = sigma^2:
    proximal-hull-limited along u, exactly t * J on u-perp."""
    s, w = to_coords(y)
    return mx.freg(s, sigma) + 0.5 * sigma ** 2 * (w ** 2 / LAM[None, :]).sum(axis=1)


def J_true(y):
    """The true prior J = -log p up to its additive constant; nonconvex
    along u."""
    s, w = to_coords(y)
    return mx.J(s) + 0.5 * (w ** 2 / LAM[None, :]).sum(axis=1)


def jbvs(y, sigma):
    """The backward viscosity solution J_BVS = f_reg / t."""
    return freg(y, sigma) / sigma ** 2


def freg_u_slice(s_grid, sigma):
    """f_reg along the u-slice (w = 0): the mixture factor alone."""
    return mx.freg(np.asarray(s_grid, float), sigma)


def freg_perp_coeffs(sigma):
    """The exact quadratic coefficients of f_reg on u-perp: t / lambda_k."""
    return sigma ** 2 / LAM
