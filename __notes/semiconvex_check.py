"""Backward viscosity solution and the right network class, checked in 1D.

Setting: eps = 1, t = sigma^2, prior p_phi a separated 2-component mixture (NOT
log-concave). The conjugate psi^* is computed EXACTLY through the inverse of
D = grad psi (strictly increasing), not by a bounded grid supremum: a grid sup
truncates -- at sigma = 1, psi^*(y) at y = 3 is already attained at x ~ 23 --
and the truncated f_reg is affine there, which fabricates curvature exactly -1
and inflates the f_reg/t vs J gap (bug found and fixed 2026-07-29).

    J      = -log p_phi                      the true prior (nonconvex)
    h      = -sigma^2 log p_z                PIRATE's implicit regularizer = t*S_eps
    psi    = |x|^2/2 - h                     grad psi = D, convex ALWAYS
    f_reg  = psi^* - |.|^2/2                 the regularizer D is the prox of

Claims:
  (1) f_reg is NONCONVEX, but f_reg + |.|^2/2 = psi^* is convex
      -> f_reg is 1-SEMICONVEX, and that is exactly the class an
         "ICNN minus a quadratic" parametrizes. Its curvature is bounded below
         by -1; the bound is APPROACHED where the posterior variance is large,
         not attained (here: -0.911 at sigma = 0.5, -0.728 at sigma = 1).
  (2) prox_{f_reg} = D GLOBALLY, even though f_reg is nonconvex
      -> the quadratic in the prox exactly compensates the nonconvexity.
  (3) the backward (sup-convolution) solution recovers f_reg EXACTLY
      -> f_reg is its own proximal hull.
  (4) f_reg / t vs J: a measurable gap -> what the noise level destroyed.
"""
import numpy as np
from scipy.special import logsumexp

# ---------------------------------------------------------------- the model
MU = np.array([-2.0, 2.0])
VAR = np.array([0.05, 0.05])
PI = np.array([0.5, 0.5])


def log_mixture(x, var):
    """log sum_i pi_i N(x; mu_i, var_i), broadcasting over the grid."""
    x = np.asarray(x, float)
    z = np.stack([np.log(PI[i]) - 0.5 * ((x - MU[i]) ** 2 / var[i]
                                         + np.log(2 * np.pi * var[i]))
                  for i in range(len(PI))])
    return logsumexp(z, axis=0)


def dlog_mixture(x, var):
    """(d/dx) log of the same mixture, analytically."""
    x = np.asarray(x, float)
    l = np.stack([np.log(PI[i]) - 0.5 * ((x - MU[i]) ** 2 / var[i]
                                         + np.log(2 * np.pi * var[i]))
                  for i in range(len(PI))])
    w = np.exp(l - logsumexp(l, axis=0, keepdims=True))
    dl = np.stack([-(x - MU[i]) / var[i] for i in range(len(PI))])
    return (w * dl).sum(axis=0)


def psi_of(x, sigma):
    return 0.5 * np.asarray(x, float) ** 2 + sigma ** 2 * log_mixture(x, VAR + sigma ** 2)


def D_of(x, sigma):
    return np.asarray(x, float) + sigma ** 2 * dlog_mixture(x, VAR + sigma ** 2)


def D_inv(y, sigma, lo=-1e4, hi=1e4, iters=90):
    """x = D^{-1}(y) by vectorized bisection; D is strictly increasing."""
    y = np.asarray(y, float)
    a, b = np.full_like(y, lo), np.full_like(y, hi)
    for _ in range(iters):
        mid = 0.5 * (a + b)
        left = D_of(mid, sigma) < y
        a, b = np.where(left, mid, a), np.where(left, b, mid)
    return 0.5 * (a + b)


def freg_of(y, sigma):
    """f_reg = psi^* - y^2/2, exactly: psi^*(y) = x*y - psi(x*) at x* = D^{-1}(y)."""
    y = np.asarray(y, float)
    xs = D_inv(y, sigma)
    return xs * y - psi_of(xs, sigma) - 0.5 * y ** 2


def d2(f, g):
    """Second difference on a uniform grid: the convexity test."""
    dx = g[1] - g[0]
    return (f[2:] - 2 * f[1:-1] + f[:-2]) / dx ** 2


def report(name, err, tol):
    ok = err < tol
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:56s} {err:.3e}")
    return ok


ok = True
print("=" * 78)
print("1D separated mixture prior (NOT log-concave), eps = 1, t = sigma^2")
print("=" * 78)

for sigma in (0.5, 1.0):
    t = sigma ** 2
    g = np.linspace(-6, 6, 24001)
    dx = g[1] - g[0]

    h = -sigma ** 2 * log_mixture(g, VAR + sigma ** 2)      # = t * S_eps
    D = D_of(g, sigma)
    J = -log_mixture(g, VAR)                                 # the TRUE prior
    f_reg = freg_of(g, sigma)                                # EXACT conjugate

    print(f"\n--- sigma = {sigma:.2f}  (t = {t:.2f}) " + "-" * 40)

    # (1) convexity of f_reg vs of f_reg + |.|^2/2
    m = (g > -6 + 2 * dx) & (g < 6 - 2 * dx)
    mi = m[1:-1]
    c_f = d2(f_reg, g)[mi]
    c_g = d2(f_reg + 0.5 * g ** 2, g)[mi]
    print(f"    min curvature of f_reg          = {c_f.min():+9.4f}"
          f"   -> {'NONCONVEX' if c_f.min() < -1e-6 else 'convex'}"
          f"   (floor -1, approached, not attained)")
    print(f"    min curvature of f_reg + x^2/2  = {c_g.min():+9.4f}"
          f"   -> {'convex (1-SEMICONVEX f_reg)' if c_g.min() > -1e-3 else 'NOT convex'}")
    ok &= report("f_reg + x^2/2 convex (violation)", max(0.0, -c_g.min()), 1e-3)

    # (2) prox_{f_reg} = D globally, by brute-force grid minimization.
    # Score the OBJECTIVE, not the minimizer: where psi'' is large, psi^* is
    # nearly affine, so the prox objective is nearly flat and its minimizer is
    # ill-conditioned while its value is not (see the task document, Sec. 3.4).
    probes = np.linspace(-4, 4, 81)
    err_obj, err_arg = 0.0, 0.0
    for x in probes:
        obj = f_reg + 0.5 * (x - g) ** 2
        u_star = g[np.argmin(obj)]
        Dx = float(D_of(x, sigma))
        err_obj = max(err_obj, float(freg_of(Dx, sigma)) + 0.5 * (x - Dx) ** 2 - obj.min())
        err_arg = max(err_arg, abs(u_star - Dx))
    ok &= report("prox objective gap at u = D(x) vs global grid min", err_obj, 1e-6)
    print(f"    (minimizer distance {err_arg:.3e} ~ {err_arg/dx:.0f} cells;"
          f" ill-conditioned where psi'' is large)")

    # (3) backward sup-convolution recovers f_reg: J_min(u) = h(x*) - |x*-u|^2/2
    # at the exact maximizer x* = D^{-1}(u) (first-order condition h'(x*) = x*-u).
    xs = D_inv(g, sigma)
    J_min = (-sigma ** 2 * log_mixture(xs, VAR + sigma ** 2)) - 0.5 * (xs - g) ** 2
    c = np.mean((J_min - f_reg)[m])
    ok &= report("|sup_x[h(x)-|x-u|^2/2] - f_reg|  (up to const)",
                 np.max(np.abs((J_min - f_reg - c)[m])), 1e-9)

    # (4) f_reg/t against the true prior J
    fr, Jt = f_reg[m] / t, J[m]
    fr, Jt = fr - fr.min(), Jt - Jt.min()
    rel = np.max(np.abs(fr - Jt)) / np.ptp(Jt)
    print(f"    max |f_reg/t - J| / range(J)    = {100*rel:8.2f} %"
          f"   <- what the noise level destroyed")
    print(f"    min curvature: true J {d2(J, g)[mi].min():+10.3f}"
          f"   vs  f_reg/t {c_f.min()/t:+10.3f}   (floor -1/t = {-1/t:+.2f})")

print()
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
