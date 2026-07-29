"""Step 0: the exact n=2 quadrature, and THE GATE on the sampler with TV ON.

    ~/miniforge3/envs/lpn_env/bin/python tv_pm/tests/test_quadrature.py   (~40 s)

WHAT THIS IS FOR (DESIGN.md, Step 0). `test_sampler.py` leaves exactly one hole,
and it is the one that matters: it pins the TV arithmetic exactly, and pins the
kernel against a closed form only at w = 0 -- TV switched OFF. The experiment runs
at w = 1. A chain with exact energies can still sample the wrong distribution, and
finite burn-in makes u_PM BIASED rather than merely noisy.

Bias is not a detail here, it is the one failure the whole design cannot see. The
held-out prox residual is scored against the same y-hat used to fit; if y-hat is
biased we fit the prox of the wrong function and the residual stays low anyway. So
either the bias is bounded here, at n = 2, where u_PM is a 2-D integral needing no
MCMC, or it is never bounded at all.

THE GATE is `test_sampler_is_unbiased_with_tv`. DESIGN.md sets it at "~0.1% from
quadrature", but a threshold in % cannot separate bias from Monte Carlo scatter:
a single chain at m = 8000 carries ~1% noise, so it MISSES 0.1% for reasons that
are expected and harmless. So the pooled error is tested against the sampler's OWN
standard error, estimated from the spread of 128 independent chains per x. The
z-statistic is the real gate; the % is reported next to it for the record.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tvpm.quadrature as Q
from tvpm.sampler import pm_truncated_gaussian, sample_pm

SIGMA, LAM = 10 / 256, 32 / 256          # the MATLAB's tabulated pair


def _x_grid(n, seed=0):
    """Test points spanning what TV actually responds to.

    Uniform on (0.02, 0.98)^2: includes near-diagonal x (smooth, TV inactive),
    far-off-diagonal x (an edge, TV fully active), and x near the box faces,
    where the [0,1] constraint bites and a missing guard would show.
    """
    return np.random.default_rng(seed).uniform(0.02, 0.98, (n, 2))


def test_quadrature_is_converged():
    """The r-rule must be converged, or the 'exact' reference is not exact.

    Gauss-Legendre on each side of the kink separately: the integrand is analytic
    there, so this converges spectrally and 100 nodes is already at round-off.
    """
    x = _x_grid(6)
    _, a = Q.log_z_and_pm(x, SIGMA, LAM, w=1.0, n_gl=100)
    _, b = Q.log_z_and_pm(x, SIGMA, LAM, w=1.0, n_gl=400)
    err = np.abs(a - b).max()
    assert err < 1e-12, f"quadrature not converged in n_gl: {err:.3e}"
    print(f"  u_PM stable to {err:.1e} between n_gl = 100 and 400 (spectral, "
          f"kink on the split point)")


def test_quadrature_matches_brute_force():
    """The rotation + closed-form-s trick, against a dumb dense 2-D grid.

    The grid integrates straight across the |u1-u2| kink with a midpoint rule, so
    ITS error (~1e-6) dominates: this bounds a blunder in the rotation, the
    diamond limits or the erf moments, not the last digit. Agreement between two
    methods sharing no algebra is the point.
    """
    x = _x_grid(6)
    _, fast = Q.log_z_and_pm(x, SIGMA, LAM, w=1.0)
    slow = Q.u_pm_grid(x, SIGMA, LAM, w=1.0, n_grid=2048)
    err = np.abs(fast - slow).max()
    assert err < 1e-5, f"rotated quadrature disagrees with the dense grid: {err:.3e}"
    print(f"  agrees with a 2048^2 dense grid to {err:.1e} (grid-limited, as "
          f"expected across the kink)")


def test_quadrature_matches_truncated_gaussian_at_w0():
    """At w = 0 the 2-D integral has a closed form; the quadrature must nail it.

    Shares the w=0 oracle with `test_sampler.py`, which is what lets the two
    reference implementations be compared at all.
    """
    x = _x_grid(6)
    _, got = Q.log_z_and_pm(x, SIGMA, LAM, w=0.0)
    exact = pm_truncated_gaussian(x, SIGMA)
    err = np.abs(got - exact).max()
    assert err < 1e-12, f"quadrature wrong at w=0 by {err:.3e}"
    print(f"  w=0: matches the exact truncated-Gaussian mean to {err:.1e}")


def _jacobian(x, h=1e-5):
    """Dy/dx at each x, by central differences. Symmetric (it is Hess psi)."""
    n = x.shape[0]
    J = np.empty((n, 2, 2))
    for j in range(2):
        e = np.zeros((1, 2))
        e[0, j] = h
        _, yp = Q.log_z_and_pm(x + e, SIGMA, LAM)
        _, ym = Q.log_z_and_pm(x - e, SIGMA, LAM)
        J[:, :, j] = (yp - ym) / (2 * h)
    return J


def test_grad_psi_is_u_pm():
    """The identity chain, link by link: grad psi = u_PM.

    psi = ||x||^2/2 - t S_eps and S_eps = -eps log(Z/(2 pi t eps)^{n/2}) are built
    from Z; u_PM is built from the first moment. They are computed by different
    code paths, so this tests the log-partition normalization AND the moment
    integrals against each other. If it fails, everything the experiment regresses
    on is wrong.
    """
    x = _x_grid(8, seed=1)
    h = 1e-5
    grad = np.empty_like(x)
    for j in range(2):
        e = np.zeros((1, 2))
        e[0, j] = h
        grad[:, j] = (Q.psi(x + e, SIGMA, LAM) - Q.psi(x - e, SIGMA, LAM)) / (2 * h)
    _, u_pm = Q.log_z_and_pm(x, SIGMA, LAM)
    err = np.abs(grad - u_pm).max()
    assert err < 1e-6, f"grad psi != u_PM, off by {err:.3e}"
    print(f"  grad psi == u_PM to {err:.1e} (FD-limited) -- with TV on")


def test_f_reg_gradient_is_the_prox_target():
    """The target itself: grad f_reg(y) = x - y at y = u_PM(x).

    This is what the network regresses on at n = 64, where f_reg cannot be
    evaluated at all (it needs S_eps, and the sampler returns the mean, not the
    partition function). Here both sides exist, so the target can be checked
    rather than assumed.

    f_reg is only reachable along the curve y = u_PM(x), so differentiate
    F(x) = f_reg(u_PM(x)) and peel the chain rule off with the Jacobian:
    dF/dx = (Dy/dx)^T grad f_reg(y), and Dy/dx = Hess psi is symmetric positive
    definite, hence invertible.
    """
    x = _x_grid(8, seed=2)
    h = 1e-5
    dF = np.empty_like(x)
    for j in range(2):
        e = np.zeros((1, 2))
        e[0, j] = h
        _, fp = Q.f_reg(x + e, SIGMA, LAM)
        _, fm = Q.f_reg(x - e, SIGMA, LAM)
        dF[:, j] = (fp - fm) / (2 * h)

    J = _jacobian(x, h)
    assert np.abs(J - np.swapaxes(J, 1, 2)).max() < 1e-6, "Dy/dx is not symmetric"
    assert np.all(np.linalg.eigvalsh(J) > 0), "Hess psi is not positive definite"

    grad_f = np.linalg.solve(J, dF[:, :, None])[:, :, 0]
    y, _ = Q.f_reg(x, SIGMA, LAM)
    err = np.abs(grad_f - (x - y)).max()
    assert err < 1e-4, f"grad f_reg(y) != x - y, off by {err:.3e}"
    print(f"  grad f_reg(y) == x - y to {err:.1e} at y = u_PM(x) -- the regression "
          f"target is the right one")


def _shrinkage(target, target_exact):
    """Systematic relative error in the LENGTH of the target, and its std error.

    The sharp probe, because burn-in bias has a known DIRECTION here. The chains
    start at u0 = x, so an under-burned chain has not finished moving away from x;
    y comes out too close to x and the target x - y too SHORT. Burn-in bias is
    therefore a shrinkage of the target, not a random perturbation of it.

    So project each chain's target onto the exact one and pool over every x at
    once. A per-component z has 2 numbers per x to average; this has one scalar
    per (x, chain) and pools all 100 components, buying another sqrt(100) of
    resolution for free. Returns (b, se) with

        b = sum_k <pooled_k, t_k> / sum_k ||t_k||^2  -  1

    so b < 0 is exactly the under-burned signature, and b = 0 is unbiased.
    """
    proj = np.einsum("krd,kd->kr", target, target_exact)     # (n_x, reps)
    denom = np.sum(target_exact ** 2)
    b = proj.mean(axis=1).sum() / denom - 1.0
    var = np.sum(proj.var(axis=1, ddof=1) / proj.shape[1])   # chains independent
    return b, np.sqrt(var) / denom


def test_sampler_is_unbiased_with_tv():
    """THE GATE. MCMC vs exact quadrature at w = 1, the case nothing else covers.

    Design: 50 x, 128 independent chains each. Pooling kills the MC error as
    1/sqrt(128) while leaving any bias untouched, and -- the point -- the SPREAD of
    those 128 chains estimates the sampler's own standard error, so the pooled
    deviation is tested against a noise scale measured in the same run rather than
    a constant carried in from `noise_diagnostic.ipynb`.

    Scored on x - y, the gradient target, not on u_PM: u_PM ~ 0.5 while x - y ~
    0.03, so a % error on u_PM would flatter the sampler by an order of magnitude
    and would not be the quantity the experiment actually fits.

    Two independent readings, both required:
    Three readings, all required:
      * z = bias/standard-error, per component. Unbiased => z ~ N(0,1), so
        RMS(z) ~ 1 REGARDLESS of m. A bias shows as RMS(z) growing with m: the
        error stops falling while the yardstick keeps shrinking.
      * the relative error itself, which must fall like m^{-1/2}. A bias floors it.
      * the SHRINKAGE statistic, which is the sharp one -- see `_shrinkage`.

    NOT gated on the pooled error against DESIGN.md's literal "~0.1%". That
    number cannot be a property of the sampler: the pooled error is delta/sqrt(reps),
    so thresholding it would test how many chains this test happens to run. The
    quantity DESIGN.md meant is the bias, and the bias is what is bounded here.
    """
    n_x, reps = 50, 128
    x = _x_grid(n_x, seed=3)
    _, y_exact = Q.log_z_and_pm(x, SIGMA, LAM, w=1.0)
    target_exact = x - y_exact
    scale = np.linalg.norm(target_exact)

    xb = np.repeat(x[:, None, :], reps, axis=1).reshape(-1, 1, 2)   # (n_x*reps,1,2)

    rows = []
    for m in (500, 2000, 8000, 32000):
        t0 = time.time()
        u = sample_pm(xb, SIGMA, LAM, sweeps=m, w=1.0, seed=11)["u_pm"]
        chains = u.reshape(n_x, reps, 2)
        target = x[:, None, :] - chains                    # (n_x, reps, 2)

        pooled = target.mean(axis=1)
        bias = pooled - target_exact
        se = target.std(axis=1, ddof=1) / np.sqrt(reps)
        z = bias / se
        rel = np.linalg.norm(bias) / scale
        delta = float(np.mean(target.std(axis=1, ddof=1)) /
                      np.mean(np.abs(target_exact)))
        b, b_se = _shrinkage(target, target_exact)
        rows.append((m, rel, np.sqrt(np.mean(z ** 2)), delta, b, b_se,
                     time.time() - t0))

    print(f"  {'sweeps':>7} {'pooled err':>11} {'RMS z':>7} {'1-chain':>9} "
          f"{'shrinkage b':>13} {'b/se':>7}")
    for m, rel, zrms, delta, b, b_se, dt in rows:
        print(f"  {m:>7} {100*rel:>10.3f}% {zrms:>7.2f} {100*delta:>8.2f}% "
              f"{100*b:>+11.3f}% {b/b_se:>+7.2f}   ({dt:.0f}s)")

    zs = [r[2] for r in rows]
    rels = [r[1] for r in rows]

    # Per-component: z must not grow with m. Unbiased chains hold RMS(z) ~ 1 at
    # every m; a fixed bias makes it climb like sqrt(m) as the error floors.
    assert max(zs) < 3.0, (f"sampler is BIASED with TV on: RMS z = "
                           f"{[f'{v:.2f}' for v in zs]} over m = 500..32000 -- "
                           f"a value growing past 1 means the deviation from "
                           f"quadrature stopped falling while the noise kept falling")

    # Corroboration: the error must decay like m^-1/2, not sit on a floor.
    assert rels[-1] < rels[0] / 4, (f"pooled error did not decay like m^-1/2: "
                                    f"{[f'{100*v:.3f}%' for v in rels]} -- a floor "
                                    f"means bias")

    # THE GATE, at the m the experiment plans to use: bound the target's
    # systematic shrinkage. Signed, so it also catches the right failure mode.
    m8, b8, se8 = rows[2][0], rows[2][4], rows[2][5]
    assert abs(b8 / se8) < 3.0, (f"target systematically off by {100*b8:+.3f}% at "
                                 f"m = {m8} ({b8/se8:+.1f} standard errors) -- "
                                 f"burn-in bias is real; raise `burn` before Step 1")
    bound = 100 * (abs(b8) + 2 * se8)
    assert bound < 0.1, (f"cannot bound the bias below 0.1% at m = {m8}: "
                         f"|b| + 2se = {bound:.3f}%. Not necessarily biased -- "
                         f"this test may just need more chains.")
    print(f"\n  GATE PASSED: at m = 8000 the target's systematic shrinkage is "
          f"{100*b8:+.3f}% +/- {100*se8:.3f}%, i.e. |bias| < {bound:.3f}% at 2 se.")
    print(f"  That is ~{rows[2][3]/(abs(b8) + 2*se8):.0f}x below the {100*rows[2][3]:.1f}% "
          f"noise of the ONE chain per x_k that Step 1 will run, so the prox "
          f"residual is read against noise, not bias.")


if __name__ == "__main__":
    for fn in (test_quadrature_is_converged,
               test_quadrature_matches_truncated_gaussian_at_w0,
               test_quadrature_matches_brute_force,
               test_grad_psi_is_u_pm,
               test_f_reg_gradient_is_the_prox_target,
               test_sampler_is_unbiased_with_tv):
        print(f"{fn.__name__}:")
        fn()
    print("\nOK: quadrature exact at n=2; sampler unbiased with TV on.")
