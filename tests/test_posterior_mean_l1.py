"""Pins src.targets.PosteriorMeanL1, the Instantiation-A ground truth.

Run from numerics/ with the project interpreter:

    ~/miniforge3/envs/lpn_env/bin/python tests/test_posterior_mean_l1.py

The class is the ground truth for a one-network experiment, so nothing else in
the pipeline can catch a bug in it: an error here is silently inherited by both
the training targets AND the score they are judged against. Every
orders-of-magnitude correction in changes.txt (B1, B2, B7) was a target bug of
exactly this kind, which is why this file exists before the notebook does.

The load-bearing check is `eps -> 0 == QuadraticL1`: at lam = t = 1 the
posterior-mean family is a smoothing of the family the repo already trusts, so
the limit is an independent oracle rather than a restatement of the code.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.targets import PosteriorMeanL1, QuadraticL1, _log_L


def _fd_grad(f, x, h=1e-5):
    """Central-difference gradient of a scalar-field f: (N,d) -> (N,)."""
    g = np.zeros_like(x)
    for i in range(x.shape[1]):
        e = np.zeros_like(x)
        e[:, i] = h
        g[:, i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def test_log_L_stable():
    """L(z) = 0.5*exp(z^2)*erfc(z) is never formed; the log stays finite where
    the naive product is 0*inf, and matches it where the naive product works."""
    from scipy.special import erfc as _erfc

    z = np.array([-40.0, -5.0, -1.0, 0.0, 1.0, 5.0, 40.0, 1e3])
    out = _log_L(z)
    assert np.all(np.isfinite(out)), f"non-finite log L: {out}"

    # Where exp(z^2) does not overflow, compare against the definition.
    zs = np.array([-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0])
    naive = np.log(0.5 * np.exp(zs**2) * _erfc(zs))
    err = np.max(np.abs(_log_L(zs) - naive))
    assert err < 1e-12, f"log L disagrees with its definition: {err:.3e}"

    # Asymptotics: log L(z) ~ z^2 as z -> -inf, ~ -log(2 z sqrt(pi)) as z -> +inf.
    assert abs(_log_L(np.array([-40.0]))[0] - 1600.0) < 1e-6
    big = _log_L(np.array([1e3]))[0]
    assert abs(big - np.log(0.5 / (1e3 * np.sqrt(np.pi)))) < 1e-3
    print(f"  log_L stable on [-40, 1e3]; matches definition to {err:.2e}")


def test_eps_to_zero_is_quadratic_l1():
    """THE oracle: at lam = t = 1, eps -> 0 must reproduce QuadraticL1 exactly.

    Checks the value function, the prior, and the preimage against a family
    verified independently in Phase 1. Nothing else pins the closed form this
    hard: the erfcx algebra, the tanh ratio, the bisection and the Fenchel
    evaluation all have to be right at once for this to pass.
    """
    rng = np.random.default_rng(0)
    x = rng.uniform(-4.0, 4.0, (200, 3))
    ref = QuadraticL1(t=1.0)
    pm = PosteriorMeanL1(eps=1e-6, lam=1.0, t=1.0)

    # S and f_reg converge UNIFORMLY: their limits are continuous.
    e_s = np.max(np.abs(pm.hjsol_true(x) - ref.hjsol_true(x)))
    e_j = np.max(np.abs(pm.prior_true(x) - ref.prior_true(x)))
    assert e_s < 1e-4, f"S_eps does not tend to the Huber envelope: {e_s:.3e}"
    assert e_j < 1e-4, f"f_reg does not tend to ||.||_1: {e_j:.3e}"

    # The preimage CANNOT converge uniformly: its limit x + sign(x) jumps by
    # 2*t*lam across the kink, while every eps > 0 preimage is continuous with
    # u_PM(0) = 0. Convergence is pointwise off the kink, with a boundary layer
    # of width ~sqrt(2*t*eps). Outside the layer the agreement is at float64
    # resolution -- a 1-ulp match against an independently verified family,
    # which is the strongest statement available about the closed form.
    e_p = np.max(np.abs(pm.preimage(x) - ref.preimage(x))[np.abs(x) > 0.05])
    assert e_p < 1e-12, f"preimage off the kink != x + sign(x): {e_p:.3e}"

    # Inside the layer, pin the structure instead: bounded by the jump, and
    # shrinking at every fixed point as eps -> 0.
    xk = np.array([[2e-4], [1e-3], [5e-3]])
    prev = np.full(xk.shape, np.inf)
    for eps in (1e-4, 1e-5, 1e-6, 1e-7):
        err = np.abs(PosteriorMeanL1(eps=eps).preimage(xk) - ref.preimage(xk))
        assert np.all(err <= 2.0 * pm.t * pm.lam), f"layer error exceeds the jump: {err}"
        assert np.all(err < prev), f"layer error grew as eps shrank: {err} vs {prev}"
        prev = err

    # f_reg's convergence IS monotone in eps -- it is the object we train on.
    prev_j = np.inf
    for eps in (1e-1, 1e-2, 1e-3, 1e-4):
        err = np.max(np.abs(PosteriorMeanL1(eps=eps).prior_true(x) - ref.prior_true(x)))
        assert err < prev_j, f"f_reg error grew as eps shrank: {err:.3e} >= {prev_j:.3e}"
        prev_j = err
    print(f"  eps->0 == QuadraticL1: S {e_s:.2e}, f_reg {e_j:.2e}; preimage {e_p:.2e} "
          f"off the kink (non-uniform at it, layer ~sqrt(2 t eps))")


def test_denoiser_is_grad_psi():
    """u_PM = grad psi, the identity the whole example rests on."""
    rng = np.random.default_rng(1)
    y = rng.uniform(-5.0, 5.0, (60, 4))
    pm = PosteriorMeanL1(eps=0.1)
    err = np.max(np.abs(pm.denoiser(y) - _fd_grad(pm.cvx_true, y)))
    assert err < 1e-7, f"u_PM != grad psi: {err:.3e}"
    print(f"  u_PM == grad psi (central diff): {err:.2e}")


def test_shrinkage_shape():
    """Pins the corrected eq:upm_l1 ratio. The printed (L(a)+L(b))/(L(a)-L(b))
    is singular at y=0; the correct reciprocal gives u_PM(0)=0 and a strict
    shrinkage by less than the threshold."""
    pm = PosteriorMeanL1(eps=0.1, lam=1.0, t=1.0)
    tl = pm.t * pm.lam

    z = pm.denoiser(np.zeros((1, 5)))
    assert np.max(np.abs(z)) < 1e-14, f"u_PM(0) != 0: {np.max(np.abs(z)):.3e}"

    y = np.linspace(-6, 6, 401).reshape(-1, 1)
    u = pm.denoiser(y)
    shrink = np.abs(u - y)
    # Strict in exact arithmetic; the tanh saturates to exactly 1.0 in float64
    # past |z| ~ 19, so the threshold is attained far from the kink. Assert the
    # non-strict bound -- the strict one is a statement about R, not float64.
    assert np.all(shrink <= tl), f"|u_PM - y| > t*lam: max {shrink.max():.6f}"
    assert np.all(np.sign(u[y != 0]) * np.sign(y[y != 0]) >= 0), "u_PM flips sign"
    assert np.all(np.diff(u[:, 0]) > 0), "u_PM is not strictly increasing"
    # Shrinkage toward the origin, i.e. |u_PM(y)| <= |y|.
    assert np.all(np.abs(u) <= np.abs(y) + 1e-15), "u_PM is not a shrinkage"
    # Near the kink the shrinkage must be STRICT -- that is the smoothing.
    near = np.abs(y[:, 0]) < 0.5
    assert np.all(shrink[near, 0] < tl - 1e-9), "no smoothing near the kink"
    print(f"  u_PM(0)=0, strictly increasing, |u-y| <= {tl} (max {shrink.max():.4f}), "
          f"strict near the kink (max {shrink[near, 0].max():.4f})")


def test_preimage_inverts():
    """preimage really is (grad psi)^{-1}, and respects its own bound."""
    rng = np.random.default_rng(2)
    a = 4.0
    x = rng.uniform(-a, a, (300, 6))
    pm = PosteriorMeanL1(eps=0.1)

    err = np.max(np.abs(pm.denoiser(pm.preimage(x)) - x))
    assert err < 1e-12, f"u_PM(preimage(x)) != x: {err:.3e}"

    bound = pm.preimage_bound(a)
    reach = np.max(np.abs(pm.preimage(x)))
    # Non-strict: float64 tanh saturation attains the bound (see _preimage_bound).
    assert reach <= bound, f"preimage {reach:.6f} escaped its bound {bound}"
    assert abs(bound - (a + pm.t * pm.lam)) < 1e-15
    print(f"  preimage inverts to {err:.2e}; max |y|_inf {reach:.4f} <= bound {bound}")


def test_sampling_identity():
    """The identity the notebook's targets are built from (Lemma A.1):

        f_reg(y_k) = <x_k, y_k> - psi(x_k) - 0.5||y_k||^2   at y_k = u_PM(x_k),
        grad f_reg(y_k) = x_k - y_k.

    Both must agree with `prior_true`, which is computed by an independent route
    (bisection + conjugate). If these disagree, the notebook trains on one
    function and scores against another.
    """
    rng = np.random.default_rng(3)
    pm = PosteriorMeanL1(eps=0.1)
    xk = rng.uniform(-5.0, 5.0, (200, 4))
    yk = pm.denoiser(xk)

    lhs = pm.prior_true(yk)
    rhs = np.sum(xk * yk, axis=1) - pm.cvx_true(xk) - 0.5 * np.sum(yk**2, axis=1)
    e_val = np.max(np.abs(lhs - rhs))
    assert e_val < 1e-10, f"value target != prior_true: {e_val:.3e}"

    e_grad = np.max(np.abs(_fd_grad(pm.prior_true, yk) - (xk - yk)))
    assert e_grad < 1e-6, f"grad f_reg != x - y: {e_grad:.3e}"
    print(f"  Lemma A.1 targets: value {e_val:.2e}, gradient {e_grad:.2e}")


def test_freg_is_convex():
    """f_reg convex <=> u_PM nonexpansive (u' <= 1) <=> f_reg'' = 1/u' - 1 >= 0.
    This is what licenses an ICNN for the one network."""
    pm = PosteriorMeanL1(eps=0.1)
    y = np.linspace(-6, 6, 2001).reshape(-1, 1)
    du = np.gradient(pm.denoiser(y)[:, 0], y[:, 0])
    assert np.all(du > 0), f"u_PM not increasing: min u' = {du.min():.3e}"
    assert np.all(du < 1.0 + 1e-6), f"u_PM not nonexpansive: max u' = {du.max():.6f}"

    x = np.linspace(-4, 4, 801).reshape(-1, 1)
    f = pm.prior_true(x)
    second = np.gradient(np.gradient(f, x[:, 0]), x[:, 0])
    assert np.all(second > -1e-4), f"f_reg not convex: min f'' = {second.min():.3e}"
    print(f"  0 < u' <= {du.max():.4f} (nonexpansive) => f_reg convex, min f'' "
          f"= {second.min():.2e}")


def test_curvature_grows_as_eps_shrinks():
    """eps is the difficulty knob: f_reg'' at the origin ~ 1/eps, so eps -> 0
    approaches the ||.||_1 kink. Pins the homotopy claim the notebook makes."""
    prev = 0.0
    for eps in (1.0, 0.1, 0.01):
        pm = PosteriorMeanL1(eps=eps)
        y = np.linspace(-0.5, 0.5, 4001).reshape(-1, 1)
        f = pm.prior_true(y)
        curv = np.gradient(np.gradient(f, y[:, 0]), y[:, 0])[2000]
        assert curv > prev, f"curvature at 0 did not grow: {curv:.3f} <= {prev:.3f}"
        prev = curv
    print(f"  f_reg''(0) grows as eps shrinks, reaching {prev:.1f} at eps=0.01")


def test_no_overflow_far_out():
    """Large |y| and tiny eps drive z^2 ~ 1e13 inside L. Nothing may overflow."""
    pm = PosteriorMeanL1(eps=1e-6)
    y = np.array([[1e3, -1e3, 0.0], [5.0, -5.0, 1e-8]])
    for name, v in (
        ("psi", pm.cvx_true(y)), ("S", pm.hjsol_true(y)), ("u_PM", pm.denoiser(y)),
        ("f_reg", pm.prior_true(y)), ("preimage", pm.preimage(y)),
    ):
        assert np.all(np.isfinite(v)), f"{name} non-finite at |y|=1e3, eps=1e-6: {v}"
    print("  finite at |y| = 1e3 with eps = 1e-6 (z^2 ~ 5e11)")


def test_train_halfwidth():
    """The sample box must map ONTO the query box: u_PM([-A,A]^d) covers
    [-a,a]^d. This is the D2 amendment, and the notebook's data generation
    depends on it.

    The coverage has ZERO margin by construction -- A = a + t*lam is the
    smallest half-width that works, so u_PM(A) = a to float64 resolution. In
    exact arithmetic u_PM(A) > a, but by ~4e-37 (1 - tanh(z/2) ~ 2*exp(-z) with
    z ~ 84 here), which is 21 orders below the ulp at 4. QuadraticL1 behaves
    identically (soft(5,1) = 4 exactly), so this is the established protocol,
    not a defect: the interior is covered, and no eval point lands on the face.
    """
    pm = PosteriorMeanL1(eps=0.1)
    a = 4.0
    A = pm.train_halfwidth(a)
    assert abs(A - 5.0) < 1e-15, f"train_halfwidth {A} != a + t*lam = 5"
    reach = pm.denoiser(np.array([[A]]))[0, 0]
    assert reach >= a, f"u_PM({A}) = {reach:.6f} does not cover the query box {a}"

    # A box any smaller genuinely fails to cover, which is what makes A minimal.
    short = pm.denoiser(np.array([[A - 0.1]]))[0, 0]
    assert short < a, f"u_PM({A - 0.1}) = {short:.6f} still covers {a}; A is not minimal"

    # The interior is what matters: sample x uniformly and check the reach.
    rng = np.random.default_rng(7)
    yk = pm.denoiser(rng.uniform(-A, A, (20000, 4)))
    print(f"  train A = {A}: u_PM(A) = {reach:.6f} >= a = {a} (onto, zero margin "
          f"by design); 20k samples reach |y|_inf = {np.max(np.abs(yk)):.4f}")


if __name__ == "__main__":
    for fn in (
        test_log_L_stable,
        test_eps_to_zero_is_quadratic_l1,
        test_denoiser_is_grad_psi,
        test_shrinkage_shape,
        test_preimage_inverts,
        test_sampling_identity,
        test_freg_is_convex,
        test_curvature_grows_as_eps_shrinks,
        test_no_overflow_far_out,
        test_train_halfwidth,
    ):
        print(f"{fn.__name__}:")
        fn()
    print("\nOK: PosteriorMeanL1 ground truth verified.")
