"""Computable duality sandwich for the Hopf formula (research avenue E.2).

Answers referee R2.3 ("plot the error decay in K/N") and supplies the numerical
half of the exponent claim in the appendix (WP3: the rate is K^{-2/n}, not the
n/2 printed in the manuscript).

THE SANDWICH
------------
For H(p) = 0.5||p||^2 and prior J, the Hopf formula gives the viscosity solution
of S_t + H(grad S) = 0, S(.,0) = J, as

    S(y, t) = sup_p { <p, y> - J*(p) - (t/2)||p||^2 }.                     (H)

Fix any finite set of slopes P = {p_1, ..., p_K} contained in dom J*. Restricting
the sup in (H) to P gives the max-plus / PAM approximant

    Gamma_K(y) = max_{k <= K} { <p_k, y> - J*(p_k) - (t/2)||p_k||^2 },      (L)

a max of affine functions of y. Each term is one admissible competitor in a sup,
so Gamma_K <= S pointwise, for EVERY K and every choice of P -- no sampling
condition, no smoothness. Gamma_K is nondecreasing under P -> P union {p}, and
Gamma_K -> S as P becomes dense in dom J*. This is `gamma_K` below.

Conversely S(., t) is convex in y (a sup of affine functions of y). So for any
sample points y_1, ..., y_M with known values s_m = S(y_m, t), Jensen gives, for
every y in conv{y_m},

    S(y, t) <= U_M(y) := min { sum_m lam_m s_m : lam in simplex,
                                                 sum_m lam_m y_m = y }.     (U)

U_M is the lower convex envelope of the point cloud {(y_m, s_m)} -- the convex
piecewise-linear interpolant of S on conv{y_m}. This is `convex_upper_bound`.

Together, on conv{y_m},

    Gamma_K(y)  <=  S(y, t)  <=  U_M(y).                                    (S)

NEITHER SIDE USES GROUND TRUTH. (L) needs only J* at the sampled slopes; (U)
needs only the values of S at the sampled points, which the two-network pipeline
already produces. So U_M - Gamma_K is a computable a-posteriori certificate that
brackets the true error, and it is exactly the object R2.3 asks to see decay.

TANGENCY
--------
If the slopes are chosen as the exact maximizers p_k = argmax of (H) at the
sample points y_k -- equivalently p_k = grad_y S(y_k, t), the exact supporting
hyperplane -- then the k-th term of (L) attains S(y_k) and

    Gamma_K(y_k) = S(y_k) = U_M(y_k),

so the sandwich closes at every sample and the certificate vanishes there. That
is the "exact supporting hyperplanes, grad g(y_k) = x_k" condition of E.2.

THE RATE, AND WHY IT HAS TWO REGIMES
------------------------------------
Take J = ||x||_1, t = 1. Then J* = iota_{||p||_inf <= 1}, the sup in (H) is
separable, S is the Moreau envelope of ||.||_1 (the separable Huber), and the
maximizer is p*(y) = clip(y, -1, 1). Because the integrand is
f_y(p) = 0.5||y||^2 - 0.5||p - y||^2, the pointwise error of (L) is EXACTLY

    S(y) - Gamma_K(y) = 0.5 * min_k ||p_k - y||^2 - 0.5 * ||p*(y) - y||^2,   (E)

a nearest-neighbour quantity (`huber_gamma_error_exact`).

The exponent is NOT determined by K and d alone. It is determined by where the
maximizer sits, because (E) is a difference of squared distances:

  * p*(y) INTERIOR to dom J* (i.e. ||y||_inf <= 1). The constraint is inactive,
    grad f_y(p*) = 0, so (E) is quadratic in the distance from p* to the nearest
    slope: error ~ 0.5 * dist(p*, P)^2 ~ delta^2 ~ K^{-2/d}.

  * p*(y) on the BOUNDARY of dom J* (i.e. some |y_i| > 1). The constraint is
    active, grad f_y(p*) = y - p* != 0, so (E) is FIRST order in that distance:
    error ~ <y - p*, p* - p_nn> ~ delta ~ K^{-1/d}.

On the query box [-4,4]^d the maximizer is interior only when ||y||_inf <= 1, a
fraction 4^{-d} of the box. So the boundary regime governs the RMS error, and a
slope set that fails to resolve d(dom J*) loses HALF the exponent. Measured
(tail fit, K <= 4096, `bin/gamma_decay.py`):

    sampler                 d=2      d=3      d=4     predicted
    grid (resolves dC)    -1.09    -0.77    -0.63    -2/d  (interior regime)
    uniform p in dom J*   -0.45      --     -0.32    -1/d  (boundary regime)
    uniform p, interior q -1.01    -0.69    -0.55    -2/d  (control)

This is why the sampler matters:

  * `slopes_grid_l1` contains the endpoints +-1 on every axis, hence every face
    of the cube, hence p*(y) exactly whenever |y_i| > 1. It attains the interior
    rate K^{-2/d}.
  * `slopes_random_l1` (i.i.d. uniform on the cube) misses d(dom J*) with
    probability one and pays K^{-1/d}. Adding just the 2^d corners does NOT
    repair it -- faces of intermediate codimension remain unresolved.

SCOPE (checked against the manuscript, 2026-07-10). All of the above concerns
the HOPF-side scheme, which approximates S itself. It is VACUOUS for the paper's
own PAM, which approximates g_t = t*J_BVS + q, whose conjugate is finite on all
of R^d, so the maximizer is never on a boundary. The relevance is prospective
(a general Hamiltonian makes the Hopf route the natural one), plus the fact that
Gaubert et al.'s theorem assumes C^2, which fails for J = ||.||_1.

`slopes_tangent_l1` OBEYS NO SINGLE POWER LAW -- do not fit a rate to it
(retracted 2026-07-10; an earlier version of this file claimed it restores
K^{-2/d}, and blamed its d>=8 behavior on the 3^d faces of dom J*). Both claims
are wrong. Those slopes lie on the (d-1)-dimensional boundary and split into
~a^{-d} K interior points and the rest on dC. The two populations serve DISJOINT
query regions and decay at different rates, so the aggregate is a mixture at
EVERY d. Splitting the RMS by query location at K <= 4096:
    d=2: queries inside C -0.674,  queries outside C -1.766
    d=4: queries inside C -0.444,  queries outside C -0.695
The aggregate is dominated by the interior queries although they are 5% (d=2)
and 0.4% (d=4) of the box, because the boundary contribution collapses; the
aggregate fit moves between -0.68 and -0.90 at d=2 across query draws.

CAVEAT. These are asymptotic statements in the fill distance. Preasymptotic
traps, all live at the K we can afford:
  * On a tensor grid delta = 2/(m-1) while K = m^d, so small m reads too steep.
    Fit the largest-K tail (`fit_loglog_slope(..., tail=4)`).
  * The grid's fitted/predicted ratio drifts up with d (1.09, 1.16, 1.26 at
    d=2,3,4) because the largest affordable m falls with d.
  * A random slope set has fill distance ~(log K / K)^{1/d}, so a fit against
    log K alone understates the exponent.
"""
import numpy as np

# ----------------------------------------------------------------------------
# The lower bound: restrict the Hopf sup to a finite slope set.
# ----------------------------------------------------------------------------


def gamma_K(y, P, jstar=None, t=1.0):
    """Max-plus approximant (L): Gamma_K(y) = max_k {<p_k,y> - J*(p_k) - t/2 |p_k|^2}.

    A LOWER bound on the Hopf solution S(y, t) for any slope set P contained in
    dom J*, since every term is an admissible competitor in the sup (H).

    Parameters
    ----------
    y : (n, d) query points.
    P : (K, d) slopes, each in dom J*.
    jstar : (K,) values J*(p_k), or None for the indicator case (J* = 0 on its
        domain), which covers J = ||.||_1 where dom J* is the unit inf-ball.
    t : time.

    Returns
    -------
    (n,) array of Gamma_K(y).
    """
    y, P = np.atleast_2d(y), np.atleast_2d(P)
    jstar = np.zeros(len(P)) if jstar is None else np.asarray(jstar)
    aff = y @ P.T - jstar - 0.5 * t * np.sum(P * P, axis=1)   # (n, K)
    return np.max(aff, axis=1)


def gamma_K_argmax(y, P, jstar=None, t=1.0):
    """Index of the active slope at each y. Useful for plotting the tessellation."""
    y, P = np.atleast_2d(y), np.atleast_2d(P)
    jstar = np.zeros(len(P)) if jstar is None else np.asarray(jstar)
    aff = y @ P.T - jstar - 0.5 * t * np.sum(P * P, axis=1)
    return np.argmax(aff, axis=1)


# ----------------------------------------------------------------------------
# Slope sets for J = ||x||_1 (dom J* = [-1,1]^d, J* = 0 there).
# ----------------------------------------------------------------------------


def slopes_grid_l1(d, m):
    """Uniform tensor grid of m points per axis on [-1,1]^d.  K = m^d.

    The clean geometry for the rate: fill distance is delta = 2/(m-1), so
    K^{-1/d} ~ delta and the predicted error ~ K^{-2/d} is testable directly.
    Only usable while m^d stays small.
    """
    if m < 2:
        raise ValueError("m >= 2")
    ax = np.linspace(-1.0, 1.0, m)
    return np.stack(np.meshgrid(*([ax] * d), indexing="ij"), axis=-1).reshape(-1, d)


def slopes_random_l1(d, K, rng):
    """K uniform random slopes in [-1,1]^d. Decouples K from d, at the price of a
    (log K / K)^{1/d} fill distance -- the rate picks up the usual log factor."""
    return rng.uniform(-1.0, 1.0, (K, d))


def slopes_tangent_l1(y_samples):
    """Exact supporting slopes p_k = grad S(y_k) = clip(y_k, -1, 1) for J=||.||_1.

    Gives the tangency Gamma_K(y_k) = S(y_k) exactly (see module docstring), so
    the lower bound interpolates S at the samples rather than merely minorizing.
    """
    return np.clip(np.asarray(y_samples), -1.0, 1.0)


# ----------------------------------------------------------------------------
# Exact reference for J = ||x||_1, t = 1: the separable Huber.
# ----------------------------------------------------------------------------


def huber_S(y, t=1.0):
    """S(y,t) = min_x { 0.5||x-y||^2 + t||x||_1 }, the separable Huber.

    Matches src.targets.QuadraticL1.hjsol_true (verified to 4e-15); duplicated
    here so the bounds module has no dependency on the training stack.
    """
    y = np.atleast_2d(y)
    a = np.abs(y)
    return np.sum(np.where(a <= t, 0.5 * y**2, t * a - 0.5 * t**2), axis=1)


def huber_gamma_error_exact(y, P, t=1.0):
    """Closed form for S - Gamma_K when J = ||.||_1 and P is in the cube.

    Since f_y(p) = 0.5||y||^2 - 0.5||p-y||^2 for the L1 Hopf integrand,
        S(y) - Gamma_K(y) = 0.5 * min_k ||p_k - y||^2 - 0.5 * ||p*(y) - y||^2,
    with p*(y) = clip(y, -t, t). A nearest-neighbour distance, so the rate is a
    covering-radius statement about P -- which is why it is K^{-2/d}.

    Used only to cross-check the generic path; carries no free parameters.
    """
    y, P = np.atleast_2d(y), np.atleast_2d(P)
    d2 = np.sum((y[:, None, :] - P[None, :, :]) ** 2, axis=2)   # (n, K)
    near = np.min(d2, axis=1)
    p_star = np.clip(y, -t, t)
    return 0.5 * near - 0.5 * np.sum((p_star - y) ** 2, axis=1)


# ----------------------------------------------------------------------------
# The upper bound: lower convex envelope of sampled values (U).
# ----------------------------------------------------------------------------


def convex_upper_bound(y_query, y_samples, s_samples, method="lp"):
    """U_M(y) = min { sum lam_m s_m : lam in simplex, sum lam_m y_m = y }.

    An UPPER bound on any convex S agreeing with s_samples at y_samples, valid on
    conv{y_samples} and +inf outside it (reported as np.inf).

    One small LP per query: K variables, d+1 equality constraints. Exact, and it
    needs no triangulation, so it does not inherit Delaunay's dimensional blowup
    -- but it is O(n_query) LPs, so keep n_query modest.

    Returns (n,) array; np.inf where y lies outside conv{y_samples}.
    """
    from scipy.optimize import linprog

    y_query = np.atleast_2d(y_query)
    Y = np.atleast_2d(y_samples)
    s = np.asarray(s_samples, dtype=float).ravel()
    M, d = Y.shape
    if method != "lp":
        raise ValueError("only method='lp' is implemented")

    # equality: [Y^T; 1^T] lam = [y; 1]
    A_eq = np.vstack([Y.T, np.ones((1, M))])
    out = np.empty(len(y_query))
    for i, y in enumerate(y_query):
        b_eq = np.concatenate([y, [1.0]])
        r = linprog(c=s, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        out[i] = r.fun if r.status == 0 else np.inf
    return out


def in_convex_hull(y_query, y_samples):
    """Mask of queries that admit a convex representation by the samples."""
    u = convex_upper_bound(y_query, y_samples, np.zeros(len(y_samples)))
    return np.isfinite(u)


# ----------------------------------------------------------------------------
# The decay study.
# ----------------------------------------------------------------------------


def decay_curve(d, Ks, y_query, sampler, t=1.0, rng=None):
    """Error of Gamma_K against the exact Huber S, as K grows.

    `sampler(d, K, rng) -> (K', d)` returns the slope set; K' may differ from K
    for the grid sampler (K' = m^d), so the realized K is returned.

    Returns a list of dicts: realized K, sup error, RMS error, mean error.
    """
    rng = rng or np.random.default_rng(0)
    S = huber_S(y_query, t=t)
    rows = []
    for K in Ks:
        P = sampler(d, K, rng)
        G = gamma_K(y_query, P, jstar=None, t=t)
        err = S - G                     # >= 0 by construction
        assert err.min() > -1e-9, f"Gamma_K exceeded S by {-err.min():.2e}"
        rows.append({
            "d": d, "K": len(P),
            "sup_err": float(err.max()),
            "rms_err": float(np.sqrt(np.mean(err**2))),
            "mean_err": float(err.mean()),
        })
    return rows


def fit_loglog_slope(K, err, tail=None):
    """Least-squares slope of log(err) against log(K). Predicted: -2/d.

    ``tail`` restricts the fit to the largest-K points. Use it. The rate is
    asymptotic in the fill distance delta, and for a tensor grid delta = 2/(m-1)
    while K = m^d: at m = 2 -> 4 the spacing falls by 3x but K only doubles, so
    a fit that includes the smallest grids reads systematically too steep. The
    bias is pure preasymptotics, not a failure of the K^{-2/d} law -- fitting
    the tail removes it.
    """
    K, err = np.asarray(K, float), np.asarray(err, float)
    keep = (err > 0) & np.isfinite(err)
    K, err = K[keep], err[keep]
    if tail is not None and len(K) > tail:
        order = np.argsort(K)[-tail:]
        K, err = K[order], err[order]
    if len(K) < 2:
        return np.nan
    return float(np.polyfit(np.log(K), np.log(err), 1)[0])


def fit_loglog_slope_effective(K, err, d, tail=None):
    """Slope against the EFFECTIVE sample count K/log(K), for random slopes.

    K i.i.d. uniform points in [-1,1]^d have fill distance ~ (log K / K)^{1/d},
    not K^{-1/d}. The squared-distance error therefore decays like
    (K/log K)^{-2/d}, so regressing on log K alone understates the exponent by
    the log factor. This regresses on log(K/log K), against the same -2/d.
    """
    K, err = np.asarray(K, float), np.asarray(err, float)
    keep = (err > 0) & np.isfinite(err) & (K > 1)
    K, err = K[keep], err[keep]
    if tail is not None and len(K) > tail:
        order = np.argsort(K)[-tail:]
        K, err = K[order], err[order]
    if len(K) < 2:
        return np.nan
    return float(np.polyfit(np.log(K / np.log(K)), np.log(err), 1)[0])


def grid_error_closed_form(y, m, t=1.0):
    """Exact S - Gamma_K on the tensor grid of m points per axis of [-1,1]^d.

    Separability collapses the nearest-neighbour term coordinatewise, and for
    |y_i| > t the maximizer p*_i = +-t is itself a grid node, contributing zero:

        S(y) - Gamma_K(y) = 0.5 * sum_{i: |y_i| <= t} dist(y_i, axis grid)^2.

    So only the coordinates that land INSIDE the cube generate error -- on
    average d/4 of them when y is uniform on [-4,4]^d. Independent derivation of
    the same number `gamma_K` computes; used to check it.
    """
    y = np.atleast_2d(y)
    ax = np.linspace(-t, t, m)
    inside = np.abs(y) <= t
    dist = np.min(np.abs(y[:, :, None] - ax[None, None, :]), axis=2)
    return 0.5 * np.sum(np.where(inside, dist**2, 0.0), axis=1)
