"""Ground-truth targets for each experiment family.

Each Problem exposes three functions of an ``(N, dim)`` array of query points:

    hjsol_true(y)  -- the HJ value function S(y, t=1),
    prior_true(y)  -- the initial data / prior J(y),
    cvx_true(y)    -- the convex potential the first LPN regresses,
                      psi(y) = 0.5*||y||^2 - S(y, 1).

Every target here is the corrected form; the bugs found in Phase 1 (a spurious
+n t/2 in the quadratic-L1 Moreau envelope, and a dimension-independent
constant in the NegL1 value function) are fixed. Correct references verified in
Phase 1 (exp_4_2_1_1D, the minplus and concave-quadratic families) are carried
over unchanged.
"""
import numpy as np
from scipy.special import erfc, erfcx


def euclid_norm(x):
    return np.linalg.norm(x, ord=2, axis=1)


def euclid_norm_sq(x):
    if x.ndim == 1:
        return np.sum(x * x)
    return np.sum(x * x, axis=1)


def _prox_l1(y, tau):
    """Element-wise soft-thresholding: prox of tau*||.||_1."""
    return np.sign(y) * np.maximum(np.abs(y) - tau, 0.0)


class Problem:
    """Base class: cvx_true is fixed once from hjsol_true.

    Subclasses also expose the preimage map and the training box it forces:

        preimage(x)         -- the y solving grad psi(y) = x,
        train_halfwidth(a)  -- smallest A with grad psi([-A,A]^d) covering the
                               query box [-a,a]^d.

    Why the training box is not the query box (D2 amendment, 2026-07-09).
    LPN Iterative recovery evaluates psi_theta at y = (grad psi)^{-1}(x); One-shot recovery trains G at
    inputs y_k = grad psi(x_k), so G's support is range(grad psi) restricted to
    the training box. BOTH recoveries therefore need the training box chosen so that
    grad psi maps it ONTO the query box. Sampling psi on the query box itself
    silently starves both networks near the boundary: measured in 2D on
    QuadraticL1, iterative-recovery RMSE 4.745 and one-shot-recovery RMSE 0.337, versus 0.034 and
    0.059 once the box is enlarged. Since preimage is componentwise monotone
    (grad psi is monotone, psi convex) the max over the box is attained at a
    corner, so each bound below is analytic.
    """

    def cvx_true(self, y):
        return 0.5 * euclid_norm_sq(y) - self.hjsol_true(y)

    def preimage(self, x):
        raise NotImplementedError

    def _preimage_bound(self, a):
        """max ||preimage(x)||_inf over the query box [-a,a]^d."""
        raise NotImplementedError

    def preimage_bound(self, a):
        """Exact max ||preimage(x)||_inf over [-a,a]^d, for the TRUE psi.

        The divergence tripwire compares against this, not against the training
        half-width. The two coincide for the expanding families (QuadraticL1,
        Minplus) but not for the contracting ones: ConcaveQuad's preimage is
        x/2, bounded by a/2 = 2, while its training box is a = 4. Testing
        against the box would leave the tripwire 3x too loose.

        The bound holds for the exact psi. A learned psi_theta may overshoot it
        slightly without diverging, so callers allow a margin.
        """
        return float(self._preimage_bound(a))

    def train_halfwidth(self, a):
        # Never shrink below the query box: psi's own fit is reported there.
        return max(float(a), self.preimage_bound(a))


class QuadraticL1(Problem):
    """J(x) = ||x||_1, H(p) = 0.5||p||^2.

    S(y, 1) is the Moreau envelope of ||.||_1, i.e. the separable Huber
    function sum_i huber(y_i), huber(z) = z^2/2 if |z|<=1 else |z|-1/2.

    FIX (Phase 1, finding 2): the notebook target returned
    sum_i |prox(y_i)| + n t/2 with n hardcoded to 1 -- it dropped the Moreau
    curvature term 0.5*(prox-y)^2 and added a dimension-independent constant.
    """

    def __init__(self, t=1.0):
        self.t = t

    def hjsol_true(self, y, t=None):
        t = self.t if t is None else t
        prox = _prox_l1(y, t)
        return np.sum(0.5 * (prox - y) ** 2 + t * np.abs(prox), axis=1)

    def prior_true(self, y):
        return np.sum(np.abs(y), axis=1)

    def preimage(self, x):
        """grad psi = soft(., t), so y* = x + t*sign(x). EXPANDS by t."""
        x = np.asarray(x)
        return x + self.t * np.sign(x)

    def _preimage_bound(self, a):
        return a + self.t


class NegL1(Problem):
    """J(x) = -||x||_1, H(p) = 0.5||p||^2.

    Per-coordinate min of -|u| + (y-u)^2/(2t) gives S(y, 1) = -||y||_1 - n t/2.

    FIX (Phase 1): the notebook target returned -(t)/2 - ||y||_1, missing the
    dimension factor n; the error was exactly (n-1) t/2.
    """

    def __init__(self, t=1.0):
        self.t = t

    def hjsol_true(self, y, t=None):
        t = self.t if t is None else t
        n = y.shape[1]
        return -np.sum(np.abs(y), axis=1) - n * t / 2.0

    def prior_true(self, y):
        return -np.sum(np.abs(y), axis=1)

    def preimage(self, x):
        """psi = q + ||.||_1 + nt/2, so grad psi = y + sign(y) and the inverse
        is soft-thresholding. CONTRACTS: this family needs no enlarged box.

        Caveat: grad psi omits the open gap (-1,1) per coordinate (psi is
        nonsmooth at 0), so the one-shot-recovery samples y_k have an interior HOLE
        around the origin rather than a boundary deficit. Logged as a
        follow-up; it is not fixed by the box margin.
        """
        return _prox_l1(np.asarray(x), 1.0)

    def _preimage_bound(self, a):
        return a  # |soft(x,1)| <= a


class ConcaveQuad(Problem):
    """J(x) = -0.25 ||x||^2, H(p) = 0.5||p||^2.

    S(y, t) = -||y||^2 / (2(2 - t)); at t = 1, S = -0.5||y||^2, so psi = ||y||^2.
    Verified correct in Phase 1 (concave-quadratic family), carried over.
    """

    def __init__(self, t=1.0):
        self.t = t

    def hjsol_true(self, y):
        return -euclid_norm_sq(y) / (2.0 * (2.0 - self.t))

    def prior_true(self, x):
        return -0.25 * euclid_norm_sq(x)

    def preimage(self, x):
        """grad psi = y*(3-t)/(2-t) (= 2y at t=1), so y* = x*(2-t)/(3-t).
        CONTRACTS; this family needs no enlarged box."""
        return np.asarray(x) * (2.0 - self.t) / (3.0 - self.t)

    def _preimage_bound(self, a):
        return a * (2.0 - self.t) / (3.0 - self.t)


def _log_L(z):
    """log L(z) for L(z) = 0.5 * exp(z^2) * erfc(z), computed without overflow.

    L is the profile function of the Cole-Hopf solution for an L1 prior. It is
    never formed directly: exp(z^2) overflows past z ~ 27 in float64 exactly
    where erfc(z) underflows, so the product is a 0*inf in disguise even though
    L itself is O(1/z) there. Two stable branches:

        z >= 0:  L(z) = 0.5 * erfcx(z),  erfcx the SCALED complementary error
                 function, which absorbs the exp(z^2) analytically.
        z <  0:  erfc(z) = 2 - erfc(-z) gives L(z) = exp(z^2) * (1 - 0.5*erfc(-z)),
                 hence log L(z) = z^2 + log1p(-0.5*erfc(-z)). The log1p argument
                 lies in (-0.5, 0], so no cancellation.

    L(z) > 0 for every real z, so the log is always defined.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty(z.shape, dtype=float)
    pos = z >= 0.0
    out[pos] = np.log(0.5 * erfcx(z[pos]))
    zneg = z[~pos]
    out[~pos] = zneg * zneg + np.log1p(-0.5 * erfc(-zneg))
    return out


class PosteriorMeanL1(Problem):
    """Posterior-mean (Cole-Hopf) smoothing of J(x) = lam * ||x||_1, H(p) = 0.5||p||^2.

    The imaging example of work2.tex Instantiation A: the given object is the
    Gaussian posterior-mean DENOISER u_PM, and the prior we recover is its
    implicit smooth regularizer f_reg = K_eps^* - 0.5||.||^2, which is the prox
    map's regularizer by Darbon-Langlois (JMIV) Prop. 3.2. Unlike the four
    families above, f_reg has no elementary closed form in its own argument --
    it is defined through a conjugate -- but it is computable to machine
    precision via the tangency point, which is what `prior_true` does.

    ARGUMENT CONVENTION (this class follows the module's, which is the OPPOSITE
    of work2.tex's). Here `y` is the denoiser INPUT = psi's argument, and `x` is
    the denoised point = the conjugate variable at which the prior is reported.
    work2.tex swaps the two letters.

    The closed forms, with a_i = (y_i + t*lam)/sqrt(2*t*eps) and
    b_i = (-y_i + t*lam)/sqrt(2*t*eps):

        S_eps(y,t) = ||y||^2/(2t) - eps * sum_i log(L(a_i) + L(b_i)),
        psi(y) = K_eps(y,t) = 0.5||y||^2 - t*S_eps(y,t)
                            = t*eps * sum_i log(L(a_i) + L(b_i)),
        u_PM(y) = grad psi(y),  (u_PM)_i = y_i + t*lam*tanh((log L(a_i) - log L(b_i))/2).

    The quadratic in S cancels against the one in psi EXACTLY, which is why
    `cvx_true` is overridden rather than inherited: the base class would form
    0.5||y||^2 - S and lose precision subtracting two large near-equal numbers.
    The override is also the only form correct at t != 1 -- the base class's
    identity assumes psi = 0.5||y||^2 - S, i.e. it silently drops the factor t.

    The shrinkage formula corrects work2.tex eq:upm_l1, which prints the ratio
    (L(a)+L(b))/(L(a)-L(b)). That is singular at y_i = 0, where a = b. The ratio
    is the RECIPROCAL of the printed one: differentiating log(L(a)+L(b)) and
    using L'(z) = 2z*L(z) - 1/sqrt(pi) (the -1/sqrt(pi) cancels in the
    difference) gives (L(a)-L(b))/(L(a)+L(b)), written above as a tanh of the
    log-difference -- the algebraically identical form, and the only one that is
    stable when one of L(a), L(b) overflows. Then u_PM(0) = 0 by symmetry and
    |u_PM(y) - y| < t*lam strictly, so u_PM is a smooth shrinkage that recovers
    soft-thresholding as eps -> 0. At lam = t = 1 the eps -> 0 limit is EXACTLY
    the QuadraticL1 family above; `tests/test_posterior_mean_l1.py` pins that.
    """

    def __init__(self, eps=0.1, lam=1.0, t=1.0):
        self.eps = float(eps)
        self.lam = float(lam)
        self.t = float(t)

    def _log_pair(self, y):
        """(log L(a_i), log L(b_i)), elementwise, same shape as y."""
        scale = np.sqrt(2.0 * self.t * self.eps)
        tl = self.t * self.lam
        y = np.asarray(y, dtype=float)
        return _log_L((y + tl) / scale), _log_L((tl - y) / scale)

    def _log_sum(self, y):
        """log(L(a_i) + L(b_i)), elementwise. logaddexp never forms either L."""
        la, lb = self._log_pair(y)
        return np.logaddexp(la, lb)

    def hjsol_true(self, y):
        y = np.asarray(y, dtype=float)
        return euclid_norm_sq(y) / (2.0 * self.t) - self.eps * np.sum(
            self._log_sum(y), axis=1
        )

    def cvx_true(self, y):
        """psi(y) = t*eps*sum_i log(L(a_i)+L(b_i)); see the class docstring for
        why this is not left to the base class."""
        return self.t * self.eps * np.sum(self._log_sum(y), axis=1)

    def denoiser(self, y):
        """u_PM(y) = grad psi(y): the smooth shrinkage. Elementwise in y."""
        la, lb = self._log_pair(y)
        return np.asarray(y, dtype=float) + self.t * self.lam * np.tanh(
            0.5 * (la - lb)
        )

    def preimage(self, x, iters=100):
        """The y solving grad psi(y) = u_PM(y) = x. EXPANDS by less than t*lam.

        u_PM has no closed-form inverse, but it is separable and each coordinate
        map is a strictly increasing bijection of R with |u_PM(y) - y| < t*lam
        (the tanh factor is bounded by 1 in modulus). So the root is bracketed by
        (x - t*lam, x + t*lam) with no search, and 100 halvings of that width-2*t*lam
        bracket land far below float64 resolution. Bisection rather than Newton:
        it cannot leave the bracket, and the cost is irrelevant next to training.

        The sign test is written without a bracket precondition on purpose. For
        |x| >> t*lam the root sits within an ulp of an endpoint (u_PM(x + t*lam)
        rounds to x), so requiring a strict sign change at the endpoints would
        fail on exactly the points where the answer is the endpoint.

        NOTE the eps -> 0 limit is NOT uniform here, unlike for S and f_reg. The
        limit x + t*lam*sign(x) jumps by 2*t*lam across the kink, while every
        eps > 0 preimage is continuous (u_PM(0) = 0 forces preimage(0) = 0), so
        no continuous family can converge to it uniformly. Convergence is
        pointwise off the kink -- measured at 1 ulp for |x| > 0.05 at eps = 1e-6
        -- with a boundary layer of width ~sqrt(2*t*eps) around it. This does not
        affect prior_true, whose limit ||.||_1 is continuous.
        """
        x = np.asarray(x, dtype=float)
        tl = self.t * self.lam
        lo, hi = x - tl, x + tl
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            below = self.denoiser(mid) < x
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return 0.5 * (lo + hi)

    def prior_true(self, x):
        """f_reg(x) = psi^*(x) - 0.5||x||^2, the denoiser's implicit regularizer.

        psi^*(x) = <y, x> - psi(y) at y = preimage(x), exact because
        grad psi(y) = x attains the sup defining the conjugate. This is the same
        identity `src.recovery.conjugate_samples` uses to build the one-shot-recovery
        targets, evaluated here on the TRUE psi instead of a learned one.
        """
        x = np.asarray(x, dtype=float)
        y = self.preimage(x)
        return (
            np.sum(y * x, axis=1) - self.cvx_true(y) - 0.5 * euclid_norm_sq(x)
        )

    def _preimage_bound(self, a):
        """In exact arithmetic |u_PM(y) - y| < t*lam is strict, so this is a
        supremum no query attains. In float64 it IS attained: the tanh saturates
        to +-1.0 once |log L(a) - log L(b)| > ~38, which happens inside the query
        box for small eps. Callers must not test the bound strictly."""
        return a + self.t * self.lam


def _project_simplex(V):
    """Row-wise Euclidean projection onto the probability simplex."""
    n, m = V.shape
    U = -np.sort(-V, axis=1)
    css = np.cumsum(U, axis=1) - 1.0
    idx = np.arange(1, m + 1)
    cond = U - css / idx > 0
    rho = m - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = css[np.arange(n), rho] / (rho + 1.0)
    return np.maximum(V - theta[:, None], 0.0)


class MaxPlus(Problem):
    """Piecewise-linear (max-plus / Hopf) prior J(x) = max_i {<p_i,x> - gamma_i}.

    NOT REPORTED IN main.pdf. Ported from legacy/old_notebooks/exp_4_1_3 (gamma = 0) and
    legacy/old_notebooks/exp_4_1_4 (gamma_i = 0.5||p_i||^2).

    FIX (2026-07-09): both notebooks had the WRONG HJ solution -- the sign of
    the t*H(p) term was flipped, giving S(y,1) = max_i{<p_i,y> + 0.5||p_i||^2}
    (case 3) and max_i <p_i,y> (case 4). Both violate the Moreau bound S <= J at
    EVERY test point (mean error ~1.0 vs the grid-computed envelope), so neither
    can be a viscosity solution. The Hopf formula that reproduces our verified
    QuadraticL1 target carries a MINUS:

        S(y,t) = sup_p { <p,y> - J*(p) - (t/2)||p||^2 }.

    Since J* (Px) = min{ lambda^T gamma : lambda in simplex, P^T lambda = p },
    the sup collapses to a concave QP over the simplex,

        S(y,1) = max_{lambda in Delta} { <P^T lambda, y> - lambda^T gamma
                                          - 0.5||P^T lambda||^2 },

    solved below by projected gradient ascent (m is small). Restricting lambda
    to the VERTICES gives max_i{<p_i,y> - gamma_i - 0.5||p_i||^2}, which is the
    max-plus approximant Gamma_K of section 3 -- a lower bound, not the exact
    solution. That is what makes these two experiments interesting for Phase 5.
    """

    def __init__(self, P, gamma=None, t=1.0, iters=500):
        self.P = np.asarray(P, dtype=float)
        self.gamma = (np.zeros(len(self.P)) if gamma is None
                      else np.asarray(gamma, dtype=float))
        self.t = t
        self.iters = iters

    def _solve(self, y):
        """argmax over the simplex; returns (S, lambda)."""
        y = np.atleast_2d(np.asarray(y, dtype=float))
        Q = self.P @ self.P.T                      # (m, m), PSD
        b = y @ self.P.T - self.gamma              # (n, m)
        L = max(np.linalg.eigvalsh(Q)[-1] * self.t, 1e-12)
        lam = np.full((len(y), len(self.P)), 1.0 / len(self.P))
        for _ in range(self.iters):
            grad = b - self.t * (lam @ Q)          # d/dlam of the objective
            lam = _project_simplex(lam + grad / L)
        p = lam @ self.P
        S = np.sum(p * y, axis=1) - lam @ self.gamma - 0.5 * self.t * np.sum(p * p, axis=1)
        return S, lam

    def hjsol_true(self, y):
        return self._solve(y)[0]

    def prior_true(self, x):
        return np.max(np.asarray(x) @ self.P.T - self.gamma, axis=1)

    def maxplus_approx(self, y):
        """Vertex (max-plus) approximant Gamma_K; a LOWER bound on hjsol_true."""
        y = np.asarray(y)
        return np.max(y @ self.P.T - self.gamma
                      - 0.5 * self.t * np.sum(self.P * self.P, axis=1), axis=1)

    def preimage(self, x):
        """grad psi(y) = y - P^T lambda*(y), so the preimage solves
        y = x + P^T lambda*(y). The map is nonexpansive (a projection composed
        with a translation), so a Krasnoselskii-Mann average converges."""
        x = np.asarray(x, dtype=float)
        y = x.copy()
        for _ in range(200):
            _, lam = self._solve(y)
            y = 0.5 * y + 0.5 * (x + lam @ self.P)
        return y

    def _preimage_bound(self, a):
        # P^T lambda lies in conv{p_i}, so ||y||_inf <= ||x||_inf + max_i ||p_i||_inf
        return a + float(np.max(np.abs(self.P)))


class Minplus(Problem):
    """Two-component min-plus (mixture) prior; H(p) = 0.5||p||^2.

    Verified correct in Phase 1 (minplus family), carried over verbatim.
    """

    def __init__(self, mu1, mu2, sigma1, sigma2):
        self.mu1, self.mu2 = np.asarray(mu1), np.asarray(mu2)
        self.sigma1, self.sigma2 = sigma1, sigma2

    def hjsol_true(self, y):
        return np.minimum(*self._branches(y))

    def prior_true(self, y):
        val1 = 0.5 * euclid_norm(y - self.mu1) ** 2 / self.sigma1
        val2 = 0.5 * euclid_norm(y - self.mu2) ** 2 / self.sigma2
        return np.minimum(val1, val2)

    def _branches(self, y):
        val1 = 0.5 * euclid_norm(y - self.mu1) ** 2 / (1 + self.sigma1)
        val2 = 0.5 * euclid_norm(y - self.mu2) ** 2 / (1 + self.sigma2)
        return val1, val2

    def preimage(self, x):
        """Solve grad psi(y) = x for psi = max_i (q - val_i), a max of two
        convex quadratics. On branch i, grad psi(y) = (sigma_i y + mu_i)/(1+sigma_i),
        so the branch candidate is

            y_i = ((1 + sigma_i) x - mu_i) / sigma_i,

        which EXPANDS by (1+sigma_i)/sigma_i when sigma_i < 1 (by 2 at sigma_i = 1).

        A branch candidate is only the answer if that branch is active there. On
        the RIDGE {psi_1 = psi_2} the potential is nonsmooth and neither
        candidate is consistent; there d(psi)(y) is the segment
        conv{g_1(y), g_2(y)} and we must solve x = (1-c) y + c*m(lam) with
        m(lam) = lam*mu1 + (1-lam)*mu2, lam in [0,1], subject to y on the ridge.
        Ignoring the ridge leaves a residual of order 1e-1 on a positive-measure
        set of x, so it is not negligible.

        For sigma_1 = sigma_2 (all configs) the quadratic terms of psi_1 - psi_2
        cancel, the ridge is the hyperplane <y, mu2 - mu1> = b bisecting the
        modes, and lam has a closed form.
        """
        x = np.asarray(x, dtype=float)
        s1, s2 = self.sigma1, self.sigma2
        mu1, mu2 = self.mu1, self.mu2
        cands = [((1 + s) * x - mu) / s for mu, s in ((mu1, s1), (mu2, s2))]

        # branch i is active at y iff val_i(y) <= val_j(y)
        v1, v2 = self._branches(cands[0]), self._branches(cands[1])
        ok1, ok2 = v1[0] <= v1[1], v2[1] <= v2[0]
        y = np.where(ok1[:, None], cands[0], cands[1])

        ridge = ~(ok1 | ok2)
        if not ridge.any():
            return y
        if not np.isclose(s1, s2):
            raise NotImplementedError(
                "ridge preimage needs sigma1 == sigma2 (unequal sigmas give a "
                "spherical, not affine, ridge)"
            )
        d = mu2 - mu1
        dd = float(d @ d)
        if dd == 0.0:
            return y  # coincident modes: psi is smooth, branches agree
        s = s1
        c = 1.0 / (1.0 + s)          # weight on mu in grad psi
        one_minus_c = s / (1.0 + s)
        b = 0.5 * (float(mu2 @ mu2) - float(mu1 @ mu1))
        A = float(mu1 @ d)
        lam = 1.0 - (x[ridge] @ d - c * A - b * one_minus_c) / (c * dd)
        lam = np.clip(lam, 0.0, 1.0)
        m = lam[:, None] * mu1 + (1.0 - lam)[:, None] * mu2
        y[ridge] = (x[ridge] - c * m) / one_minus_c
        return y

    def _preimage_bound(self, a):
        return max(
            ((1 + s) * a + float(np.max(np.abs(mu)))) / s
            for mu, s in ((self.mu1, self.sigma1), (self.mu2, self.sigma2))
        )
