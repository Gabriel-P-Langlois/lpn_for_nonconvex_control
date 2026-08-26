"""Duality-sandwich certificate on sample triples (y_k, g(y_k), x_k).

Companion to CLAUDE/theoretical_work/convergence_work.tex. Given the conjugate
samples the pipeline already emits -- y_k = grad psi(x_k), g(y_k) = <x_k,y_k> -
psi(x_k) (a sample of psi* = g), slope x_k in dg(y_k) -- this module builds the
two envelopes of convergence_work.tex Sec. 2 and their computable gap:

    L_K(y) = max_k { g(y_k) + <x_k, y - y_k> }              (lower bound on g)
    U_K(y) = min { sum_k lam_k g(y_k) : lam in simplex,     (upper bound on
                   sum_k lam_k y_k = y }                      conv{y_k})

    L_K <= g <= U_K,  and  gap = U_K - L_K >= 0  is the certificate.

The functions take raw arrays and are agnostic to the source of the triples:
the closed-form oracle in __new_theory_exps/helpers.py feeds them exact samples
(where g is known and the fitted rates are checkable), and bin/certificate_fig.py
feeds them the triples of a TRAINED psi network. One source of truth for the
envelope math, so the paper figure inherits the oracle's unit tests.

NOTE: src/maxplus_bounds.py has an older, unchecked LP for U_K that returns
infeasible "optimal" points on degenerate instances; use THIS module's
convex_upper_bound_checked instead. Only fit_loglog_slope{,_effective} are
reused from there (pure log-log fits, no LP).
"""
import numpy as np

from .maxplus_bounds import fit_loglog_slope, fit_loglog_slope_effective


# ---------------------------------------------------------------------------
# The two envelopes.
# ---------------------------------------------------------------------------


def lower_envelope(y_query, y_samples, g_samples, slopes):
    """L_K(y) = max_k { g(y_k) + <x_k, y - y_k> }, evaluated at each query.

    Exact and cheap (no solver): a max over the K supporting hyperplanes whose
    slopes x_k are recorded subgradients of g at the samples y_k.
    """
    y_query = np.atleast_2d(y_query)
    aff = g_samples + np.einsum("kd,qd->qk", slopes, y_query) - np.sum(
        slopes * y_samples, axis=1
    )
    return np.max(aff, axis=1)


def convex_upper_bound_checked(y_query, y_samples, s_samples, tol=1e-6):
    """U_K by LP, with a feasibility post-check the maxplus_bounds version lacks.

    scipy's default 'highs' LP occasionally returns status 0 with a badly
    infeasible solution on degenerate instances (near-duplicate columns;
    observed equality residual 3.6e-2 at K = 2048, n = 1). We verify the
    equality residual of the returned point and fall through a method list; a
    query where no method passes gets np.nan (solver failure, distinct from
    np.inf = outside the hull).
    """
    from scipy.optimize import linprog

    y_query = np.atleast_2d(y_query)
    Y = np.atleast_2d(y_samples)
    s = np.asarray(s_samples, dtype=float).ravel()
    A_eq = np.vstack([Y.T, np.ones((1, len(Y)))])
    out = np.empty(len(y_query))
    for i, y in enumerate(y_query):
        b_eq = np.concatenate([y, [1.0]])
        val, infeasible = np.nan, 0
        for method in ("highs-ds", "highs-ipm", "highs"):
            r = linprog(c=s, A_eq=A_eq, b_eq=b_eq, bounds=(0, None),
                        method=method)
            if r.status == 2:
                infeasible += 1
                continue
            if r.status == 0:
                resid = float(np.max(np.abs(A_eq @ r.x - b_eq)))
                if resid <= tol * (1.0 + float(np.max(np.abs(b_eq)))):
                    val = r.fun
                    break
        out[i] = np.inf if (np.isnan(val) and infeasible > 0) else val
    return out


def certificate(y_query, y_samples, g_samples, slopes):
    """Return (L_K, U_K, in-hull mask) at the queries.

    U_K = +inf off conv{y_k}; nan marks an LP solver failure (excluded from the
    mask, to be counted by the caller). The certificate is U_K - L_K on the
    masked queries.
    """
    L = lower_envelope(y_query, y_samples, g_samples, slopes)
    U = convex_upper_bound_checked(y_query, y_samples, g_samples)
    return L, U, np.isfinite(U)


def a_posteriori_bound(g_hat, L_K, U_K):
    """Prop. 5.1: |g_hat(y) - g(y)| <= max(|g_hat - L_K|, |g_hat - U_K|).

    A computable, ground-truth-free error bar for any estimate g_hat (e.g. the
    trained second network) at queries inside the hull, since g in [L_K, U_K].
    """
    return np.maximum(np.abs(g_hat - L_K), np.abs(g_hat - U_K))


# ---------------------------------------------------------------------------
# Fill distance and rate fitting.
# ---------------------------------------------------------------------------


def fill_distance(probes, points):
    """sup over probes of the nearest-point distance (Monte-Carlo estimate)."""
    d2 = np.sum((probes[:, None, :] - points[None, :, :]) ** 2, axis=2)
    return float(np.sqrt(np.min(d2, axis=1).max()))


def tail_slope(rows, xkey, ykey, tail=4, effective_n=None):
    """Tail slope of log(y) vs log(x), optionally with the (log K) correction.

    effective_n given: fit against x/log(x) (the covering-radius correction for
    random samples, x ~ (log N / N)^{1/n}); otherwise a plain log-log slope.
    """
    xs = [r[xkey] for r in rows]
    ys = [r[ykey] for r in rows]
    if effective_n is not None:
        return fit_loglog_slope_effective(xs, ys, effective_n, tail=tail)
    return fit_loglog_slope(xs, ys, tail=tail)
