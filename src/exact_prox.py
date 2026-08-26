"""Exact-prox ablation helpers (experiments_plan.tex, Experiment P).

Pure, I/O-free, training-free building blocks. Given a ``Problem`` whose exact
proximal map is known in closed form, this module forms the exact conjugate
triples ``(y_k, g(y_k), x_k)`` and reduces a recovery to relative-L2 errors. It
deletes the first network of the reported pipeline (term E1 in the plan): the
triples are built from the TRUE psi, not a learned one.

Conventions (identical to src/recovery.py and the reported pipeline):

    psi(x)   = cvx_true(x),  the convex potential the FIRST network learns.
    grad psi = prox_{tJ},    the forward proximal map -> ``grad_psi`` below.
    g(y)     = psi^*(y),      the conjugate the SECOND network G fits.
    grad g   = (grad psi)^{-1} = Problem.preimage,   the INVERSE prox.

So the learned second network recovers ``g``; its GRADIENT recovers the inverse
prox ``grad g = preimage`` (verified in recovery.py: route2_preimage = grad G).
That is the map scored by :func:`rel_prox_error`.

``grad_psi`` is the only new mathematics here; it is the analytic inverse of the
existing ``Problem.preimage``, and ``tests/test_exact_prox.py`` pins the
round-trip ``preimage(grad_psi(x)) == x`` and the Fenchel identity to ~1e-12.
"""
import numpy as np

from src.targets import QuadraticL1, NegL1, ConcaveQuad, Minplus


def _t(problem):
    return float(getattr(problem, "t", 1.0))


# --------------------------------------------------------------------------
# Forward proximal map grad psi = prox_{tJ}, closed form per family.
# --------------------------------------------------------------------------
def grad_psi(problem, x):
    """Exact forward prox grad psi(x) = prox_{tJ}(x), family-dispatched.

    This is the map the first (psi) network learns in the reported pipeline; the
    ablation supplies it exactly and removes that network. It is the analytic
    inverse of ``Problem.preimage`` (which is grad g = (grad psi)^{-1}).
    """
    x = np.asarray(x, dtype=float)
    if isinstance(problem, QuadraticL1):
        t = _t(problem)
        # prox of t||.||_1 = soft-threshold; CONTRACTS by t.
        return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)
    if isinstance(problem, NegL1):
        # psi = q + ||.||_1 + n t/2 => grad psi = x + sign(x); EXPANDS, and omits
        # the open gap (-1,1) per coordinate (the interior-hole caveat).
        return x + np.sign(x)
    if isinstance(problem, ConcaveQuad):
        t = _t(problem)
        # psi = ||.||^2 at t=1 => grad psi = 2x; in general (3-t)/(2-t). EXPANDS.
        return x * (3.0 - t) / (2.0 - t)
    if isinstance(problem, Minplus):
        return _grad_psi_minplus(problem, x)
    raise NotImplementedError(
        f"forward prox grad_psi not implemented for {type(problem).__name__}"
    )


def _grad_psi_minplus(problem, x):
    """grad psi for psi = max_i psi_i, psi_i(x) = 0.5||x||^2 - val_i(x).

    The active branch is argmax_i psi_i = argmin_i val_i (the (1+sigma) branch
    values of Minplus._branches). On branch i, grad psi(x) = (sigma_i x + mu_i)/
    (1+sigma_i). The ridge {psi_1 = psi_2} is measure zero, so for continuous x
    either branch's gradient serves; the round-trip test tolerates it.
    """
    x = np.asarray(x, dtype=float)
    s1, s2 = problem.sigma1, problem.sigma2
    mu1, mu2 = problem.mu1, problem.mu2
    v1, v2 = problem._branches(x)  # 0.5||x-mu_i||^2/(1+sigma_i), shape (N,)
    grad1 = (s1 * x + mu1) / (1.0 + s1)
    grad2 = (s2 * x + mu2) / (1.0 + s2)
    return np.where((v1 <= v2)[:, None], grad1, grad2)


# --------------------------------------------------------------------------
# Exact conjugate triples and the exact conjugate value at arbitrary points.
# --------------------------------------------------------------------------
def build_triples(problem, x):
    """Exact triples ``(y, g, x)`` at psi-domain inputs ``x`` (shape (N, d)).

        y_k = grad psi(x_k) = prox_{tJ}(x_k)     (forward, exact)
        g_k = <x_k, y_k> - psi(x_k) = psi^*(y_k) (Fenchel value; exact since
                                                  y_k = grad psi(x_k) attains the sup)
        x_k in dg(y_k)                            (the recorded slope/subgradient)

    Returns ``(y, g, x)`` with ``y, x`` of shape ``(N, d)`` and ``g`` of shape
    ``(N,)``. The same trio ``src.recovery.conjugate_samples`` emits from a
    LEARNED psi, here from the TRUE psi.
    """
    x = np.asarray(x, dtype=float)
    y = grad_psi(problem, x)
    g = np.sum(x * y, axis=1) - problem.cvx_true(x)
    return y, g, x


def g_exact(problem, z):
    """g(z) = psi^*(z) at arbitrary z, via g(z) = <x*, z> - psi(x*), x* = grad g(z).

    ``x* = Problem.preimage(z)`` solves grad psi(x*) = z, which attains the sup
    defining psi^*. Correct for EVERY family, including NegL1, where g yields the
    convex J_BVS rather than the nonconvex prior J = ``prior_true``.
    """
    z = np.asarray(z, dtype=float)
    xstar = problem.preimage(z)
    return np.sum(xstar * z, axis=1) - problem.cvx_true(xstar)


def jbvs_exact(problem, z):
    """The recovered prior's ground truth: J_BVS(z) = (g(z) - 0.5||z||^2)/t.

    This is what the recovered prior ``G(z) - 0.5||z||^2`` targets, and it is NOT
    ``prior_true`` for the nonconvex NegL1 family (J_BVS is the convex envelope).
    Name and reference kept explicit per the recovery-error convention.
    """
    z = np.asarray(z, dtype=float)
    quad = 0.5 * np.sum(z * z, axis=1)
    return (g_exact(problem, z) - quad) / _t(problem)


# --------------------------------------------------------------------------
# Relative-L2 reducers (Experiment R form).
# --------------------------------------------------------------------------
def rel_l2(approx, truth):
    """||approx - truth||_2 / ||truth||_2, flattened (Frobenius for vector fields).

    The single accuracy reducer for Experiment C: a scalar prior field (N,) or a
    vector prox field (N, d) alike. Both recovered objects are scored with it
    against closed-form ground truth on the held-out test set.
    """
    approx = np.asarray(approx, dtype=float)
    truth = np.asarray(truth, dtype=float)
    denom = np.linalg.norm(truth)
    return float(np.linalg.norm(approx - truth) / denom) if denom > 0 else float("nan")
