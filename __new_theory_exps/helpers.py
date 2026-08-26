"""Local helpers for the convergence-theory verification experiments.

Companion to CLAUDE/theoretical_work/convergence_work.tex. Everything here is
NEW code local to __new_theory_exps; existing modules (src.maxplus_bounds) are
imported read-only and never modified.

Objects, in the notation of convergence_work.tex (t fixed):

    g(y)   = t*J_BVS(y) + 0.5||y||^2        (alpha-strongly convex, alpha = 1
                                             for both families below, whose
                                             priors are convex so J_BVS = J)
    psi(x) = 0.5||x||^2 - t*S(x,t),  grad psi = prox_{tJ}
    samples: y_k = prox_{tJ}(x_k),  slope x_k in dg(y_k),  g(y_k) exact.

    L_K = max of supporting hyperplanes (lower bound on g everywhere)
    U_K = lower convex envelope of the sampled points (upper bound on conv{y_k})

Families (both with every quantity in closed form, so no network enters):

  * "l1":    J = ||.||_1. g is nonsmooth (kinks on the axes): Lemma B regime,
             predicted certificate rate h ~ K^{-1/n}.
  * "huber": J(y) = sum_i H_beta(y_i), H_beta the Huber function. g is C^{1,1}
             with L = 1 + t/beta: Lemma A regime, predicted rate h^2 ~ K^{-2/n}.
"""
import sys
from pathlib import Path

import numpy as np

# Read-only imports from the existing stack.
_NUMERICS_ROOT = Path(__file__).resolve().parents[1]
if str(_NUMERICS_ROOT) not in sys.path:
    sys.path.insert(0, str(_NUMERICS_ROOT))

# The envelope/sandwich math now lives in src.certificate (one source of truth,
# shared with bin/certificate_fig.py); this file keeps only the closed-form
# family generators that feed it exact samples. Re-imported into the helpers
# namespace so helpers.<name> keeps working for the notebook.
from src.certificate import (  # noqa: E402, F401
    certificate,
    convex_upper_bound_checked,
    fill_distance,
    lower_envelope,
    tail_slope,
)

# ----------------------------------------------------------------------------
# Families: samples and exact g.
# ----------------------------------------------------------------------------


def soft_threshold(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def huber_prior(y, beta):
    """H_beta applied coordinatewise and summed: J(y) = sum_i H_beta(y_i)."""
    a = np.abs(y)
    return np.sum(np.where(a <= beta, y**2 / (2 * beta), a - beta / 2), axis=-1)


def huber_prox(x, t, beta):
    """prox_{tJ}(x) for the separable Huber prior, coordinatewise closed form.

    Stationarity y + (t/beta) y = x in the quadratic zone (|y| <= beta, i.e.
    |x| <= beta + t), else y = x - t sign(x).
    """
    inner = np.abs(x) <= beta + t
    return np.where(inner, x * beta / (beta + t), x - t * np.sign(x))


def make_samples(x, family, t=1.0, beta=1.0):
    """From x-samples, return (y_k, g(y_k), slopes x_k) for the family.

    y_k = prox_{tJ}(x_k) = grad psi(x_k), and x_k in dg(y_k) by Fenchel--Young;
    g(y_k) is evaluated exactly from the closed-form prior (J = J_BVS since the
    priors here are convex).
    """
    x = np.atleast_2d(x)
    if family == "l1":
        y = soft_threshold(x, t)
        g = t * np.sum(np.abs(y), axis=1) + 0.5 * np.sum(y * y, axis=1)
    elif family == "huber":
        y = huber_prox(x, t, beta)
        g = t * huber_prior(y, beta) + 0.5 * np.sum(y * y, axis=1)
    else:
        raise ValueError(family)
    return y, g, x


def g_exact(y, family, t=1.0, beta=1.0):
    y = np.atleast_2d(y)
    if family == "l1":
        return t * np.sum(np.abs(y), axis=1) + 0.5 * np.sum(y * y, axis=1)
    if family == "huber":
        return t * huber_prior(y, beta) + 0.5 * np.sum(y * y, axis=1)
    raise ValueError(family)


def L_smoothness(family, t=1.0, beta=1.0):
    """Upper curvature bound L of g on R^n (inf for the kink family)."""
    return np.inf if family == "l1" else 1.0 + t / beta


# ----------------------------------------------------------------------------
# Experiment 1: certificate decay in K. (Envelopes: src.certificate, imported
# above; certificate(), lower_envelope(), convex_upper_bound_checked() live
# there now.)
# ----------------------------------------------------------------------------


def make_samples_generic_l1(y, t=1.0):
    """Generic y-samples for the l1 family: exact values and one subgradient.

    Unlike the prox pushforward, generic (uniform) y-samples MISS the kink set
    {y_i = 0} with probability one, which is the regime of Lemma B (first-order
    rate). Subgradient: x = y + t*sign(y), with sign(0) := +1 (measure zero).
    """
    y = np.atleast_2d(y)
    g = t * np.sum(np.abs(y), axis=1) + 0.5 * np.sum(y * y, axis=1)
    sgn = np.where(y >= 0, 1.0, -1.0)
    return y, g, y + t * sgn


def exp1_decay(family, n, Ks, n_query=200, box=4.0, qbox=2.5, t=1.0, beta=1.0,
               seed=1, sampling="pushforward", replicates=1):
    """sup and RMS of U_K - L_K over in-hull queries, as K grows.

    sampling="pushforward": x-samples uniform on [-box, box]^n (the training
    protocol) pushed through prox_{tJ}. For J = ||.||_1 this places sample
    ATOMS on the kink set of g (every coordinate with |x_i| <= t maps to 0),
    so the certificate contracts at the SMOOTH rate: the pipeline's own
    sampling resolves the kinks.

    sampling="generic": y-samples uniform on [-qbox-0.5, qbox+0.5]^n (l1 only),
    which miss the kinks almost surely -- the Lemma B regime, rate -1/n.

    Queries uniform on [-qbox, qbox]^n filtered to conv{y_k}.
    """
    rng = np.random.default_rng(seed)
    yq_all = rng.uniform(-qbox, qbox, (n_query, n))
    if sampling == "generic":
        yq_kink = rng.uniform(-qbox, qbox, (n_query // 2, n))
        yq_kink[np.arange(len(yq_kink)), rng.integers(0, n, len(yq_kink))] = 0.0
        yq_all = np.vstack([yq_all, yq_kink])
    rows = []
    for K in Ks:
        reps = []
        for _ in range(replicates):
            if sampling == "pushforward":
                x = rng.uniform(-box, box, (K, n))
                y, g, s = make_samples(x, family, t=t, beta=beta)
            elif sampling == "generic" and family == "l1":
                y0 = rng.uniform(-qbox - 0.5, qbox + 0.5, (K, n))
                y, g, s = make_samples_generic_l1(y0, t=t)
            else:
                raise ValueError((family, sampling))
            L, U, ok = certificate(yq_all, y, g, s)
            gap = U[ok] - L[ok]
            gtrue = g_exact(yq_all[ok], family, t=t, beta=beta)
            viol_L = float(np.max(L[ok] - gtrue))
            viol_U = float(np.max(gtrue - U[ok]))
            # L is evaluated analytically (exact); U comes from a checked LP
            # (see convex_upper_bound_checked) -- allow a scale-aware
            # tolerance and RECORD the violation, do not fail on solver noise.
            scale = 1.0 + float(np.max(np.abs(gtrue)))
            assert viol_L <= 1e-8 * scale, f"L exceeded g by {viol_L:.2e}"
            assert viol_U <= 1e-5 * scale, f"U fell below g by {viol_U:.2e}"
            reps.append({
                "n_inhull": int(ok.sum()),
                "sup_gap": float(gap.max()),
                "rms_gap": float(np.sqrt(np.mean(gap**2))),
                "sup_err_L": float(np.max(gtrue - L[ok])),
                "sup_err_U": float(np.max(U[ok] - gtrue)),
                "lp_viol": max(viol_U, 0.0),
            })
        # Geometric mean over replicates: rates are fitted in log space, and
        # nearest-neighbour sup statistics fluctuate multiplicatively.
        gmean = lambda key: float(np.exp(np.mean(np.log([r[key] for r in reps]))))
        rows.append({
            "family": family, "n": n, "K": K, "sampling": sampling,
            "replicates": replicates,
            "n_inhull": reps[0]["n_inhull"],
            "sup_gap": gmean("sup_gap"),
            "rms_gap": gmean("rms_gap"),
            "sup_err_L": gmean("sup_err_L"),
            "sup_err_U": gmean("sup_err_U"),
            "lp_viol": max(r["lp_viol"] for r in reps),
        })
    return rows


def exp1_tangency(family, n, K=400, box=4.0, t=1.0, beta=1.0, seed=1):
    """max_k |L_K(y_k) - g(y_k)| and |U_K(y_k) - g(y_k)|: both should be ~ 0."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-box, box, (K, n))
    y, g, s = make_samples(x, family, t=t, beta=beta)
    L, U, ok = certificate(y, y, g, s)
    return float(np.max(np.abs(L - g))), float(np.max(np.abs(U[ok] - g[ok])))


# ----------------------------------------------------------------------------
# Experiment 2: fill distances and their pushforward.
# ----------------------------------------------------------------------------


def exp2_fill(family, n, Ns, box=4.0, t=1.0, beta=1.0, n_probe=4000, seed=2):
    """h_X (x-side) and h_Y (y-side, over the image region) against N.

    grad psi = prox_{tJ} is nonexpansive (alpha = 1 for these convex priors),
    so the theory predicts h_Y <= h_X; both should scale as (log N / N)^{1/n}.
    """
    rng = np.random.default_rng(seed)
    probes_x = rng.uniform(-box, box, (n_probe, n))
    probes_y, _, _ = make_samples(probes_x, family, t=t, beta=beta)
    rows = []
    for N in Ns:
        x = rng.uniform(-box, box, (N, n))
        y, _, _ = make_samples(x, family, t=t, beta=beta)
        rows.append({
            "family": family, "n": n, "N": N,
            "h_X": fill_distance(probes_x, x),
            "h_Y": fill_distance(probes_y, y),
            "pred": float((np.log(N) / N) ** (1.0 / n)),
        })
    return rows


# ----------------------------------------------------------------------------
# Experiment 3: argmin stability (Lemma "Stability of partial minimization").
# ----------------------------------------------------------------------------


def _refine_argmin(f, lo, hi, n_grid=20001, rounds=3):
    """Nested-grid argmin of a scalar function on [lo, hi]; ~1e-12 resolution."""
    for _ in range(rounds):
        ys = np.linspace(lo, hi, n_grid)
        j = int(np.argmin(f(ys)))
        step = (hi - lo) / (n_grid - 1)
        lo, hi = ys[j] - 2 * step, ys[j] + 2 * step
    return 0.5 * (lo + hi)


def _prox_of_perturbed(xq, eps, t, beta, n, span=6.0):
    """1-D reference: minimizers of phi1 = g - x*y and phi2 = phi1 + eps*cos(y),
    by nested-grid refinement. Only used for n = 1 checks."""
    def phi1(ys):
        return g_exact(ys[:, None], "huber", t=t, beta=beta) - xq * ys

    def phi2(ys):
        return phi1(ys) + eps * np.cos(ys)

    return (_refine_argmin(phi1, -span, span),
            _refine_argmin(phi2, -span, span))


def exp3_smooth_perturbation(eps_list, t=1.0, beta=1.0, n_x=41, seed=3):
    """Smooth perturbation eps*cos(y) of g (Huber family, n = 1).

    sup-norm gap is eps; both objectives are (1 - eps)-strongly convex for
    eps < 1. The lemma bounds the minimizer shift by 2*sqrt(eps/(1-eps)); a
    smooth perturbation should sit well inside the bound with shift O(eps).
    """
    xs = np.linspace(-4.0, 4.0, n_x)
    rows = []
    for eps in eps_list:
        shifts = [abs(np.subtract(*_prox_of_perturbed(x, eps, t, beta, 1)))
                  for x in xs]
        rows.append({
            "eps": eps,
            "max_shift": float(np.max(shifts)),
            "bound": float(2 * np.sqrt(eps / (1 - eps))),
        })
    return rows


def exp3_dip_perturbation(eps_list, sigma=1.0):
    """Near-worst-case perturbation: phi1 = sigma/2 y^2, phi2 = phi1 - eps*bump
    centered at d = 0.99*sqrt(2 eps / sigma), bump height eps, narrow support.

    The dip captures the minimizer, so the shift is ~ sqrt(2 eps / sigma) =
    (1/sqrt(2)) * bound: the lemma's sqrt(eps) scaling is attained up to the
    constant, and the ratio shift/bound should be ~ 0.7 across eps.
    """
    rows = []
    for eps in eps_list:
        d = 0.99 * np.sqrt(2 * eps / sigma)
        w = 0.05 * d
        ys = np.linspace(-2 * d - 5 * w, 2 * d + 5 * w, 400001)
        phi1 = 0.5 * sigma * ys**2
        bump = np.exp(-0.5 * ((ys - d) / w) ** 2)
        phi2 = phi1 - eps * bump
        v1, v2 = ys[np.argmin(phi1)], ys[np.argmin(phi2)]
        rows.append({
            "eps": eps,
            "shift": float(abs(v2 - v1)),
            "bound": float(2 * np.sqrt(eps / sigma)),
        })
    return rows


# Reporting: tail_slope lives in src.certificate (imported above).
