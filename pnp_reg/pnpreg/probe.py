"""Matrix-free Jacobian diagnostics: asymmetry ratio and extreme eigenvalues.

Experiment 2 (task document, Sections 5.2 and 6.2) asks, of a denoiser D at a
test input z, whether J = grad D(z) is (1) symmetric, (2) positive
semidefinite, and (3) bounded above by the identity -- the three nested
conditions under which D is the proximal operator of some regularizer, of a
possibly nonconvex one, and of a convex one. Nothing here knows what D is:
every estimator sees only Jacobian-vector and vector-Jacobian products through
the `LinOp` duck type, so the SAME code path runs on a 2x2 quadrature
Jacobian, a 64-dimensional ICNN prox, and a 2.6-million-dimensional CNN.
That single code path is what the calibration rows certify.

Estimators:

  * `hutchinson_asymmetry` -- rho = ||J - J^T||_F / (2 ||J||_F) in [0, 1],
    from E||Jv||^2 = ||J||_F^2 and E||Jv - J^T v||^2 = ||J - J^T||_F^2 for
    v ~ N(0, I). Reports Monte Carlo standard errors; the SE of rho itself is
    a paired bootstrap, because the delta method degenerates as rho -> 0.
  * the noise floor (kill criterion: "asymmetry at the level of the probe
    noise" must be reported as such) has two components, both produced by
    `hutchinson_asymmetry` itself. (a) `identity_max`: the largest relative
    violation of <v_i, J v_j> = <J^T v_i, v_j> over probe pairs -- an
    identity that holds for EVERY matrix, so its size reads the pure
    jvp-vs-vjp arithmetic inconsistency that also contaminates the b_j; it
    is the resolvable-asymmetry scale in rho's own units (an error delta in
    Jv - J^T v of relative size identity_max shifts rho by about
    identity_max / 2). (b) the SURROGATE ROW (probe_targets.target_floor): a
    map of the same architecture class whose Jacobian is symmetric by weight
    construction, g(z) = z - C^T elu(C z + b) with C the released first conv
    layer; its measured rho is an end-to-end zero-test of the estimator.
    A two-path floor (recomputing J v by a different autodiff composition)
    was designed first and measured DEGENERATE: forward-mode, reverse-over-
    reverse, and reverse-over-forward all reduce to the same conv /
    conv_transpose kernels and agree BITWISE, on CPU and MPS, in float32 and
    float64 (2026-07-30) -- an autodiff composition cannot supply an
    arithmetically independent second path for a convolutional operator.
  * `lanczos_symmetric` / `lanczos_multistart` -- extreme eigenvalues of the
    symmetric part S = (J + J^T)/2 via Lanczos with full reorthogonalization
    (classical Gram-Schmidt applied twice), the tridiagonal solved in float64,
    and Ritz residual bounds res = beta_k |y_k| reported with every estimate.

All reductions accumulate in float64 regardless of the operator's dtype.
"""
import numpy as np
import torch
from scipy.linalg import eigh_tridiagonal


class LinOp:
    """Jacobian access at one test point, matrix-free.

    Required: `n` (flat dimension), `dtype`, `device`, `jvp(v) -> J v`,
    `vjp(u) -> J^T u`, both mapping flat 1-D tensors to flat 1-D tensors.

    Subclasses override `jvp`/`vjp`; `sym` is derived.
    """

    n: int
    dtype: torch.dtype
    device: str

    def jvp(self, v):
        raise NotImplementedError

    def vjp(self, u):
        raise NotImplementedError

    def sym(self, v):
        """S v with S = (J + J^T)/2."""
        return 0.5 * (self.jvp(v) + self.vjp(v))


class DenseOp(LinOp):
    """A dense matrix wrapped as a LinOp (calibration rows and tests)."""

    def __init__(self, A):
        A = torch.as_tensor(A)
        self.A = A
        self.n = A.shape[0]
        self.dtype = A.dtype
        self.device = str(A.device)

    def jvp(self, v):
        return self.A @ v

    def vjp(self, u):
        return self.A.T @ u


def gaussian_probes(n, m, seed):
    """(m, n) standard Gaussian probes, float64, generated on CPU.

    CPU generation with an explicit torch.Generator makes the probes bitwise
    identical across devices, so an MPS run and its CPU cross-check see the
    same vectors. Callers cast/move as needed.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(m, n, generator=g, dtype=torch.float64)


def _bootstrap_rho_se(a, b, n_boot, seed):
    """SE of rho = sqrt(mean b) / (2 sqrt(mean a)) by paired resampling.

    Pairs (a_j, b_j) share the probe v_j and are resampled together. Used
    instead of the delta method, whose variance formula divides by rho and
    degenerates exactly where we operate (rho near 0).
    """
    rng = np.random.default_rng(seed)
    m = len(a)
    idx = rng.integers(0, m, size=(n_boot, m))
    am = a[idx].mean(axis=1)
    bm = b[idx].mean(axis=1)
    r = np.sqrt(np.maximum(bm, 0.0)) / (2.0 * np.sqrt(np.maximum(am, 1e-300)))
    return float(r.std(ddof=1))


def hutchinson_asymmetry(op, m_probes=16, seed=0, n_boot=2000):
    """Estimate rho = ||J - J^T||_F / (2 ||J||_F) with Monte Carlo errors.

    Per probe v_j ~ N(0, I): a_j = ||J v_j||^2 (unbiased for ||J||_F^2) and
    b_j = ||J v_j - J^T v_j||^2 (unbiased for ||J - J^T||_F^2 = 4 ||K||_F^2),
    so rho_hat = sqrt(mean b) / (2 sqrt(mean a)) targets exactly the ratio of
    the task document (divisor 2||J||_F, range [0, 1]).

    Returns a dict with rho, rho_se (paired bootstrap), F2, F2_se, K2, K2_se,
    the per-probe arrays a and b, and identity_max: the largest relative
    violation of <v_i, J v_j> = <J^T v_i, v_j> over probe pairs -- an identity
    that holds for EVERY matrix, so its size reads pure jvp-vs-vjp arithmetic
    noise, at no extra products (see the module docstring on the floor).
    """
    transpose = op.vjp
    probes = gaussian_probes(op.n, m_probes, seed)
    a = np.empty(m_probes)
    b = np.empty(m_probes)
    # vectors are kept only for the identity diagnostic, and only for the
    # first few probes: at n = 2.6e6 a full float64 set is ~660 MB, which
    # matters on a 16 GB machine (the run also holds a ~1.5 GB vjp graph)
    n_keep = min(m_probes, 4)
    Jv_all = []
    JTv_all = []
    for j in range(m_probes):
        v = probes[j].to(dtype=op.dtype)
        if op.device != "cpu":
            v = v.to(op.device)
        # move to CPU before the float64 cast: MPS has no float64
        Jv = op.jvp(v).detach().cpu().double()
        JTv = transpose(v).detach().cpu().double()
        d = Jv - JTv
        a[j] = float((Jv ** 2).sum())
        b[j] = float((d ** 2).sum())
        if j < n_keep:
            Jv_all.append(Jv)
            JTv_all.append(JTv)
        del Jv, JTv, d

    F2 = float(a.mean())
    K2 = float(b.mean())
    rho = float(np.sqrt(max(K2, 0.0)) / (2.0 * np.sqrt(F2)))
    m = m_probes
    out = {
        "rho": rho,
        "rho_se": _bootstrap_rho_se(a, b, n_boot, seed + 1),
        "F2": F2,
        "F2_se": float(a.std(ddof=1) / np.sqrt(m)) if m > 1 else float("nan"),
        "K2": K2,
        "K2_se": float(b.std(ddof=1) / np.sqrt(m)) if m > 1 else float("nan"),
        "a": a.tolist(),
        "b": b.tolist(),
        "m_probes": m,
    }
    # <v_i, J v_j> = <J^T v_i, v_j> for any J: cross-pair arithmetic check,
    # vectorized over the kept probes (lhs and rhs are both V J V^T).
    P = probes[:n_keep].double()
    JV = torch.stack(Jv_all)                       # (n_keep, n)
    JTV = torch.stack(JTv_all)                     # (n_keep, n)
    lhs = P @ JV.T
    rhs = JTV @ P.T
    den = torch.clamp(P.norm(dim=1)[:, None] * JV.norm(dim=1)[None, :], min=1e-300)
    rel = ((lhs - rhs).abs() / den).fill_diagonal_(0.0)
    out["identity_max"] = float(rel.max())
    return out


def lanczos_symmetric(op, k=30, seed=0, basis_device="cpu", return_ritz=False):
    """Extreme eigenvalues of S = (J + J^T)/2 by k Lanczos steps.

    Full reorthogonalization: classical Gram-Schmidt against the whole stored
    basis, applied twice ("twice is enough", Parlett), which keeps the basis
    orthogonal to the storage dtype's precision -- the calibration rows and
    `orth_err` verify it. The basis lives on `basis_device` (default CPU) in
    the OPERATOR'S dtype (float64 rows keep LAPACK-grade gates; float32 rows
    halve the ~600 MB a float64 basis would cost at n = 2.6e6, and their
    accuracy is bounded by the operator's own float32 arithmetic anyway);
    the recurrence scalars and the k x k tridiagonal are always float64.
    Ritz residual bounds: |theta - eig(S)| <= beta_k |y[k-1]| for the Ritz
    pair (theta, y), reported for both extremes.

    `return_ritz=True` additionally returns the two extreme Ritz VECTORS,
    assembled from the stored basis (v = sum_j Y[j, extreme] q_j, ~10 MB
    each in float32 at n = 2.6e6) under keys `ritz_vec_lmin`,
    `ritz_vec_lmax` -- used by Experiment 3.1 to display which deformation
    patterns the implicit regularizer penalizes and favors.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    v0 = torch.randn(op.n, generator=g, dtype=torch.float64)
    bdt = op.dtype if op.dtype in (torch.float32, torch.float64) else torch.float64

    def to_op(x):
        x = x.to(dtype=op.dtype)
        return x.to(op.device) if op.device != "cpu" else x

    def to_basis(x):
        return x.detach().to(basis_device).to(bdt)

    def dot(x, y):
        return float(x.double() @ y.double())

    Q = []
    q = to_basis(v0)
    q = q / float(q.double().norm())
    alphas, betas = [], []
    beta_last = 0.0
    n_iter = 0
    for i in range(k):
        Q.append(q)
        w = to_basis(op.sym(to_op(q))).double()
        if i > 0:
            w = w - betas[-1] * Q[i - 1].double()
        alpha = dot(q, w)
        w = w - alpha * q.double()
        # full reorthogonalization, classical Gram-Schmidt twice
        for _ in range(2):
            for qj in Q:
                w = w - dot(qj, w) * qj.double()
        alphas.append(alpha)
        n_iter = i + 1
        beta = float(w.norm())
        if i < k - 1:
            scale = max(abs(a) for a in alphas)
            if beta < 1e-10 * max(scale, 1.0):
                beta_last = beta
                break  # invariant subspace: Ritz values exact
            betas.append(beta)
            q = (w / beta).to(bdt)
        else:
            beta_last = beta

    al = np.array(alphas)
    be = np.array(betas[: len(al) - 1])
    if len(al) == 1:
        theta = al.copy()
        Y = np.ones((1, 1))
    else:
        theta, Y = eigh_tridiagonal(al, be)
    res = beta_last * np.abs(Y[-1, :])
    # Gram matrix by pairwise float64 dots: no stacked float64 copy of the
    # basis (~600 MB at n = 2.6e6, k = 30)
    kq = len(Q)
    orth = np.zeros((kq, kq))
    for i in range(kq):
        for j in range(i + 1):
            orth[i, j] = orth[j, i] = dot(Q[i], Q[j])
    orth -= np.eye(kq)
    orth = torch.from_numpy(orth)
    ritz_vecs = {}
    if return_ritz:
        for key, col in (("ritz_vec_lmin", 0), ("ritz_vec_lmax", -1)):
            v = torch.zeros(op.n, dtype=torch.float64)
            for j in range(kq):
                v += float(Y[j, col]) * Q[j].double()
            ritz_vecs[key] = (v / v.norm()).to(bdt)
    del Q
    return {
        "ritz": theta.tolist(),
        "lmin": float(theta[0]),
        "lmax": float(theta[-1]),
        "res_lmin": float(res[0]),
        "res_lmax": float(res[-1]),
        "orth_err": float(orth.abs().max()),
        "n_iter": n_iter,
        "alphas": al.tolist(),
        "betas": be.tolist(),
        **ritz_vecs,
    }


def lanczos_multistart(op, k=30, starts=3, seed0=0, basis_device="cpu"):
    """Extremes over `starts` independent Lanczos runs.

    Lanczos converges to extreme eigenvalues from inside the spectrum, so the
    conservative aggregate is min over starts for lambda_min and max over
    starts for lambda_max; the cross-start spread is reported as a
    convergence diagnostic alongside the Ritz residuals.
    """
    runs = [lanczos_symmetric(op, k=k, seed=seed0 + s, basis_device=basis_device)
            for s in range(starts)]
    lmins = [r["lmin"] for r in runs]
    lmaxs = [r["lmax"] for r in runs]
    i_min = int(np.argmin(lmins))
    i_max = int(np.argmax(lmaxs))
    return {
        "lmin": lmins[i_min],
        "lmax": lmaxs[i_max],
        "res_lmin": runs[i_min]["res_lmin"],
        "res_lmax": runs[i_max]["res_lmax"],
        "spread_lmin": float(max(lmins) - min(lmins)),
        "spread_lmax": float(max(lmaxs) - min(lmaxs)),
        "orth_err": float(max(r["orth_err"] for r in runs)),
        "per_start": [{"lmin": r["lmin"], "lmax": r["lmax"],
                       "res_lmin": r["res_lmin"], "res_lmax": r["res_lmax"],
                       "n_iter": r["n_iter"]} for r in runs],
    }
