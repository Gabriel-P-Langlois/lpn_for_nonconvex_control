"""Experiment 3.1: is the denoiser affine, and if so, what is its regularizer?

Experiment 2 measured a Jacobian spectrum for PIRATE+ that is constant to
about 1e-4 across operating-distribution test points. A constant Jacobian
means the denoiser is AFFINE there, D(z) = A z + b, and an affine denoiser
has a QUADRATIC implicit regularizer, computable in closed form from
measured quantities with no training:

    with S = sym(A) (the skew part is discarded; rho certifies how much),
    the prox identity z - y = grad f_reg(y) at y = D(z) inverts to

        grad f_reg(y) = S^{-1}(y - b) - y,

    i.e. f_reg is the quadratic with Hessian H = S^{-1} - I. Everything is
    matrix-free: S v = (jvp + vjp)/2, S^{-1} v by conjugate gradients ---
    well conditioned, since spec(S) in [0.945, 1.054] for PIRATE+; the
    ill-conditioned (I - S)^{-1} is never needed. H's extreme eigenvalues
    are 1/lambda - 1 at the measured extremes of S (monotone map).

Three measurements (driver: `pnpreg.affine_run`):

  * `affinity` -- err = ||D(z_i) - D(z_j) - J(z_mid)(z_i - z_j)|| /
    ||D(z_i) - D(z_j)|| over pairs of operating-distribution inputs, one
    forward-mode jvp at each midpoint; plus a direct Jacobian-variation
    check and the constant term b with its cross-point spread.
  * `selfcheck` -- the prox identity, checkable without ground truth:
    grad f_reg(D(z)) should equal z - D(z) at a HELD-OUT z. Its relative
    residual is the experiment's falsifiable prediction: it should land at
    the scale of (affinity error + rho). Run on both networks; for the
    AWGN denoiser (rho = 0.44) it must be LARGE -- that contrast is the
    point.
  * `ritz_vectors` -- the extreme Ritz vectors of S, displayed as
    deformation-field slices: the pattern the implicit prior most favors
    (eigenvalue of S above 1, negative curvature of f_reg) and the one it
    most penalizes (below 1, positive curvature).

Affinity is claimed on the operating distribution only (the released field
plus unit noise); the optional scale sweep in the driver probes how far the
affine region extends.
"""
import numpy as np
import torch

from . import probe
from . import probe_targets as pt

SEED = 14            # matches Experiment 2's test inputs exactly
SEED_HELDOUT = 114   # never used by Experiment 2


def _f64(x):
    return x.detach().cpu().double()


def apply_D(model, z):
    with torch.no_grad():
        return z - model(z)


def jvp_D(model, z, v):
    """Forward-mode J(z) v for D(z) = z - model(z)."""
    _, jv = torch.func.jvp(lambda x: x - model(x), (z,), (v,))
    return jv.detach()


def operating_inputs(field, n_points, seed, device, dtype):
    """The same z_i = field + N(0, I) construction (and seeds) as
    probe_targets.target_pirate, so results line up with the Exp-2 table."""
    zs = []
    for i in range(n_points):
        g = torch.Generator(device="cpu").manual_seed(seed + i)
        noise = torch.randn(field.shape, generator=g, dtype=torch.float32)
        zs.append((field + noise).to(device=device, dtype=dtype))
    return zs


def affinity_from_ops(D_fn, jvp_fn, zs, n_pairs, n_var_checks=2):
    """The affinity metric over the first `n_pairs` index pairs (i < j,
    lexicographic -- deterministic), given callables D_fn(z) and
    jvp_fn(z, v). Returns per-pair errors, the Jacobian-variation check,
    and the constant term b with its spread."""
    Ds = [D_fn(z) for z in zs]
    pairs = [(i, j) for i in range(len(zs)) for j in range(i + 1, len(zs))][:n_pairs]
    errs = []
    var_checks = []
    for k, (i, j) in enumerate(pairs):
        v = zs[i] - zs[j]
        mid = 0.5 * (zs[i] + zs[j])
        Jv = jvp_fn(mid, v)
        dD = _f64(Ds[i]) - _f64(Ds[j])
        errs.append(float((dD - _f64(Jv)).norm() / dD.norm()))
        if k < n_var_checks:
            Jv_i = jvp_fn(zs[i], v)
            var_checks.append(float((_f64(Jv_i) - _f64(Jv)).norm() / _f64(Jv).norm()))
    # constant term of the LOCAL affine model at the reference point:
    # b = D(z_ref) - J(z_ref) z_ref. The per-point values b_i are kept as a
    # DIAGNOSTIC only (their spread mixes Jacobian variation along the large
    # field vector into the estimate, which is why they are not averaged --
    # found in the smoke run of 2026-07-30).
    z_ref = zs[0]
    bs = [_f64(Ds[i]) - _f64(jvp_fn(z_ref, zs[i])) for i in range(len(zs))]
    b_hat = bs[0]
    b_spread = max(float((b - b_hat).norm()) for b in bs) / max(float(b_hat.norm()), 1e-300)
    return {
        "pairs": pairs,
        "err": errs,
        "err_max": max(errs),
        "err_median": float(np.median(errs)),
        "jac_variation": var_checks,
        "b_norm": float(b_hat.norm()),
        "b_spread": b_spread,
        "D_scale": float(np.median([float(_f64(d).norm()) for d in Ds])),
    }, b_hat


def affinity(which, n_points=8, n_pairs=12, device="cpu", dtype=torch.float32,
             crop=None, model=None, seed=SEED):
    """The affinity test for one released network."""
    if model is None:
        model = pt.load_pirate_dncnn(which, device=device, dtype=dtype)
    field = pt.load_field(crop)
    if dtype == torch.float64:
        field = field.double()
    zs = operating_inputs(field, n_points, seed, device, dtype)
    out, b_hat = affinity_from_ops(lambda z: apply_D(model, z),
                                   lambda z, v: jvp_D(model, z, v),
                                   zs, n_pairs)
    out.update(which=which, n_points=n_points, device=device,
               dtype=str(dtype), seed=seed, crop=list(crop) if crop else None)
    return out, b_hat, model, field


def cg_solve_S(op, rhs, tol=1e-8, max_iter=200):
    """w with S w = rhs by conjugate gradients on the matrix-free symmetric
    part; spec(S) well inside (0, inf) for the cases this runs on."""
    rhs64 = _f64(rhs)

    def matvec(x64):
        x = x64.to(dtype=op.dtype)
        if op.device != "cpu":
            x = x.to(op.device)
        return _f64(op.sym(x))

    w = torch.zeros_like(rhs64)
    r = rhs64 - matvec(w)
    p = r.clone()
    rs = float(r @ r)
    b_norm = float(rhs64.norm())
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        Ap = matvec(p)
        alpha = rs / float(p @ Ap)
        w = w + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)
        if np.sqrt(rs_new) <= tol * max(b_norm, 1e-300):
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return w, n_iter, float(np.sqrt(rs_new) / max(b_norm, 1e-300))


def selfcheck(which, b_hat, device="cpu", dtype=torch.float32, crop=None,
              model=None, seed=SEED_HELDOUT, cg_tol=1e-8, cg_max_iter=200):
    """The prox-identity residual at a held-out input:
    grad f_reg(y) = S^{-1}(y - b) - y must equal z - y at y = D(z).
    Exact for a symmetric affine denoiser; its size here reads
    (affinity error + skew part) in the same units."""
    if model is None:
        model = pt.load_pirate_dncnn(which, device=device, dtype=dtype)
    field = pt.load_field(crop)
    if dtype == torch.float64:
        field = field.double()
    z = operating_inputs(field, 1, seed, device, dtype)[0]
    op = pt.PirateOp(model, z)
    y = op.apply_D(z)
    ra = residual_asymmetry(op)
    rhs = _f64(y).reshape(-1) - b_hat.reshape(-1)
    w, n_iter, cg_resid = cg_solve_S(op, rhs, tol=cg_tol, max_iter=cg_max_iter)
    grad_f = w - _f64(y).reshape(-1)
    target = _f64(z).reshape(-1) - _f64(y).reshape(-1)
    resid = float((grad_f - target).norm() / target.norm())
    return {
        "which": which,
        "seed": seed,
        "resid": resid,
        "target_scale_rel": float(target.norm() / _f64(z).reshape(-1).norm()),
        "cg_iters": n_iter,
        "cg_resid": cg_resid,
        **ra,
    }


def ritz_vectors(which, k=30, device="cpu", crop=None, model=None, seed=SEED):
    """Extreme Ritz vectors of S at the first Exp-2 test point, reshaped to
    the field, with their S-eigenvalues and the implied f_reg curvatures."""
    if model is None:
        model = pt.load_pirate_dncnn(which, device=device)
    field = pt.load_field(crop)
    z = operating_inputs(field, 1, seed, device, torch.float32)[0]
    op = pt.PirateOp(model, z)
    lan = probe.lanczos_symmetric(op, k=k, seed=probe_seed(which),
                                  return_ritz=True)
    shape = tuple(field.shape)
    out = {}
    for key, lam in (("lmin", lan["lmin"]), ("lmax", lan["lmax"])):
        v = lan[f"ritz_vec_{key}"].reshape(shape)
        out[key] = {"lambda_S": lam, "curvature_freg": 1.0 / lam - 1.0,
                    "vec": v}
    out["res"] = {"lmin": lan["res_lmin"], "lmax": lan["res_lmax"]}
    return out


def probe_seed(which):
    """A fixed, distinct Lanczos seed per network (recorded in the JSON)."""
    return 3000 + (0 if which == "pirate" else 1)


class ResidualOp(probe.LinOp):
    """The Jacobian of the RESIDUAL map R(z) = z - D(z), i.e. I - A.

    The regularizer's gradient is the residual, not the denoiser, so the
    dimensionless asymmetry that decides whether an objective approximately
    exists is rho applied to THIS operator: for a near-identity denoiser the
    identity part dominates ||A|| and makes rho(A) misleadingly small (the
    smoke run of 2026-07-30 measured exactly this -- rho(A) = 0.016 against
    a residual-relative asymmetry ~30x larger)."""

    def __init__(self, op):
        self.inner = op
        self.n = op.n
        self.dtype = op.dtype
        self.device = op.device

    def jvp(self, v):
        return v - self.inner.jvp(v)

    def vjp(self, u):
        return u - self.inner.vjp(u)


def residual_asymmetry(op, m_probes=8, seed=77):
    """rho of I - A: ||K||_F / (2 ||I - A||_F), plus the norm ratio
    ||I - A||_F / ||A||_F that converts between the two normalizations."""
    r_res = probe.hutchinson_asymmetry(ResidualOp(op), m_probes=m_probes, seed=seed)
    r_full = probe.hutchinson_asymmetry(op, m_probes=m_probes, seed=seed)
    return {
        "rho_res": r_res["rho"],
        "rho_res_se": r_res["rho_se"],
        "rho_full": r_full["rho"],
        "norm_ratio_res_over_full": float(np.sqrt(r_res["F2"] / r_full["F2"])),
    }
