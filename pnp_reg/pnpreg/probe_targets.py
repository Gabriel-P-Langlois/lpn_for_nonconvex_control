"""The operators Experiment 2 probes: three calibrations, PIRATE, PIRATE+.

Each `target_*` function returns a list of `PointCase` -- one frozen denoiser
Jacobian per test point, wrapped as a `probe.LinOp` -- plus, for the
calibration rows, the exact answers the estimators must reproduce (task
document, kill criterion 1: an uncalibrated estimator is worthless).

Rows (see DESIGN.md, decisions of 2026-07-30):

  cal_mixture     the exact 1-D mixture MMSE denoiser (`mixture.D`), applied
                  coordinatewise on R^n: diagonal Jacobian diag(dD(z_i)),
                  float64. Symmetric, PSD, and FAILS condition 3 by design
                  (a coordinate is pinned at z = 0, where dD > 1).
  cal_quadrature  the exact n = 2 TV posterior mean (`tvpm.quadrature`),
                  Jacobian by central finite differences of a spectrally
                  convergent quadrature, float64. All three conditions hold
                  by theorem (grad u_PM = hess psi, between 0 and I).
  cal_icnn        the trained tv_pm convex ICNN's proximal operator on R^64:
                  D(x) = argmin_u J_theta(u) + ||u - x||^2/2, so
                  grad D = (I + hess J_theta(u_hat))^{-1} -- symmetric, PSD,
                  and at most I by architecture. The probed operator solves
                  (I + H) w = v by CG with Hessian-vector products (jvp slot)
                  and by a dense factorization (vjp slot): two genuinely
                  different arithmetic paths for the same symmetric matrix.
  pirate, pirate_plus
                  the released DnCNN weights (numerics/ext/pirate/, verbatim
                  upstream -- imported, never edited). The network predicts
                  the noise, so the denoiser is D(z) = z - dnn(z) and
                  grad D = I - grad dnn. Test inputs are the released
                  registration field plus unit-variance Gaussian noise, the
                  operating distribution of their own training loop
                  (ext/pirate/train_denoiser.py: sigma = 1, noise standard
                  deviation sigma^2 = 1).
"""
import os
from dataclasses import dataclass, field as dc_field

import numpy as np
import torch

from . import paths
from . import mixture as mx
from .probe import LinOp, DenseOp

SIGMA_TV = 20.0 / 256.0          # the tv_pm production pair sigma = t
T_TV = 20.0 / 256.0


@dataclass
class PointCase:
    """One probed Jacobian: the operator, its exact answers (calibration
    rows only), and bookkeeping the JSON records.

    Large operators (PIRATE: a retained vjp graph per point) are built
    lazily through `op_factory` and freed with `release()`; probe_run
    consumes points strictly one at a time."""
    name: str
    index: int
    op: LinOp | None = None
    exact: dict | None = None
    meta: dict = dc_field(default_factory=dict)
    op_factory: object = None

    def get_op(self):
        if self.op is None:
            self.op = self.op_factory()
        return self.op

    def release(self):
        if self.op_factory is not None:
            self.op = None


# ---------------------------------------------------------------- cal_mixture

class DiagOp(LinOp):
    """diag(d): the coordinatewise mixture denoiser's Jacobian. jvp and vjp
    are the same product but kept as separate calls so both estimator slots
    are exercised."""

    def __init__(self, d):
        self.d = torch.as_tensor(d, dtype=torch.float64)
        self.n = len(self.d)
        self.dtype = torch.float64
        self.device = "cpu"

    def jvp(self, v):
        return self.d * v

    def vjp(self, u):
        return self.d * u


def target_cal_mixture(n=64, sigma=0.5, n_points=8, seed=11):
    """Diagonal Jacobians diag(dD(z)) at z with iid p_z coordinates.

    Point 0 gets one coordinate forced to z = 0, between the modes, where
    dD(0, 0.5) is approximately 11.3 > 1: the guaranteed condition-3
    violation the calibration must detect."""
    cases = []
    for i in range(n_points):
        z = mx.sample_pz(n, sigma, seed + i)
        if i == 0:
            z[0] = 0.0
        d = mx.dD(z, sigma)
        exact = {
            "rho": 0.0,
            "lmin": float(d.min()),
            "lmax": float(d.max()),
            "F2": float((d ** 2).sum()),
        }
        cases.append(PointCase("cal_mixture", i, DiagOp(d), exact,
                               {"sigma": sigma, "seed": seed + i,
                                "forced_zero": i == 0}))
    return cases


# ------------------------------------------------------------ cal_quadrature

def _quad_jacobian(x, sigma, lam, h):
    """2x2 Jacobian of the exact posterior mean by central differences.

    The quadrature (tvpm/quadrature.py) is spectrally convergent, so the FD
    error is pure truncation, O(h^2) ~ 1e-10 at h = 1e-5."""
    from tvpm import quadrature as Q
    pts = []
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        pts.append(x + e)
        pts.append(x - e)
    _, u = Q.log_z_and_pm(np.array(pts), sigma, lam, w=1.0)
    J = np.empty((2, 2))
    for j in range(2):
        J[:, j] = (u[2 * j] - u[2 * j + 1]) / (2 * h)
    return J


def target_cal_quadrature(sigma=SIGMA_TV, n_points=16, seed=12, h=1e-5):
    """Exact n = 2 TV posterior-mean Jacobians, sigma = t = 20/256, lam = 2t."""
    lam = 2.0 * T_TV
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n_points):
        u = rng.uniform(0.0, 1.0, 2)
        x = u + rng.normal(0.0, sigma, 2)
        J = _quad_jacobian(x, sigma, lam, h)
        J2 = _quad_jacobian(x, sigma, lam, 2 * h)
        S = 0.5 * (J + J.T)
        eig = np.linalg.eigvalsh(S)
        K = 0.5 * (J - J.T)
        exact = {
            "rho": float(np.linalg.norm(K) / np.linalg.norm(J)),
            "lmin": float(eig[0]),
            "lmax": float(eig[-1]),
            "F2": float((J ** 2).sum()),
        }
        cases.append(PointCase("cal_quadrature", i, DenseOp(J), exact,
                               {"sigma": sigma, "lam": lam, "h": h,
                                "x": x.tolist(),
                                "fd_h_vs_2h": float(np.abs(J - J2).max())}))
    return cases


# ----------------------------------------------------------------- cal_icnn

class IcnnProxOp(LinOp):
    """grad D = (I + H)^{-1} for the ICNN prox, two arithmetic paths.

    jvp: conjugate-gradient solve of (I + H) w = v with matrix-free
    Hessian-vector products of the scalar potential (double backprop).
    vjp: dense Cholesky solve of the SAME symmetric matrix. The estimator's
    b_j then reads the CG-vs-dense discrepancy, calibrating the two-slot
    code path honestly."""

    def __init__(self, model, mu, s, u_hat, H_dense, cg_tol):
        self.model = model
        self.mu = mu
        self.s = s
        self.u_hat = u_hat.detach()
        self.n = u_hat.numel()
        self.dtype = u_hat.dtype
        self.device = "cpu"
        A = torch.eye(self.n, dtype=self.dtype) + H_dense
        self.chol = torch.linalg.cholesky(A)
        self.cg_tol = cg_tol

    def _hvp(self, p):
        """H p at u_hat, matrix-free: H = diag(1/s) hess[scalar] diag(1/s)."""
        z = ((self.u_hat - self.mu) / self.s).clone().requires_grad_(True)
        g = torch.autograd.grad(self.model.scalar(z.unsqueeze(0)).sum(), z,
                                create_graph=True)[0]
        hz = torch.autograd.grad((g * (p / self.s)).sum(), z)[0]
        return hz / self.s

    def jvp(self, v):
        b = v
        w = torch.zeros_like(b)
        r = b - (w + self._hvp(w))
        p = r.clone()
        rs = float(r @ r)
        bnorm = float(b.norm())
        for _ in range(4 * self.n):
            Ap = p + self._hvp(p)
            alpha = rs / float(p @ Ap)
            w = w + alpha * p
            r = r - alpha * Ap
            rs_new = float(r @ r)
            if np.sqrt(rs_new) <= self.cg_tol * max(bnorm, 1e-300):
                break
            p = r + (rs_new / rs) * p
            rs = rs_new
        return w

    def vjp(self, u):
        return torch.cholesky_solve(u.reshape(-1, 1), self.chol).reshape(-1)


def _solve_prox(model, mu, s, x, tol=1e-12, iters=500):
    """u_hat = argmin scalar((u - mu)/s) + ||u - x||^2 / 2 by L-BFGS,
    replicating tvpm/denoise.py:prox_of_Jtheta in the operator's dtype.
    Returns (u_hat, optimality residual ||grad J(u_hat) + u_hat - x||)."""
    u = x.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([u], lr=1.0, max_iter=iters, tolerance_grad=tol,
                            tolerance_change=0.0, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        obj = model.scalar(((u - mu) / s).unsqueeze(0)).sum() \
            + 0.5 * ((u - x) ** 2).sum()
        obj.backward()
        return obj

    opt.step(closure)
    z = ((u - mu) / s).detach().requires_grad_(True)
    g = torch.autograd.grad(model.scalar(z.unsqueeze(0)).sum(), z)[0] / s
    resid = float((g + (u.detach() - x)).norm())
    return u.detach(), resid


def load_tvpm_icnn():
    """The tv_pm production fc checkpoint at sigma = t = 20/256 (beta 20,
    250k steps, m = 8000); module defaults would return None."""
    from tvpm import recover
    p = recover.find_checkpoint(arch="fc", sweeps=8000, beta=20.0,
                                steps=250000, sigma=SIGMA_TV, t=T_TV)
    if p is None:
        raise FileNotFoundError(
            "tv_pm fc checkpoint (beta=20, steps=250000, sigma=t=20/256) not "
            "found; see numerics/logs/ckpt/")
    model, units, ck = recover.load_checkpoint(p)
    return model, units, ck, p


def target_cal_icnn(n_points=8, seed=13, dtype=torch.float64, cg_tol=1e-12):
    """ICNN prox Jacobians at noisy 8x8 cameraman patches (sigma = 20/256).

    float64 by default (the model is cast; convexity is dtype-independent);
    the float32 sensitivity repeat passes dtype=torch.float32, cg_tol=1e-6."""
    from scipy.io import loadmat
    from tvpm.paths import IMAGES
    from tvpm.denoise import tile

    model, units, ck, ckpt_path = load_tvpm_icnn()
    model = model.double() if dtype == torch.float64 else model.float()
    mu = torch.as_tensor(np.asarray(units.mu), dtype=dtype).reshape(-1)
    s = torch.as_tensor(np.asarray(units.s), dtype=dtype).reshape(-1)

    img = np.asarray(loadmat(os.path.join(IMAGES, "cameraman_256x256_d.mat"))
                     ["cameraman_256x256_d"], dtype=float)
    patches = tile(img)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(patches), size=n_points, replace=False)
    cases = []
    for i, pi in enumerate(idx):
        xc = patches[pi].reshape(-1)
        x = np.clip(xc + rng.normal(0, SIGMA_TV, xc.shape), 0, 1)
        xt = torch.as_tensor(x, dtype=dtype)
        u_hat, resid = _solve_prox(model, mu, s, xt)

        z = ((u_hat - mu) / s).detach()
        Hz = torch.autograd.functional.hessian(
            lambda zz: model.scalar(zz.unsqueeze(0)).sum(), z)
        H = Hz / (s.reshape(-1, 1) * s.reshape(1, -1))
        H = 0.5 * (H + H.T)                      # symmetrize roundoff
        mu_eigs = torch.linalg.eigvalsh(H).numpy()
        inv_eigs = 1.0 / (1.0 + mu_eigs)
        exact = {
            "rho": 0.0,
            "lmin": float(inv_eigs.min()),
            "lmax": float(inv_eigs.max()),
            "F2": float((inv_eigs ** 2).sum()),
            "H_eig_min": float(mu_eigs.min()),
        }
        op = IcnnProxOp(model, mu, s, u_hat, H, cg_tol)
        cases.append(PointCase("cal_icnn", i, op, exact,
                               {"patch": int(pi), "prox_resid": resid,
                                "sigma": SIGMA_TV, "seed": seed,
                                "ckpt": os.path.basename(ckpt_path),
                                "dtype": str(dtype)}))
    return cases


# ------------------------------------------------------- pirate, pirate_plus

N_PIRATE_PARAMS = 453_059


def load_pirate_dncnn(which, device="cpu", dtype=torch.float32):
    """The released DnCNN, frozen. which in {"pirate", "pirate_plus"}.

    The PIRATE+ checkpoint stores the SAME 12 tensors under a "PIRATE.dnn."
    key prefix (the DEQ fine-tune wraps the denoiser); strip it and load
    strict."""
    from ext.pirate.model.base import DnCNN
    path = {"pirate": paths.PIRATE_CKPT_AWGN,
            "pirate_plus": paths.PIRATE_CKPT_PLUS}[which]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    if which == "pirate_plus":
        sd = {k[len("PIRATE.dnn."):]: v for k, v in sd.items()
              if k.startswith("PIRATE.dnn.")}
    model = DnCNN()
    model.load_state_dict(sd, strict=True)
    n_par = sum(p.numel() for p in model.parameters())
    if n_par != N_PIRATE_PARAMS:
        raise RuntimeError(f"{which}: {n_par} parameters, expected {N_PIRATE_PARAMS}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device=device, dtype=dtype)


def load_field(crop=None):
    """The released registration field, (1, 3, 80, 96, 112) float32; `crop`
    center-crops the spatial dims (smoke mode)."""
    import h5py
    with h5py.File(paths.PIRATE_FIELD, "r") as f:
        x = torch.from_numpy(np.array(f["fieldData"], dtype="float32"))
    if crop is not None:
        _, _, D, H, W = x.shape
        d, h, w = crop
        x = x[:, :, (D - d) // 2:(D - d) // 2 + d,
              (H - h) // 2:(H - h) // 2 + h,
              (W - w) // 2:(W - w) // 2 + w]
    return x


class PirateOp(LinOp):
    """J = grad D(z) for D(z) = z - model(z), via torch.func.

    vjp: linearized once at z (the retained graph is the expensive part,
    ~1.5 GB at full field size -- one PirateOp per point, freed after use).
    jvp: forward-mode. NOTE (measured 2026-07-30): alternative autodiff
    compositions of the same product (reverse-over-reverse, reverse-over-
    forward) agree BITWISE with these -- conv kernels are transposed exactly
    -- so they cannot serve as an independent second path; the noise floor is
    the surrogate row plus identity_max instead (probe.py docstring)."""

    def __init__(self, model, z):
        self.model = model
        self.z = z
        self.shape = z.shape
        self.n = z.numel()
        self.dtype = z.dtype
        self.device = str(z.device).split(":")[0]
        self._D = lambda x: x - self.model(x)
        _, self._vjp_fn = torch.func.vjp(self._D, z)

    def apply_D(self, x):
        with torch.no_grad():
            return self._D(x)

    def jvp(self, v):
        _, jv = torch.func.jvp(self._D, (self.z,), (v.reshape(self.shape),))
        return jv.reshape(-1)

    def vjp(self, u):
        return self._vjp_fn(u.reshape(self.shape))[0].reshape(-1)


class SymSurrogate(torch.nn.Module):
    """A one-layer map whose Jacobian is symmetric BY WEIGHT CONSTRUCTION.

    g(z) = C^T elu(C z + b) with (C, b) the released first convolution of the
    PIRATE DnCNN: grad g = C^T diag(elu'(Cz+b)) C, exactly symmetric and PSD.
    Wrapped in PirateOp, D(z) = z - g(z) has an exactly symmetric Jacobian
    computed through the same conv / conv_transpose kernels as the real rows,
    so its measured rho is the estimator's end-to-end zero-test (the floor
    row of the deliverable table)."""

    def __init__(self, w, b):
        super().__init__()
        self.register_buffer("w", w)
        self.register_buffer("b", b)

    def forward(self, x):
        import torch.nn.functional as F
        return F.conv_transpose3d(F.elu(F.conv3d(x, self.w, self.b, padding=1)),
                                  self.w, padding=1)


def target_floor(n_points=2, seed=14, device="cpu", dtype=torch.float32,
                 crop=None, model=None):
    """The symmetric-surrogate floor row, at the SAME test inputs (same
    seeds) as target_pirate's first points. exact rho = 0 by construction."""
    if model is None:
        model = load_pirate_dncnn("pirate", device=device, dtype=dtype)
    surr = SymSurrogate(model.dncnn_3[0].weight.detach(),
                        model.dncnn_3[0].bias.detach()).to(device=device, dtype=dtype)
    field = load_field(crop)
    cases = []
    for i in range(n_points):
        def factory(i=i):
            g = torch.Generator(device="cpu").manual_seed(seed + i)
            noise = torch.randn(field.shape, generator=g, dtype=torch.float32)
            z = (field + noise).to(device=device, dtype=dtype)
            return PirateOp(surr, z)
        cases.append(PointCase("floor", i, None,
                               {"rho": 0.0},
                               {"seed": seed + i, "device": device,
                                "dtype": str(dtype),
                                "crop": list(crop) if crop else None},
                               op_factory=factory))
    return cases


def target_pirate(which, n_points=8, seed=14, device="cpu",
                  dtype=torch.float32, crop=None, model=None):
    """PIRATE/PIRATE+ Jacobians at field + N(0, I) test inputs.

    Noise is CPU-seeded (seed + point index) so an MPS run and its CPU
    cross-check probe identical inputs. Pass `model` to reuse a loaded net.
    NOTE the memory cost: each PointCase holds a retained vjp graph; consume
    and free sequentially (probe_run does)."""
    if model is None:
        model = load_pirate_dncnn(which, device=device, dtype=dtype)
    field = load_field(crop)
    cases = []
    for i in range(n_points):
        def factory(i=i):
            g = torch.Generator(device="cpu").manual_seed(seed + i)
            noise = torch.randn(field.shape, generator=g, dtype=torch.float32)
            z = (field + noise).to(device=device, dtype=dtype)
            return PirateOp(model, z)
        cases.append(PointCase(which, i, None, None,
                               {"seed": seed + i, "device": device,
                                "dtype": str(dtype),
                                "crop": list(crop) if crop else None},
                               op_factory=factory))
    return cases
