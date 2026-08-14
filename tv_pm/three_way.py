"""Three-way (four-way) denoiser comparison on cameraman, at sigma = t = 20/256.

METHODS
  u_PM              MCMC posterior-mean denoiser (ground truth), m = 8000 sweeps
  u_t (Route 1)     grad psi_theta(x)              -- one forward pass
                    prox_{J_1} collapses to grad psi_theta exactly:
                      J_1 = psi* - 0.5||.||^2, so grad J_1(u) = w(u) - u with
                      grad psi(w) = u; stationarity of argmin_u J_1(u)+0.5||u-x||^2
                      gives w(u_hat) = x, i.e. u_hat = grad psi_theta(x).
  u_t (Route 2)     solve grad G_theta(u) = x      -- convex solve on the conjugate net
                      prox_{J_2}(x) = argmin_u G(u) - <u,x>,  J_2 = G - 0.5||.||^2
  u_hat (direct)    prox_{J_theta}(x) by L-BFGS    -- the existing notebook method

Both routes rest on ONE trained denoiser, psi_theta, fitted by gradient
supervision  grad psi_theta(x_k) ~= u_PM(x_k)  on the cached m=8000 splits.
G_theta is trained on pairs manufactured from psi_theta (no new data).
"""
import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tvpm import dataset
from tvpm.paths import DATA, FIGS, LOG_CKPT, RESULTS
from tvpm.sampler import from_sigma_t, sample_pm
from tvpm.denoise import tile, untile, psnr, rel_l2, prox_of_Jtheta
from src.network import LPN
from src.gradfit import train_grad, Units, net_grad
from src.recovery import conjugate_samples

SIGMA = 20 / 256
T = 20 / 256
SWEEPS = 8000
WIDTH, LAYERS, BETA = 256, 2, 20
IMAGE = "cameraman_256x256_d"
# Outputs follow tvpm/paths.py, not a bespoke folder: figures that travel go to
# results/figs/, their numbers to results/, and the cached sampler output to
# data/ -- the same places recover_tv.ipynb writes.
FIG_DIR, RES_DIR, CACHE_DIR = FIGS, RESULTS, DATA
for _d in (FIG_DIR, RES_DIR, CACHE_DIR):
    os.makedirs(_d, exist_ok=True)

SUF = os.environ.get("TW_SUFFIX", "")          # "_smoke" for the end-to-end dry run
PSI_CK = os.path.join(LOG_CKPT, f"threeway_psi_w{WIDTH}_L{LAYERS}_b{BETA}{SUF}.pth")
G_CK = os.path.join(LOG_CKPT, f"threeway_G_w{WIDTH}_L{LAYERS}_b{BETA}{SUF}.pth")


def flat(a):
    return a.reshape(a.shape[0], -1)


def load_splits():
    kw = dict(sweeps=SWEEPS, sigma=SIGMA, t=T)
    return (dataset.load("train", **kw), dataset.load("val", **kw),
            dataset.load("eval", **kw))


# --------------------------------------------------------------------------
# stage 1: psi_theta, the LPN denoiser.  grad psi_theta(x_k) ~= u_PM(x_k)
# --------------------------------------------------------------------------
def train_psi(steps, seed=1):
    tr, va, _ = load_splits()
    x, y = flat(tr["x"]), flat(tr["y"])
    xv, yv = flat(va["x"]), flat(va["y"])
    u = Units(x, standardize=True)
    model = LPN(in_dim=x.shape[1], hidden=WIDTH, layers=LAYERS, beta=BETA)
    t0 = time.time()
    hist = train_grad(model, u.z(x), u.target(y), u.z(xv), u.target(yv),
                      steps=steps, seed=seed)
    torch.save({"state": model.state_dict(), "mu": u.mu, "s": u.s,
                "in_dim": x.shape[1], "hidden": WIDTH, "layers": LAYERS,
                "beta": BETA, "steps": steps, "best_val": hist["best_val"],
                "sigma": SIGMA, "t": T, "sweeps": SWEEPS,
                "kind": "psi (grad psi ~ u_PM)"}, PSI_CK)
    print(f"[psi] {steps} steps in {(time.time()-t0)/60:.1f} min, "
          f"best val {hist['best_val']:.4e} -> {os.path.basename(PSI_CK)}", flush=True)
    return model, u


def load_psi():
    ck = torch.load(PSI_CK, weights_only=False)
    m = LPN(in_dim=ck["in_dim"], hidden=ck["hidden"], layers=ck["layers"], beta=ck["beta"])
    m.load_state_dict(ck["state"]); m.eval()
    return m, Units.from_saved(ck["mu"], ck["s"]), ck


# --------------------------------------------------------------------------
# stage 2: G_theta ~= psi_theta*, trained on pairs made from psi_theta alone
# --------------------------------------------------------------------------
def conjugate_pairs_x(psi, u, x):
    """(y_k, G_k) = (grad_x psi(x_k), <y_k, x_k> - psi(x_k)), in X-UNITS.

    WHY X-UNITS AND NOT psi's z-units. G approximates psi*, and psi* = J + 0.5||.||^2.
    In x-units that quadratic is present at full strength, so G is ~1-strongly
    convex and  argmin_u G(u) - <u,x>  (Route 2's denoiser) is a well-posed
    strongly convex program. In psi's standardized z-units the same conjugate
    carries curvature ~1/s^2 concentrated on a domain of radius ~||s*u_PM||, and
    the ICNN flattens outside it, so G(v) - <v,z> is unbounded below and the
    solve runs away (observed: NaN). The units are a change of variables, not a
    modelling choice, but they decide whether Route 2's own program is coercive.
    """
    xt = torch.tensor(np.asarray(u.z(x))).float().requires_grad_(True)
    psi_x = psi.scalar(xt)                             # psi_tilde(z) == psi(x)
    gz = torch.autograd.grad(psi_x.sum(), xt)[0]
    y = (gz / torch.tensor(u.s).float()).detach().numpy()          # grad_x psi = u_PM
    psi_val = psi_x.detach().numpy()
    Gk = np.sum(y * np.asarray(x), axis=1, keepdims=True) - psi_val
    return y, Gk


def train_G(steps, seed=1):
    """G_theta regresses psi_theta* on the conjugate pairs, in x-units.

    Value regression, exactly as bin/_run.py does for Route 2; the pairs are
    manufactured from psi_theta alone, so no new data enters.
    """
    psi, u, _ = load_psi()
    tr, va, _ = load_splits()
    yk_tr, Gk_tr = conjugate_pairs_x(psi, u, flat(tr["x"]))
    yk_va, Gk_va = conjugate_pairs_x(psi, u, flat(va["x"]))
    print(f"[G] conjugate pairs {yk_tr.shape}: y in [{yk_tr.min():.3f}, "
          f"{yk_tr.max():.3f}], G in [{Gk_tr.min():.3f}, {Gk_tr.max():.3f}]", flush=True)
    G = LPN(in_dim=yk_tr.shape[1], hidden=WIDTH, layers=LAYERS, beta=BETA)
    from src.train import train_potential
    t0 = time.time()
    hist = train_potential(G, yk_tr, Gk_tr, yk_va, Gk_va, steps=steps)
    torch.save({"state": G.state_dict(), "in_dim": yk_tr.shape[1], "hidden": WIDTH,
                "layers": LAYERS, "beta": BETA, "steps": steps,
                "best_val": hist.get("best_val"),
                "yk_reach": float(np.abs(yk_tr).max()),
                "kind": "G ~= psi* in x-units"}, G_CK)
    print(f"[G] {steps} steps in {(time.time()-t0)/60:.1f} min, "
          f"best val {hist.get('best_val')} -> {os.path.basename(G_CK)}", flush=True)
    return G


GG_CK = os.path.join(LOG_CKPT, f"threeway_Ggrad_w{WIDTH}_L{LAYERS}_b{BETA}{SUF}.pth")


def train_G_grad(steps, seed=1):
    """G_theta fitted to the conjugate's GRADIENT: grad G(y_k) ~= x_k.

    WHY THIS VARIANT EXISTS. bin/_run.py's Route 2 regresses psi*'s VALUES, and
    the synthetic notebooks score Route 2 on prior VALUES, so a value fit is all
    that protocol ever needs. Route 2's DENOISER, however, is the inversion of G,
    which depends on grad G. Measured on the trained value-fitted G:

        G value fit              8.4e-4 relative          (excellent)
        grad G(y_k) vs x_k       13.8 % median, 25.1 % p90 (poor)

    so the denoiser inherits a ~25 % error from a network whose values are right
    to four digits. This variant hands Route 2 exactly the supervision its own
    denoiser needs, so the comparison cannot be dismissed as having crippled it.
    No standardization: G must keep the 0.5||.||^2 of psi* = J + 0.5||.||^2 at
    full strength, or its inversion stops being coercive.
    """
    psi, u, _ = load_psi()
    tr, va, _ = load_splits()
    xtr, xva = flat(tr["x"]), flat(va["x"])
    yk_tr, _ = conjugate_pairs_x(psi, u, xtr)
    yk_va, _ = conjugate_pairs_x(psi, u, xva)
    G = LPN(in_dim=yk_tr.shape[1], hidden=WIDTH, layers=LAYERS, beta=BETA)
    t0 = time.time()
    hist = train_grad(G, yk_tr, xtr, yk_va, xva, steps=steps, seed=seed)
    torch.save({"state": G.state_dict(), "in_dim": yk_tr.shape[1], "hidden": WIDTH,
                "layers": LAYERS, "beta": BETA, "steps": steps,
                "best_val": hist["best_val"],
                "yk_reach": float(np.abs(yk_tr).max()),
                "kind": "G ~= psi*, GRADIENT-supervised"}, GG_CK)
    print(f"[Ggrad] {steps} steps in {(time.time()-t0)/60:.1f} min, "
          f"best val {hist['best_val']:.4e} -> {os.path.basename(GG_CK)}", flush=True)
    return G


def load_G(path=None):
    ck = torch.load(path or G_CK, weights_only=False)
    m = LPN(in_dim=ck["in_dim"], hidden=ck["hidden"], layers=ck["layers"], beta=ck["beta"])
    m.load_state_dict(ck["state"]); m.eval()
    return m, ck


# --------------------------------------------------------------------------
# the three learned denoisers
# --------------------------------------------------------------------------
def denoise_route1(psi, u, x):
    """u_t(Route 1) = grad psi_theta(x). Exact prox of Route 1's recovered prior."""
    return u.grad(psi, flat(x)).reshape(x.shape)


def denoise_route2(G, u, x, iters=1500, tol=1e-12):
    """u_t(Route 2) = argmin_u G_theta(u) - <u, x>, i.e. grad G(u_hat) = x.

    J_2 = G - 0.5||.||^2 is Route 2's recovered prior, so
      prox_{J_2}(x) = argmin_u J_2(u) + 0.5||u-x||^2 = argmin_u G(u) - <u,x> + const.
    Since G ~= psi* = J + 0.5||.||^2, the program is ~1-strongly convex. Warm-started
    at u = x (the answer lies within ~13% of it) and solved by L-BFGS. The
    optimality residual ||grad G(u_hat) - x|| is returned and certifies the solve
    independently of how good G is.
    """
    xf = torch.tensor(np.asarray(flat(x))).float()
    v = xf.clone().requires_grad_(True)
    # Adam, not L-BFGS: G is an ICNN fitted on range(u_PM) ~ [0,1]^64 and grows
    # only LINEARLY outside it (measured slope ~6.5 vs ||x|| ~3.8), so the
    # objective is coercive but barely; strong-Wolfe line search overshoots into
    # the flat region and overflows. This is the same reason src/invert.py drives
    # Route 1's inversion with Adam rather than a line-searched method, so using
    # Adam here keeps the two routes on the same footing.
    opt = torch.optim.Adam([v], lr=1e-2)
    prev = None
    for i in range(iters):
        opt.zero_grad()
        obj = G.scalar(v).sum() - (v * xf).sum()
        obj.backward()
        opt.step()
        cur = obj.item()
        if prev is not None and abs(cur - prev) < tol * max(1.0, abs(prev)):
            break
        prev = cur
    vv = v.detach().clone().requires_grad_(True)
    g = torch.autograd.grad(G.scalar(vv).sum(), vv)[0]
    r = ((g - xf).norm(dim=1) / xf.norm(dim=1).clamp(min=1))
    resid = {"median": float(r.median()), "max": float(r.max())}
    return v.detach().numpy().reshape(x.shape), resid


def denoise_direct(ckpt_path, x):
    """u_hat = prox_{J_theta}(x), the existing notebook method."""
    ck = torch.load(ckpt_path, weights_only=False)
    m = LPN(in_dim=ck["in_dim"], hidden=ck["hidden"], layers=ck["layers"], beta=ck["beta"])
    m.load_state_dict(ck["state"]); m.eval()
    mu, s = torch.tensor(ck["mu"]).float(), torch.tensor(ck["s"]).float()
    uh, resid = prox_of_Jtheta(m, mu, s, x)
    return uh, resid, ck


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["psi", "G", "Ggrad", "compare"])
    ap.add_argument("--steps", type=int, default=250_000)
    a = ap.parse_args()
    if a.stage == "psi":
        train_psi(a.steps)
    elif a.stage == "Ggrad":
        train_G_grad(a.steps)
    elif a.stage == "G":
        train_G(a.steps)
    else:
        from compare_impl import compare
        compare()


if __name__ == "__main__":
    main()
