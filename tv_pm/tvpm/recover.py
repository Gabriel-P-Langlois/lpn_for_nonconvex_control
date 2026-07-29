"""Steps 2-3: fit one convex net to the denoiser's gradients, and score it.

    ~/miniforge3/envs/lpn_env/bin/python -m tvpm.recover                     # ~10 min, from tv_pm/
    ~/miniforge3/envs/lpn_env/bin/python -m tvpm.recover --sweeps 2000

Reads only the cached splits from `dataset.py`; never samples. Lifted from
`../posterior_mean_l1.ipynb` (`train_grad` and its scoring) with the mathematics
unchanged, so the loop is importable and testable and the notebook stays thin.
The l1 notebook keeps its inline copy: it is executed and frozen, and the two are
separate experiments.

WHAT IS BEING FIT. One convex net J_theta, gradient-only, no psi, no inversion, no
conjugation:

    loss = MSE(grad J_theta(y_k), x_k - y_k) / var(x - y)

Each sample supplies n = 64 constraints, not the 1 a value loss would: ~1.3M
constraints against ~150k parameters.

WHY A LOSS WITH NO GROUND TRUTH IN IT IS STILL FALSIFIABLE. There is no ground
truth for f_reg -- evaluating it needs S_eps, the log-partition function, and the
sampler returns the posterior MEAN, not the normalizing constant. But the denoiser
is a ground truth and pins f_reg hard: if prox_f = prox_g then grad f = grad g on
the range of the prox, so f = g + c there. A PROX DETERMINES ITS REGULARIZER UP TO
A CONSTANT on range(u_PM), which is connected here. So a held-out prox residual is
not a validation loss -- it is a theorem's hypothesis, and the constant is the only
thing left unidentified (and prox_{f+c} = prox_f, so it does not matter).

THE FLOOR. The check has a known floor: the sampler's own noise delta. Read the
residual against it (DESIGN.md):

    ~ delta   the net extracted everything the data contains
    >> delta  the net failed -- capacity, budget, or a bug
    << delta  impossible; suspect a leak or a self-referential score

NORMALIZATION, and it is not a detail. delta is measured RELATIVE TO THE TARGET,
||noise|| / ||x-y||. The repo's D2 residual is ||grad J(y) - (x-y)|| / max(1,||x||).
Here ||x|| ~ 4 and ||x-y|| ~ 0.3, so the two differ by more than an order of
magnitude and comparing them against each other would be meaningless. Both are
reported; ONLY `resid_rel` may be read against delta.
"""
import argparse
import os
import time

import numpy as np
import torch

from . import dataset
from .icnn import ConvICNN
from .paths import LOG_CKPT, ROOT, find_ckpt
from .sampler import from_sigma_t, sample_pm
from src.network import LPN            # noqa: E402  (paths.py puts ROOT on sys.path)

WIDTH, LAYERS, BETA = 256, 2, 5      # D2's network, unchanged
BATCH, LR, STEPS = 512, 1e-3, 50_000


def train_grad(model, X, G, Xv, Gv, batch_size=BATCH, steps=STEPS, lr=LR,
               lr_decay_at=(0.5, 0.75), eval_every=500, seed=1, quiet=False):
    """Fit a convex net to observed gradients: loss = MSE(grad net(y_k), G) / var(G).

    Normalized by the target variance so the loss reads the same at any scale.
    The input-gradient is in the loss, so it is built with create_graph=True and
    then differentiated w.r.t. the parameters (double backprop).

    No early stopping (D2): we under-fit, not overfit. The best-validation
    CHECKPOINT is kept, which costs nothing and makes the reported val loss
    describe the model actually returned.
    """
    dev = next(model.parameters()).device
    t_ = lambda a: torch.tensor(np.asarray(a)).float().to(dev)
    X, Xv, G, Gv = t_(X), t_(Xv), t_(G), t_(Gv)
    s = float(G.var().item())
    n = X.shape[0]

    def loss_at(xb, gb, create_graph):
        xin = xb.detach().requires_grad_(True)
        grad = torch.autograd.grad(model.scalar(xin).sum(), xin, create_graph=create_graph)[0]
        return (grad - gb).pow(2).mean() / s

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    decay = {int(f * steps) for f in lr_decay_at}
    gen = torch.Generator(device="cpu").manual_seed(seed)
    hist = {"train": [], "val": [], "steps": [], "best_val": None}
    best, best_state = np.inf, None
    step, running, seen = 0, 0.0, 0
    perm, cursor = torch.randperm(n, generator=gen).to(dev), 0

    while step < steps:
        if cursor + batch_size > n:                       # reshuffle on epoch exhaustion
            perm, cursor = torch.randperm(n, generator=gen).to(dev), 0
        idx = perm[cursor:cursor + batch_size]
        cursor += batch_size
        if step in decay and step > 0:
            for gp in opt.param_groups:
                gp["lr"] *= 0.1

        opt.zero_grad()
        loss = loss_at(X[idx], G[idx], create_graph=True)
        loss.backward()
        opt.step()
        model.wclip()                                      # convexity, every step
        running += loss.item() * idx.shape[0]
        seen += idx.shape[0]
        step += 1

        # ALWAYS evaluate on the last step, not only on the eval_every grid: a
        # budget below eval_every (a smoke run) would otherwise never validate,
        # leaving best_val None and the returned net unchecked -- which breaks
        # the invariant that we return the best-validation checkpoint. At the
        # production budgets steps is a multiple of eval_every, so this is a
        # no-op there.
        if step % eval_every == 0 or step == steps:
            vl = loss_at(Xv, Gv, create_graph=False).item()
            hist["train"].append(running / seen); hist["val"].append(vl)
            hist["steps"].append(step); running, seen = 0.0, 0
            if vl < best - 1e-12:
                best = vl
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if not quiet and step % (eval_every * 20) == 0:
                print(f"    step {step:6d}  train {hist['train'][-1]:.4e}  val {vl:.4e}")

    if best_state is not None:
        model.load_state_dict(best_state)
        hist["best_val"] = float(best)
    return hist


def net_value(model, y):
    with torch.no_grad():
        return model.scalar(torch.tensor(np.asarray(y)).float()).cpu().numpy().ravel()


def net_grad(model, y):
    yb = torch.tensor(np.asarray(y)).float().requires_grad_(True)
    return torch.autograd.grad(model.scalar(yb).sum(), yb)[0].detach().cpu().numpy()


def atv(u):
    """Anisotropic TV of each patch, the sampler's ATV: unordered 4-neighbour pairs."""
    return (np.abs(np.diff(u, axis=1)).sum(axis=(1, 2)) +
            np.abs(np.diff(u, axis=2)).sum(axis=(1, 2)))


def estimate_delta(x, sweeps, reps=8, n_x=16, seed=77,
                   sigma=dataset.SIGMA, t=dataset.T):
    """The floor: relative noise on the target x - y, from the SPREAD of `reps`
    independent chains per x.

    Measured on the actual evaluation patches rather than carried over from
    `noise_diagnostic.ipynb`, so the floor and the residual describe the same
    data. Returns the mean over x of std(x-y)/||x-y||, i.e. what one chain's
    target is worth -- exactly the quantity the residual must be read against.
    """
    xs = x[:n_x]
    xb = np.repeat(xs[:, None], reps, axis=1).reshape(-1, *xs.shape[1:])
    _, lam = from_sigma_t(sigma, t)
    u = sample_pm(xb, sigma, lam, sweeps=sweeps, w=1.0, seed=seed)["u_pm"]
    tgt = xs[:, None] - u.reshape(n_x, reps, *xs.shape[1:])       # (n_x, reps, H, W)
    spread = tgt.std(axis=1, ddof=1)                              # per-pixel, over chains
    mag = np.linalg.norm(tgt.mean(axis=1).reshape(n_x, -1), axis=1)
    return float(np.mean(np.linalg.norm(spread.reshape(n_x, -1), axis=1) / mag))


class Units:
    """Choice of input units: z = (y - mu)/s. Identity unless `standardize`.

    NOT cosmetic, and not a modelling choice -- it is a change of units, and the
    only reason it is optional is that it is on trial. J(y) = J_tilde((y-mu)/s) is
    convex in y exactly when J_tilde is convex in z (affine precomposition), so
    the ICNN guarantee is untouched, and the chain rule gives

        grad_y J = grad_z J_tilde / s,      target in z-units = s * (x - y)

    WHY IT MIGHT BE NEEDED HERE and nowhere else in the repo: D2's families feed
    the net points uniform on [-4,4]^d. These y are 8x8 natural-image patches, and
    they vary by only std ~0.039 WITHIN a patch while the patch mean ranges over
    0.064..0.921. Softplus(beta=5) bends over ~1/beta = 0.2 of its argument, so at
    Kaiming init the informative directions move the pre-activation by ~5x LESS
    than the activation's transition width: the net is near-linear in exactly the
    directions carrying the signal.

    GLOBAL (glob=True) is required by the conv-ICNN: per-pixel mu, s would give a
    different offset at every site and BREAK the shift-invariance the convolution
    is built on. It costs nothing here -- s came out 0.197..0.200 uniformly across
    all 64 pixels, so a single scalar is numerically what per-pixel already was,
    which is also why the fully-connected baseline is unaffected by the choice.
    """

    def __init__(self, y, standardize, glob=False):
        if standardize and glob:
            self.mu = np.full(y.shape[1], float(y.mean()))
            self.s = np.full(y.shape[1], float(y.std()) + 1e-8)
        elif standardize:
            self.mu = y.mean(axis=0)
            self.s = y.std(axis=0) + 1e-8
        else:
            self.mu, self.s = np.zeros(y.shape[1]), np.ones(y.shape[1])

    @classmethod
    def from_saved(cls, mu, s):
        """Rebuild the units from a checkpoint's stored mu, s (no data needed)."""
        obj = cls.__new__(cls)
        obj.mu, obj.s = np.asarray(mu), np.asarray(s)
        return obj

    def z(self, y):
        return (y - self.mu) / self.s

    def target(self, g):
        return g * self.s                      # grad in z-units

    def grad(self, model, y):
        return net_grad(model, self.z(y)) / self.s      # back to y-units


def score(model, d, u, delta=None):
    """Held-out scoring, in the ORIGINAL units. No ground truth is used or exists."""
    y, g = d["y"], d["g"]
    n = y.shape[0]
    yf, gf = y.reshape(n, -1), g.reshape(n, -1)

    r = np.linalg.norm(u.grad(model, yf) - gf, axis=1)
    # Relative to the TARGET: the only normalization comparable to delta.
    out = {"resid_rel": float(np.median(r / np.linalg.norm(gf, axis=1)))}
    # J_theta vs TV: strong correlation expected, but NOT equality -- f_reg is a
    # SMOOTHED TV and the gap is the point (no staircasing).
    out["tv_corr"] = float(np.corrcoef(net_value(model, u.z(yf)), atv(y))[0, 1])
    if delta is not None:
        out["delta"] = delta
        out["ratio"] = out["resid_rel"] / delta
    return out


def build_model(arch, n, width, layers, beta, channels, blocks):
    """The fitter. Both expose scalar(flat_x)->(N,1) and wclip(), so everything
    downstream (train_grad, net_grad, score) is architecture-agnostic."""
    if arch == "fc":
        return LPN(in_dim=n, hidden=width, layers=layers, beta=beta)
    if arch == "conv":
        hw = int(round(n ** 0.5))
        assert hw * hw == n, f"conv-ICNN needs a square patch; n={n} is not a square"
        return ConvICNN(hw=(hw, hw), channels=channels, blocks=blocks, beta=beta)
    raise ValueError(f"unknown arch {arch!r}")


def checkpoint_name(n, arch, sweeps, width, layers, channels, blocks, beta, steps,
                    sigma, t, legacy=False):
    """The checkpoint filename for one configuration.

    Every hyperparameter that changes the trained net is IN the name, so runs
    that settle a choice can proceed concurrently without clobbering each other.

    `legacy=True` returns the scheme the SHIPPED fc checkpoint was written under
    (no `_fc_`, no `_L2`). It has to stay reachable: without it a default run
    would fail to find `results/ckpt/tv_pm_64D_m8000_w256_b20.0_s250000.pth`
    and retrain for ~2 h to reproduce a net that is already on disk. The conv
    scheme never changed, so its legacy and current names coincide.
    """
    tag = (f"w{width}_L{layers}" if arch == "fc" else f"C{channels}_B{blocks}")
    stem = (f"tv_pm_{n}D_m{sweeps}_w{width}" if legacy and arch == "fc"
            else f"tv_pm_{n}D_{arch}_m{sweeps}_{tag}")
    return f"{stem}_b{float(beta)}_s{steps}{dataset.tag(sigma, t)}.pth"


def load_checkpoint(path):
    """Rebuild (model, Units) from a saved checkpoint -- no data, no training.
    Lets a notebook score/plot a shipped net without the ~2-13 h retrain."""
    ck = torch.load(path, weights_only=False)
    model = build_model(ck.get("arch", "fc"), ck["in_dim"], ck.get("hidden", WIDTH),
                        ck.get("layers", LAYERS), ck["beta"],
                        ck.get("channels", 32), ck.get("blocks", 2))
    model.load_state_dict(ck["state"]); model.eval()
    return model, Units.from_saved(ck["mu"], ck["s"]), ck


def find_checkpoint(n=64, arch="fc", sweeps=dataset.SWEEPS, width=WIDTH,
                    layers=LAYERS, channels=32, blocks=2, beta=BETA, steps=STEPS,
                    sigma=dataset.SIGMA, t=dataset.T):
    """Path of an existing checkpoint for this configuration, or None.

    Tries the current naming scheme, then the legacy one, in both checkpoint
    directories.
    """
    for legacy in (False, True):
        name = checkpoint_name(n, arch, sweeps, width, layers, channels, blocks,
                               beta, steps, sigma, t, legacy=legacy)
        p = find_ckpt(name)
        if os.path.exists(p):
            return p
    return None


def ensure_model(n=64, arch="fc", sweeps=dataset.SWEEPS, width=WIDTH,
                 layers=LAYERS, channels=32, blocks=2, beta=BETA, steps=STEPS,
                 sigma=dataset.SIGMA, t=dataset.T, standardize=True, seed=1,
                 scale=1.0, force=False, quiet=False):
    """Return (model, units, meta) for this configuration, training if needed.

    The training counterpart of dataset.ensure(): a cached checkpoint is loaded
    in seconds, an absent one is trained once and saved under a name carrying
    every hyperparameter, so the next call finds it. Callers never have to
    declare whether they intend to train -- the configuration decides.
    """
    existing = find_checkpoint(n, arch, sweeps, width, layers,
                               channels, blocks, beta, steps, sigma, t)
    p = None if force else existing
    if p is None:
        if not quiet:
            # Distinguish a forced retrain (a checkpoint exists but force=True
            # asked us to overwrite it) from a genuinely missing one, so the
            # message never contradicts what the plan cell reported.
            why = ("forced retrain (force=True), overwriting the cached checkpoint"
                   if existing is not None else "no checkpoint for this configuration")
            print(f"  {why} -> training ({steps} steps, arch={arch})")
        run(sweeps=sweeps, steps=steps, width=width, seed=seed, save=True,
            standardize=standardize, beta=beta, layers=layers, arch=arch,
            channels=channels, blocks=blocks, sigma=sigma, t=t, scale=scale)
        p = find_checkpoint(n, arch, sweeps, width, layers, channels, blocks,
                            beta, steps, sigma, t)
        if p is None:
            raise RuntimeError("training finished but no checkpoint was written")
    elif not quiet:
        print(f"  cached checkpoint: {os.path.relpath(p, ROOT)}")
    return load_checkpoint(p)


def run(sweeps=dataset.SWEEPS, steps=STEPS, width=WIDTH, seed=1, save=True,
        standardize=False, n_train=None, beta=BETA, layers=LAYERS,
        arch="fc", channels=32, blocks=2, sigma=dataset.SIGMA, t=dataset.T,
        scale=1.0):
    eps, lam = from_sigma_t(sigma, t)
    tr, va, ev = dataset.ensure(sweeps=sweeps, sigma=sigma, t=t, scale=scale)
    flat = lambda a: a.reshape(a.shape[0], -1)
    ytr, gtr = flat(tr["y"]), flat(tr["g"])
    if n_train:                                  # subsample: tests N, samples nothing
        ytr, gtr = ytr[:n_train], gtr[:n_train]
    n = ytr.shape[1]
    u = Units(ytr, standardize, glob=(arch == "conv"))   # conv needs GLOBAL units

    torch.manual_seed(seed)
    model = build_model(arch, n, width, layers, beta, channels, blocks)
    spec = (f"width={width}, layers={layers}" if arch == "fc"
            else f"channels={channels}, blocks={blocks}")
    print(f"  arch={arch}, n={n}, N={ytr.shape[0]}, "
          f"{sum(p.numel() for p in model.parameters())} parameters, {steps} steps, "
          f"{spec}, beta={beta}, standardize={standardize}{' (global)' if arch=='conv' else ''}")
    print(f"  sigma={sigma:.5f}, t={t:.5f} -> eps={eps:.5f}, lam={lam:.5f}")

    t0 = time.time()
    hist = train_grad(model, u.z(ytr), u.target(gtr),
                      u.z(flat(va["y"])), u.target(flat(va["g"])), steps=steps, seed=seed)
    # best_val can still be None if EVERY validation loss was NaN (nothing ever
    # beat inf). Say so rather than dying in the format string.
    bv = hist["best_val"]
    print(f"  trained in {(time.time()-t0)/60:.1f} min; best val loss "
          + (f"{bv:.4e}" if bv is not None else "n/a (no validation improved -- NaN?)"))

    delta = estimate_delta(ev["x"], sweeps, sigma=sigma, t=t)
    s_val = score(model, va, u, delta)
    s_ev = score(model, ev, u, delta)

    print(f"\n  sampler noise floor delta ({sweeps} sweeps, measured on eval patches)"
          f" : {100*delta:.2f} %")
    for name, s in (("val (Barbara)", s_val), ("eval (cameraman)", s_ev)):
        print(f"  {name:18s} resid/|x-y| {100*s['resid_rel']:6.2f} %  "
              f"= {s['ratio']:5.2f} x delta   |  corr(J,TV) {s['tv_corr']:+.3f}")

    if save:
        # arch and hyperparameters are IN the name: sweeps that settle them run
        # concurrently, and a shared name would have them clobber each other.
        # Built by checkpoint_name() -- the SAME function find_checkpoint() reads
        # with, so the two cannot drift (e.g. b20 vs b20.0 for an int beta).
        ck = os.path.join(LOG_CKPT,
                          checkpoint_name(n, arch, sweeps, width, layers,
                                          channels, blocks, beta, steps, sigma, t))
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        torch.save({"state": model.state_dict(), "arch": arch, "in_dim": n,
                    "hidden": width, "layers": layers, "channels": channels,
                    "blocks": blocks, "beta": beta, "sweeps": sweeps,
                    "steps": steps, "mu": u.mu,
                    "s": u.s, "sigma": sigma, "t": t, "eps": eps, "lam": lam}, ck)
        print(f"  -> {ck}")
    return model, hist, {"val": s_val, "eval": s_ev, "delta": delta}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweeps", type=int, default=dataset.SWEEPS,
                    help="which cached m to train on (the bias sweep)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--standardize", action="store_true",
                    help="z = (y-mu)/s; a change of units, see Units")
    ap.add_argument("--n-train", type=int, default=None,
                    help="subsample the cached train split; tests N, samples nothing")
    ap.add_argument("--beta", type=float, default=BETA,
                    help="Softplus sharpness; higher = sharper (-> ReLU). f_reg's "
                         "TV kink is sharp, so beta=5 may be too smooth to resolve it")
    ap.add_argument("--layers", type=int, default=LAYERS,
                    help="fc only: ICNN hidden core layers (depth); D2 default is 2")
    ap.add_argument("--arch", choices=("fc", "conv"), default="fc",
                    help="fc = dense ICNN (LPN); conv = convolutional ICNN "
                         "(locality + shift-invariance for the TV prior)")
    ap.add_argument("--channels", type=int, default=32, help="conv only: feature channels")
    ap.add_argument("--blocks", type=int, default=2, help="conv only: hidden conv blocks")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--sigma", type=float, default=dataset.SIGMA,
                    help="noise level; must match a cached dataset at this (sigma, t)")
    ap.add_argument("--t", type=float, default=dataset.T,
                    help="denoising strength; eps=sigma^2/t, lam=2t")
    a = ap.parse_args()
    run(sweeps=a.sweeps, steps=a.steps, width=a.width, save=not a.no_save,
        standardize=a.standardize, n_train=a.n_train, beta=a.beta, layers=a.layers,
        arch=a.arch, channels=a.channels, blocks=a.blocks, sigma=a.sigma, t=a.t)


if __name__ == "__main__":
    main()
