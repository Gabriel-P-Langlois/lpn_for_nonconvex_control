"""Gradient-supervised fitting of a convex scalar network, plus its helpers.

Shared by every experiment that observes gradients of a potential rather than
its values (tv_pm, pnp_reg). The model is duck-typed: it must expose
``scalar(x) -> (N, 1)``, ``wclip()``, and the usual ``parameters()`` /
``state_dict()`` / ``load_state_dict()``. Both ``src.network.LPN`` and
``src.conv_icnn.ConvICNN`` satisfy this.

Moved verbatim from ``tv_pm/tvpm/recover.py`` on 2026-07-29 (changes.txt C16);
the mathematics is unchanged. History before the move: lifted from the l1
posterior-mean notebook with the mathematics unchanged.
"""
import numpy as np
import torch

BATCH, LR, STEPS = 512, 1e-3, 50_000


def train_grad(model, X, G, Xv, Gv, batch_size=BATCH, steps=STEPS, lr=LR,
               lr_decay_at=(0.5, 0.75), eval_every=500, seed=1, quiet=False):
    """Fit a convex net to observed gradients: loss = MSE(grad net(y_k), G) / var(G).

    Normalized by the target variance so the loss reads the same at any scale.
    The input-gradient is in the loss, so it is built with create_graph=True and
    then differentiated w.r.t. the parameters (double backprop).

    No early stopping (repository protocol): we under-fit, not overfit. The best-validation
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


class Units:
    """Choice of input units: z = (y - mu)/s. Identity unless `standardize`.

    NOT cosmetic, and not a modelling choice -- it is a change of units.
    J(y) = J_tilde((y-mu)/s) is convex in y exactly when J_tilde is convex in z
    (affine precomposition), so the ICNN guarantee is untouched, and the chain
    rule gives

        grad_y J = grad_z J_tilde / s,      target in z-units = s * (x - y)

    WHEN IT IS NEEDED: when the inputs concentrate on a thin, shifted set at a
    scale the activation cannot resolve. Softplus(beta) bends over ~1/beta of
    its argument; if the informative directions move the pre-activation by much
    less than that, the net is near-linear exactly where the signal is (this is
    what happened to tv_pm's natural-image patches, std ~0.039 within a patch).

    GLOBAL (glob=True) is required by the conv-ICNN: per-pixel mu, s would give
    a different offset at every site and BREAK the shift-invariance the
    convolution is built on.
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
