"""Inversion of the learned proximal map (LPN Iterative recovery support).

We invert psi_theta by solving the convex program  min_x psi_theta(x) - <z, x>,
whose stationarity condition is grad psi_theta(x) = z. Two convex solvers are
provided: gradient descent (``cvx_gd``, the default) and SciPy conjugate
gradient (``cvx_cg``).

The least-squares inverter ``invert_ls`` from the old ``lib/`` is intentionally
removed (Phase 1, finding 11'): min_y ||grad psi(y) - z||^2 is nonconvex and
degenerate on the affine pieces of piecewise-linear priors.
"""
import numpy as np
import torch


def _objective(x, model, z, alpha=0.0):
    """psi_theta(x) + (alpha/2)||x||^2 - <z, x>, summed over the batch.

    ``alpha`` is the strong-convexity regularizer of Fang et al.'s inversion
    (upstream bakes it into the network: lpn_128.py adds alpha*||x||^2 to the
    potential; our paper writes it in the objective). It is NOT free: at a
    minimizer, grad psi(x) = z - alpha*x, so the recovered object is the prox of
    a DIFFERENT prior, biased at order alpha. With alpha = 0 the objective need
    not be coercive and descent can run away; with alpha > 0 it always converges
    and silently returns a distorted answer. Neither setting is "safe"; the
    prox residual reported below is what exposes the trade -- it equals exactly
    alpha*||y|| at an exact alpha-regularized solution.
    """
    b = x.shape[0]
    quad = 0.5 * alpha * x.reshape(b, -1).pow(2).sum() if alpha else 0.0
    return (
        model.scalar(x).squeeze(1).sum()
        + quad
        - torch.sum(z.reshape(b, -1) * x.reshape(b, -1))
    )


def invert_cvx_gd(
    x, model, max_iters=20000, lr=1e-3, tol=1e-8, alpha=0.0, verbose=False
):
    """Invert by Adam on psi(y) + (alpha/2)||y||^2 - <x, y>.

    NO box projection and no other safeguard: LPN Iterative recovery must be run exactly as
    Fang et al. specify it, with alpha its only knob. Constraining the iterate
    to the training box would give LPN Iterative recovery information about where psi_theta is
    valid that One-shot recovery never receives, which is preferential treatment, not a
    fair baseline. Whatever this solver returns is what the method returns; the
    prox residual (see src.recovery.prox_residual) says how much to trust it,
    and the SAME residual is computed for One-shot recovery.

    Stops when the batch objective stops changing; that is a shared criterion,
    so per-point quality must be read from the residual, not from stopping.
    """
    device = next(model.parameters()).device
    z = torch.tensor(np.asarray(x)).float().to(device)
    xv = torch.zeros_like(z, requires_grad=True)
    optimizer = torch.optim.Adam([xv], lr=lr)

    prev = None
    for i in range(max_iters):
        optimizer.zero_grad()
        loss = _objective(xv, model, z, alpha)
        loss.backward()
        optimizer.step()
        cur = loss.item()
        if prev is not None and abs(cur - prev) < tol * max(1.0, abs(prev)):
            break
        prev = cur
    if verbose:
        print(f"[invert_cvx_gd] stopped at iter {i + 1} (alpha={alpha})")
    return xv.detach().cpu().numpy()


def invert_cvx_cg(x, model):
    """Invert by SciPy conjugate gradient, one point at a time."""
    from scipy.optimize import fmin_cg

    device = next(model.parameters()).device
    x = np.asarray(x)

    def f(xi, z):
        xt = torch.tensor(xi).view(1, -1).float().to(device)
        zt = torch.tensor(z).view(1, -1).float().to(device)
        return _objective(xt, model, zt).item()

    def gradf(xi, z):
        xt = torch.tensor(xi).view(1, -1).float().to(device)
        xt.requires_grad_(True)
        zt = torch.tensor(z).view(1, -1).float().to(device)
        v = _objective(xt, model, zt)
        v.backward()
        return xt.grad.cpu().numpy().flatten()

    y = np.zeros(x.shape)
    for i in range(x.shape[0]):
        z = x[i]
        y[i] = fmin_cg(f, z.copy().flatten(), fprime=gradf, args=(z,), disp=0).reshape(
            z.shape
        )
    return y


def invert(x, model, inv_alg="cvx_gd", **kwargs):
    """Dispatch to a convex inverter. ``inv_alg`` in {'cvx_gd', 'cvx_cg'}.

    Only ``cvx_gd`` supports the box projection and the convergence report.
    """
    if inv_alg == "cvx_gd":
        return invert_cvx_gd(x, model, **kwargs)
    if inv_alg == "cvx_cg":
        if kwargs.get("alpha"):
            raise ValueError("cvx_cg does not implement the alpha regularizer")
        return invert_cvx_cg(x, model)
    raise ValueError(f"Unknown or retired inversion algorithm: {inv_alg!r}")
