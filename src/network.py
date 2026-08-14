"""Single learned proximal network (LPN) with a Softplus activation.

This is the one canonical network for all experiments (D2 convention). The
former Mish variant in ``legacy/old_notebooks/lib/network.py`` broke the input-convexity
guarantee and is deliberately not reproduced here.

The network represents a convex potential ``psi_theta`` via ``scalar`` and its
gradient (the learned proximal map) via ``forward``. Non-negativity of the
inner linear weights, enforced by ``wclip`` after each optimizer step, together
with the convex, non-decreasing Softplus activation, makes ``scalar`` an input
convex neural network (ICNN).
"""
import torch
import torch.nn as nn


def hidden_width(dim):
    """Width, FIXED at 256 for every dimension (reverted 2026-07-09).

    The manuscript's four tables all report "2 layers and 256 neurons", and all
    24 notebooks of the reported families use hidden=256, layers=2, beta=5 at
    every dimension from 2 to 64. The tiered width (d<=4: 64; {8,16}: 128;
    {32,64}: 256) was introduced by D2 on 2026-07-08 and is a DEVIATION from
    the paper, never justified by an experiment.

    It is reverted because it confounds the dimension study: N now grows with d
    (N = 15000*d) and |J| ~ 2d, so letting capacity jump at d=8 and d=32 as
    well would leave an error trend across dimensions attributable to nothing.
    With the width fixed, d is the only variable. Cost is modest: the CPU is
    overhead-bound, so width 256 is 14x the parameters of width 64 but only
    2.7x the time.

    ``dim`` is retained in the signature so callers need not change.
    """
    return 256


class LPN(nn.Module):
    def __init__(self, in_dim, hidden=None, layers=2, beta=5):
        super().__init__()
        if hidden is None:
            hidden = hidden_width(in_dim)
        self.hidden = hidden
        self.lin = nn.ModuleList(
            [
                nn.Linear(in_dim, hidden, bias=False),
                *[nn.Linear(hidden, hidden, bias=False) for _ in range(layers)],
                nn.Linear(hidden, 1, bias=False),
            ]
        )
        self.res = nn.ModuleList(
            [*[nn.Linear(in_dim, hidden) for _ in range(layers)], nn.Linear(in_dim, 1)]
        )
        self.act = nn.Softplus(beta=beta)

    def scalar(self, x):
        """Evaluate the convex potential psi_theta(x); shape (n, 1)."""
        y = self.act(self.lin[0](x))
        for core, res in zip(self.lin[1:-1], self.res[:-1]):
            y = self.act(core(y) + res(x))
        y = self.lin[-1](y) + self.res[-1](x)
        return y

    # No init_weights(). Upstream defines a log-normal positive initializer for
    # the inner weights and never calls it; we measured it (2026-07-09) and it
    # is strictly worse. The default (Kaiming) init draws ~50% of the inner
    # weights negative, and wclip pins those to exactly 0 at the first step,
    # which looks like lost capacity but is not: at d=2, psi val MSE is
    # 1.10e-3 (default) vs 6.25e-3 (|default|, same magnitudes, no negatives)
    # vs 9.51e-2 (log-normal). The sign-clamping acts as useful sparsification.

    def wclip(self):
        """Clip the inner ('lin') weights to be non-negative to keep convexity."""
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.clamp_(0)

    def forward(self, x, create_graph=False):
        """Return grad_x psi_theta(x), the learned proximal map; shape (n, in_dim).

        ``create_graph=True`` is needed ONLY to differentiate through the proximal
        map (a second-order quantity). Nothing in this codebase does: training
        regresses ``scalar``, and every caller of ``forward`` (conjugate sampling,
        prox residuals, one-shot-recovery preimages) consumes the gradient as a value. It
        used to be unconditionally True, which built and retained a second-order
        graph on every call.

        Does not mutate ``x``: an input that does not already require grad is
        detached into a fresh leaf, so callers never see their tensor's
        ``requires_grad`` flipped underneath them.
        """
        with torch.enable_grad():
            xin = x if x.requires_grad else x.detach().requires_grad_(True)
            y = self.scalar(xin)
            grad = torch.autograd.grad(
                y.sum(), xin, create_graph=create_graph
            )[0]
        return grad if create_graph else grad.detach()
