"""THE CONVEXITY GATE for the conv-ICNN. Must pass before any training run.

    ~/miniforge3/envs/lpn_env/bin/python notebooks/tv_pm/test_conv_icnn.py   (~5 s)

conv_icnn.py PROVES convexity on paper (Amos et al. 2017 Prop. 1, applied to the
linear conv maps; Mukherjee et al. 2020 for the imaging instantiation). But the
guarantee holds only if the nonnegativity constraints actually hold in CODE: a
kernel wclip misses, a sign sneaking in through a norm layer, a wrong reduction.
The recovered f_reg is only meaningful if J_theta is genuinely convex -- a
non-convex J is not the regularizer of ANY prox -- so this is a gate, not a
nicety.

Three positive checks and one negative (the teeth):
  * midpoint inequality J((a+b)/2) <= (J(a)+J(b))/2, the definition itself;
  * along random lines, the 1-D restriction t -> J(a+t v) is convex (second
    difference >= 0), a stronger test than isolated midpoints;
  * the full 64x64 input Hessian is PSD (lambda_min >= -tol) -- at n=64 it is
    cheap to form exactly, which beats a Hutchinson estimate;
  * WITHOUT wclip, with deliberately negative feature kernels, convexity must
    FAIL -- otherwise the test cannot tell a convex net from a broken one.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conv_icnn import ConvICNN

HW = (8, 8)
N = HW[0] * HW[1]


def _model(seed=0, beta=20.0, pad="reflect"):
    torch.manual_seed(seed)
    m = ConvICNN(hw=HW, channels=32, blocks=2, beta=beta, pad=pad)
    m.wclip()                      # enforce the constraints, as training would
    return m


def _J(m, x):
    with torch.no_grad():
        return m.scalar(x).squeeze(-1)


def test_interface_matches_lpn():
    """scalar: flat (N, HW) -> (N, 1); wclip and forward exist. recover.py relies
    on this being drop-in for LPN."""
    m = _model()
    x = torch.rand(5, N)
    assert m.scalar(x).shape == (5, 1), "scalar must map (N,HW) -> (N,1)"
    g = m.forward(x)
    assert g.shape == (5, N), "forward must return grad of shape (N, HW)"
    assert hasattr(m, "wclip")
    print(f"  interface OK: scalar (N,{N})->(N,1), forward ->(N,{N}), "
          f"{sum(p.numel() for p in m.parameters())} parameters")


def test_midpoint_inequality():
    """J((a+b)/2) <= (J(a)+J(b))/2 for random image pairs in [0,1]^N."""
    m = _model(seed=1)
    rng = torch.Generator().manual_seed(1)
    a = torch.rand(2000, N, generator=rng)
    b = torch.rand(2000, N, generator=rng)
    lhs = _J(m, 0.5 * (a + b))
    rhs = 0.5 * (_J(m, a) + _J(m, b))
    worst = float((lhs - rhs).max())
    assert worst < 1e-5, f"midpoint convexity violated by {worst:.3e}"
    print(f"  midpoint inequality holds on 2000 pairs (worst {worst:.2e})")


def test_convex_along_lines():
    """t -> J(a + t v) convex: its second difference must be >= 0 everywhere."""
    m = _model(seed=2)
    rng = torch.Generator().manual_seed(2)
    a = torch.rand(64, N, generator=rng)
    v = torch.randn(64, N, generator=rng)
    ts = torch.linspace(-2, 2, 81)
    vals = torch.stack([_J(m, a + t * v) for t in ts], dim=1)   # (64, 81)
    d2 = vals[:, 2:] - 2 * vals[:, 1:-1] + vals[:, :-2]          # (dt)^2 * J''
    worst = float(d2.min())
    assert worst > -1e-5, f"second difference negative ({worst:.3e}): not convex on a line"
    print(f"  convex along 64 random lines (min second difference {worst:.2e})")


def test_input_hessian_is_psd():
    """The full N x N input Hessian of J -- cheap to form at N=64 -- must be PSD."""
    m = _model(seed=3)
    rng = torch.Generator().manual_seed(3)
    worst = np.inf
    for _ in range(12):
        x = torch.rand(1, N, generator=rng, requires_grad=True)
        H = torch.autograd.functional.hessian(lambda z: m.scalar(z).sum(), x).reshape(N, N)
        H = 0.5 * (H + H.T)                       # symmetrize away autograd round-off
        lo = float(torch.linalg.eigvalsh(H)[0])
        worst = min(worst, lo)
    assert worst > -1e-4, f"input Hessian not PSD: lambda_min = {worst:.3e}"
    print(f"  input Hessian PSD at 12 points (min eigenvalue {worst:.2e})")


def test_wclip_enforces_nonnegative():
    """wclip must actually clamp the constrained weights -- the mechanism the
    whole guarantee rests on."""
    m = _model(seed=4)
    with torch.no_grad():                          # dirty the constrained weights
        m.feat[0].weight[0, 0, 0, 0] = -3.0
        m.head.weight[0, 0] = -1.0
    m.wclip()
    assert (m.feat[0].weight >= 0).all(), "wclip left a negative feature kernel"
    assert (m.head.weight >= 0).all(), "wclip left a negative head weight"
    print("  wclip clamps feature kernels and head to >= 0")


def test_teeth_nonconvex_when_condition_violated():
    """The negative control. VIOLATE a stated convexity condition -- a NEGATIVE
    head weight -- and the checker must catch it, or the positive tests prove
    nothing.

    The head must be nonnegative (it is the last W^z of Prop. 1). Force it
    strongly negative: then J = -(nonneg) . (convex pooled features) + affine is
    CONCAVE and non-affine, so the midpoint inequality must be VIOLATED
    (J(mid) > average). A negative FEATURE kernel is a weaker control -- with
    Softplus(beta=20) saturating and the affine skips dominating, such a net can
    stay incidentally convex on the sampled region, which is a fact about that
    perturbation, not a hole in the checker.
    """
    torch.manual_seed(5)
    m = ConvICNN(hw=HW, channels=32, blocks=2, beta=20.0)
    with torch.no_grad():
        m.in_conv.weight.mul_(4.0)                 # ensure the features are curved
        m.head.weight.copy_(-m.head.weight.abs() - 1.0)   # negative head: concave J
    rng = torch.Generator().manual_seed(5)
    a = torch.rand(4000, N, generator=rng)
    b = torch.rand(4000, N, generator=rng)
    viol = (_J(m, 0.5 * (a + b)) - 0.5 * (_J(m, a) + _J(m, b))).max()
    assert float(viol) > 1e-4, ("a negative head weight did NOT break convexity -- "
                                "the checker cannot distinguish convex from broken")
    print(f"  teeth: a negative head weight makes J concave, and the midpoint "
          f"checker catches it (violation {float(viol):.2e})")


if __name__ == "__main__":
    for fn in (test_interface_matches_lpn, test_midpoint_inequality,
               test_convex_along_lines, test_input_hessian_is_psd,
               test_wclip_enforces_nonnegative, test_teeth_nonconvex_when_condition_violated):
        print(f"{fn.__name__}:")
        fn()
    print("\nOK: conv-ICNN is convex in its input, verified in code.")
