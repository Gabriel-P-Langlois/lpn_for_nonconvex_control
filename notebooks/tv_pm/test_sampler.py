"""Pins the batched TV posterior-mean sampler.

    ~/miniforge3/envs/lpn_env/bin/python notebooks/tv_pm/test_sampler.py

The sampler is the ground truth AND the data source for the TV example: there is
no closed-form u_PM to check it against at w = 1, and no exact f_reg at all. So
a bug here is unfalsifiable downstream -- it would corrupt the targets and the
certificate together. Two independent checks bracket it:

  * the TV term, via the incremental dE the MH step uses vs `energy` from
    scratch -- exact, deterministic, no sampling;
  * everything else, via w = 0, where the posterior collapses to independent
    truncated Gaussians with a closed-form mean.

Between them every line of the kernel is covered.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sampler import (_delta_energy, _shift, checkerboard, energy, params,
                     pm_truncated_gaussian, sample_pm)


def test_neighbours_do_not_wrap():
    """Boundaries must not be periodic: the MATLAB guards every neighbour with
    i>1 / i<size, so an edge pixel has 3 neighbours and a corner 2."""
    u = np.arange(9, dtype=float).reshape(1, 3, 3)
    nb, valid = _shift(u, 1, +1)          # neighbour above
    assert not valid[0, :].any(), "top row must have no neighbour above"
    assert valid[1:, :].all()
    assert np.allclose(nb[0, 1:, :], u[0, :-1, :]), "shift is misaligned"

    # Total incident edges over a 3x3 lattice = 2*3*2 = 12 unordered pairs.
    count = np.zeros((1, 3, 3))
    for axis, off in ((1, +1), (2, +1)):
        _, v = _shift(count, axis, off)
        count += v
    assert count.sum() == 12, f"wrong edge count {count.sum()}, expected 12"
    print("  no wraparound; 3x3 lattice has 12 unordered neighbour pairs")


def test_atv_matches_matlab_definition():
    """ATV = sum over unordered pairs. The MATLAB sums each pixel's neighbours
    then halves, which is the same thing; check on a hand-computable case."""
    u = np.array([[[0.0, 1.0], [1.0, 0.0]]])      # 2x2 checkerboard
    # edges: (00,01)=1, (10,11)=1, (00,10)=1, (01,11)=1  -> ATV = 4
    tv = energy(u, np.zeros_like(u), t=1e12, w=1.0)[0]   # kill the quadratic
    assert abs(tv - 4.0) < 1e-9, f"ATV = {tv}, expected 4"

    u = np.array([[[0.2, 0.2], [0.2, 0.2]]])      # constant -> ATV = 0
    assert abs(energy(u, np.zeros_like(u), t=1e12, w=1.0)[0]) < 1e-12
    print("  ATV counts each unordered pair once (checkerboard 2x2 -> 4)")


def test_delta_energy_is_exact():
    """The MH step's incremental dE must equal a from-scratch energy difference.

    Only valid one colour at a time -- that IS the conditional independence the
    checkerboard scan relies on, so this test also pins the scan's correctness.
    """
    rng = np.random.default_rng(0)
    N, H, W = 5, 6, 7
    x = rng.uniform(0, 1, (N, H, W))
    u = rng.uniform(0, 1, (N, H, W))
    t, w = 0.0625, 1.0
    colour = checkerboard(H, W)

    worst = 0.0
    for c in (0, 1):
        mask = colour == c
        d = rng.uniform(-0.1, 0.1, (N, H, W)) * mask
        pred = np.sum(_delta_energy(u, x, d, t, w) * mask, axis=(1, 2))
        true = energy(u + d, x, t, w) - energy(u, x, t, w)
        worst = max(worst, np.max(np.abs(pred - true)))
    assert worst < 1e-9, f"incremental dE wrong by {worst:.3e}"

    # And it must FAIL if both colours move at once -- proof the test has teeth
    # and that the scan may not update everything simultaneously.
    d = rng.uniform(-0.1, 0.1, (N, H, W))
    pred = np.sum(_delta_energy(u, x, d, t, w), axis=(1, 2))
    true = energy(u + d, x, t, w) - energy(u, x, t, w)
    assert np.max(np.abs(pred - true)) > 1e-6, "test cannot detect a bad scan"
    print(f"  incremental dE == exact dE per colour to {worst:.2e} "
          f"(and diverges if colours are mixed, as it must)")


def test_pm_matches_truncated_gaussian():
    """THE oracle. At w = 0 the posterior is a product of N(x_i, sigma^2)
    truncated to [0,1]; its mean is closed-form. Exercises the MH kernel, the
    acceptance rule, the box guard, the batching and burn-in at once.

    Run every chain on the SAME x and POOL them. A single chain's u_pm carries
    MC error ~sigma/sqrt(sweeps/tau) ~ 2e-3, which no tolerance can distinguish
    from a small bias; pooling 512 chains drives the MC error to ~1e-4 and
    leaves bias as the only thing a failure could mean. x straddles both box
    edges, where truncation actually bites and a missing guard would show.
    """
    sigma, lam = 10 / 256, 32 / 256
    x1 = np.array([[0.5, 0.02], [0.98, 0.25]])
    x = np.stack([x1] * 512)

    out = sample_pm(x, sigma, lam, sweeps=3000, w=0.0, seed=2)
    pooled = out["u_pm"].mean(axis=0)              # pooled over independent chains
    exact = pm_truncated_gaussian(x1[None], sigma)[0]
    err = np.max(np.abs(pooled - exact))
    assert err < 5e-4, f"u_PM biased off the truncated-Gaussian mean by {err:.3e}"
    assert 0.2 < out["accept"] < 0.9, f"acceptance {out['accept']:.3f} implausible"

    # Unbiased, not merely close: the error must FALL like 1/sqrt(sweeps). A bias
    # would sit on a floor instead.
    errs = []
    for sw in (500, 2000, 8000):
        p = sample_pm(x[:128], sigma, lam, sweeps=sw, w=0.0, seed=3)["u_pm"].mean(axis=0)
        errs.append(np.max(np.abs(p - exact)))
    assert errs[-1] < errs[0] / 2, f"error did not decay with sweeps: {errs}"
    print(f"  w=0: pooled u_PM matches the exact truncated-Gaussian mean to "
          f"{err:.2e} (accept {out['accept']:.3f})")
    print(f"       decays with sweeps: " +
          " -> ".join(f"{e:.1e}" for e in errs) + "  (500/2000/8000)")


def test_pm_is_a_shrinkage_toward_smooth():
    """At w = 1 there is no closed form, so pin what must be true anyway:
    u_PM stays strictly inside the box (the constraint is in the prior) and TV
    denoising reduces total variation."""
    rng = np.random.default_rng(3)
    sigma, lam = 10 / 256, 32 / 256
    clean = np.tile(np.repeat([[0.25, 0.75]], 4, axis=1), (4, 1))[None]  # 4x8 step
    x = (clean + rng.normal(0, sigma, (32,) + clean.shape[1:])).clip(0, 1)
    out = sample_pm(x, sigma, lam, sweeps=3000, seed=4)
    y = out["u_pm"]

    assert np.all((y > 0.0) & (y < 1.0)), "u_PM must map into the OPEN box"
    tv = lambda a: np.sum(np.abs(np.diff(a, axis=1)), axis=(1, 2)) + \
                   np.sum(np.abs(np.diff(a, axis=2)), axis=(1, 2))
    assert np.mean(tv(y)) < np.mean(tv(x)), "denoising did not reduce TV"
    assert np.mean(np.abs(y - x)) < 3 * sigma, "u_PM moved implausibly far from x"
    print(f"  w=1: u_PM in (0,1), TV {np.mean(tv(x)):.2f} -> {np.mean(tv(y)):.2f}, "
          f"mean |y-x| = {np.mean(np.abs(y - x)):.4f} (sigma = {sigma:.4f})")


def test_params():
    sigma, lam = 10 / 256, 32 / 256
    eps, t = params(sigma, lam)
    assert abs(t - lam / 2) < 1e-15, "t must equal lam/2"
    assert abs(np.sqrt(t * eps) - sigma) < 1e-15, "sqrt(t*eps) must equal sigma"
    print(f"  eps = {eps:.6f}, t = {t:.6f} = lam/2, sqrt(t*eps) = sigma")


def test_chains_are_independent():
    """The batching claim: chain k must depend on x_k alone. If the vectorized
    update leaked across the batch axis, every result would be silently wrong."""
    rng = np.random.default_rng(5)
    sigma, lam = 10 / 256, 32 / 256
    x = rng.uniform(0.2, 0.8, (4, 4, 4))
    full = sample_pm(x, sigma, lam, sweeps=300, seed=7)["u_pm"]
    # Re-run chain 2 alone, with the batch rebuilt around it.
    x2 = np.stack([x[2]] * 4)
    alone = sample_pm(x2, sigma, lam, sweeps=300, seed=7)["u_pm"]
    # Not bit-identical (the RNG stream is shared), so compare statistically:
    # chain 2 of the full batch must be closer to its own re-run than to any
    # other chain's result.
    d_self = np.mean(np.abs(full[2] - alone[2]))
    d_other = min(np.mean(np.abs(full[2] - alone[j])) for j in (0, 1, 3))
    assert d_self <= d_other + 1e-12, "batched chains are not independent"
    print(f"  chains independent: self {d_self:.2e} vs cross {d_other:.2e}")


if __name__ == "__main__":
    for fn in (test_params, test_neighbours_do_not_wrap,
               test_atv_matches_matlab_definition, test_delta_energy_is_exact,
               test_chains_are_independent, test_pm_matches_truncated_gaussian,
               test_pm_is_a_shrinkage_toward_smooth):
        print(f"{fn.__name__}:")
        fn()
    print("\nOK: TV posterior-mean sampler verified.")
