# `tv_pm/` — TV posterior-mean denoiser (work2.tex Instantiation B)

Standalone. Recovers `f_reg`, the implicit regularizer of the anisotropic-TV
posterior-mean denoiser, from denoiser evaluations alone, with **one convex
network**. Status: **complete through the demonstration** — the recovered prior's
prox reproduces the denoiser (see `results/`). Everything is **preliminary and
single-seed** (observations, not conclusions — see `DESIGN.md`).

**Start with `recover_tv.ipynb`** (the experiment, `ARCH = fc|conv`, committed
executed) or **`MANUAL.md`** (copy-paste commands to reproduce).

| file | what it is |
|---|---|
| **`recover_tv.ipynb`** | **the experiment end to end** — data → model → score → figures → denoising; `ARCH` selects fc/conv (default fc); committed executed |
| `MANUAL.md` | short reproduce-from-scratch / from-checkpoints instructions |
| `DESIGN.md` | the plan and full audit trail — **read this for the story**; colleague summary at the top, Conv-ICNN section at the bottom |
| `sampler.py` | batched posterior-mean sampler for anisotropic TV, ported from MATLAB |
| `test_sampler.py` | pins the sampler — must pass before anything here is believed |
| `quadrature.py` | exact `u_PM`, `S_ε`, `ψ`, `f_reg` at `n=2`, TV present — the only exact reference in this example |
| `test_quadrature.py` | the Step-0 gate: **the sampler is unbiased with TV on** (~40 s) |
| `dataset.py` | Step 1: patches → `x_k` → `y_k = u_PM(x_k)`, cached to `data/` (`--sweeps`, `--scale`) |
| `recover.py` | Steps 2–3: train one convex net on the gradients + score (`--arch fc\|conv`, `--beta`, `--standardize`, …) |
| `conv_icnn.py` | convolutional ICNN — locality + shift-invariance for the TV prior; provably convex, 19.5k params |
| `test_conv_icnn.py` | the conv-ICNN **convexity gate** — must pass before training |
| `step4_figures.py`, `denoise_demo.py` | the qualitative figures and the denoising demonstration |
| `noise_diagnostic.ipynb` | measures whether the gradient target survives sampler noise |
| `results/` | shipped figures, `metrics.csv`, checkpoints, and the numbers writeup |

## Why this is separate

The `posterior_mean_l1` example recovers a prior from denoiser evaluations alone,
using the exact target `∇f_reg(y_k) = x_k − y_k` at `y_k = u_PM(x_k)`. There
`u_PM` is a closed form. Here it is an MCMC estimate, so `y_k` carries an error
`δ` that hits **both** the target and the point it attaches to at first order —
and unlike the `ℓ¹` case there is **no ground truth** for `f_reg` at all, so the
prox-residual certificate would be the only score. Those are different enough
problems to keep the code apart.

## Provenance

`sampler.py` ports `ideas/old_files/proj1_mcmc_alg_pme.m` (Langlois), preserving
the model exactly:

```
E(u) = ‖u−x‖²/(2t) + w·ATV(u)      on [0,1]^n,     posterior ∝ exp(−E/ε)
ε = 2σ²/λ,   t = σ²/ε = λ/2,   √(tε) = σ
```

with single-site random-walk Metropolis, a uniform proposal, and pixels clipped
to `[0,1]`.

**`w` is ours, not the paper's**, and it is the only free parameter added. The
MATLAB hardwires the TV coefficient to 1, so **`w=1` is the model**. `w=0` is a
test hook: it keeps the box but drops TV, leaving the prior
`J = indicator([0,1]^n)` — still a genuine convex prior, but one whose posterior
factorizes into truncated Gaussians with a closed-form mean. That is the only
exact statement available anywhere in this example, which is why the sampler test
and the calibration stage in `DESIGN.md` are both built on it.

**One structural change.** The MATLAB computes `u_PM` at *one* `x` — 20000 sweeps
× 65536 sites × 2 chains ≈ 2.6e9 steps for the noisy Barbara. Prior recovery
needs `u_PM(x_k)` at `N` *different* `x_k`, which that structure cannot deliver.
Since the `N` chains are independent and anisotropic TV is a nearest-neighbour
MRF, `sampler.py` updates one checkerboard colour at a time across every chain
at once — exactly, since same-colour pixels are conditionally independent. The
chain is unchanged; only the scan order is, and random-with-replacement becomes
a proper sweep.

The `[0,1]` box is part of the model, not a guard rail: it makes the effective
prior `TV + indicator`, so `u_PM` maps into the **open** box and `f_reg` is
defined only there.

## How the sampler is checked

There is no closed-form `u_PM` at `w=1`, so three independent checks bracket it.
The first two (`test_sampler.py`, 7 assertions) leave a hole; the third closes it.

- **the TV term** — the incremental `ΔE` the MH step uses must equal `energy()`
  computed from scratch (1.1e-14). The same test confirms it *fails* when both
  colours move at once, which is the conditional independence the scan needs.
- **everything else, at `w=0`** — the posterior collapses to independent
  `N(x_i, σ²)` truncated to `[0,1]`, whose mean is closed-form. Pooled over 512
  chains, `u_PM` matches it to 1.2e-4, and the error decays like `m^{-1/2}`, so
  the sampler is unbiased rather than merely close.
- **the distribution itself, at `w=1`** (`test_quadrature.py`) — the two above
  cover the arithmetic and the `w=0` kernel, but not the claim that the chain
  samples the right distribution *with TV on*. It does. At `n=2`, `u_PM` is a 2-D
  integral needing no MCMC, so it is computed exactly (rotate to `s,r`: TV touches
  `r` alone, the `s`-integral is a truncated-Gaussian moment in closed form, and
  Gauss–Legendre on each side of the kink converges spectrally). Against it, over
  50 `x` × 128 chains, **the target's systematic shrinkage at `m=8000` is
  `−0.018% ± 0.018%`, so `|bias| < 0.055%`** — ~40× under the 2.2% noise of a
  single chain. The same file verifies the identity chain with TV on: `∇ψ = u_PM`
  (3.9e-11) and `∇f_reg(y) = x−y` (5.2e-11).

That last one matters more than its size suggests: it is the only check that can
see **bias**, and the experiment's own certificate structurally cannot. The
held-out prox residual is scored against the same `ŷ` used to fit, so a biased
`ŷ` means fitting the prox of the wrong function while the residual stays low.

## What the diagnostic found

Relative error on the gradient target `x_k − y_k`, from the spread of 8
independent chains per `x_k` (16 distinct `x_k`, Barbara + noise, σ=10/256,
λ=32/256):

| sweeps | n=64 (8×8) | n=256 (16×16) |
|---|---|---|
| 500 | 4.75 % | 4.32 % |
| 2000 | 2.37 % | 2.16 % |
| 8000 | 1.20 % | 1.08 % |
| 32000 | 0.59 % | 0.54 % |

Decays like `m^{-1/2}`, and is **essentially independent of n** — signal and
noise both scale like `√n`, so the SNR is dimension-free. The image-size limit
is therefore compute, not accuracy. Projected cost for `N=20000` targets on one
CPU: ~11 min at 1.2% (8×8), ~33 min at 1.1% (16×16), ~43 min at 0.6% (8×8).

**Verdict: affordable.** What it does not settle is whether ~1% target noise is
tolerable — the `ℓ¹` example reached 0.0076% with exact targets. Regression over
`N` samples averages the noise down, but `δ` also moves the *location* `ŷ_k`,
which is errors-in-variables and biases a fit rather than averaging out of it.
Sizing that needs the curvature of `f_reg`, which is exactly what we have no
ground truth for. Only a real run answers it.

## Running

```
~/miniforge3/envs/lpn_env/bin/python notebooks/tv_pm/test_sampler.py      # ~3 s
~/miniforge3/envs/lpn_env/bin/python notebooks/tv_pm/test_quadrature.py   # ~40 s
```
The notebook runs from this directory (`sys.path` assumes it) and takes ~90 s.
