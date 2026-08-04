# Experimental design — recovering the TV denoiser's implicit regularizer

## Goal

Recover `f_reg`, the implicit regularizer of the anisotropic-TV posterior-mean
denoiser, from denoiser evaluations alone, with the one-network gradient-only
design validated on the `ℓ¹` posterior-mean example (`archive/`). This is
`work2.tex` Instantiation B: a prior whose *existence* is all Louchet and
Gribonval supply, with no representation and no closed form.

## Model and notation

The denoiser is the posterior mean `u_PM(x)` under the posterior `∝ exp(−E/ε)`
with

```
E(u) = ‖u−x‖²/(2t) + w·ATV(u)     on [0,1]^n,     ε = σ²/t,   λ = 2t,   √(tε) = σ
```

where `ATV` is the sum of `|u_i − u_j|` over unordered 4-neighbor pairs. There
are two degrees of freedom, `(σ, t)`; the configured pair is `σ = t = 20/256`
(the results currently in `results/` were produced at `σ = 10/256`,
`t = 16/256`).

`w` is a TV weight we added; the MATLAB source hardwires it to 1, so `w = 1` is
the model. `w = 0` is a test hook only: it drops TV but keeps the box, and the
posterior then factorizes into truncated Gaussians with a closed-form mean —
the one exact statement available at every `n`.

The `[0,1]` box is part of the model: the effective prior is
`TV + indicator([0,1]^n)`, so `u_PM` maps into the open box and `f_reg` is
defined only there.

## Method

`S_ε` is never observed, so the training target is prox optimality:

```
∇f_reg(y_k) = x_k − y_k     at   y_k = u_PM(x_k).
```

One convex network `J_θ` is trained on these gradients — no `ψ_θ`, no
inversion, no conjugation. The additive constant of `f_reg` is unidentifiable
and irrelevant (`prox_{f+c} = prox_f`).

## Why the design is falsifiable

There is no ground truth for `f_reg`: evaluating it requires the log-partition
function, and the sampler returns the posterior mean, not the normalizing
constant. The denoiser itself, however, serves as ground truth, and it pins `f_reg`:
if `prox_f = prox_g`, then `∇f = ∇g` on the range of the prox, so `f = g + c`
there. A prox determines its regularizer up to a constant on `range(u_PM)`,
which is connected here.

The held-out prox residual — `‖∇J_θ(y) − (x−y)‖` relative to `‖x−y‖` at fresh
`x` — is therefore a recovery certificate, not merely a validation loss, and it
has a known floor: the sampler's own noise `δ`. The decision rule is fixed in
advance:

| held-out residual | reading |
|---|---|
| `≈ δ` | the network extracted everything the data contains |
| `≫ δ` | the network failed: capacity, budget, or a bug |
| `≪ δ` | impossible; suspect a leak or a self-referential score |

The certificate is blind to one failure mode: **bias**. Finite burn-in makes
`ŷ` biased rather than merely noisy, and a biased `ŷ` means fitting the prox of
the wrong function while the residual — measured against the same `ŷ` — stays
low. Step 0 exists to exclude this.

## The sampler

`tvpm/sampler.py` ports `proj1_mcmc_alg_pme.m` (Langlois): single-site
random-walk Metropolis with a uniform proposal, pixels clipped to `[0,1]`. One
structural change: prior recovery needs `u_PM(x_k)` at `N` different `x_k`, so
the sampler runs `N` independent chains and updates one checkerboard color at
a time across all of them. This is exact — same-color pixels are conditionally
independent under a nearest-neighbor MRF — and changes only the scan order.

Three independent checks validate it:

1. **Arithmetic.** The incremental `ΔE` used by the Metropolis step equals
   `energy()` computed from scratch (1.1e-14), and the same test confirms the
   check *fails* when both checkerboard colors move at once — the conditional
   independence the scan relies on (`tests/test_sampler.py`).
2. **The `w = 0` kernel.** The posterior collapses to independent truncated
   Gaussians with a closed-form mean; pooled over 512 chains, `u_PM` matches it
   to 1.2e-4 and the error decays like `m^{-1/2}`.
3. **The distribution with TV on** — Step 0 below.

## Step 0 — sampler bias gate

At `n = 2`, `u_PM` is a two-dimensional integral and needs no MCMC: rotate
coordinates so TV acts on one variable, integrate the other as a
truncated-Gaussian moment in closed form, and apply Gauss–Legendre on each side
of the kink (`tvpm/quadrature.py`; an independent dense-grid witness agrees to
3.6e-7). Against this exact reference, over 50 `x` × 128 chains
(`tests/test_quadrature.py`):

| sweeps `m` | single-chain noise | shrinkage `b` |
|---|---|---|
| 500 | 8.65 % | +0.023 % (+0.3 se) |
| 2000 | 4.44 % | +0.016 % (+0.4 se) |
| 8000 | 2.21 % | −0.018 % (−1.0 se) |
| 32000 | 1.10 % | +0.011 % (+1.2 se) |

The bias is bounded by the signed shrinkage statistic `b`, which projects each
chain's target onto the exact one and pools over all `x`. Burn-in bias has a
known sign — chains start at `u0 = x`, so an under-burned chain leaves the
target too short, `b < 0` — and nothing of the kind appears: `b` flips sign and
stays within ±1.2 standard errors. At `m = 8000`, `|bias| < 0.055%`, roughly
40× below single-chain noise, so the prox residual is read against noise, not
bias. The same test verifies, with TV on, the chain of identities the method
rests on: `∇ψ = u_PM` (3.9e-11) and `∇f_reg(y) = x − y` (5.2e-11).

## Step 1 — data

```
n     = 64                   (8×8 patches)
train = 20000 random Barbara patches + N(0,σ²), clipped to [0,1]    seed 1
val   =  4000 more, at disjoint patch positions                     seed 2
eval  =  4000 cameraman patches + noise (transfer split)            seed 3
m     =  8000 sweeps per chain
```

The sampler is the expensive stage; its output is cached to `data/*.npz`
(gitignored) and nothing downstream retriggers it. The sampler noise on the
gradient target, measured from the spread of independent chains, decays like
`m^{-1/2}` and is essentially independent of `n` (signal and noise both scale
like `√n`): `δ ≈ 1.2%` at `m = 8000`. Sampling the training split costs about
11 minutes on one CPU.

One concern is not resolved by the noise level alone: `δ` also perturbs the
*location* `ŷ_k`, which is an errors-in-variables effect and biases a
regression rather than averaging out of it. Quantifying it would require the
curvature of `f_reg`, which is exactly what is unknown; the bias sweep in
Step 3 is the empirical answer.

## Step 2 — training

One ICNN, trained on the gradient targets:

- fully connected (`--arch fc`): the learned proximal network (LPN), width 256, 2 layers, 181k parameters;
- convolutional (`--arch conv`): 3×3 kernels, 32 channels, 2 blocks, 19.5k
  parameters (see the conv-ICNN section).

```
loss   = MSE(∇J_θ(ŷ_k), x_k − ŷ_k) / var(x−ŷ)
Adam, batch 512, lr 1e-3 ×0.1 at 50% and 75% of S = 250000 steps
wclip() every step; no early stopping; keep the best-validation checkpoint
input standardized (globally for conv); Softplus β = 20
```

Four empirical findings (single seed) set these values:

1. **Input standardization is required.** The patches concentrate on a thin
   set — standard deviation 0.039 within a patch, mean ranging 0.06–0.92 —
   while Softplus bends over a scale of ~0.2, so the raw network was nearly
   linear in the informative directions and matched a plain linear
   least-squares fit. Standardization moved the residual from approximately
   70% to below 20%.
2. **The budget is steps at `lr = 1e-3`.** The inherited `S = 50000` decayed
   the learning rate too early; `S = 250000` with decays at 50%/75% tracks the
   validation loss.
3. **`β = 20`.** `f_reg` retains a TV kink, measured at `n = 2` to be rounded
   over a scale of ~0.038; `Softplus(β=5)` is too smooth to resolve it.
   `β = 20` reduced the residual from 13.4% to 10.7%, with the gain appearing
   at the value the kink scale predicts.
4. **Capacity did not help.** Width (256/512/1024) and depth (2/3 layers) did
   not improve the residual, and width 512 diverged. The fixes that worked matched
   the network to the problem's scales; this motivates the convolutional
   architecture, which changes structure rather than size.

## Step 3 — evaluation

- **Held-out prox residual** against the noise floor `δ`, per the decision rule
  above. This is the primary metric.
- **Bias sweep.** Retrain at `m ∈ {2000, 8000, 32000}` from cached data. If the
  residual tracks `δ(m)`, the recovery is sampler-noise-limited; if it plateaus
  above, the network is the bottleneck; if the recovered prior itself keeps
  moving as `m` grows, burn-in bias is real and the certificate cannot see it.
- **Transfer.** Train on Barbara, evaluate on cameraman. `f_reg` is a property
  of the denoiser, not of an image, so it must transfer; failure would mean the
  network learned the patch distribution instead.

## Step 4 — qualitative figures

- `J_θ(y)` against `TV(y)` on held-out patches: strong correlation is expected,
  equality is not — `f_reg` is a *smoothed* TV, and the gap is the point.
- `J_θ` on smooth versus textured patches.
- A conditional cross-section: one pixel varies, the others drawn from the
  patch distribution — never fixed at a constant (axis slices are misleading in
  high dimension; see `../numerics_audit.tex`).
- Conv only: the learned first-layer kernels. Difference stencils `[+1, −1]`
  emerging from denoiser evaluations alone are direct evidence that the network
  rediscovered TV structure.

## The convolutional ICNN

`f_reg` is approximately a sum over sites of a local functional: **local** and
**shift-invariant**. A dense network must learn both facts from data; a
convolutional ICNN builds them in (3×3 kernels, weight sharing) and learns the
local functional once rather than once per site, with 19.5k parameters against
the dense network's 181k. It assumes only locality and shift-invariance — true
of any image prior and known here — not pairwise interactions, differences, or
TV itself.

**Convexity.** By Amos et al. (2017, Prop. 1), a feed-forward network is convex
in its input if the feature-path weights are nonnegative and the activations
are convex and non-decreasing; a convolution is linear, so the proposition
applies to the kernel entries (as in Mukherjee et al. 2020). In `../src/conv_icnn.py`,
the input and skip convolutions are unconstrained (affine in `y`); the feature
convolutions and the pooling head are kept nonnegative by `wclip` at each step;
the activation is Softplus. `tests/test_icnn.py` is the gate — midpoint
inequality, convexity along random lines, input-Hessian PSD, and a negative
control — and must pass before training.

**Implementation constraints.**

- *Global* standardization: per-pixel offsets would differ by site and break
  shift-invariance.
- *Mean* pooling, not sum: both are convexity-safe and representationally
  equivalent, but summing 64 sites inflated the initial gradient far past the
  target scale and training failed to recover.
- *Reflect* padding: the sampler's TV does not wrap, so circular padding would
  impose a periodic TV; fixed paddings are linear and convexity-safe.

## Design decisions

1. **`x_k` live on the natural-patch region.** The denoiser is probed where
   images live, which is what a plug-and-play deployment uses; `f_reg` is
   unconstrained off that region, and the cameraman split tests transfer.
   (Uniform sampling of `[0,1]^64` was rejected: every uniform sample is a
   noise image, so it probes a regime not encountered in practice.)
2. **No training-box amendment**, unlike the four production families: training
   and evaluation `y` both come from `u_PM` applied to the same kind of input,
   so coverage holds by construction.
3. **8×8 patches.** Eight times the validated `n = 8`; 16×16 is affordable and
   no less accurate per the noise measurements, but a failure there would be
   ambiguous between the design and the scale. If 8×8 works, 16×16 is a cheap
   follow-up.
4. **`x_k` clipped to `[0,1]`.** Matches the input a deployed denoiser
   receives; a scope restriction, stated rather than hidden.

## Failure criteria (fixed in advance)

- The Step-0 gate fails: the sampler is biased; nothing downstream is worth
  running.
- The residual is `≫ δ` and insensitive to width and steps: the design does not
  scale from `n = 8` to `n = 64`; report that as the finding.
- The recovered prior moves as `m` grows: burn-in bias dominates; report the
  `m`-dependence, not a single number.
- The residual is `≈ δ` but `J_θ` is uncorrelated with TV: an error the
  certificate cannot localize; debug, do not ship.

## Open issue

The `[0,1]` box causes `f_reg` to become unbounded at the boundary, and an ICNN cannot
represent a barrier. In practice the `y_k` avoid the boundary, so this may
never bind — but if the evaluation shows error concentrating near the box
faces, this is why. No pre-emptive fix.
