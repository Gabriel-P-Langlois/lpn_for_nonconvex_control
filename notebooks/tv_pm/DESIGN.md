# Plan: recover the TV denoiser's implicit regularizer

Status (2026-07-19): **complete through the demonstration.** Steps 0–4 done;
production recovery ~10.7% held-out prox residual (fc and conv tie), and its prox
**reproduces the posterior-mean denoiser** (`results/`, `denoise_demo.py`). The
conv-ICNN is built and convexity-verified (see "Conv-ICNN" at the bottom). The
`beta=20` m-sweep is done. Experiment notebook: `recover_tv.ipynb`. Everything is
**single-seed and preliminary — observations, not conclusions.**

---

## For a colleague picking this up: what actually moved the number

**These are observations, not conclusions** — single seed, no repeats, no error
bars; the "floor"/"κ" are a fit to a few points, and the architecture and transfer
comparisons are single runs. Read them as what these experiments SHOWED, and
confirm (seeds, repeats, more images/`m`) before treating any of it as a claim in
the paper.

The recovery went from a broken 70% prox residual to ~10.7% through four fixes, and
the pattern is the lesson. Full detail and numbers in `../changes.txt` (entries
2026-07-17/19); the short version:

1. **Input standardization** (`recover.py:Units`). The patches live in a thin
   cigar — std 0.039 within a patch, mean ranging 0.06–0.92 — and `Softplus`
   bends over ~0.2, so the net was near-LINEAR in the informative directions.
   Proof it was that bad: the raw net matched a plain linear least-squares fit.
   Fix cut the residual 70% → (with the next two) the teens.
2. **Step budget.** The inherited `S=50000` decayed the LR 2.1× too early; what
   tracks the val loss is *steps at lr=1e-3*, not total steps (a repo-wide lesson,
   `changes.txt` entry 17). `S=250000` with decays at 50%/75%.
3. **Activation sharpness `beta`** (the user's insight). `f_reg` is a TV kink,
   measured sharp at n=2 (rounded over ~0.038); `Softplus(beta=5)` is 5× too
   smooth to resolve the median neighbour difference. `beta=20` is the measured
   sweet spot: 13.4% → 10.7%, and the curve turns exactly where predicted.
4. **What did NOT work: capacity.** Width (256/512/1024) and depth (2/3 layers)
   both failed — 512 even diverged. The three fixes that worked all MATCHED the
   net to the problem's scales; adding parameters did not. That is the whole
   argument for the conv-ICNN below: give the net the right STRUCTURE (locality +
   shift-invariance), not more capacity.

Reproduce the production FC number:
`python recover.py --arch fc --standardize --beta 20 --steps 250000` (~1.5–2.6 h).

## Notation

The sampler's posterior is `∝ exp(−E/ε)` with

```
E(u) = ‖u−x‖²/(2t) + w·ATV(u)     on [0,1]^n,     ε = 2σ²/λ,   t = λ/2,   √(tε) = σ
```

`ATV` = sum of `|u_i − u_j|` over unordered 4-neighbour pairs. `w` is the TV
weight and is **ours, not the paper's** — the MATLAB hardwires it to 1.
**`w = 1` throughout this plan.** `w=0` survives only as a unit test inside
`test_sampler.py`, where it pins the MH kernel against a closed form; it is not
part of the experiment.

## Goal

Recover `f_reg`, the implicit regularizer of the anisotropic-TV posterior-mean
denoiser, from **denoiser evaluations alone**, with the one-network gradient-only
design validated on `ℓ¹` in `../posterior_mean_l1.ipynb`. This is `work2.tex`
Instantiation B: a prior whose *existence* is all Louchet and Gribonval supply,
with no representation and no closed form.

## What is already settled

- **Regime.** `S_ε` is never observed, so the target is prox optimality,
  `∇f_reg(y_k) = x_k − y_k` at `y_k = u_PM(x_k)`. One convex net, gradient-only,
  no `ψ_θ`, no inversion, no conjugation.
- **The sampler works** (`sampler.py`, `test_sampler.py`, 7 checks) and batches:
  `N` independent chains update one checkerboard colour at a time, exactly.
- **It is affordable** (`noise_diagnostic.ipynb`): `N=20000` targets at 1.2%
  relative noise costs ~11 min at 8×8. Noise decays like `m^{-1/2}` and is
  **independent of `n`** — signal and noise both scale like `√n` — so image size
  is a compute choice, not an accuracy one.
- **The constant is unidentifiable** and irrelevant (`prox_{f+c} = prox_f`).

## Why this is falsifiable, despite no ground truth for `f_reg`

There is no ground truth for `f_reg`: evaluating it needs `S_ε`, the
log-partition function, and the sampler returns the posterior *mean*, not the
normalizing constant. Having the clean Barbara does not help — `f_reg` is a
property of the **denoiser**, not of any image.

But the denoiser *is* a ground truth, and it pins `f_reg` hard:

> If `prox_f = prox_g` then `∇f = ∇g` on the range of the prox, so `f = g + c`
> there. **A prox determines its regularizer up to a constant** on
> `range(u_PM)`, which is connected here.

So the held-out prox residual is not merely a validation loss. If
`∇J_θ(y) ≈ x − y` at fresh `x`, then `J_θ ≈ f_reg + c` on the region covered —
that is a theorem. And the check has a **known floor**: the sampler's own noise
`δ`, measured at 1.2% for `m=8000`. Hence the decision rule:

| held-out residual | reading |
|---|---|
| `≈ δ` | the net extracted everything the data contains |
| `≫ δ` | the net failed — capacity, budget, or a bug |
| `≪ δ` | impossible; suspect a leak or a self-referential score |

**The one hole this cannot see: bias.** Finite burn-in makes `ŷ` slightly
*biased*, not just noisy. A biased `ŷ` means we fit the prox of the wrong
function, and the residual — measured against the same biased `ŷ` — stays low.
Step 0 exists for this and nothing else.

## Tomorrow, in order

### Step 0 — is the sampler unbiased when TV is on? — **DONE, PASSED** (2026-07-17)

> **Result.** It is unbiased. `quadrature.py` gives exact `u_PM` at `n=2` with TV
> present; `test_quadrature.py` is the gate (~40 s, 6 checks). Over 50 `x` × 128
> independent chains, scored on the target `x−y`:
>
> | sweeps | pooled err | RMS z | 1-chain noise | shrinkage `b` |
> |---|---|---|---|---|
> | 500 | 0.729 % | 0.97 | 8.65 % | +0.023 % (+0.3 se) |
> | 2000 | 0.370 % | 0.96 | 4.44 % | +0.016 % (+0.4 se) |
> | 8000 | 0.187 % | 0.99 | 2.21 % | −0.018 % (−1.0 se) |
> | 32000 | 0.089 % | 0.94 | 1.10 % | +0.011 % (+1.2 se) |
>
> `RMS z ≈ 1` at every `m` — the deviation from quadrature is exactly one standard
> error, i.e. pure Monte Carlo scatter — and the error falls like `m^{-1/2}` with
> no floor. **At `m=8000`, `|bias| < 0.055%` at 2 se, ~40× under the 2.2% noise of
> the single chain per `x_k` that Step 1 runs.** So the prox residual will be read
> against noise, not bias, which is what this step existed to establish.
>
> The identity chain is verified too, with TV on: `∇ψ = u_PM` to 3.9e-11 and
> `∇f_reg(y) = x−y` to 5.2e-11 at `y = u_PM(x)` — so the quantity Step 2 regresses
> on is the right one, checked rather than assumed.
>
> **The gate as written below (~0.1% pooled) was replaced, and the plan was wrong,
> not the sampler.** The pooled error is `δ/√reps`: thresholding it tests how many
> chains the test happens to run, not a property of the sampler. It reads 0.187%
> at `m=8000` purely because 128 chains were pooled. What Step 0 meant to bound is
> the *bias*, so the gate now bounds the bias directly, via a signed **shrinkage**
> statistic `b` that projects each chain's target onto the exact one and pools over
> every `x` at once (another √100 of resolution, free). Signed because burn-in bias
> has a known direction: chains start at `u0 = x`, so an under-burned chain leaves
> `y` too close to `x` and the target too *short* — `b < 0`. Nothing of the kind
> appears: `b` flips sign and stays inside ±1.2 se.
>
> Two things worth carrying forward. **(i)** Single-chain noise at `n=2` is 2.2% at
> `m=8000`, against 1.2% at `n=64` in `noise_diagnostic.ipynb`. Not a contradiction
> — the diagnostic's `√n` signal/noise argument is asymptotic and `n=2` is as far
> from it as one can get — but it means `n=2` is a *pessimistic* noise proxy, and
> the diagnostic's `n=64` figure stands. **(ii)** `u_pm_grid` (dumb dense 2-D grid,
> straight across the kink) agrees to 3.6e-7 and is kept as an independent witness:
> the rotation trick is checked against something sharing none of its algebra.

Original plan, for the record:

`test_sampler.py` proves the TV arithmetic is right (incremental `ΔE` vs
`energy`, 1.1e-14) but **not that the chain samples the right distribution with
`w=1`**. The `w=0` oracle cannot check this: it switches TV off.

At `n=2` (a 1×2 image) `u_PM` needs no MCMC — it is a 2-D integral over `[0,1]²`:

```
u_PM(x) = ∫ u e^{−E(u)/ε} du / Z,      Z = ∫ e^{−E(u)/ε} du
```

by quadrature, machine precision, **TV fully present**. Cheap: `σ ≈ 0.039`
confines the integrand to `±6σ` around `x`, so a 512² local grid per `x` suffices,
and `e^{−ATV(u)/ε}` is independent of `x` so it precomputes. Compare against
`sample_pm` at each `m ∈ {500, 2000, 8000, 32000}` over ~200 `x`.

Also worth doing here, since `ψ = ½‖x‖² − tS_ε` and `S_ε = −ε ln(Z/(2πtε)^{n/2})`
are both quadrature-available at `n=2`: confirm `∇ψ = u_PM` and that
`f_reg(y_k) = ⟨x_k,y_k⟩ − ψ(x_k) − ½‖y_k‖²` is consistent. That is the whole
identity chain, verified once, with TV on.

**Gate:** if the MCMC mean sits more than ~0.1% from quadrature at the `m` we
plan to use, fix the sampler before spending anything else.

### Step 1 — data (~15 min compute)

```
n        = 64                      (8×8; see open decisions)
σ, λ     = 10/256, 32/256          → ε = 0.024414, t = 0.0625   (the MATLAB's tabulated pair)
N        = 20000 random 8×8 Barbara patches + N(0,σ²), clipped to [0,1]   seed 1
val      = 4000 more, disjoint patches                                    seed 2
eval     = 4000 patches from cameraman_256x256_d.mat + noise               seed 3
m        = 8000 sweeps  (δ ≈ 1.2%; raise if Step 0 says burn-in bias needs it)
```

Run `sample_pm` once per split → `ŷ_k`. **Cache to `tv_pm/data/*.npz`** (~10 MB
each): the sampler is the expensive part and nothing below should ever retrigger
it. Gitignore that directory.

No box amendment (decided; see above).

### Step 2 — train (~10 min)

One `LPN`, gradient-only, exactly the `ℓ¹` design:

```
loss   = MSE(∇J_θ(ŷ_k), x_k − ŷ_k) / var(x−ŷ)
width  = 256, layers 2, β = 5, wclip() every step
Adam, batch 512, lr 1e-3 ×0.1 at 50%/75% of S = 50000 steps
no early stopping; keep the best-validation checkpoint
```

Lift `train_grad` from `../posterior_mean_l1.ipynb` into **`tv_pm/recover.py`**
(with the scoring), so the notebook stays thin and the loop is importable and
testable. The `ℓ¹` notebook keeps its inline copy — it is executed and frozen,
and the two are separate experiments.

Width 256 is D2's default; the `ℓ¹` run used 128 at `n=8`, and nothing has been
measured in between. Gradient supervision gives `64·N ≈ 1.3M` constraints against
~150k parameters, so this is not obviously under-determined.

### Step 3 — score

- **Held-out prox residual** vs the `δ` floor from `noise_diagnostic.ipynb`. The
  headline; read it with the table above.
- **Bias sweep.** Retrain at `m ∈ {2000, 8000, 32000}` from cached data. If the
  residual tracks `δ(m)` the story is consistent; if it plateaus above `δ`, the
  net is the bottleneck; if the *recovered prior* keeps moving as `m` grows, the
  burn-in bias is real.
- **Off-distribution.** Barbara-trained, cameraman-evaluated. `f_reg` is a
  property of the denoiser, so it must transfer; if it does not, we learned the
  patch distribution instead.

### Step 4 — qualitative (what the paper wants)

- `J_θ(y)` against `TV(y)` on held-out patches: strong correlation expected, but
  **not** equality — `f_reg` is a *smoothed* TV, and the gap is the point
  (R2.4/19/24, no staircasing).
- `J_θ` on smooth vs textured patches.
- A conditional cross-section: one pixel varies, the rest drawn from the patch
  distribution — never fixed at a constant (`numerics_audit.tex`: axis slices lie
  in high `n`).

## Decided (2026-07-16, user) — do not reopen

1. **The `x_k` live on the natural-patch region.** `x_k` = random Barbara 8×8
   patches + `N(0,σ²)`, clipped. So **`f_reg` is recovered on the natural-patch
   region of `(0,1)^n`, not on all of it.** That is a scope statement to make up
   front in the writeup, not a caveat to bury: the denoiser is probed where images
   actually live, which is what a plug-and-play deployment needs, and `f_reg` is
   simply unconstrained off that region. The cameraman eval is what tests whether
   it transfers. (Rejected: uniform `x_k` on `[0,1]^64` — a more complete recovery
   of `f_reg` as a mathematical object, but in 64 dimensions every uniform sample
   is a noise image, so it probes TV in a regime nobody deploys and weakens the
   imaging story.)
2. **No box amendment**, unlike D2. Train and eval `y` both come from `u_PM`
   applied to the same kind of input, so coverage is automatic by construction.
   Say so explicitly — otherwise it reads as an oversight against the four
   production families, where the amendment was load-bearing.

## Decided (2026-07-17, user) — the two open sizing questions

3. **Image size: 8×8 (`n=64`).** 8× the validated `n=8`, ~11 min of sampling.
   16×16 (`n=256`) is affordable (~33 min) and, per the diagnostic, *no less
   accurate* — but it is 4× past the largest `n` at which anything in this repo
   has worked, so a failure there would be ambiguous between the design and the
   scale. If 8×8 works, 16×16 is a cheap follow-up from the same code.
4. **`x_k` clipped to `[0,1]`.** Matches the MATLAB's `imnoise` and the input a
   deployed denoiser actually receives. It restricts which region of `f_reg` we
   see; say so alongside the natural-patch scope statement (#1), as scope rather
   than as a defect.

## Still open

1. **The `[0,1]` barrier.** The box is in the model, so `f_reg` blows up at
   `∂[0,1]^n` and an ICNN cannot represent a barrier. In practice the `y_k` avoid
   the boundary, so this may never bite — but if Step 3 shows the error piling up
   near the box faces, that is why. Do not add a fix pre-emptively.

## Kill criteria

Decide now, not after seeing numbers.

- **Step 0 gate fails** (MCMC mean ≠ quadrature at `n=2`, `w=1`): the sampler is
  biased. Nothing downstream is worth running. Fix or stop.
- **Held-out residual ≫ δ and does not improve with width or steps:** the design
  does not scale from `n=8` to `n=64`. Report that — it is a real finding about
  the method, and more informative than the `ℓ¹` success alone.
- **The recovered prior keeps moving as `m` grows:** burn-in bias dominates, and
  the certificate cannot see it. Report the `m`-dependence as the result; do not
  ship a single number.
- **Residual ≈ δ but `J_θ` uncorrelated with `TV`:** something is wrong that the
  certificate cannot localise. Debug; do not ship.

## Cost

| step | |
|---|---|
| 0 quadrature check | ~1 h build, minutes to run |
| 1 data (3 splits, `m=8000`, `n=64`) | ~15 min |
| 2 train | ~10 min |
| 3 bias sweep (2 more `m`, cached) | ~1 h sampling + ~20 min training |

CPU throughout; nothing here needs a GPU.

## Files this will touch

```
tv_pm/quadrature.py        DONE exact u_PM, S_eps, psi, f_reg at n=2 by quadrature (Step 0)
tv_pm/test_quadrature.py   DONE the Step-0 gate as an assertion -- PASSED
tv_pm/dataset.py           DONE Step 1: patches → x_k → y_k = u_PM(x_k), cached.
                                NOT in the original list, which routed data through
                                recover.py. Split off so the expensive run-once
                                stage stays out of the importable training code and
                                the bias sweep can re-enter at one m: `--sweeps`.
tv_pm/recover.py           DONE train_grad + scoring; --arch fc|conv, --beta, etc.
tv_pm/conv_icnn.py         DONE convolutional ICNN (see the Conv-ICNN section)
tv_pm/test_conv_icnn.py    DONE the convexity gate for the conv-ICNN -- PASSED
tv_pm/step4_figures.py     DONE qualitative panels + learned-kernel figure
tv_pm/denoise_demo.py      DONE the prior in action: prox-denoise vs u_PM
tv_pm/recover_tv.ipynb     DONE the experiment end to end; ARCH fc|conv; executed
tv_pm/MANUAL.md            DONE short reproduce instructions
tv_pm/results/            DONE  shipped figures, metrics.csv, checkpoints, README
                                (tracked via a numerics/.gitignore negation)
tv_pm/data/                DONE cached sampler output; gitignored (regenerable)
../../changes.txt          DONE running record (doubles as commit text)
```

Nothing in `src/`, `bin/`, or the four production notebooks is touched.

---

## Conv-ICNN (2026-07-18) — structure instead of capacity

### Why
Width and depth both failed to move the FC-ICNN's ~10.7% floor (see the colleague
summary at the top). `f_reg` is `≈ Σ_sites φ(local neighbourhood)`: **local** and
**shift-invariant**. A dense net must learn both facts from data and estimate a
shift-invariant function *without knowing it is one* — plausibly the floor itself.
A conv-ICNN builds locality (3×3 kernels) and shift-invariance (weight sharing)
in, so it learns `φ` once, not once per site. It has **19.5k parameters vs the
FC's 181k** and reuses every kernel across all 64 sites.

### Not circular
It assumes only **locality + shift-invariance** — true of any image prior, and
here actually *known* (the denoiser is anisotropic TV + a box). It does NOT assume
pairwise, differences, or TV; a 3×3 conv can learn any local shift-invariant
convex functional. Contrast handing the net `y_i − y_j` directions, which *would*
assume the TV answer.

### Convexity — proven and TESTED
Amos et al. 2017 Prop. 1: a feed-forward net is convex in its input if the
feature-path weights are nonnegative and the activations are convex &
non-decreasing. A convolution is linear, so the proposition applies verbatim, read
on the **kernel entries** (Mukherjee et al. 2020, arXiv:2008.02839, do exactly
this for imaging). In `conv_icnn.py`: input conv and skip convs unconstrained
(affine in `y`); feature convs and the pooling head **nonnegative** (`wclip` each
step); `Softplus(beta=20)` (convex, increasing — a non-monotone activation like
Mish would break it, cf. B5). `test_conv_icnn.py` is the **gate** — midpoint
inequality, convexity along random lines, full 64×64 input-Hessian PSD, `wclip`
clamps, and a negative-control that a negative head weight makes `J` concave and
the checker catches it. Must pass before training.

### Two things a replicator must not miss
- **Global standardization** (`Units(..., glob=True)`, auto-selected for conv).
  Per-pixel `mu, s` would give a different offset per site and BREAK
  shift-invariance. Free here (`s = 0.197..0.200` uniform), but required.
- **MEAN pool, not sum.** Both are convexity-safe and representationally
  equivalent (the head absorbs the H·W factor), but summing 64 sites inflated the
  initial gradient ~1000× past the target and Adam could not recover. Mean starts
  `J` at O(1), matching the FC net's init scale. This was a real failed first
  attempt (val loss 9779, corr ≈ 0) — do not repeat it.

### Boundary
The sampler's TV does not wrap, so padding is `reflect` (invents no edge), NOT
`circular` (which would impose periodic TV). All fixed paddings are linear, hence
convexity-safe; this is a fidelity choice, checked at the patch faces.

### Run and expectation
`python recover.py --arch conv --standardize --beta 20 --steps 250000`
(`--channels 32 --blocks 2` default; global units and mean-pool are automatic).
A/B against the FC's 10.7% / 0.0073 val loss, everything else identical (same
cached `m=8000, N=20000` data, loss, budget). Best case the floor drops toward the
sampler-noise-limited regime (`κδ ≈ 8%`), letting us say *MCMC target noise, not
the network, is the limit*. Watch training stability (nonneg convs + double
backprop + sharp Softplus is a narrow regime; fallback is grad-clip / lower LR).
Bonus deliverable: inspect the learned first-layer kernels — if they emerge as
difference stencils `[+1,−1]`, that is direct evidence the net rediscovered TV
structure from denoiser evaluations alone.
