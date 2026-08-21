# Review of `pnp_reg/` — two findings, and what the experiments can and cannot support

Reviewed: `README.md`, `DESIGN.md` (490 lines), `pnpreg/{mixture,readout,experiment,probe,probe_targets,affine,bimodal}.py`,
`results/{metrics,metrics_uniform,affine_metrics,probe_table.md}`. New run added:
`pnpreg/threeway_mixture.py` and `pnpreg/threeway_bimodal.py` (below).

The engineering standard here is high — the three-row calibration gate in Experiment 2, one row carrying a *designed*
condition-3 violation, plus a symmetric-by-construction floor row, is better practice than most published numerical
work. What follows is about what the experiments can support, not how carefully they were run.

---

## Why the LPN iterative method belonged in the comparison

Experiment 1 as shipped compares two fits:

- **fit (a)** ICNN `G_θ` on `∇G(y_k) = x_k`, read out `J_a = G_θ − ½y²`
- **fit (b)** plain ICNN on `∇J_b(y_k) = x_k − y_k` — its docstring calls it *"the tv_pm design"*

Both are direct gradient fits with **no inversion**. They differ only in the function class. That confounds two
independent axes — *is there a recovery step?* and *which class is `J` in?* — and leaves the folder unable to say where
the semiconvexity comes from. With only those two rows, "semiconvex readout beats convex readout" is compatible with
the wrong explanation, that the `−½‖·‖²` is a trick of the one-shot route.

Adding the third method settles it. `pnpreg/threeway_mixture.py` runs all three at Experiment 1's own budget, data and
network (σ = 0.5, hidden 64, 20 000 steps each). The 1-D inversion is a monotone bisection, exact to machine
precision, so **the solver is removed from the comparison** and any gap between the two LPN routes is the recovery
mathematics rather than an optimizer artifact.

| method | class | rel-L2 | min curvature | held-out residual | eval cost |
|---|---|---|---|---|---|
| **LPN Iterative recovery** | 1-semiconvex | **0.02 %** | **−0.9126** | 0.37 % | 7.2 s (inversion) |
| **One-shot recovery** | 1-semiconvex | 1.22 % | −0.9657 | 0.25 % | 0.05 s (forward) |
| **Direct fit** | convex | **100.96 %** | **−0.0024** | 99.78 % | 0.06 s (forward) |
| *exact* `f_reg = t·J_BVS` | 1-semiconvex | — | −0.9113 | — | — |

The two shipped rows reproduce (one-shot 1.22 % vs the recorded 1.2 %; direct 100.96 % vs 101 %), so this is the same
experiment with one row added. Figure: `results/figs/threeway_mixture_sigma0.5.png`.

---

## Finding 1 — the readout class is not negotiable (a mathematical fact)

`J = ψ* − ½‖·‖²` with `ψ*` convex is **1-semiconvex by construction**. That `−½‖·‖²` sits in the Fenchel formula that
**both** LPN routes share — it is not a property of skipping the inversion. So both routes carry the correct class for
free, and the table shows it: both find the dip (−0.9126 and −0.9657 against an exact −0.9113).

The direct fit parameterizes `J` as a plain ICNN, so it **assumes `f_reg` is convex**. That assumption is undocumented,
and it is a statement about the prior, not about the method:

- on a **log-concave** prior `J_BVS = J`, `f_reg` is convex, the class contains the target — and the direct fit is the
  better-conditioned estimator and wins (TV: **8.07 %** vs 13.73 % one-shot vs 14.48 % iterative);
- on a **nonconvex** prior `f_reg` has curvature down to `−1/t` and the class cannot contain it — the direct fit reads
  **101 %** and its curvature is pinned at 0.000 where the truth is −0.911.

No budget, dimension or optimizer changes this; a convex function cannot have negative curvature. Experiment 3.2 is
the controlled crossover inside one 64-D problem: the convex control matches the 63 convex directions at 1.13 % and
fails categorically on the single nonconvex one.

**Consequence: "the direct fit is best" is conditional, and the condition is convexity of `f_reg`.** Stated
unconditionally it is wrong, and Experiment 1 in this repository is the counterexample.

## Finding 2 — among the two valid routes, the ranking is a solver question (a numerical fact)

Both LPN routes are in the right class, so choosing between them is not about representation. It is about the
inversion:

| setting | inversion | Iterative | One-shot |
|---|---|---|---|
| 1-D mixture, exact bisection | free, machine precision | **0.02 %** | 1.22 % |
| 64-D TV, Adam solve, no safe α | 12.7 s/query; 0.45 % of solves leave the valid box | 14.48 % | **13.73 %** |

With an exact inversion the iterative route is the more accurate of the two — it inherits only `ψ_θ`'s error and the
Fenchel step adds nothing. At field dimension the inversion becomes the expensive, α-dependent step with no safe
setting (α = 0 trips the preimage tripwire, α = 0.1 biases the prior to 99.55 %), and it loses — at ~400× the query
cost.

**Consequence: the iterative-vs-one-shot ordering is a statement about the inversion's conditioning at that dimension,
not about the recovery mathematics.** Reporting either ordering without the dimension and the solver attached
overclaims.

These two findings are independent and should stay separate in the paper: Finding 1 is about which function class you
must use, Finding 2 is about which of the two admissible estimators is cheaper to make accurate.

---

## Experiment 3.2 extended: the missing cell, at field dimension

`pnpreg/threeway_bimodal.py` runs the same three methods on the 64-D bimodal prior at Experiment 3.2's data,
network, budget (50 000 steps) and scoring. The inversion is `src.invert.invert_cvx_gd` — the same Adam inverter,
lr and alpha `tv_pm` uses — so this is the fourth cell of the grid:

| | `f_reg` convex | `f_reg` nonconvex |
|---|---|---|
| **1-D, exact inversion** | — | Exp 1: Iterative **0.02 %** best |
| **64-D, solver inversion** | TV: Direct **8.07 %** best, Iterative 14.48 % worst | **this run** |

| method | class | cost | rel-L2 | u curvature | perp spectrum | resid median | resid p90 |
|---|---|---|---|---|---|---|---|
| LPN Iterative recovery | 1-semiconvex | inversion/query | **2.33 %** | −0.8807 | 13.63 % | 13.24 % | 22.75 % |
| One-shot recovery | 1-semiconvex | forward | 7.27 % | **−0.9142** | **0.47 %** | **3.49 %** | **5.01 %** |
| Direct fit | convex | forward | 16.00 % | **0.0012** | 1.02 % | 5.97 % | 16.36 % |
| *exact* | 1-semiconvex | — | — | −0.9113 | — | — | — |

Inversion certificate: median 7.6e-4, max 2.6e-3, max |w|∞ = 2.41 — the solve converged, so the iterative row is the
method, not the optimizer. The direct row reproduces Experiment 3.2's shipped fit (b) (16.00 % vs the recorded
15.4 %) and one-shot is fit (a). Figure: `results/figs/threeway_bimodal_s0.5.png`.

**Finding 1 holds at field dimension, and the failure is exactly localized.** The direct fit's u-curvature is
**0.0012** against an exact −0.9113 — it bridges the dip with a flat segment, visible in the figure — while matching
the 63 convex directions at 1.02 %. Its 16.00 % overall error is that one direction. Both semiconvex methods
reproduce the dip (−0.88, −0.91).

**Finding 2 holds too: the inversion degrades the iterative route at dimension.** It has the best aggregate value
error (2.33 %) but the worst gradient-level metrics by a wide margin — perp spectrum 13.63 % against 0.47 %, residual
p90 22.75 % against 5.01 % — and it pays an inversion per query. Those gradient-level metrics are the ones that matter
if the recovered `J` is to be used as a prior, so the ordering matches `tv_pm`.

### Seed variance, measured by accident

This configuration was run twice (identical data, budget and architecture; `br.fit` does not seed the weight init).
The value errors moved:

| | run 1 | run 2 |
|---|---|---|
| LPN Iterative recovery | 2.17 % | 2.33 % |
| One-shot recovery | 3.47 % | **7.27 %** |
| Direct fit | 15.62 % | 16.00 % |

Iterative and direct are stable to ~0.4 points; **one-shot's value error moved by a factor of two.** The
gradient-level metrics were far steadier (spectrum 0.36 % → 0.47 %, residual p90 4.71 % → 5.01 %), so the instability
is in the *value* reconstruction, which integrates gradient error over the domain. This is a direct measurement of
the single-seed shortfall below: a 3.47 % headline would not have replicated. **Any value-level percentage in the
paper needs at least three seeds.**

### Anchoring: a defect this run found in `tv_pm/prior_routes`

`G` is trained on the denoiser's own outputs `y_k = D(z_k)` — which *are* the training data. `tv_pm/prior_routes`
instead manufactures `G`'s inputs from `ψ_θ` (`conjugate_pairs_x`), the literal reading of `bin/_run.py`'s
conjugate-sampling step, which had no denoiser data to anchor on. Measured here at n = 64, that costs a factor of
**19** in value error — ψ_θ's error enters through the conjugate and accumulates as a systematic value offset while
the gradients stay fine. Data-anchoring is free and strictly better.

**The fix is in `tv_pm/prior_routes/three_way.py`, `train_G_grad`, lines 178–179:**

```python
    yk_tr, _ = conjugate_pairs_x(psi, u, xtr)      # ψ-anchored: ψ_θ's outputs
    yk_va, _ = conjugate_pairs_x(psi, u, xva)
```
becomes
```python
    yk_tr, yk_va = flat(tr["y"]), flat(va["y"])    # the sampler's own u_PM
```

and line 175's `load_psi()` becomes dead — the one-shot route stops depending on ψ_θ entirely. Retraining `G` is
~90 min; ψ_θ and J_θ are untouched. This may move the reported 13.73 % materially, possibly below the direct fit's
8.07 %, which would change that comparison's conclusion — worth doing before the TV numbers go in the paper.

### Reproducing

```
PY=~/miniforge3/envs/lpn_env/bin/python      # from pnp_reg/
$PY -m pnpreg.threeway_bimodal --smoke       # ~25 min (a 400-step psi never
                                             #   converges the inversion -- expected)
$PY -m pnpreg.threeway_bimodal --sigma 0.5   # ~35 min: 3 fits, then seconds on cached
                                             #   checkpoints; --force to retrain
```

---

## Shortfalls

### The real-denoiser experiments cannot exercise Finding 1

- **PIRATE (AWGN, σ = 1) fails conservativity**: ρ = 0.444, seven orders above the floor, λ_min(S) = −0.72. `D` is the
  gradient of nothing; no regularizer with `prox = D` exists. There is nothing to recover.
- **PIRATE+ passes PSD but is affine to 0.20 %** ⇒ its implicit regularizer is a **quadratic**, a Gaussian implicit
  prior. `DESIGN.md` draws the right conclusion: the nonparametric fit is *"SUPERSEDED for PIRATE+"*.

One is not a gradient field, the other is a Gaussian. **Neither provides a nonconvex-recovery testbed.** All evidence
for Finding 1 is synthetic by construction (Exps 1 and 3.2). That is a defensible way to validate a solver, but the
paper must not imply PIRATE demonstrates nonconvex recovery — it demonstrates the *diagnostic*, a separate
contribution.

### `probe_table.md` reports the misleading asymmetry, and the ordering inverts

The deliverable table carries only `ρ_full`:

| | ρ_full (in the table) | ρ_res (JSON only) |
|---|---|---|
| PIRATE (AWGN) | 0.444 | **0.098** |
| PIRATE+ | 0.0156 | **0.702** |

Experiment 3.1 established that the scale-free question is `ρ(I − A)`, because the regularizer's gradient is
`R(z) = z − D(z)`. In those units the ordering **reverses**. `DESIGN.md` says so and flags the "28× closer to
symmetric" headline — but the table, which is what gets read and cited, shows only the diluted number.
**Add a `ρ_res` column**: cheapest fix, largest correctness gain, numbers already in `affine_metrics.json`.

### The certificate can read 2.2 % on a 108 %-wrong fit

Recorded twice in `DESIGN.md`: uniform sampling at σ = 0.5, the convex control's **median residual is 2.2 % while its
value error is 108 %**; Exp 3.2, median 4.5 % but **q95 16.5 %**. The held-out prox residual is the ground-truth-free
score the whole method is sold on, so this is a limitation of the central instrument. **Report a high quantile with
every residual** — in `prior_routes` the p90 is what separates the methods (21 % vs 98–371 %), and the median alone
would have hidden it.

### Statistical and budget

- **Single seed everywhere.** Fine for the categorical claims (a convex net cannot bend downward); not fine for
  1.2 %, 3.2 %, 0.32 %, "5× class separation", which will end up in the abstract with no uncertainty. Three seeds
  costs minutes at these budgets.
- **Budget deviations**: Exp 1 at width 64 / 20 000 steps and Exp 3.2 at 50 000 against the protocol's 256 / 250 000 —
  recorded honestly. Within an experiment the fits share a budget so the comparisons are fair, but Exp 1's 1.2 % and
  `prior_routes`' 8.07 % are 20k- and 250k-step numbers on different problems and must never share a table.
- **Coverage dominates at low σ**: Exp 1 at σ = 0.25 reads 47 % under `p_z` sampling and **0.04 %** under uniform — a
  1000× swing from the sampling scheme alone. Whichever appears must name the scheme in the same sentence.

### Smaller

- `DESIGN.md` gives fit (b)'s perp spectrum as **0.76 %** in one paragraph and **1.13 %** a few lines later
  (`README.md` says 1.13 %); the 0.76 % looks like a stale first-run value.
- The AWGN `selfcheck_verdict = True` comes from a rule that only asserts `resid ≥ ρ_res/3` — a verdict that cannot
  fail should not print beside a genuine pass.
- No σ-ladder on real weights (one σ released). For PIRATE+ the family follows analytically since quadratics are
  closed under the viscous HJ semigroup — a nice result, but available *only because* the denoiser turned out affine.

---

## Recommendations, ordered

1. **Add `ρ_res` to `probe_table.{md,tex}`** — the current table supports the opposite of the correct reading.
2. **State the convexity condition wherever "direct fit wins" appears**, citing Exp 3.2's u-direction. One sentence
   prevents a wrong general claim.
3. **Keep Findings 1 and 2 separate** — class vs solver. Merging them produces the unfalsifiable "our method is best".
4. **Three seeds** for Exps 1 and 3.2 so the headline percentages carry uncertainty.
5. **Report a high quantile with every residual.**
6. Reconcile the 0.76 % / 1.13 % inconsistency in `DESIGN.md`.
7. If real-data nonconvex recovery is wanted, it needs a denoiser that is neither non-conservative nor affine — i.e.
   **retrain one** (`DESIGN.md` prices a σ-ladder at ~30 min/σ on MPS), labelled as your training, not theirs.
   Otherwise state plainly that the nonconvex evidence is synthetic by design.

## Reproducing the added run

```
PY=~/miniforge3/envs/lpn_env/bin/python      # from pnp_reg/
$PY -m pnpreg.threeway_mixture --smoke       # ~40 s end-to-end check
$PY -m pnpreg.threeway_mixture --sigma 0.5   # ~8 min: the table and the figure
```

Outputs: `results/threeway_mixture_sigma0.5.json`, `results/figs/threeway_mixture_sigma0.5.{png,pdf}`,
checkpoints and curves under `results/ckpt/` and `results/threeway_mixture_curves_s0.5.npz` so the figure can be
redrawn without retraining.
