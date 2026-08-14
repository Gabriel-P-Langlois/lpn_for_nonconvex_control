# Learned proximal networks for (non-)convex high-dimensional Hamilton–Jacobi PDEs

Numerics for the SIAP revision. Based on
[What's in a Prior? Learned Proximal Networks for Inverse Problems](https://openreview.net/pdf?id=kNPcOaqC5r)
(Fang, Buchanan, Sulam, ICLR 2024); the repository began as a fork of the
Sulam-Group implementation.

**Read `numerics_audit.tex` (9 pp) first.** It states the current protocol, the
bugs fixed, the claims retracted, and the open items. `changes.txt` is the
chronological record and lists every superseded number. Figures and tables that
predate the protocol in the audit should not be cited.

---

## The two recoveries

We learn a convex potential ψ(y) = ½‖y‖² − S(y,1) = (J + ½‖·‖²)\*(y), so that
∇ψ = prox_J. From ψ we recover the prior J in two ways:

| | method | cost | parameters |
|---|---|---|---|
| **LPN Iterative recovery** (baseline) | invert ∇ψ(y) = x per query, then J(x) = ⟨x,y⟩ − ψ(y) − ½‖x‖² | one optimization problem per query | α (a regularization weight that biases the recovered prior at order α) |
| **One-shot recovery** (ours) | fit a second network G ≈ ψ\*, then J(x) = G(x) − ½‖x‖² | one forward pass | none |

LPN Iterative recovery is Algorithm 2 of Fang et al. Under a symmetric protocol the two recoveries
are comparable: in the most recent sweep, LPN Iterative recovery attained the lower error in 7
of 12 configurations. We therefore claim that One-shot recovery matches a fully tuned
inversion baseline while requiring no per-query optimization; we do not claim
that either recovery dominates the other.

---

## Layout

```
src/            single source of truth
  network.py      the Softplus ICNN (one network, used for both ψ and G)
  conv_icnn.py    the convolutional ICNN (locality + shift invariance)
  targets.py      exact S, J, ψ, and the preimage map, per family
  train.py        mini-batch MSE training, step-based budget
  gradfit.py      gradient-supervised training (train_grad) + Units
  recovery.py     both recoveries + the shared certificates
  invert.py       the convex inverter (LPN Iterative recovery)
  plotting.py     cross-sections
  diagnostics.py  in-distribution figures (see below)

bin/            thin configurations; the mathematics lives in src/
  _run.py         the pipeline: data → ψ → G → both recoveries → figures
  quadratic_l1.py  negl1.py  concave_quad.py  minplus.py    (--dim d)
  run_all.py      sweep driver → logs/summary.csv
  plot.py         regenerate figures from a checkpoint, without retraining
  not_in_paper/   maxplus_case3/4 — complete experiments, not reported in the manuscript

notebooks/      the paper's experiments, one executed notebook per family
tv_pm/          the TV posterior-mean (imaging) experiment; its own README
pnp_reg/        the plug-and-play registration experiments; its own README
tests/          regression tests for the targets and the pipeline
ext/            the upstream LPN repository, verbatim, + PROVENANCE.md + its license
legacy/         old_notebooks/ (the original notebooks, superseded) + slurm/
logs/           run output. Tracked: ckpt/, *_metrics.json, summary*.csv,
                superseded/ (provenance for the audit's numbers). Ignored:
                transcripts and per-run figures (regenerable via bin/plot.py)
figs/           the figures the audit embeds  (committed)
```

`numerics_audit.tex` compiles from this directory (its figures live in `figs/`).

## Install

```bash
conda env create -f environment.yml     # python >= 3.10, torch (CPU)
conda activate lpn_env
```

No `pip install` step is required: the code imports as `from src...`, and the
scripts run from this directory.

## Run

```bash
python bin/quadratic_l1.py --dim 8            # one family, one dimension
python bin/quadratic_l1.py --dim 2 --smoke    # ~1 min sanity check
python bin/run_all.py --dims 2 4 8            # sweep → logs/summary.csv
```

Each run writes `logs/ckpt/<run>_{psi,G}.pth`, `logs/<run>_metrics.json`, and a
cross-section figure. A configuration takes ≈ 30 min on CPU (width 256,
S = 250 000 steps).

Runs above d = 8 are opt-in: `run_all.py` refuses them without
`--allow-high-dim`. They are costly, and the high-dimensional diagnostics
require care (see `numerics_audit.tex`).

## Figures

Because every run checkpoints, figures regenerate from weights in ≈ 3 minutes
instead of a ≈ 28-minute retrain:

```bash
python bin/plot.py --family quadratic_l1 --dim 16 --open
```

This produces, for any family and dimension:

- **cross-section** — the classical axis slice. Reliable at d = 2 but
  misleading at d ≳ 8: the slice pins the remaining d − 1 coordinates at zero,
  a region that carries no training data in high dimension (see
  `numerics_audit.tex`).
- **conditional cross-section** — varies x₁ while drawing the remaining
  coordinates from the training distribution, with each curve centered at
  x₁ = 0; the in-distribution replacement for the axis slice. For separable
  priors the exact curves collapse to a single line.
- **typical ray** — the profile along t·𝟙/√d, which lies in the bulk of the
  training distribution.
- **predicted vs. true** — Ĵ(x) against J(x) over the test set, one panel per
  route; independent of any choice of slice.
- **prox scatter** — the learned ∇ψ against the exact prox_J. This diagnoses
  the first network, which both recoveries share: LPN Iterative recovery inverts it and One-shot recovery is
  fitted to its image, so its error bounds both.
- **preimage scatter** — ∇G(x) against the analytic preimage y\*(x). This
  checks the one-shot recovery's defining identity, ∇G = (∇ψ)⁻¹, directly.

## Experimental protocol

- **Training box.** ψ is trained on [−A,A]^d, where A is the smallest
  half-width such that ∇ψ maps the training box onto the query box [−4,4]^d;
  `problem.train_halfwidth(a)` computes A for each family. Both recoveries require
  this: LPN Iterative recovery evaluates ψ at preimages that leave the query box, and One-shot recovery
  trains G at inputs that lie strictly inside it. All errors are reported on
  the query box.
- **Training budget.** The budget is S = 250 000 optimizer steps with
  N = 15 000·d training samples and mini-batches of 512. The learning rate is
  10⁻³, reduced by a factor of 10 at 50 % and 75 % of S. Epochs are a derived
  quantity and are not reported. The schedule of the reference implementation
  is not reused: that implementation trains full batch, so one of its epochs
  corresponds to one gradient step, and its decay cadence presumes exact
  gradients.
- **No early stopping.** The networks under-fit rather than overfit, and early
  stopping truncated training on validation noise. The checkpoint with the
  best validation error is kept instead.
- **LPN Iterative recovery reporting.** LPN Iterative recovery is reported at the better of α ∈ {0, 0.1},
  with diverged solves excluded. A zero prox residual does not certify
  convergence, since ∇ψ(y) → x along a diverging minimizing sequence.
- **Error metric.** Absolute RMSE is not comparable across dimensions, because
  ‖J‖ grows like 2d; relative error is reported.

## License

MIT (see `LICENSE`), except `ext/lpn/`, which is the upstream LPN code and
keeps its own Apache 2.0 license (`ext/lpn/LICENSE`).
