# Learned proximal networks for (non-)convex high-dimensional Hamilton–Jacobi PDEs

Numerics for the SIAP revision. Based on
[What's in a Prior? Learned Proximal Networks for Inverse Problems](https://openreview.net/pdf?id=kNPcOaqC5r)
(Fang, Buchanan, Sulam, ICLR 2024); this repo was forked from Sulam-Group.

**Read `numerics_audit.tex` (9 pp) first.** It states the current protocol, the
bugs fixed, the claims retracted, and what is open. `changes.txt` is the
chronological record and lists every superseded number. If a figure or a table
predates the protocol in the audit, do not cite it.

---

## The two routes

We learn a convex potential ψ(y) = ½‖y‖² − S(y,1) = (J + ½‖·‖²)\*(y), so that
∇ψ = prox_J. From ψ we recover the prior J two ways:

| | how | cost | knobs |
|---|---|---|---|
| **Route 1** (baseline) | invert ∇ψ(y) = x per query, then J(x) = ⟨x,y⟩ − ψ(y) − ½‖x‖² | an optimization per query | α (a regularizer that biases the answer) |
| **Route 2** (ours) | fit a second network G ≈ ψ\*, then J(x) = G(x) − ½‖x‖² | one forward pass | none |

Route 1 is Fang et al.'s Algorithm 2. On a symmetric protocol the two are
comparable. **Do not claim Route 2 dominates**: under the last fair table Route 1
won 7 of 12 configurations. The defensible claim is that Route 2 *matches* a
fully-tuned inversion baseline while never inverting.

---

## Layout

```
src/            single source of truth
  network.py      the Softplus ICNN (one network, used for both ψ and G)
  targets.py      exact S, J, ψ, and the preimage map, per family
  train.py        mini-batch MSE training, step-based budget
  recovery.py     both routes + the shared certificates
  invert.py       the convex inverter (Route 1)
  plotting.py     cross-sections
  diagnostics.py  in-distribution figures (see below)

bin/            thin configs; the math lives in src/
  _run.py         the pipeline: data → ψ → G → both routes → figures
  quadratic_l1.py  negl1.py  concave_quad.py  minplus.py    (--dim d)
  run_all.py      sweep driver → logs/summary.csv
  plot.py         regenerate figures from a checkpoint, no retraining
  not_in_paper/   maxplus_case3/4 — real experiments, NOT in the manuscript

ext/            the upstream LPN repo, verbatim, + PROVENANCE.md
exps/           the original 38 notebooks. Superseded; kept for reference.
legacy/slurm/   archived cluster artifacts
logs/           metrics, figures, checkpoints  (gitignored)
figs/           the few figures the audit embeds  (committed)
```

## Install

```bash
conda env create -f environment.yml     # python 3.9, torch
pip install -e .
```

## Run

```bash
python bin/quadratic_l1.py --dim 8            # one family, one dimension
python bin/quadratic_l1.py --dim 2 --smoke    # ~1 min sanity check
python bin/run_all.py --dims 2 4 8            # sweep → logs/summary.csv
```

Each run writes `logs/ckpt/<run>_{psi,G}.pth`, `logs/<run>_metrics.json`, and a
cross-section figure. A configuration takes ≈ 30 min on CPU (width 256,
S = 250 000 steps).

**Runs above d = 8 are opt-in.** `run_all.py` refuses them without
`--allow-high-dim`; they are expensive and the results need care (see below).

## Figures

Because every run checkpoints, figures regenerate from weights in ≈ 3 minutes
instead of a ≈ 28-minute retrain:

```bash
python bin/plot.py --family quadratic_l1 --dim 16 --open
```

This emits, for any family and dimension:

- **cross-section** — the classical axis slice. Valid at d = 2; **misleading at
  d ≳ 8** (below).
- **conditional cross-section** — vary x₁, draw the *other* coordinates from the
  training distribution, centre each curve at x₁ = 0. The in-distribution
  replacement. For separable priors the exact curves collapse to one line.
- **typical ray** — profile along t·𝟙/√d, which lies in the bulk.
- **predicted vs. true** — Ĵ(x) against J(x) over the test set, one panel per
  route. Cannot be gamed by a choice of slice.
- **prox scatter** — learned ∇ψ against exact prox_J. A diagnostic of the
  **shared** first network: Route 1 inverts it, Route 2 is fitted to its image,
  so it bounds both.
- **preimage scatter** — ∇G(x) against the analytic preimage y\*(x). This is
  Route 2's own identity, ∇G = (∇ψ)⁻¹, and the most direct check of our method.

### Why the classical cross-section lies in high dimension

A cross-section along axis 1 pins the other d−1 coordinates at **exactly zero**.
Under the uniform measure on [−A,A]^d that slice has volume fraction
(ε/A)^(d−1) — about 10⁻¹⁵ at d = 16 and 10⁻⁶³ at d = 64. With N = 240 000
training points, **not one lies near it**. The figure is pure extrapolation while
the reported RMSE is measured on typical points, where the network fits.

At d = 16, the *same checkpoint* gives ψ offset by −1.45 on the axis slice and
lying on the exact curve on the conditional slice. This is the mechanism behind
the manuscript's own open question about the d = 64 cross-sections; the guess
there ("probably due to the way we sample the hypercube") was right.

## Conventions that will surprise you

- **The training box is not the query box.** ψ is trained on [−A,A]^d with A
  chosen so ∇ψ maps it *onto* the query box [−4,4]^d. Both routes need this:
  Route 1 evaluates ψ at preimages that leave the query box, Route 2 trains G at
  inputs that lie strictly inside it. `problem.train_halfwidth(a)` computes A.
- **The budget is optimizer steps, not epochs.** S = 250 000, N = 15 000·d.
  Epochs are derived and never reported. Do *not* copy the paper's decay cadence
  (×0.1 every 20 %): the paper is full-batch, so one of its "epochs" is one
  gradient step, and that cadence presumes exact gradients. What tracks ψ's
  error is the number of steps at lr = 1e-3.
- **No early stopping.** We under-fit, not overfit; it truncated G on noise.
- **Route 1 is reported at the best of α ∈ {0, 0.1}, excluding diverged solves.**
  A zero prox residual does *not* certify convergence — ∇ψ(y) → x along a
  diverging minimizing sequence.
- **Absolute RMSE is not comparable across dimensions** (‖J‖ grows like 2d).
  Report relative error.

## Citation

```bib
@inproceedings{
    fang2024whats,
    title={What's in a Prior? Learned Proximal Networks for Inverse Problems},
    author={Zhenghan Fang and Sam Buchanan and Jeremias Sulam},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024}
}
```
