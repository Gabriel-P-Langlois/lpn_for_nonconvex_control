# `notebooks/` — the paper's experiments, one notebook per family

| notebook | what it is |
|---|---|
| `quadratic_l1`, `minplus`, `concave_quad`, `negl1` | the paper's four production families, thin wrappers over `bin/_run.py` |
| `posterior_mean_l1` | **standalone**, different data model — see below |
| `tv_pm/recover_tv` | **standalone** TV posterior-mean example (work2.tex Instantiation B): recover a TV denoiser's `f_reg` and denoise with it. Its own folder, docs, and `results/`; start at `tv_pm/README.md` |

`posterior_mean_l1.ipynb` is not one of the four and does not follow their rules.
It is the imaging example of `CLAUDE/_reviews/work2.tex` (Instantiation A):
recover a posterior-mean denoiser's implicit regularizer `f_reg` — a prior with
no closed form in its own argument — with **one** network, from **denoiser
evaluations alone**. The value function `S_ε` is assumed never observed, which
is why one network suffices: prox optimality makes `∇f_reg(y_k) = x_k − y_k`
exact at `y_k = u_PM(x_k)`, so there is no ψ to learn, nothing to invert and
nothing to conjugate. It carries its own training loop and evaluation because it
tests a *different data model*, so there is no `bin/` config for it to wrap.
Only its ground truth is imported (`src.targets.PosteriorMeanL1`, pinned by
`tests/test_posterior_mean_l1.py`), and `S_ε` enters there and nowhere else.
Gradients fix the prior only up to an additive constant, which the data does not
identify and the prox does not see, so its error is reported with the best
constant removed. Its numbers are a **scoping budget** — smaller N, width and
steps than D2 — and are not comparable to `logs/summary_d2_full.csv`. Everything
below concerns the four production notebooks.

These four notebooks replace the per-(family, dimension) notebooks in
`legacy/old_notebooks/`. They are thin wrappers over the shared pipeline
(`bin/_run.py`, figures via `bin/plot.py`): no mathematics lives in a notebook,
so every correction and protocol amendment lands here automatically. Each
notebook covers its family at every reported dimension d ∈ {2, 4, 8, 16, 32, 64}:
the sweep results table (read from `logs/*_metrics.json`), the diagnostic
figures regenerated from the production checkpoints in `logs/ckpt/`, and a
`RETRAIN` flag that reruns the full pipeline under the exact protocol.

They are committed **executed**, so the production numbers and figures are
visible without running anything. Protocol and audit trail:
`../numerics_audit.tex`; chronology: `../changes.txt`.

## Old → new

| old notebooks in `legacy/old_notebooks/` | replaced by |
|---|---|
| `exp_4_1_2_quadratic_{2..64}D.ipynb`, duplicate `exp_L1_prior_{2..64}D.ipynb`, `exp_4_1_2_quadratic_2D_MC.ipynb` | `quadratic_l1.ipynb` |
| `exp_1_minplus_{2..64}D.ipynb` (+ `8D copy` duplicate) | `minplus.ipynb` |
| `exp_quadratic_concave_prior_{2..64}D.ipynb` (+ `_MC_32D`, `_32D_test_v1` stale variants) | `concave_quad.ipynb` |
| `exp_NegL1_prior_{2..64}D.ipynb` | `negl1.ipynb` |

## Deliberately not rebuilt

- `exp_4_1_3_minplus_8D.ipynb`, `exp_4_1_4_minplus_8D.ipynb` — despite the
  names these are **max-plus** (Hopf) priors, not the paper's min-plus mixture,
  and they appear nowhere in the manuscript. Ported to `bin/not_in_paper/`
  (see its README for the sign bug and the training-scheme difference).
- `exp_4_2_1_1D.ipynb` — a d = 1 quadratic-ℓ₁ illustration from an older
  section numbering; not part of the current §4. `bin/quadratic_l1.py --dim 1`
  reproduces it if ever needed.
- `laplacian_train.ipynb`, `old_laplacian_experiment/` — unrelated to the paper.

## Running

Kernel: the `lpn_env` interpreter (`~/miniforge3/envs/lpn_env/bin/python`);
the system Python has no torch. With `RETRAIN = False` (default) a full
notebook run only regenerates figures from checkpoints (a few minutes per
dimension; Route-1 curves require the inversions). With `RETRAIN = True` the
pipeline retrains under the name `<family>_<d>D_retrain`, which cannot clobber
the production checkpoints. Runs at d > 8 are opt-in by project policy.
