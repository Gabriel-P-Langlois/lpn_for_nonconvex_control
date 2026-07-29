# `tv_pm/` — TV posterior-mean denoiser (work2.tex Instantiation B)

Standalone. Recovers `f_reg`, the implicit regularizer of the anisotropic-TV
posterior-mean denoiser, from denoiser evaluations alone, with one convex
network. Status: complete through the demonstration — the recovered prior's
prox reproduces the denoiser (see `results/`). All results are preliminary and
single-seed.

**Start with `recover_tv.ipynb`**, which runs the experiment for one
configuration (`ARCH = fc|conv`). The experimental design — the method, the
sampler and its validation, and the design decisions — is in `DESIGN.md`.
Numbers and figures are in `results/`.

| file | what it is |
|---|---|
| **`recover_tv.ipynb`** | the experiment: data → model → score → figures → denoising → cost |
| `DESIGN.md` | the experimental design |
| `tvpm/` | the code: `sampler.py` (MCMC `u_PM`), `quadrature.py` (exact `n=2` reference), `dataset.py` (data), `recover.py` (train + score), `icnn.py` (convolutional ICNN), `figures.py`, `denoise.py`, `paths.py` |
| `tests/` | gates; run them before trusting a result |
| `images/` | the two source images (`.mat`), tracked so the repository is self-contained |
| `data/` | cached sampler output (gitignored, regenerable) |
| `results/` | shipped figures, `metrics.csv`, checkpoints, and the numbers write-up |
| `archive/` | superseded notebooks, kept for reference only |

## Reproducing the results

Use the `lpn_env` interpreter (the system Python has no PyTorch), and run
everything from this directory so that the `tvpm` package is importable:

```
PY=~/miniforge3/envs/lpn_env/bin/python
```

**Without computing anything.** Everything reported is in `results/`: figures
in `figs/`, numbers in `metrics.csv`, a one-page summary in
`results/README.md`.

**The notebook (recommended).** Set the parameters in the configuration cell
and run all cells: it samples the data it needs, trains the network if no
checkpoint matches, and produces every number and figure. Stage 1 (data):
`SIGMA`, `T`, `PM_SWEEPS`, `SCALE`. Stage 2 (fit): `ARCH`, `BETA`,
`FIT_STEPS`. `IMAGE` selects the image that Section 4 denoises; `REFIT = True`
forces retraining. Every output is keyed on the full configuration, so changing
a parameter writes to new files and cannot overwrite an earlier run; only
`REFIT` overwrites.

**The same stages from the command line:**

```
$PY tests/test_sampler.py           # sampler unit tests (~3 s)
$PY tests/test_quadrature.py        # bias gate: sampler unbiased with TV on (~40 s)
$PY tests/test_params.py            # (sigma,t) algebra, cache keys, ckpt names (~5 s)
$PY -m tvpm.dataset                 # ~15 min: samples u_PM, writes data/*.npz
$PY -m tvpm.recover --arch fc   --standardize --beta 20 --steps 250000   # ~2 h  (CPU)
$PY tests/test_icnn.py              # convexity gate for the conv net (~5 s) — must pass
$PY -m tvpm.recover --arch conv --standardize --beta 20 --steps 250000   # ~13 h (CPU)
$PY -m tvpm.figures                 # qualitative panels + learned kernels (seconds)
$PY -m tvpm.denoise --arch conv     # denoising with the learned prior (~2 min)
```

`tvpm.dataset` and `tvpm.recover` accept `--sigma`/`--t` (`eps` and `lam`
follow); `tvpm.figures` and `tvpm.denoise` render the default `(sigma, t)`
only — for any other pair, use the notebook, which loads the matching checkpoint
explicitly. Checkpoints are written to `../logs/ckpt/` (gitignored); the
shipped copies live in `results/ckpt/`.

## The model's parameters

The posterior is `exp(-E/eps)` with `E(u) = ‖u-x‖²/(2t) + ATV(u)`, so there are
two degrees of freedom, not four. Choose `(sigma, t)`; then

```
eps = sigma^2 / t          lam = 2t          sqrt(t*eps) = sigma
```

via `tvpm.sampler.from_sigma_t` (`params()` is its inverse). `sigma` is the
noise the denoiser is built for and `t` the denoising strength — as `eps → 0`,
`u_PM` tends to the MAP, `prox_{t·ATV}` — and `eps` is the temperature: the
degree to which `f_reg` is a *smoothed* TV rather than TV.

## Settings

- Noise/smoothing: the configured pair is `σ = t = 20/256`. The results
  currently in `results/` were produced at `σ = 10/256`, `t = 16/256` and will
  be superseded by reruns at the configured pair.
- `8×8` patches (`n = 64`), `m = 8000` MCMC sweeps, `N = 20000` training
  patches.
- Network: `β = 20` Softplus, standardized input (global for conv), `wclip`
  each step. Loss: MSE on `∇J`.
- Everything runs on CPU by project convention.
