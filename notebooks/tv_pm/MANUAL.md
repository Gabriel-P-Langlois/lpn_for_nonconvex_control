# MANUAL — reproduce the TV posterior-mean example

Recover a TV denoiser's implicit regularizer `f_reg` from denoiser evaluations
alone, with one convex net. Method and findings: `DESIGN.md`. Results: `results/`.

## Environment

Use the `lpn_env` interpreter (has PyTorch; the system Python does not):

```
PY=~/miniforge3/envs/lpn_env/bin/python
```

Run everything **from this directory** (`numerics/notebooks/tv_pm/`), so that the
`tvpm` package is importable. The Barbara/cameraman source images must be present
at `../../ideas/old_files/*.mat`.

## Layout

```
recover_tv.ipynb      the experiment, self-contained (entry point)
noise_diagnostic.ipynb  sampler noise study (delta vs m)
tvpm/                 all the code
  paths.py            every filesystem location, defined once
  sampler.py          u_PM by MCMC; the (sigma, t) <-> (eps, lam) algebra
  quadrature.py       exact n=2 oracle, for the Step-0 gate
  dataset.py          patches -> u_PM, cached; ensure() provisions
  icnn.py             convolutional ICNN
  recover.py          train + score; ensure_model() provisions
  figures.py          the qualitative panels
  denoise.py          the prior in action, vs the true denoiser
tests/                gates; run them before trusting a result
data/  results/       cached sampler output; figures, checkpoints, metrics
archive/              superseded notebooks, kept for reference only
```

## 0. Just look — no compute

Everything reported is in `results/`: figures in `figs/` (PNG + PDF), numbers in
`metrics.csv`, one-page story in `results/README.md`. `recover_tv.ipynb` is
committed **executed**, so its figures and numbers are visible without running it.

## The notebook (recommended entry point)

`recover_tv.ipynb` runs the whole experiment on its own. Set the parameters in the
configuration cell and run all: it samples the data it needs, trains the network
if no checkpoint matches, and produces every number and figure. Nothing has to be
run outside it, and a cell before each stage reports what will be computed and
roughly how long it will take.

The parameters are grouped by stage. Stage 1 (data): `SIGMA`, `T`, `PM_SWEEPS`,
`SCALE`. Stage 2 (fit): `ARCH`, `BETA`, `FIT_STEPS`. Plus `IMAGE`, the image
section 4 denoises — **not** the held-out split, which is fixed in
`dataset.SPLITS`.

Every output is keyed on the whole configuration, so changing any parameter writes
to new files and cannot overwrite an earlier run's data, checkpoints, or figures.

## The same thing from the command line

Modules live in a package, so they run with `-m` from this directory.

```
$PY -m tvpm.dataset                 # ~15 min: samples u_PM, writes data/*.npz
$PY -m tvpm.figures                 # qualitative panels + learned kernels (seconds)
$PY -m tvpm.denoise --arch conv     # the prior in action, denoising (~2 min)
$PY -m tvpm.denoise --arch fc       # (same for the fully-connected prior)
```

`tvpm.denoise` re-samples `u_PM` itself, so it does not need the cache;
`tvpm.figures` does (it reads `data/eval_8x8_m8000.npz`).

`tvpm.dataset` and `tvpm.recover` accept `--sigma`/`--t` (`eps`, `lam` follow).
`tvpm.figures`/`tvpm.denoise` render the default `(sigma, t)` only; for any other
pair use the notebook, which loads the matching checkpoint explicitly.

## Reproduce from scratch — training

```
$PY tests/test_sampler.py           # sampler unit tests (~3 s)
$PY tests/test_quadrature.py        # Step-0 gate: sampler unbiased with TV on (~40 s)
$PY tests/test_params.py            # (sigma,t) algebra, cache keys, ckpt names (~5 s)
$PY -m tvpm.dataset                 # ~15 min: the data (once)
$PY -m tvpm.recover --arch fc   --standardize --beta 20 --steps 250000   # ~2 h  (CPU)
$PY tests/test_icnn.py              # convexity gate for the conv net (~5 s) — MUST pass
$PY -m tvpm.recover --arch conv --standardize --beta 20 --steps 250000   # ~13 h (CPU)
```

Checkpoints are written to `../../logs/ckpt/`. Both nets reach ~10.7% held-out
prox residual; the conv reproduces the denoiser better on transfer (see
`results/README.md`).

## The model's parameters

The posterior is `exp(-E/eps)` with `E(u) = ‖u-x‖²/(2t) + ATV(u)`, so there are
**two** degrees of freedom, not four. Choose `(sigma, t)`; then

```
eps = sigma^2 / t          lam = 2t          sqrt(t*eps) = sigma
```

via `tvpm.sampler.from_sigma_t` (`params()` is its inverse). `sigma` is the noise
the denoiser is built for, `t` the denoising strength — as `eps -> 0`, `u_PM` tends
to the MAP, `prox_{t·ATV}` — and `eps` is the temperature, i.e. exactly how much
`f_reg` is a *smoothed* TV rather than TV.

## Fixed settings — do not change casually

- Noise/smoothing: `σ = 10/256`, `t = 16/256` → `λ = 32/256`, `ε = 0.024414`
  (the MATLAB's tabulated pair).
- `8×8` patches (`n = 64`), `m = 8000` MCMC sweeps, `N = 20000` train.
- Network: `β = 20` Softplus (matched to the TV kink), **standardized input**
  (global for conv), `wclip` each step. Loss: MSE on `∇J`.
- Everything is CPU by project convention.
