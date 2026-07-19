# MANUAL — reproduce the TV posterior-mean example

Recover a TV denoiser's implicit regularizer `f_reg` from denoiser evaluations
alone, with one convex net. Method and findings: `DESIGN.md`. Results: `results/`.

## Environment

Use the `lpn_env` interpreter (has PyTorch; the system Python does not):

```
PY=~/miniforge3/envs/lpn_env/bin/python
```

Run everything **from this directory** (`numerics/notebooks/tv_pm/`). The
Barbara/cameraman source images must be present at `../../ideas/old_files/*.mat`.

## 0. Just look — no compute

Everything reported is in `results/`: figures in `figs/` (PNG + PDF), numbers in
`metrics.csv`, one-page story in `results/README.md`. The notebook
`recover_tv.ipynb` is committed **executed**, so its figures and numbers are
visible without running anything.

## The notebook (recommended entry point)

`recover_tv.ipynb` runs the whole experiment as a thin wrapper over the modules
below. Set `ARCH = "fc"` (default) or `"conv"` and `RETRAIN = False` (use the
shipped checkpoint, seconds) or `True` (retrain). It needs the sampler cache — run
step 1 once first, or the notebook builds it. The steps below are the same thing
from the command line.

## 1. Regenerate the figures from the shipped checkpoints (~15 min)

The trained nets ship in `results/ckpt/` (the scripts find them automatically).
Only the sampler cache is missing — build it once, then draw:

```
$PY dataset.py                      # ~15 min: samples u_PM, writes data/*.npz
$PY step4_figures.py                # qualitative panels + learned kernels (seconds)
$PY denoise_demo.py --arch conv     # the prior in action, denoising (~2 min)
$PY denoise_demo.py --arch fc       # (same for the fully-connected prior)
```

`denoise_demo.py` re-samples `u_PM` itself, so it does not need step 1's cache;
`step4_figures.py` does (it reads `data/eval_8x8_m8000.npz`).

## 2. Reproduce from scratch — training

```
$PY test_sampler.py                 # sampler unit tests (~3 s)
$PY test_quadrature.py              # Step-0 gate: sampler unbiased with TV on (~40 s)
$PY dataset.py                      # ~15 min: the data (once)
$PY recover.py --arch fc   --standardize --beta 20 --steps 250000   # ~2 h  (CPU)
$PY test_conv_icnn.py               # convexity gate for the conv net (~5 s) — MUST pass
$PY recover.py --arch conv --standardize --beta 20 --steps 250000   # ~13 h (CPU)
```

Checkpoints are written to `../../logs/ckpt/`. Both nets reach ~10.7% held-out
prox residual; the conv reproduces the denoiser better on transfer (see
`results/README.md`).

## Fixed settings — do not change casually

- Noise/smoothing: `σ = 10/256`, `λ = 32/256` → `t = 0.0625`, `ε = 0.024414`
  (the MATLAB's tabulated pair; `√(tε) = σ`, `tε = σ²`).
- `8×8` patches (`n = 64`), `m = 8000` MCMC sweeps, `N = 20000` train.
- Network: `β = 20` Softplus (matched to the TV kink), **standardized input**
  (global for conv), `wclip` each step. Loss: MSE on `∇J`.
- Everything is CPU by project convention.

## Files

`sampler.py` (u_PM MCMC) · `quadrature.py` (exact `n=2` check) · `dataset.py`
(build data) · `recover.py` (train + score, `--arch fc|conv`) · `conv_icnn.py`
(convolutional ICNN) · `step4_figures.py`, `denoise_demo.py` (figures) ·
`test_*.py` (gates).
