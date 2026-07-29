# `tv_pm/results/` — preliminary results for the TV posterior-mean example

Recovering the anisotropic-TV denoiser's implicit regularizer `f_reg` from
denoiser evaluations alone, with one convex network (`work2.tex` Instantiation B).
Full method and audit trail in `../DESIGN.md`; chronology in `../../changes.txt`.

## These are OBSERVATIONS, not conclusions

Everything below is preliminary and empirical: **single seed, no repeats, no error
bars.** The "floor" and "κ" come from a two-parameter fit to three `m` points; the
"sampler-limited" reading is an inference from a slope; the denoising comparison is
two images, patch-wise. Numbers are what these particular runs produced, not
statistically established claims. Read them as what we have SEEN, and confirm
(repeats, seeds, more images/`m`) before anything goes in the paper as a claim.

## Headline (observed)

`f_reg` is recovered to a **~10.7% held-out prox residual** with **corr(J, TV) =
0.99** on the transfer image (train Barbara, evaluate cameraman). The fully-
connected and convolutional ICNNs tie on this metric; the conv reaches it with 9×
fewer parameters and gives the cleanest recovery (corr 0.997) — but runs 5–6× slower
(convolutions cost more FLOPs despite fewer parameters). More MCMC sweeps keep
improving the recovery (see the m-sweep below), so ~10.7% at the production budget
is not a hard limit.

## Figures (`figs/`)

| file | what it shows |
|---|---|
| `step4_core_fc.{png,pdf}` | 3-panel: J vs TV, prior-penalises-structure, conditional slices (production FC net) |
| `step4_core_conv.{png,pdf}` | the same for the conv-ICNN (corr 0.997) |
| `step4_kernels_conv.{png,pdf}` | the conv's learned first-layer 3×3 kernels — **many are `[+1,−1]` difference stencils**: the net rediscovered TV's building blocks from denoiser evaluations alone |
| `denoise_fc_cameraman.{png,pdf}` | **the prior IN ACTION**: denoise with the prox of the learned prior, compare to the sampler denoiser `u_PM`. FC: PSNR(û, u_PM) = 43.7 dB |
| `denoise_conv_cameraman.{png,pdf}` | same with the conv prior: **PSNR(û, u_PM) = 54.0 dB** — reproduces the denoiser 10 dB better than FC |

Regenerate (from checkpoints, no retraining):
`python -m tvpm.figures` (from ../)   (qualitative panels, seconds)
`python -m tvpm.denoise --arch conv` (from ../)   (denoising demo, ~2 min: sampler + prox solve)

## The demonstration that matters (`denoise_*` figures)

The posterior-mean denoiser IS the proximal operator of `f_reg`, so if the
recovery is faithful the prox of the LEARNED prior must reproduce it:
`û(x) = argmin_u J_θ(u) + ½‖u−x‖²  ≈  u_PM(x)`. It does — `u_PM` and `û` are
visually identical and their difference lives only at edges (where `f_reg` is most
nonlinear). **This is the validation that `f_reg` was actually recovered**, not
just correlated with TV.

### `û` vs `u_PM` across both images (does the prior reproduce the denoiser)

| image | arch | PSNR | SSIM | rel-L2 |
|---|---|---|---|---|
| Barbara (train dist.) | fc | 59.1 dB | 0.9998 | 0.22% |
| Barbara (train dist.) | conv | 57.7 dB | 0.9997 | 0.26% |
| cameraman (transfer) | fc | 43.7 dB | 0.9987 | 1.25% |
| cameraman (transfer) | conv | 54.0 dB | 0.9995 | 0.38% |

Two readings:
- **In-distribution (Barbara), both priors reproduce the denoiser essentially
  perfectly** — ~0.2% relative L2, SSIM 0.9998 — and are indistinguishable (fc a
  hair ahead).
- **On transfer (cameraman) the conv did better here**: 0.38% vs 1.25% rel-L2.
  A plausible reading is that its shift-invariance acts as a generalization prior
  that pays off off the training image — but this is one image each, so treat it
  as an observation to confirm, not a settled advantage.

### On the metric (PSNR/dB)

PSNR is the field-standard headline (it is just log-MSE with a reference peak),
and is reported for familiarity — but it is always paired with **SSIM** (the
perceptual companion) in the denoising literature, so both are given. For the
RECOVERY claim specifically (`û ≈ u_PM`), **relative L2** is the most informative
number: it measures how close the two maps are and is directly comparable to the
~10% held-out prox residual. The denoising here is deliberately gentle (small σ,
so noisy is already ~28 dB); the headline is the MATCH to `u_PM`, not the dB gained
over the noisy input.

## Numbers (`metrics.csv`)

Every reported run: architecture, hyperparameters, held-out prox residual (eval =
cameraman, val = Barbara), `corr(J, TV)`, and the sampler noise floor `delta`.
The story in one line each:

- **Three levers moved the number** (70% → 10.7%): input standardization, a step
  budget with enough high-LR steps, and activation sharpness `beta=20` (matched to
  `f_reg`'s sharp TV kink).
- **Four capacity/architecture levers did not**: width, depth, and the conv-ICNN
  all tie at ~10.7%. The floor is not network expressivity.
- The **m-sweep** (residual falls 22% → 10% as MCMC sweeps go 2k → 32k) fits
  `residual² = floor² + (κ·δ)²` with floor ≈ 9.7%, κ ≈ 4.5 at `beta=5`: what
  remains is the errors-in-variables cost of noisy MCMC targets, not the net.
  (An `m`-sweep at the production `beta=20`, fully-connected, is still open.)

## Reproducing from scratch

`../DESIGN.md` has the commands. Order: `dataset.py` (samples `u_PM`, ~15 min,
cached to `../data/`), then `recover.py --arch fc --standardize --beta 20 --steps
250000` (~2 h) or `--arch conv` (~13 h, CPU — conv double-backprop is 6.8× the
FLOPs of the dense net despite fewer parameters). The convexity gate
`tests/test_icnn.py` must pass before trusting any conv result.

## Handoff — what travels, and what a colleague can do with it

The repo's `.gitignore` excludes `data/`, `logs/`, and `*.pth`, but a negation
rule keeps **everything under this `results/` directory tracked**. So on a
copy-paste-and-push, a colleague receives:

- all **code** (`../*.py`, the notebooks),
- these **figures** (`figs/*.png`, `*.pdf`), **metrics** (`metrics.csv`), this README,
- the two **production checkpoints** in `ckpt/` — fc and conv, `beta=20`
  (< 1 MB each). The scripts and `../recover_tv.ipynb` look here automatically when
  `../../logs/ckpt/` is absent, so a colleague can **regenerate every figure
  without retraining**: run `../recover_tv.ipynb`, or
  `python -m tvpm.figures` (from ../) and `python -m tvpm.denoise --arch conv` (from ../).

What does NOT travel (by `.gitignore`):
- `../data/` — cached sampler output. Regenerable with `python ../dataset.py`
  (~15 min); needed only to re-*train* or re-*score*, not to regenerate the
  figures from the shipped checkpoints (`tvpm/denoise.py` re-samples `u_PM` itself;
  `tvpm/figures.py` needs `data/eval_8x8_m8000.npz`, so run `tvpm.dataset` first if
  extending the qualitative panels).
- `../../logs/` — other checkpoints and training logs (the `beta` and `m` sweeps
  were run with `--no-save`; only the two production nets are kept).
- `ideas/old_files/*.mat` — the Barbara/cameraman source images (also gitignored
  as reference material); the scripts need them, so make sure they are present.
