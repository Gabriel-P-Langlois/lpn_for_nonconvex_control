# `pnp_reg/` — the plug-and-play registration task

Experiments for the task document `../../__tasks/task_pnp_registration.tex`
(PIRATE as a Hamilton–Jacobi flow). Implemented so far: **Experiment 1, the
mixture figure** (Section 6.1 of the task document). Experiment 2 — the
PIRATE/PIRATE+ Jacobian diagnosis — is specified there and was unblocked on
2026-07-30: their weights are released and cloned at `../ext/pirate/`
(see `DESIGN.md`, decisions of 2026-07-30).

Shared machinery lives in `../src/` (`gradfit.py` for the gradient-supervision
loop, `network.py` for the ICNN); this directory holds only what is specific
to the experiment.

| file | what it is |
|---|---|
| **`experiment1_mixture.ipynb`** | Experiment 1 end to end, committed executed (~3 min to rerun) |
| **`experiment2_probe.ipynb`** | Experiment 2 end to end from saved metrics (~seconds to rerun) |
| `DESIGN.md` | configuration, decisions, and observations |
| `pnpreg/mixture.py` | the 1-D mixture prior and its exact objects: `J`, `D`, `dD`, `ψ`, `f_reg = t·J_BVS` |
| `pnpreg/readout.py` | the semiconvex readout `J_θ = G_θ − ½‖y‖²` and its gradient |
| `pnpreg/experiment.py` | Experiment 1: data → two fits → metrics (`--smoke` for a ~10 s check) |
| `pnpreg/probe.py` | Experiment 2: matrix-free estimators (asymmetry ratio, Lanczos extremes) |
| `pnpreg/probe_targets.py` | Experiment 2: the probed operators (3 calibrations, floor, PIRATE, PIRATE+) |
| `pnpreg/probe_run.py` | Experiment 2: driver — calibrate, gate, diagnose, write table |
| `pnpreg/figures.py` | the two-panel Experiment-1 figure |
| `pnpreg/paths.py` | filesystem locations, defined once |
| `tests/` | gates: `test_mixture.py`, `test_readout.py`, `test_probe.py`, `test_probe_targets.py` |
| `results/` | tracked: figures, `metrics.json`, `probe_metrics.json`, `probe_table.{md,tex}` |

## What the experiment shows

On a prior where everything is exact, the figure displays the two results that
constrain every plug-and-play method built on an MMSE denoiser: the denoiser at
noise level σ can reveal the prior only up to `f_reg = t·J_BVS` (the gap to
`tJ` in the figure), and recovering that object requires the 1-semiconvex
class — the plain convex network flattens the region between the modes, where
the target's curvature is −0.91 at σ = 0.5 (floor −1). The outcome is guaranteed by the task document's
Propositions `prop:prox` and `prop:tight`; the run is an implementation test
and a paper figure, not a hypothesis test.

## Reproducing

```
PY=~/miniforge3/envs/lpn_env/bin/python      # run from this directory
$PY tests/test_mixture.py                    # ~5 s, must pass
$PY tests/test_readout.py                    # ~2 s, must pass
$PY -m pnpreg.experiment --smoke             # ~10 s end-to-end check
$PY -m pnpreg.experiment                     # ~3 min: the figure + metrics
```

Headline numbers (σ = 0.5, β = 20): the semiconvex fit reaches 1.2% relative
L2 against the exact `f_reg` with a 0.2% held-out prox residual; the convex
control reads ~100% on both. Metrics are reported on the sampled range of the
denoiser (see `DESIGN.md`, including why σ = 0.25 behaves differently).

## Experiment 2 — the PIRATE/PIRATE+ Jacobian diagnosis

The task document's falsifiable experiment (its Section 6.2). PIRATE
(Hu, Gan, Sun, An, Kamilov, ICLR 2024) plugs a trained DnCNN denoiser `D`
into an iterative registration method and treats the update as descent on an
objective containing a regularizer whose proximal operator is `D` — a
hypothesis its paper never states or tests. The theory resolves it into three
nested conditions on the Jacobian `J = ∇D(z)` at operating inputs `z`:

1. **conservative** — `J` symmetric: `D` is the gradient of something
   (Poincaré lemma); otherwise the objective does not exist as written;
2. **proximal** — additionally `J ⪰ 0`: `D` is the prox of a possibly
   nonconvex regularizer (the semiconvex class of Experiment 1);
3. **convex-proximal** — additionally `J ⪯ I`: the regularizer is convex.

The probe measures, at each test input, the asymmetry ratio
`ρ = ‖J − Jᵀ‖_F / (2‖J‖_F) ∈ [0, 1]` (Hutchinson probes with bootstrap
standard errors) and Lanczos estimates of the extreme eigenvalues of the
symmetric part `S = (J + Jᵀ)/2` (full reorthogonalization, multistart, Ritz
residual bounds) — everything from Jacobian-vector and vector-Jacobian
products by automatic differentiation; no Jacobian is ever formed. Weights:
the released checkpoints in `../ext/pirate/` (one AWGN denoiser at their
σ = 1, plus PIRATE+ — the same twelve tensors after deep-equilibrium
fine-tuning, so the two rows are an exactly matched pair). Test inputs are
the released registration field plus unit-variance Gaussian noise, the
operating distribution of their own training script.

**Calibration gates before any PIRATE number** (the task document's kill
criterion — a number from an uncalibrated estimator is worthless): the run
aborts, with `gates_passed: false` in the JSON and no PIRATE rows, unless
the estimators reproduce three denoisers with known answers — the exact
coordinatewise mixture denoiser of Experiment 1 (symmetric, PSD, **fails**
condition 3 at a designed test point), the exact n = 2 TV posterior mean by
quadrature (all three conditions hold by theorem), and the trained `tv_pm`
convex-ICNN prox probed through a CG-vs-dense two-path operator (all three
hold by architecture; run in float64 and repeated in float32 as the
precision bridge). A fourth "floor" row — a surrogate of the same conv
architecture whose Jacobian is symmetric by weight construction — is the
estimator's end-to-end zero-test; the resolvable-asymmetry floor reported
with the table is the larger of its measured ρ and the jvp-vs-vjp bilinear
inconsistency `identity_max`, and a PIRATE ρ at or below that floor is
reported as "≤ floor", never as asymmetry.

### Reproducing

```
PY=~/miniforge3/envs/lpn_env/bin/python          # run from this directory
$PY tests/test_probe.py                          # ~15 s, must pass
$PY tests/test_probe_targets.py                  # ~2 min, must pass
$PY -m pnpreg.probe_run --smoke                  # ~1 min end-to-end check (CPU, cropped field)
$PY -m pnpreg.probe_run --rows cal_mixture cal_quadrature cal_icnn cal_icnn32 floor --tag cal
                                                 # ~10 min: calibration only (CPU)
$PY -m pnpreg.probe_run --device mps             # ~2.5-3 h: the production table
$PY -m pnpreg.probe_run --rows pirate --device cpu --n-test 1 --probes 4 --k 10 --starts 1 --tag cpu_check
                                                 # ~10 min: device cross-check
```

Outputs: `results/probe_metrics.json` (all per-point numbers, seeds, budgets,
gate records), `results/probe_table.md` / `.tex` (the deliverable table). All
randomness is CPU-seeded, so an MPS run and its CPU cross-check probe
bitwise-identical inputs; residual MPS convolution nondeterminism is bounded
by the reported Monte Carlo errors and the cross-check deltas in `DESIGN.md`.

Note: we probe the **denoiser** inside PIRATE+, not its deep-equilibrium
fixed-point iteration — the three conditions concern `∇D` alone.

## Experiment 3.1 — is PIRATE+ affine? The closed-form regularizer

Experiment 2 measured a PIRATE+ Jacobian spectrum constant to ~1e-4 across
test points. This experiment tests the stronger statement — that the
denoiser is AFFINE on its operating distribution — because an affine
denoiser has a quadratic implicit regularizer, computable in closed form
from measured quantities with no training (`pnpreg/affine.py` for the
mathematics; runs entirely on CPU).

Result (production 2026-07-30, `results/affine_metrics.json`): PIRATE+ is
affine to **0.20%** (Jacobian variation 0.8%), and the closed-form
quadratic passes its falsifiable prediction — the prox-identity residual at
a held-out input is 0.597 against a predicted ceiling of 0.704 set by the
affinity error plus the residual-relative asymmetry `rho_res = 0.702`. The
AWGN denoiser is NOT affine (pair error 0.50, Jacobian varying 93%). The
figure `results/figs/experiment31_affine.png` shows the extreme-curvature
deformation patterns of the implicit regularizer and the affine constant
term. See `DESIGN.md` (Experiment 3.1) for the rho_res normalization
finding and the consequences for the full nonparametric recovery.

```
PY=~/miniforge3/envs/lpn_env/bin/python      # run from this directory
$PY tests/test_affine.py                     # ~10 s, must pass
$PY -m pnpreg.affine_run --smoke             # ~1 min end-to-end check
$PY -m pnpreg.affine_run                     # ~80 min: the production numbers
```

## Experiment 3.2 — the backward viscosity solution recovered by our network

Route B on an exactly solvable model (`pnpreg/bimodal.py`): a bimodal field
prior on 8×8 patches, nonconvex along one bump-pattern direction `u` (the
Experiment-1 mixture), Gaussian with spectrum `λ_k = 4/k²` on the other 63
directions. The network trained on denoiser pairs `(D(z), z)` is a solver
for the backward Hamilton–Jacobi problem: its readout estimates
`f_reg = t·J_BVS`, and here the backward solution is known in closed form —
hull-limited along `u`, exactly `t·J` on the convex directions.

Result (production 2026-07-31, σ = 0.5, `results/bimodal_metrics.json`): the
semiconvex readout recovers the exact backward solution to **3.2%** in value
and **0.32%** in the 63-mode spectrum (3.1% worst mode, by gradient-slice
slopes anchored on the operating support), with the nonconvex dip reproduced
(min curvature −0.883 vs exact −0.911, floor −1); the convex control fails
exactly along `u` (min curvature 0.000, overall 15.4%) while matching the
Gaussian block at 1.13% — the class-separation result at field dimension. Figure:
`results/figs/experiment32_bimodal.png`.

```
PY=~/miniforge3/envs/lpn_env/bin/python      # run from this directory
$PY tests/test_bimodal.py                    # ~10 s, must pass
$PY -m pnpreg.bimodal_run --smoke            # ~30 s end-to-end check
$PY -m pnpreg.bimodal_run                    # production: 2 fits × 50k steps
```
