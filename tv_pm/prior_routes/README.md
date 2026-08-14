# `prior_routes/` — three ways to recover a denoiser's implicit regularizer

Self-contained comparison built on `tv_pm/`'s cached sampler splits. Nothing here
retrains the TV posterior-mean denoiser; it recovers the **prior** behind it three
ways and scores each one against `u_PM`.

## Naming (used everywhere in this repository — see `../../src/NAMING.md`)

| formerly | prose / comments / docs | plot + legend | what it does |
|---|---|---|---|
| Route 1, LPN (Fang et al.) | **LPN Iterative recovery** | **Iterative** | `J_1(x) = <x,w> - psi(w) - ½‖x‖²`, `grad psi(w) = x` — an inversion **per query point** (arXiv:2310.14344) |
| Route 2, LPN (Ours) | **One-shot recovery** | **One-shot** | `J_2(x) = G(x) - ½‖x‖²`, `G ≈ psi*` — **one forward pass** |
| Direct (Ours) | **Direct fit** | **Direct fit** | `J_theta` fitted straight from the prox optimality condition — no recovery step |

Python identifiers (`route1`, `route2`, `prior_rmse_route1`, …) and the metric keys
in `logs/*_metrics.json` were **deliberately left unchanged**: they are a data
contract with the cached logs and with `tests/test_diverged_route1.py`. Only prose,
printed output and plot text carry the new names.

## Layout

    three_way.py                networks + the two learned denoisers; CLI entry point
    compare_impl.py             the cameraman denoising comparison and its figure
    prior_compare.py            the prior-recovery comparison and its figure
    three_way_comparison.ipynb  the write-up; reads checkpoints, retrains nothing
    three_way_retrain.ipynb     TRAINS psi and G, then reruns both comparisons

`three_way.py` puts `tv_pm/` and the repo root on `sys.path` itself, anchored on
`__file__`, so every script and the notebook run from any working directory.

Outputs follow `tvpm/paths.py`, not a folder of their own:

    ../results/figs/   threeway_cameraman.{png,pdf}   prior_recovery.{png,pdf}
    ../results/        threeway_psnr.csv  threeway_metrics.json
                       prior_recovery.csv  prior_recovery.json
    ../data/           threeway_u_pm_m8000_seed3.npz     (sampler cache)
    ../../logs/ckpt/   threeway_psi_*.pth  threeway_Ggrad_*.pth

## The two notebooks

`three_way_comparison.ipynb` is the write-up: it loads the shipped checkpoints and
reproduces the reported figures in seconds.

`three_way_retrain.ipynb` rebuilds `psi_theta` and `G_theta` from the cached splits
and reruns both comparisons on the nets it just trained. It writes to **tagged**
files (`RUN_TAG` for checkpoints, `OUT_TAG` for figures and tables), so a retrain
cannot overwrite the reported results; the last section explains how to promote a
run deliberately. Start at `PRESET = "smoke"` (~4 min end to end) before spending
the ~3 h full preset.

`J_theta` is never retrained by either notebook — it is `recover_tv.ipynb`'s
checkpoint, reused unchanged, so the Direct fit stays the object that notebook reports.

Two knobs exist only because LPN Iterative recovery's cost does **not** shrink with
the training budget: an under-fitted `psi` makes its inversion converge *later*, so a
3000-step net runs the full 20000 iterations (~13 min on 2000 patches) where the
trained net early-stops in ~13 s. `TW_N_EVAL` and `TW_INVERT_ITERS` let a smoke run
shrink the eval set instead of pretending the inversion is cheap.

## Reproduction

    python3 three_way.py --stage psi    --steps 250000     # ~75 min, 2 CPU cores
    python3 three_way.py --stage Ggrad  --steps 250000     # ~90 min
    python3 three_way.py --stage compare                   # denoising  + figure
    python3 prior_compare.py                               # prior test + figure

`J_theta` is `recover_tv.ipynb`'s existing checkpoint, reused unchanged; its numbers
here reproduce `tvpm.denoise.run` exactly (27.32 / 27.09 / 46.93 dB / 0.86 %).

## Results (cameraman, sigma = t = 20/256, m = 8000, every network 250 000 steps)

Prior recovery, scored against `u_PM`'s prox optimality condition; sampler noise
floor `delta = 2.19 %`:

| recovered prior | resid / target | p90 | x delta | cost |
|---|---|---|---|---|
| LPN Iterative recovery | 14.48 % | 371 % | 6.6x | 12.7 s (inversion) |
| One-shot recovery | 13.73 % | 98 % | 6.3x | 0.03 s (forward) |
| **Direct fit** | **8.07 %** | **21 %** | **3.7x** | 0.03 s (forward) |

Denoising:

| method | PSNR vs clean | PSNR vs u_PM | rel-L2 |
|---|---|---|---|
| u_PM (MCMC) | 27.32 dB | — | — |
| `grad psi_theta` (deployed by Iterative AND One-shot) | 26.92 dB | 41.76 dB | 1.56 % |
| **Direct fit**, `prox_{J_theta}` | **27.09 dB** | **46.93 dB** | **0.86 %** |

Neither LPN recovery inverts anything to *denoise*: `grad G = (grad psi)^-1`, so both
deploy `grad psi_theta`. They differ only in how they evaluate the prior.

## Font sizes

Every figure in the repository takes its text size from `src/plotstyle.py`
(`FONT_SIZE = 11`) through rcParams. Plotting code sets no `fontsize=` of its own,
so titles, labels, ticks and legends match across all figures.
