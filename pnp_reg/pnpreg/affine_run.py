"""Experiment 3.1 driver: affinity, closed-form regularizer, Ritz figure.

    ~/miniforge3/envs/lpn_env/bin/python -m pnpreg.affine_run --smoke   (~1 min)
    ~/miniforge3/envs/lpn_env/bin/python -m pnpreg.affine_run           (~40 min, CPU)

Stages (see pnpreg/affine.py for the mathematics):
1. Affinity test on both released networks (12 operating-distribution pairs
   each), float32; a float64 repeat of a few pairs on a crop bounds the
   float32 arithmetic. Everything runs on CPU.
2. Self-check of the closed-form quadratic regularizer at a HELD-OUT input:
   the prox-identity residual, predicted to land at (affinity error + rho)
   for PIRATE+ and to be LARGE for the AWGN denoiser. The verdict against
   that prediction is printed and stored; a failed verdict means the
   quadratic account is wrong and downstream writing must not use it.
3. Extreme Ritz vectors of S per network -> the deformation patterns the
   implicit prior most favors and most penalizes, as a figure.
4. Optional --scale-sweep: Lanczos extremes at s * field + noise for
   s in {0.5, 1.5}, probing how far the affine region extends.

Outputs (tracked): results/affine_metrics.json,
results/figs/experiment31_affine.{png,pdf}. rho values are read from the
Experiment 2 production JSON (results/probe_metrics.json).
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch

from . import affine
from . import paths
from . import probe
from . import probe_targets as pt

NETS = ("pirate", "pirate_plus")


def load_exp2_rho():
    p = os.path.join(paths.RESULTS, "probe_metrics.json")
    with open(p) as f:
        M = json.load(f)
    return {k: M["rows"][k]["summary"]["rho_mean"] for k in NETS if k in M["rows"]}


def slice_mid(vec_field):
    """(1, 3, D, H, W) -> (3, H, W) central axial slice, numpy float64."""
    x = vec_field.detach().cpu().double().numpy()
    return x[0, :, x.shape[2] // 2]


def make_figure(ritz, b_slices, out_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [n for n in NETS if n in ritz]
    fig, axes = plt.subplots(len(rows), 3, figsize=(12.5, 4.1 * len(rows)))
    axes = np.atleast_2d(axes)
    step = 5
    for r, name in enumerate(rows):
        panels = [
            (f"{name}: favored pattern\n" +
             rf"$\lambda_S={ritz[name]['lmax']['lambda_S']:.3f}$, curvature "
             rf"${ritz[name]['lmax']['curvature_freg']:+.3f}$",
             slice_mid(ritz[name]["lmax"]["vec"])),
            (f"{name}: penalized pattern\n" +
             rf"$\lambda_S={ritz[name]['lmin']['lambda_S']:.3f}$, curvature "
             rf"${ritz[name]['lmin']['curvature_freg']:+.3f}$",
             slice_mid(ritz[name]["lmin"]["vec"])),
            (f"{name}: constant term $\\hat b$ (slice)", b_slices[name]),
        ]
        for c, (title, s3) in enumerate(panels):
            ax = axes[r, c]
            mag = np.linalg.norm(s3, axis=0)
            im = ax.imshow(mag, cmap="viridis")
            H, W = mag.shape
            yy, xx = np.mgrid[0:H:step, 0:W:step]
            ax.quiver(xx, yy, s3[2, ::step, ::step], -s3[1, ::step, ::step],
                      color="w", width=0.003, scale_units="xy", angles="xy")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Experiment 3.1: extreme curvature directions of the implicit "
                 "regularizer, and the affine constant term", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(paths.FIGS, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-points", type=int, default=8)
    ap.add_argument("--n-pairs", type=int, default=12)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--scale-sweep", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    crop = None
    k = a.k
    n_points, n_pairs = a.n_points, a.n_pairs
    f64_points, f64_pairs = 4, 3
    if a.smoke:
        crop, k, n_points, n_pairs = (24, 24, 24), 6, 3, 3
        f64_points, f64_pairs = 2, 1
        a.tag = a.tag or "smoke"

    try:
        rho = load_exp2_rho()
    except FileNotFoundError:
        rho = {}
        print("WARNING: Experiment 2 JSON not found; verdicts unavailable")

    out = {"config": {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__, "device": "cpu", "crop": crop,
        "n_points": n_points, "n_pairs": n_pairs, "k": k,
        "seed": affine.SEED, "seed_heldout": affine.SEED_HELDOUT,
        "rho_exp2": rho,
    }, "nets": {}}

    ritz = {}
    b_slices = {}
    for which in NETS:
        print(f"== {which}", flush=True)
        t0 = time.time()
        model = pt.load_pirate_dncnn(which, device="cpu")
        aff, b_hat, model, field = affine.affinity(
            which, n_points=n_points, n_pairs=n_pairs, device="cpu",
            crop=crop, model=model)
        print(f"  affinity: err median {aff['err_median']:.3e} max {aff['err_max']:.3e} | "
              f"jac variation {['%.1e' % v for v in aff['jac_variation']]} | "
              f"b norm {aff['b_norm']:.3e} spread {aff['b_spread']:.3e} | "
              f"{time.time()-t0:.0f}s", flush=True)

        t0 = time.time()
        # the float64 repeat exists only to bound float32 arithmetic in the
        # affinity metric, and it ALWAYS runs on a crop: a full-field float64
        # jvp transiently holds ~5 GB of double activations per call, which
        # took the 2026-07-30 run to 13 GB on the 16 GB machine
        crop64 = crop if crop is not None else (24, 24, 24)
        model64 = pt.load_pirate_dncnn(which, device="cpu", dtype=torch.float64)
        aff64, _, _, _ = affine.affinity(
            which, n_points=f64_points, n_pairs=f64_pairs, device="cpu",
            dtype=torch.float64, crop=crop64, model=model64)
        del model64
        print(f"  affinity float64 repeat (crop {crop64}): "
              f"median {aff64['err_median']:.3e} max {aff64['err_max']:.3e} | "
              f"{time.time()-t0:.0f}s", flush=True)

        t0 = time.time()
        # the AWGN row's S is indefinite (Exp 2: lambda_min = -0.72), so its
        # closed form does not exist and CG cannot converge -- cap it at 30
        # iterations; the row documents the expected failure, nothing more
        sc = affine.selfcheck(which, b_hat, device="cpu", crop=crop, model=model,
                              cg_max_iter=30 if which == "pirate" else 200)
        # the skew part enters the prox identity relative to the RESIDUAL
        # map's own size, so the prediction uses rho_res, not the full-
        # Jacobian rho (which the identity part dilutes for a near-identity
        # denoiser -- smoke-run finding, 2026-07-30)
        pred = aff["err_median"] + sc["rho_res"]
        verdict = bool(sc["resid"] <= 3.0 * pred) if which == "pirate_plus" \
            else (bool(sc["resid"] >= sc["rho_res"] / 3.0) if which in rho else None)
        print(f"  self-check: prox-identity resid {sc['resid']:.3e} "
              f"(rho_res {sc['rho_res']:.3e}, rho_full {sc['rho_full']:.3e}, "
              f"prediction {pred:.3e}, verdict {verdict}) | "
              f"residual/full norm ratio {sc['norm_ratio_res_over_full']:.3e} | "
              f"cg {sc['cg_iters']} iters | {time.time()-t0:.0f}s", flush=True)

        t0 = time.time()
        rv = affine.ritz_vectors(which, k=k, device="cpu", crop=crop, model=model)
        ritz[which] = rv
        b_slices[which] = slice_mid(b_hat)
        print(f"  ritz: lmax {rv['lmax']['lambda_S']:.4f} "
              f"(curv {rv['lmax']['curvature_freg']:+.4f}), "
              f"lmin {rv['lmin']['lambda_S']:.4f} "
              f"(curv {rv['lmin']['curvature_freg']:+.4f}) | "
              f"{time.time()-t0:.0f}s", flush=True)

        rec = {"affinity": aff, "affinity_float64": aff64, "selfcheck": sc,
               "selfcheck_prediction": pred, "selfcheck_verdict": verdict,
               "ritz": {kk: {"lambda_S": rv[kk]["lambda_S"],
                             "curvature_freg": rv[kk]["curvature_freg"]}
                        for kk in ("lmin", "lmax")},
               "ritz_res": rv["res"]}

        if a.scale_sweep:
            sweep = {}
            for s in (0.5, 1.5):
                g = torch.Generator(device="cpu").manual_seed(affine.SEED)
                noise = torch.randn(field.shape, generator=g, dtype=torch.float32)
                z = s * field + noise
                op = pt.PirateOp(model, z)
                lan = probe.lanczos_symmetric(op, k=min(k, 10), seed=affine.probe_seed(which))
                sweep[str(s)] = {"lmin": lan["lmin"], "lmax": lan["lmax"]}
                del op
            rec["scale_sweep"] = sweep
            print(f"  scale sweep: {sweep}", flush=True)

        out["nets"][which] = rec
        del model

    tag = f"_{a.tag}" if a.tag else ""
    paths.ensure_dirs()
    jpath = os.path.join(paths.RESULTS, f"affine_metrics{tag}.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {jpath}", flush=True)
    make_figure(ritz, b_slices,
                os.path.join(paths.FIGS, f"experiment31_affine{tag}"))
    print(f"-> {paths.FIGS}/experiment31_affine{tag}.png/pdf", flush=True)

    bad = [w for w in out["nets"]
           if out["nets"][w]["selfcheck_verdict"] is False]
    if bad:
        print(f"SELF-CHECK VERDICT FAILED for {bad}: the quadratic account "
              f"does not hold at the predicted scale; do not write it up.",
              flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
