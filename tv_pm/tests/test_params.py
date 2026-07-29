"""Gate on the (sigma, t) parametrization, the cache keys, and the checkpoint names.

    ~/miniforge3/envs/lpn_env/bin/python tv_pm/tests/test_params.py

These are the pieces that decide WHICH FILE a run reads and writes. A mistake
here does not raise -- it silently trains on the wrong data or reports a cached
result from another configuration -- so they are checked rather than trusted.

Cheap: no sampling, no training. The one training test runs 3 steps on random
data purely to exercise the validation path.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tvpm import dataset, recover
from tvpm.sampler import from_sigma_t, params

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail and not cond else ''}")
    if not cond:
        FAIL.append(name)


print("\n1. the algebra: two degrees of freedom, not four")
# The model is E(u) = ||u-x||^2/(2t) + ATV(u), posterior ~ exp(-E/eps), with
# eps = 2 sigma^2/lam and t = lam/2. So (sigma, t) and (sigma, lam) are two
# parametrizations of the SAME pair and must round-trip exactly.
for sigma, t in ((10 / 256, 16 / 256), (0.02, 0.05), (0.1, 1.0), (1e-3, 1e-2)):
    eps, lam = from_sigma_t(sigma, t)
    eps2, t2 = params(sigma, lam)
    check(f"round trip at sigma={sigma:g}, t={t:g}",
          np.isclose(eps, eps2, rtol=1e-12) and np.isclose(t, t2, rtol=1e-12),
          f"got eps {eps} vs {eps2}, t {t} vs {t2}")
    check(f"  sqrt(t*eps) == sigma at t={t:g}", np.isclose((t * eps) ** 0.5, sigma, rtol=1e-12))
    check(f"  lam == 2t at t={t:g}", np.isclose(lam, 2 * t, rtol=1e-12))

print("\n2. the defaults still reproduce the MATLAB's tabulated pair")
check("sigma == 10/256", np.isclose(dataset.SIGMA, 10 / 256, rtol=1e-15))
check("lam == 32/256", np.isclose(dataset.LAM, 32 / 256, rtol=1e-15))
check("EPS, LAM agree with from_sigma_t",
      np.allclose(from_sigma_t(dataset.SIGMA, dataset.T), (dataset.EPS, dataset.LAM), rtol=1e-15))

print("\n3. tag(): empty at the defaults, distinct elsewhere")
# Empty at the defaults is what keeps every shipped cache, checkpoint and figure
# reachable under the name it was written with.
check("tag() == '' at the defaults", dataset.tag() == "")
check("tag(default) == '' explicitly", dataset.tag(dataset.SIGMA, dataset.T) == "")
tags = {dataset.tag(s, t) for s, t in ((0.02, 0.05), (0.05, 0.02), (0.02, 0.06))}
check("distinct (sigma,t) give distinct tags", len(tags) == 3, str(tags))
check("a non-default tag is non-empty", all(x for x in tags))

print("\n4. cache paths key on everything that changes the data")
base = dataset.path("train")
check("default path keeps the legacy name",
      os.path.basename(base) == "train_8x8_m8000.npz", os.path.basename(base))
variants = {
    "default": dataset.path("train"),
    "sweeps": dataset.path("train", 2000),
    "sigma,t": dataset.path("train", 8000, 8, 0.02, 0.05),
    "scale": dataset.path("train", 8000, 8, dataset.SIGMA, dataset.T, 0.25),
    "split": dataset.path("val"),
}
check("all five differ", len(set(variants.values())) == 5,
      str({k: os.path.basename(v) for k, v in variants.items()}))
# scale is the one that was missing and silently aliased a 200-patch cache onto
# the 20000-patch name.
check("scale=1.0 adds nothing to the name",
      dataset.path("train", 8000, 8, dataset.SIGMA, dataset.T, 1.0) == base)

print("\n5. load() refuses a cache sampled at another (sigma, t)")
# The filename alone is not proof: caches written before (sigma,t) entered the
# name carry the DEFAULT name whatever they were sampled at.
if os.path.exists(dataset.path("eval")):
    d = dataset.load("eval")
    check("the shipped default cache loads", d["x"].shape[0] > 0)
    check("it records t and eps", "t" in d and "eps" in d)
    try:
        # same (default) filename, but ask for a t the file was not built at
        dataset.load("eval", sigma=dataset.SIGMA, t=float(d["t"]) * 1.000001)
        check("mismatched t is rejected", False, "no exception raised")
    except (ValueError, FileNotFoundError):
        check("mismatched t is rejected", True)
else:
    print("  SKIP  no default cache on disk")

print("\n6. checkpoint names: the shipped nets stay reachable")
# ensure_model() must find the shipped fc checkpoint, which predates the current
# naming scheme. If it cannot, a default run silently retrains for ~2 h.
legacy_fc = recover.checkpoint_name(64, "fc", 8000, 256, 2, 32, 2, 20, 250000,
                                    dataset.SIGMA, dataset.T, legacy=True)
check("legacy fc name matches the shipped file",
      legacy_fc == "tv_pm_64D_m8000_w256_b20.0_s250000.pth", legacy_fc)
legacy_conv = recover.checkpoint_name(64, "conv", 8000, 256, 2, 32, 2, 20, 250000,
                                      dataset.SIGMA, dataset.T, legacy=True)
cur_conv = recover.checkpoint_name(64, "conv", 8000, 256, 2, 32, 2, 20, 250000,
                                   dataset.SIGMA, dataset.T)
check("conv legacy == conv current (its scheme never changed)", legacy_conv == cur_conv)
check("conv name matches the shipped file",
      cur_conv == "tv_pm_64D_conv_m8000_C32_B2_b20.0_s250000.pth", cur_conv)
names = {recover.checkpoint_name(64, a, m, 256, 2, 32, 2, b, s, sg, t)
         for a in ("fc", "conv") for m in (2000, 8000) for b in (5, 20)
         for s in (1000, 250000) for sg, t in ((dataset.SIGMA, dataset.T), (0.02, 0.05))}
check("every hyperparameter combination gets its own name", len(names) == 32, str(len(names)))

print("\n7. find_checkpoint() locates the shipped nets")
for arch in ("fc", "conv"):
    p = recover.find_checkpoint(arch=arch, beta=20, steps=250000, sweeps=8000)
    check(f"{arch}: found on disk", p is not None and os.path.exists(p),
          str(p))
check("an absurd configuration finds nothing",
      recover.find_checkpoint(arch="fc", beta=99, steps=7, sweeps=3) is None)

print("\n8. training validates on the final step even below eval_every")
# Regression: with eval_every=500 a 300-step budget never validated, leaving
# best_val None -- which crashed the caller and returned an unchecked net.
X, G = torch.randn(64, 8), torch.randn(64, 8)
for steps in (3, 300, 500):
    m = recover.build_model("fc", 8, 16, 2, 20.0, 32, 2)
    h = recover.train_grad(m, X, G, X, G, batch_size=16, steps=steps,
                           eval_every=500, quiet=True)
    check(f"steps={steps}: best_val is set", h["best_val"] is not None)
    check(f"steps={steps}: last eval is the last step", h["steps"][-1] == steps,
          str(h["steps"]))

print("\n9. run() writes the name find_checkpoint() reads (int beta round trip)")
# Regression: run() built the name with f"b{beta}" while checkpoint_name() uses
# f"b{float(beta)}", so an int beta wrote ...b20... but the finder looked for
# ...b20.0... and ensure_model raised "training finished but no checkpoint".
# The two now share checkpoint_name(). Checked WITHOUT sampling: run() saves a
# checkpoint whose name checkpoint_name() must reproduce for BETA passed as int.
n_, arch_, sw_, w_, L_, C_, B_, steps_ = 8, "fc", 50, 16, 2, 32, 2, 30
for beta_ in (20, 20.0, 5):                       # int and float must agree
    saved = recover.checkpoint_name(n_, arch_, sw_, w_, L_, C_, B_, beta_, steps_,
                                    dataset.SIGMA, dataset.T)
    check(f"beta={beta_!r}: name is stable to int/float",
          saved == recover.checkpoint_name(n_, arch_, sw_, w_, L_, C_, B_,
                                           float(beta_), steps_, dataset.SIGMA, dataset.T))
    check(f"beta={beta_!r}: name carries b{float(beta_)}", f"_b{float(beta_)}_" in saved,
          saved)

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
sys.exit(1 if FAIL else 0)
