"""Gate for the mixture closed forms. Every quantity Experiment 1 trusts is
checked here against an independent computation. Run from pnp_reg/:

    ~/miniforge3/envs/lpn_env/bin/python tests/test_mixture.py    (~5 s)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pnpreg import mixture as mx

SIGMA = 0.5
ok = True


def report(name, err, tol):
    global ok
    good = err < tol
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] {name:58s} {err:.3e}")


g, gw = mx.grid()
dx = g[1] - g[0]

# 1. p_z against a brute-force numerical convolution of p_phi with G_sigma.
u = np.linspace(-10, 10, 40001)
du = u[1] - u[0]
pphi = np.exp(mx._log_mix(u, mx.NU))
xs = np.linspace(-5, 5, 41)
kern = np.exp(-0.5 * ((xs[:, None] - u[None, :]) / SIGMA) ** 2) / np.sqrt(2 * np.pi * SIGMA ** 2)
pz_num = (kern * pphi[None, :]).sum(axis=1) * du
report("p_z vs numerical convolution (rel sup)",
       float(np.max(np.abs(pz_num - np.exp(mx.log_pz(xs, SIGMA))) / pz_num)), 1e-6)

# 2. Analytic (log p_z)' against central finite differences.
eps = 1e-5
fd = (mx.log_pz(xs + eps, SIGMA) - mx.log_pz(xs - eps, SIGMA)) / (2 * eps)
report("(log p_z)' analytic vs finite difference (sup)",
       float(np.max(np.abs(fd - mx._dlog_mix(xs, mx.NU + SIGMA ** 2)))), 1e-5)

# 3. D = grad psi: the denoiser equals the numerical gradient of psi.
Dg = np.gradient(mx.psi(g, SIGMA), dx)
m = (g > -5) & (g < 5)
report("D vs numerical grad psi (sup, |x|<5)",
       float(np.max(np.abs(Dg[m] - mx.D(g[m], SIGMA)))), 1e-5)

# 4. f_reg + y^2/2 = psi^* is convex: nonnegative second differences.
fr = mx.freg(g, SIGMA)
conv = fr + 0.5 * g ** 2
c2 = (conv[2:] - 2 * conv[1:-1] + conv[:-2]) / dx ** 2
report("f_reg + y^2/2 convex (violation, |y|<5)", max(0.0, float(-c2[m[1:-1]].min())), 1e-3)

# 5. The prox-optimality identity grad f_reg(D(x)) = x - D(x), via finite
#    differences of f_reg at the (off-grid) points y = D(x).
xs2 = np.linspace(-4, 4, 81)
ys = mx.D(xs2, SIGMA)
dfr = (mx.freg(ys + 1e-5, SIGMA) - mx.freg(ys - 1e-5, SIGMA)) / 2e-5
report("grad f_reg(D(x)) = x - D(x) (sup)",
       float(np.max(np.abs(dfr - (xs2 - ys)))), 1e-5)

# 6. prox_{f_reg} = D globally: objective gap at u = D(x) vs the grid minimum,
#    with f_reg(D(x)) evaluated exactly (cf. __tasks/semiconvex_check.py).
gap = 0.0
for x in xs2:
    obj = fr + 0.5 * (x - g) ** 2
    Dx = float(mx.D(x, SIGMA))
    obj_at_D = float(mx.freg(Dx, SIGMA)) + 0.5 * (x - Dx) ** 2
    gap = max(gap, obj_at_D - obj.min())
report("prox objective gap at u = D(x) (global)", gap, 1e-9)

# 7. t * J_BVS = f_reg by construction of jbvs (identity check of the API).
report("t*J_BVS - f_reg (sup)",
       float(np.max(np.abs(SIGMA ** 2 * mx.jbvs(g, SIGMA) - fr))), 1e-12)

# 8. NO TRUNCATION (regression for the 2026-07-29 bug): far outside the old
#    wide grid's reach, f_reg' must equal D^{-1}(y) - y. At sigma = 0.5 and
#    y = 5.5 the maximizer sits at x = D^{-1}(5.5) ~ 24; the old bounded-grid
#    conjugate capped it at 14 and returned an affine (curvature -1) f_reg.
for y0 in (4.5, 5.5):
    d5 = (float(mx.freg(y0 + 1e-5, SIGMA)) - float(mx.freg(y0 - 1e-5, SIGMA))) / 2e-5
    xinv = float(mx.D_inverse(np.array([y0]), SIGMA)[0])
    report(f"f_reg'({y0}) = D^-1({y0}) - {y0} (beyond old grid)",
           abs(d5 - (xinv - y0)), 1e-5)

# 9. Analytic D' (added for Experiment 2's probe calibration) against central
#    finite differences of D, on a grid covering both modes and the gap.
xs3 = np.linspace(-4, 4, 161)
eps = 1e-6
fdD = (mx.D(xs3 + eps, SIGMA) - mx.D(xs3 - eps, SIGMA)) / (2 * eps)
report("dD analytic vs finite difference of D (sup)",
       float(np.max(np.abs(fdD - mx.dD(xs3, SIGMA)))), 1e-7)

# 10. dD > 0 everywhere (posterior-variance identity: D strictly increasing)
#     and dD(0) > 1 (the designed condition-3 violation between the modes).
report("dD > 0 everywhere (violation)", max(0.0, float(-mx.dD(g, SIGMA).min())), 1e-12)
report("dD(0, 0.5) > 1 (margin shortfall)", max(0.0, 1.0 - float(mx.dD(0.0, SIGMA))), 1e-12)

print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
