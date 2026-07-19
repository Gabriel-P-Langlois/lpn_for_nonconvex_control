import sys, os; sys.path.insert(0, os.path.abspath("."))
import numpy as np
from src.targets import QuadraticL1
from src.maxplus_bounds import (gamma_K, huber_S, huber_gamma_error_exact,
                                slopes_random_l1, slopes_grid_l1,
                                slopes_tangent_l1, convex_upper_bound)

rng = np.random.default_rng(0)

# (a) huber_S == the verified target
for d in (1, 3, 7):
    y = rng.uniform(-4, 4, (500, d))
    assert np.abs(huber_S(y) - QuadraticL1(t=1.0).hjsol_true(y)).max() < 1e-12
print("(a) huber_S == QuadraticL1.hjsol_true                       OK")

# (b) Gamma_K <= S for random / grid / tangent slopes
d = 3; y = rng.uniform(-4, 4, (2000, d)); S = huber_S(y)
for name, P in [("random", slopes_random_l1(d, 200, rng)),
                ("grid",   slopes_grid_l1(d, 5)),
                ("tangent", slopes_tangent_l1(rng.uniform(-4, 4, (200, d))))]:
    g = gamma_K(y, P)
    assert (g - S).max() <= 1e-9, f"{name}: Gamma_K > S by {(g-S).max():.2e}"
    print(f"(b) Gamma_K <= S  [{name:7s}]  max(Gamma-S) = {(g-S).max(): .2e}   OK")

# (c) tangency: with p_k = clip(y_k), Gamma_K(y_k) == S(y_k) exactly
yk = rng.uniform(-4, 4, (300, d))
P = slopes_tangent_l1(yk)
gap = huber_S(yk) - gamma_K(yk, P)
print(f"(c) tangency at samples: max |S-Gamma| = {np.abs(gap).max():.2e}        OK")
assert np.abs(gap).max() < 1e-10

# (d) closed-form error == generic error
P = slopes_random_l1(d, 64, rng)
e_generic = huber_S(y) - gamma_K(y, P)
e_exact = huber_gamma_error_exact(y, P)
print(f"(d) closed form vs generic: max diff = {np.abs(e_generic-e_exact).max():.2e}  OK")
assert np.abs(e_generic - e_exact).max() < 1e-9

# (e) the sandwich, d=2: Gamma_K <= S <= U_M on conv{y_m}
d = 2
ym = rng.uniform(-3.5, 3.5, (60, d))
sm = huber_S(ym)
Pt = slopes_tangent_l1(ym)
yq = rng.uniform(-1.5, 1.5, (40, d))          # well inside conv{y_m}
lo = gamma_K(yq, Pt); Sq = huber_S(yq)
hi = convex_upper_bound(yq, ym, sm)
fin = np.isfinite(hi)
print(f"(e) sandwich on {fin.sum()}/{len(yq)} in-hull queries:")
print(f"    max(Gamma - S) = {(lo[fin]-Sq[fin]).max(): .2e}   (<= 0)")
print(f"    max(S - U)     = {(Sq[fin]-hi[fin]).max(): .2e}   (<= 0)")
assert (lo[fin] - Sq[fin]).max() <= 1e-9
assert (Sq[fin] - hi[fin]).max() <= 1e-9
# certificate brackets the true error
cert = hi[fin] - lo[fin]; true_err = Sq[fin] - lo[fin]
print(f"    certificate U-Gamma >= true error S-Gamma everywhere: {bool((cert>=true_err-1e-9).all())}")
print(f"    median certificate {np.median(cert):.4f} vs median true err {np.median(true_err):.4f}")
print("\nALL BOUNDS TESTS PASS")
