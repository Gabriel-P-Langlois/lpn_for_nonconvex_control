import sys, os; sys.path.insert(0, os.path.abspath("."))
import numpy as np
from src.maxplus_bounds import gamma_K, huber_S, slopes_grid_l1, grid_error_closed_form
rng = np.random.default_rng(0)
for d in (1, 2, 3, 4):
    for m in (2, 3, 5, 8):
        if m**d > 200000: continue
        y = rng.uniform(-4, 4, (500, d))
        e_direct = huber_S(y) - gamma_K(y, slopes_grid_l1(d, m))
        e_form = grid_error_closed_form(y, m)
        dmax = np.abs(e_direct - e_form).max()
        assert dmax < 1e-9, (d, m, dmax)
    print(f"d={d}: closed form matches max-plus evaluation for m=2,3,5,8")
# and the corner claim: all |y_i|>1  =>  exactly zero error on any grid
y = rng.uniform(1.01, 4, (200, 3)) * rng.choice([-1, 1], (200, 3))
e = huber_S(y) - gamma_K(y, slopes_grid_l1(3, 4))
print(f"\nqueries with all |y_i|>1: max error = {e.max():.2e}  (grid contains p*=corner)")
