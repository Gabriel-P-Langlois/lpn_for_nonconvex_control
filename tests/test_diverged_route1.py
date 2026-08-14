"""Force every alpha to 'diverge' by shrinking the preimage bound to ~0.
Exercises the all-diverged branch of bin/_run.py: prior_rmse_route1 must be
None, not the min over the diverged runs."""
import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("bin"))
from _run import run
from src.targets import QuadraticL1

class Pathological(QuadraticL1):
    def preimage_bound(self, a):
        return 1e-3          # any solve "diverges" against this
    def train_halfwidth(self, a):
        return 5.0           # keep training identical to the real family

m = run("diverge_probe", Pathological(), dim=2, smoke=True,
        out_dir=os.environ["SCRATCH"])
print("\n--- assertions ---")
assert m["route1_all_alphas_diverged"] is True, "flag not set"
assert m["prior_rmse_route1"] is None, f"expected None, got {m['prior_rmse_route1']}"
assert all(m["route1_diverged_per_alpha"].values()), "per-alpha flags wrong"
assert m["prior_rmse_route2"] is not None, "One-shot recovery must be unaffected"
print("PASS: all alphas diverged -> prior_rmse_route1 is None, One-shot recovery intact")
print("      per-alpha RMSEs still recorded:", m["route1_rmse_per_alpha"])
