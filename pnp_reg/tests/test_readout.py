"""Gate for the semiconvex readout: the chain rule, checked by finite
differences at random initialization (task document, Section 8: getting this
wrong yields a plausible network wrong by a linear term). Run from pnp_reg/:

    ~/miniforge3/envs/lpn_env/bin/python tests/test_readout.py    (~2 s)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pnpreg import readout
from src.network import LPN

ok = True


def report(name, err, tol):
    global ok
    good = err < tol
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] {name:58s} {err:.3e}")


torch.manual_seed(0)
G = LPN(in_dim=1, hidden=32, layers=2, beta=5)
y = np.random.default_rng(0).uniform(-4, 4, 24)

# 1. Finite differences of the readout VALUE reproduce the readout GRADIENT.
# The helpers evaluate in float32, so the FD roundoff floor is ~1e-7/eps;
# eps = 1e-2 balances it against truncation. The bug this gate exists to catch
# (a wrong or missing linear term in the chain rule) would show as an O(1)
# error, three orders above the tolerance.
eps = 1e-2
fd = (readout.value(G, y + eps) - readout.value(G, y - eps)) / (2 * eps)
report("FD of J_a value vs readout.grad (sup)",
       float(np.max(np.abs(fd - readout.grad(G, y)))), 1e-3)

# 2. The readout subtracts exactly the quadratic: J_a + y^2/2 = G.
from src.gradfit import net_value
report("J_a + y^2/2 = G_theta (sup)",
       float(np.max(np.abs(readout.value(G, y) + 0.5 * y ** 2
                           - net_value(G, y.reshape(-1, 1))))), 1e-6)

# 3. The gradient subtracts exactly the identity: grad J_a + y = grad G.
from src.gradfit import net_grad
report("grad J_a + y = grad G_theta (sup)",
       float(np.max(np.abs(readout.grad(G, y) + y
                           - net_grad(G, y.reshape(-1, 1)).ravel()))), 1e-6)

print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
