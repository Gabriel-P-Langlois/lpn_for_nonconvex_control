"""Gate for the Experiment 3.1 machinery, on operators where everything is
exact. No released networks. Run from pnp_reg/:

    ~/miniforge3/envs/lpn_env/bin/python tests/test_affine.py    (~10 s)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pnpreg import affine
from pnpreg.probe import DenseOp

ok = True


def report(name, err, tol):
    global ok
    good = err < tol
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] {name:58s} {err:.3e}")


torch.manual_seed(0)
rng = np.random.default_rng(0)
n = 32

# A symmetric matrix with spectrum well inside (0, 2): S = I + 0.05 * sym(M).
M = rng.normal(size=(n, n))
S = np.eye(n) + 0.05 * 0.5 * (M + M.T)
S_t = torch.tensor(S)
b = torch.tensor(rng.normal(size=n))

# 1. Affinity metric is exactly 0 for an affine map (jvp of an affine torch
#    function is exact), on the same code path the experiment uses.
zs = [torch.tensor(rng.normal(size=n)) for _ in range(4)]
aff, b_hat = affine.affinity_from_ops(lambda z: S_t @ z + b,
                                      lambda z, v: S_t @ v, zs, n_pairs=4)
report("affine operator: affinity error", aff["err_max"], 1e-12)
report("affine operator: b recovered", float((b_hat - b).norm() / b.norm()), 1e-12)
report("affine operator: b spread", aff["b_spread"], 1e-12)

# 2. A genuinely nonlinear map (cubic perturbation) reads O(separation),
#    far above any floor -- the metric can tell the difference.
eps = 0.1
aff_nl, _ = affine.affinity_from_ops(
    lambda z: S_t @ z + b + eps * z ** 3,
    lambda z, v: S_t @ v + 3 * eps * (z ** 2) * v, zs, n_pairs=4)
report("nonlinear operator: affinity error detected (shortfall vs 1e-3)",
       max(0.0, 1e-3 - aff_nl["err_max"]), 1e-12)

# 3. Closed-form inversion: for D(z) = S z + b, the quadratic with gradient
#    grad f(y) = S^{-1}(y - b) - y satisfies the prox identity exactly:
#    argmin_u f(u) + ||u - z||^2/2 = S z + b for every z.
#    Verify via the optimality condition grad f(y) + y - z = 0 at y = Sz + b.
Sinv = np.linalg.inv(S)
for z in zs[:2]:
    y = S_t @ z + b
    grad_f = torch.tensor(Sinv) @ (y - b) - y
    report("prox identity of the closed-form quadratic",
           float((grad_f + y - z).norm() / z.norm()), 1e-10)

# 4. The Hessian of that quadratic is S^{-1} - I with extremes 1/lambda - 1
#    at the extremes of S (the monotone-map shortcut the experiment uses).
ev = np.linalg.eigvalsh(S)
H_ev = np.linalg.eigvalsh(Sinv - np.eye(n))
report("H extremes = 1/lambda - 1 (lmax side)",
       abs(H_ev[-1] - (1.0 / ev[0] - 1.0)), 1e-10)
report("H extremes = 1/lambda - 1 (lmin side)",
       abs(H_ev[0] - (1.0 / ev[-1] - 1.0)), 1e-10)

# 5. CG-on-S (the matrix-free solve used at field size) vs dense solve.
op = DenseOp(S_t)
rhs = torch.tensor(rng.normal(size=n))
w, iters, cg_resid = affine.cg_solve_S(op, rhs, tol=1e-12)
w_dense = torch.tensor(np.linalg.solve(S, rhs.numpy()))
report("CG on S vs dense solve", float((w - w_dense).norm() / w_dense.norm()), 1e-8)

print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
