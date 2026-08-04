"""Gate for the probe estimators, on dense matrices where everything is
exact. No networks, no data. Run from pnp_reg/:

    ~/miniforge3/envs/lpn_env/bin/python tests/test_probe.py    (~15 s)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pnpreg import probe

ok = True


def report(name, err, tol):
    global ok
    good = err < tol
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] {name:58s} {err:.3e}")


N = 50
rng = np.random.default_rng(0)
A = rng.normal(size=(N, N)) + 0.05 * rng.normal(size=(N, N))
S_half = 0.5 * (A + A.T)
exact_rho = np.linalg.norm(A - A.T) / (2 * np.linalg.norm(A))
ev = np.linalg.eigvalsh(S_half)
op = probe.DenseOp(torch.tensor(A))

# 1. rho_hat vs exact within 4 bootstrap SE at M = 5000, and the bootstrap SE
#    itself validated against the empirical spread of 20 independent repeats
#    (an SE off by more than 2x in either direction fails).
r = probe.hutchinson_asymmetry(op, m_probes=5000, seed=1)
report("rho_hat vs exact (units of bootstrap SE)",
       abs(r["rho"] - exact_rho) / r["rho_se"], 4.0)
reps = [probe.hutchinson_asymmetry(op, m_probes=100, seed=100 + i)["rho"]
        for i in range(20)]
se_emp = float(np.std(reps, ddof=1))
se_boot = probe.hutchinson_asymmetry(op, m_probes=100, seed=1)["rho_se"]
ratio = se_boot / se_emp
report("bootstrap SE vs empirical spread (|log2 ratio|)",
       abs(np.log2(ratio)), 1.0)

# 2. Symmetric matrix: rho_hat identically ~0 in float64 (b_j cancels to
#    roundoff), and identity_max at machine precision.
opS = probe.DenseOp(torch.tensor(S_half))
rS = probe.hutchinson_asymmetry(opS, m_probes=64, seed=2)
report("symmetric matrix: rho_hat == 0", rS["rho"], 1e-12)
report("symmetric matrix: identity_max", rS["identity_max"], 1e-12)

# 3. Lanczos at k = N reproduces the dense spectrum's extremes to LAPACK
#    accuracy, with an orthogonal basis.
l = probe.lanczos_multistart(op, k=N, starts=2, seed0=5)
report("lanczos k=n lmin vs eigh", abs(l["lmin"] - ev[0]), 1e-10)
report("lanczos k=n lmax vs eigh", abs(l["lmax"] - ev[-1]), 1e-10)
report("lanczos basis orthogonality", l["orth_err"], 1e-12)

# 4. Ritz residual bound validity at k = 15 (unconverged): for each extreme,
#    |theta - nearest exact eigenvalue| <= residual (Ritz bound theorem).
l15 = probe.lanczos_symmetric(op, k=15, seed=7)
gap_min = float(np.min(np.abs(ev - l15["lmin"])))
gap_max = float(np.min(np.abs(ev - l15["lmax"])))
report("ritz bound holds for lmin (gap - res)", max(0.0, gap_min - l15["res_lmin"]), 1e-12)
report("ritz bound holds for lmax (gap - res)", max(0.0, gap_max - l15["res_lmax"]), 1e-12)

# 5. Rank-3 symmetric matrix: breakdown terminates early with the exact
#    nonzero extremes (invariant subspace found).
V = rng.normal(size=(N, 3))
low = V @ np.diag([3.0, -1.0, 0.5]) @ V.T
low = 0.5 * (low + low.T)
evl = np.linalg.eigvalsh(low)
lb = probe.lanczos_symmetric(probe.DenseOp(torch.tensor(low)), k=40, seed=9)
report("rank-3 breakdown: early termination (iters - 10)",
       max(0.0, lb["n_iter"] - 10.0), 0.5)
report("rank-3 breakdown: lmin exact", abs(lb["lmin"] - evl[0]), 1e-9)
report("rank-3 breakdown: lmax exact", abs(lb["lmax"] - evl[-1]), 1e-9)

# 6. The estimator resolves an asymmetry of size 1e-6 planted on a symmetric
#    matrix (rho_hat lands within a factor 2 of the exact tiny ratio) --
#    the smallest asymmetry the PIRATE rows could need to certify.
E = rng.normal(size=(N, N))
K = 0.5e-6 * (E - E.T)
Apert = S_half + K
exact_tiny = np.linalg.norm(K) / (np.sqrt(np.linalg.norm(S_half) ** 2
                                          + np.linalg.norm(K) ** 2))
rt = probe.hutchinson_asymmetry(probe.DenseOp(torch.tensor(Apert)),
                                m_probes=256, seed=11)
report("planted 1e-6 asymmetry resolved (|log2 rho/exact|)",
       abs(np.log2(rt["rho"] / exact_tiny)), 1.0)

# 7. float32 operator: Lanczos with float64 recurrence scalars recovers the
#    extremes of the float32-rounded matrix to ~1e-5; the basis is STORED in
#    float32 (memory: ~600 MB saved at n = 2.6e6), so its orthogonality
#    bottoms out at float32 eps, not float64 -- gate accordingly.
A32 = torch.tensor(S_half, dtype=torch.float32)
ev32 = np.linalg.eigvalsh(A32.numpy().astype(np.float64))
l32 = probe.lanczos_multistart(probe.DenseOp(A32), k=N, starts=2, seed0=13)
report("float32 lanczos lmin", abs(l32["lmin"] - ev32[0]), 1e-4)
report("float32 lanczos lmax", abs(l32["lmax"] - ev32[-1]), 1e-4)
report("float32 lanczos orthogonality (float32 basis)", l32["orth_err"], 1e-6)

# 8. Ritz vectors (return_ritz=True): at k = n the extreme Ritz vectors
#    equal the eigh eigenvectors up to sign, and satisfy the eigen-equation.
lr = probe.lanczos_symmetric(probe.DenseOp(torch.tensor(S_half)), k=N, seed=17,
                             return_ritz=True)
ew, EV = np.linalg.eigh(S_half)
for key, col, theta in (("ritz_vec_lmin", 0, ew[0]), ("ritz_vec_lmax", -1, ew[-1])):
    v = lr[key].numpy()
    align = abs(float(v @ EV[:, col]))
    report(f"{key} aligned with eigh eigenvector (1 - |cos|)", 1.0 - align, 1e-8)
    report(f"{key} eigen-residual ||Sv - theta v||",
           float(np.linalg.norm(S_half @ v - theta * v)), 1e-8)

print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
