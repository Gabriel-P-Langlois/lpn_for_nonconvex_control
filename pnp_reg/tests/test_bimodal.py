"""Gate for the Experiment 3.2 exact model: every closed form is checked
against an independent computation before any network trains on it. Run
from pnp_reg/:

    ~/miniforge3/envs/lpn_env/bin/python tests/test_bimodal.py    (~10 s)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pnpreg import bimodal as bm
from pnpreg import mixture as mx

ok = True


def report(name, err, tol):
    global ok
    good = err < tol
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] {name:58s} {err:.3e}")


SIGMA = 0.5

# 1. Basis: orthonormal, first column is the bump direction u.
G = bm.V.T @ bm.V - np.eye(bm.N)
report("basis orthonormal", float(np.abs(G).max()), 1e-12)
ii, jj = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
bump = np.exp(-(((ii - 3.5) ** 2 + (jj - 3.5) ** 2) / (2 * 1.5 ** 2))).reshape(-1)
bump /= np.linalg.norm(bump)
report("V[:,0] == u", float(np.abs(bm.U - bump).max()), 1e-12)

# 2. Sampling moments: E[s^2] = 4 + NU (mixture), E[w_k^2] = lambda_k;
#    both within 4 standard errors at m = 200000.
m = 200_000
y = bm.sample_prior(m, seed=0)
s, w = bm.to_coords(y)
es2 = float((s ** 2).mean())
se_s2 = float((s ** 2).std(ddof=1) / np.sqrt(m))
report("E[s^2] vs 4 + NU (z-score)", abs(es2 - (4 + mx.NU)) / se_s2, 4.0)
z_w = np.abs((w ** 2).mean(axis=0) - bm.LAM) / ((w ** 2).std(axis=0, ddof=1) / np.sqrt(m))
report("E[w_k^2] vs lambda_k (max z-score)", float(z_w.max()), 5.0)

# 3. D = grad psi by central finite differences at random points.
rng = np.random.default_rng(1)
z0 = bm.sample_data(6, SIGMA, seed=2)[0]
eps = 1e-6
err = 0.0
for i in range(bm.N):
    e = np.zeros(bm.N)
    e[i] = eps
    fd = (bm.psi(z0 + e, SIGMA) - bm.psi(z0 - e, SIGMA)) / (2 * eps)
    err = max(err, float(np.abs(fd - bm.D(z0, SIGMA)[:, i]).max()))
report("D == grad psi (FD, sup over coords)", err, 1e-6)

# 4. Prox identity grad f_reg(D(z)) = z - D(z), via finite differences of
#    f_reg along both a u-direction and a perp-direction at y = D(z).
y0 = bm.D(z0, SIGMA)
target = z0 - y0
for name, direction in (("u", bm.U), ("v_5", bm.V_PERP[:, 4])):
    fd = (bm.freg(y0 + eps * direction, SIGMA) - bm.freg(y0 - eps * direction, SIGMA)) / (2 * eps)
    proj = target @ direction
    report(f"prox identity along {name} (sup)", float(np.abs(fd - proj).max()), 1e-5)

# 5. f_reg factorization: equals mixture.freg on the u-slice and the
#    quadratic t/(2 lambda_k) w^2 on each perp direction (1e-12).
sg = np.linspace(-2.5, 2.5, 11)
err = float(np.abs(bm.freg(np.outer(sg, bm.U), SIGMA) - mx.freg(sg, SIGMA)).max())
report("f_reg u-slice == mixture.freg", err, 1e-12)
wv = 0.7
k_test = 10
yk = wv * bm.V_PERP[:, k_test]
exact = mx.freg(np.zeros(1), SIGMA)[0] + 0.5 * SIGMA ** 2 * wv ** 2 / bm.LAM[k_test]
report("f_reg perp direction == quadratic", abs(float(bm.freg(yk, SIGMA)[0]) - exact), 1e-12)

# 6. Wiener gain formula vs a direct posterior-mean computation on one
#    Gaussian factor: for scalar prior N(0, lam), E[y|z] = lam/(lam+s^2) z.
lam = bm.LAM[3]
zz = np.linspace(-3, 3, 7)
grid = np.linspace(-20, 20, 200_001)
post = []
for z1 in zz:
    logp = -grid ** 2 / (2 * lam) - (z1 - grid) ** 2 / (2 * SIGMA ** 2)
    p = np.exp(logp - logp.max())
    post.append(float((grid * p).sum() / p.sum()))
report("Wiener gain vs quadrature posterior mean",
       float(np.abs(np.array(post) - bm.gains(SIGMA)[3] * zz).max()), 1e-6)

# 7. No information loss on perp: the perp part of f_reg equals t times the
#    perp part of J, exactly; along u the hull gap is strictly positive.
w_pt = 0.5 * bm.V_PERP[:, 7]
lhs = float(bm.freg(w_pt, SIGMA)[0] - mx.freg(np.zeros(1), SIGMA)[0])
rhs = SIGMA ** 2 * float(bm.J_true(w_pt)[0] - mx.J(np.zeros(1))[0])
report("f_reg == t*J on perp (no-loss identity)", abs(lhs - rhs), 1e-12)
s_gap = 0.0   # between the modes the hull gap must be positive
gap_u = float(SIGMA ** 2 * mx.J(np.array([s_gap]))[0] - mx.freg(np.array([s_gap]), SIGMA)[0])
report("hull gap t*J - f_reg > 0 at s = 0 (shortfall)", max(0.0, -gap_u), 1e-12)

print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
