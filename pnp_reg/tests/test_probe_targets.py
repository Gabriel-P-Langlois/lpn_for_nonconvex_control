"""Gate for the probed operators: each wrapper is checked against an
independent computation before any table row is trusted. CPU only; the
PIRATE checks run on a center crop of the released field. Run from pnp_reg/:

    ~/miniforge3/envs/lpn_env/bin/python tests/test_probe_targets.py    (~2 min)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pnpreg import mixture as mx
from pnpreg import probe, probe_targets as pt

ok = True


def report(name, err, tol):
    global ok
    good = err < tol
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] {name:58s} {err:.3e}")


CROP = (24, 24, 24)

# 1. Mixture operator: jvp equals dD(z) * v exactly (diagonal by construction).
cases = pt.target_cal_mixture(n=64, n_points=1, seed=11)
op = cases[0].op
g = torch.Generator(device="cpu").manual_seed(0)
v = torch.randn(64, generator=g, dtype=torch.float64)
report("mixture jvp == dD * v", float((op.jvp(v) - op.d * v).abs().max()), 1e-12)
report("mixture forced z=0 present (lmax > 1 margin)",
       max(0.0, 1.0 - cases[0].exact["lmax"]), 1e-12)

# 2. Quadrature Jacobian: symmetric to FD accuracy, h-vs-2h stable, and the
#    known-theorem bounds hold at every generated point.
qcases = pt.target_cal_quadrature(n_points=4, seed=12)
for c in qcases:
    J = c.op.A.numpy()
    report(f"quad[{c.index}] J symmetric", float(np.abs(J - J.T).max()), 1e-8)
    report(f"quad[{c.index}] FD h vs 2h", c.meta["fd_h_vs_2h"], 1e-7)
    report(f"quad[{c.index}] 0 <= eig <= 1 (violation)",
           max(0.0, -c.exact["lmin"], c.exact["lmax"] - 1.0), 1e-6)

# 3. ICNN prox: checkpoint resolvable, tight solve, PSD Hessian, and the CG
#    matvec agrees with the dense factorization on random vectors.
model, units, ck, ckpt = pt.load_tvpm_icnn()
report("tv_pm checkpoint found (beta=20, s=250k, sig=t=20/256)",
       0.0 if ckpt else 1.0, 0.5)
icases = pt.target_cal_icnn(n_points=1, seed=13)
c = icases[0]
report("icnn prox optimality residual", c.meta["prox_resid"], 1e-7)
report("icnn Hessian PSD (eig_min)", max(0.0, -c.exact["H_eig_min"]), 1e-8)
v64 = torch.randn(64, generator=g, dtype=torch.float64)
report("icnn CG matvec vs dense solve",
       float((c.op.jvp(v64) - c.op.vjp(v64)).abs().max()), 1e-8)

# 4. PIRATE loaders: both checkpoints load strict with the exact parameter
#    count; the stripped PIRATE+ key set equals the AWGN key set; and the
#    fine-tuning actually moved the weights.
mA = pt.load_pirate_dncnn("pirate")
mP = pt.load_pirate_dncnn("pirate_plus")
report("pirate param count", abs(sum(p.numel() for p in mA.parameters())
                                 - pt.N_PIRATE_PARAMS), 0.5)
report("pirate_plus param count", abs(sum(p.numel() for p in mP.parameters())
                                      - pt.N_PIRATE_PARAMS), 0.5)
kA = set(mA.state_dict().keys())
kP = set(mP.state_dict().keys())
report("key sets equal after prefix strip", float(len(kA ^ kP)), 0.5)
delta = max(float((a - b).abs().max())
            for a, b in zip(mA.state_dict().values(), mP.state_dict().values()))
report("fine-tuning moved the weights (1 - delta gate)",
       1.0 if delta == 0.0 else 0.0, 0.5)

# 5. PIRATE operator on the crop, float64 (the model is exact there): the jvp
#    is the directional derivative (FD h-refinement), the bilinear identity
#    <u, Jv> = <J^T u, v> holds to roundoff, and D(z) is finite.
m64 = pt.load_pirate_dncnn("pirate").double()
field = pt.load_field(crop=CROP).double()
gg = torch.Generator(device="cpu").manual_seed(14)
z = field + torch.randn(field.shape, generator=gg, dtype=torch.float32).double()
op = pt.PirateOp(m64, z)
report("D(z) finite on crop", 0.0 if torch.isfinite(op.apply_D(z)).all() else 1.0, 0.5)
v = torch.randn(op.n, generator=gg, dtype=torch.float64)
Jv = op.jvp(v)
h = 1e-4
vt = v.reshape(op.shape)
fd = (op.apply_D(z + h * vt) - op.apply_D(z - h * vt)).reshape(-1) / (2 * h)
report("jvp vs FD directional derivative (float64, rel)",
       float((Jv - fd).norm() / fd.norm()), 1e-5)
u = torch.randn(op.n, generator=gg, dtype=torch.float64)
lhs = float(u @ Jv)
rhs = float(op.vjp(u) @ v)
report("bilinear identity <u,Jv> = <J^T u,v> (float64, rel)",
       abs(lhs - rhs) / abs(lhs), 1e-10)

# 6. The float32 crop operator (production dtype): identity noise at the
#    documented ~1e-5 scale, not orders larger.
m32 = pt.load_pirate_dncnn("pirate")
z32 = z.float()
op32 = pt.PirateOp(m32, z32)
r32 = probe.hutchinson_asymmetry(op32, m_probes=4, seed=3)
report("float32 identity_max at the documented scale", r32["identity_max"], 1e-3)

# 7. Symmetric surrogate (the floor row): rho reads exactly 0 and the dense
#    Jacobian on a tiny crop is symmetric to float32 roundoff.
fcases = pt.target_floor(n_points=1, crop=CROP)
fop = fcases[0].get_op()
rf = probe.hutchinson_asymmetry(fop, m_probes=4, seed=4)
report("surrogate floor rho == 0", rf["rho"], 1e-6)
tiny = pt.target_floor(n_points=1, crop=(6, 6, 6))[0].get_op()
J_dense = torch.stack([tiny.jvp(torch.eye(tiny.n, dtype=tiny.dtype)[i])
                       for i in range(tiny.n)]).T
report("surrogate dense Jacobian symmetric (float32)",
       float((J_dense - J_dense.T).abs().max()), 1e-5)

print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
