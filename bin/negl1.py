"""NegL1 family: J(x) = -||x||_1 (nonconvex prior), H(p) = 0.5||p||^2.

S(y,1) = -||y||_1 - n t/2; the dimension factor n was missing in the notebooks
(Phase 1 finding). grad psi(y) = y + sign(y) CONTRACTS, so the training box
equals the query box: this family is unaffected by the D2 box amendment.

CAVEAT (open, see changes.txt): grad psi omits the open gap (-1,1) per
coordinate, so the Route-2 samples y_k = grad psi(x_k) have an interior HOLE
around the origin. The box margin does not fix this; Route-2 accuracy near
x = 0 should be read with that in mind.

Migrated from legacy/old_notebooks/exp_NegL1_prior_{2,4,8,16,32,64}D.ipynb.

    python bin/negl1.py --dim 16
"""
from _run import cli
from src.targets import NegL1

DIMS = (2, 4, 8, 16, 32, 64)


def config(dim):
    return NegL1()


if __name__ == "__main__":
    cli("negl1", config, DIMS)
