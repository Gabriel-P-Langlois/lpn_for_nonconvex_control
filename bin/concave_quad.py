"""Concave-quadratic family: J(x) = -0.25||x||^2, H(p) = 0.5||p||^2.

S(y,t) = -||y||^2 / (2(2-t)); at t = 1, psi(y) = ||y||^2. Verified correct in
Phase 1 and carried over unchanged. grad psi = 2y CONTRACTS (preimage x/2), so
the training box equals the query box: unaffected by the D2 box amendment.

Migrated from legacy/old_notebooks/exp_quadratic_concave_prior_{2,4,8,16,32,64}D.ipynb. The
*_MC_32D and *_32D_test_v1 notebooks were stale hyperparameter variants
(beta 10/100, hidden 50/128, layers 4), not different math; D2 supersedes them.

    python bin/concave_quad.py --dim 32
"""
from _run import cli
from src.targets import ConcaveQuad

DIMS = (2, 4, 8, 16, 32, 64)


def config(dim):
    return ConcaveQuad(t=1.0)


if __name__ == "__main__":
    cli("concave_quad", config, DIMS)
