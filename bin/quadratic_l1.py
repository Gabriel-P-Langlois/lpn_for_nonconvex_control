"""Quadratic-L1 family: J(x) = ||x||_1, H(p) = 0.5||p||^2.

S(y,1) is the separable Huber (Moreau envelope of ||.||_1). The corrected target
lives in src.targets.QuadraticL1; grad psi = soft(.,1) EXPANDS, so _run.py trains
psi on [-5,5]^d to cover the [-4,4]^d query box.

Migrated from exps/exp_4_1_2_quadratic_{2,4,8,16,32,64}D.ipynb (corrupted Moreau
target, finding 2) and the duplicate exps/exp_L1_prior_*.ipynb (same problem,
correct target). d=1 is exps/exp_4_2_1_1D.ipynb. The *_2D_MC notebook was a
stale hyperparameter variant, not a Monte-Carlo method; D2 supersedes it.

    python bin/quadratic_l1.py --dim 8
    python bin/quadratic_l1.py --dim 2 --smoke
"""
from _run import cli
from src.targets import QuadraticL1

DIMS = (1, 2, 4, 8, 16, 32, 64)


def config(dim):
    return QuadraticL1()


if __name__ == "__main__":
    cli("quadratic_l1", config, DIMS)
