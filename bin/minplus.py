"""Min-plus (two-component mixture) family; H(p) = 0.5||p||^2.

Mode geometry is the same at every dimension, as in the notebooks:
mu1 = e_1, mu2 = 1/sqrt(d), sigma1 = sigma2 = 1. With sigma = 1 the preimage
map y* = 2x - mu EXPANDS by 2, so _run.py trains psi on [-9,9]^d to cover the
[-4,4]^d query box. psi = max(psi_1, psi_2) is nonsmooth on the ridge; the
preimage there has a closed form only because sigma1 == sigma2.

Migrated from legacy/old_notebooks/exp_1_minplus_{2,4,8,16,32,64}D.ipynb ('8D copy' is a
duplicate). NOT migrated: exp_4_1_3 / exp_4_1_4 (max-plus Hopf targets trained
with LPN's proximal matching loss on a gamma schedule, not MSE on psi) -- these
need a training-scheme decision and stay in legacy/old_notebooks/ for now.

    python bin/minplus.py --dim 8
"""
import numpy as np

from _run import cli
from src.targets import Minplus

DIMS = (2, 4, 8, 16, 32, 64)


def config(dim):
    mu1 = np.zeros(dim)
    mu1[0] = 1.0
    mu2 = np.ones(dim) / np.sqrt(dim)
    return Minplus(mu1=mu1, mu2=mu2, sigma1=1.0, sigma2=1.0)


if __name__ == "__main__":
    cli("minplus", config, DIMS)
