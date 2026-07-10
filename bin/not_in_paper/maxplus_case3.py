"""Max-plus prior J(x) = max_i <p_i, x>  (gamma = 0).   NOT IN main.pdf.

Ported from exps/exp_4_1_3_minplus_8D.ipynb -- which, despite its filename, is a
MAX-plus (Hopf) prior, not the paper's min-plus mixture of quadratics. The
notebook's S(y,1) = max_i{<p_i,y> + 0.5||p_i||^2} is wrong (it exceeds J
everywhere); src.targets.MaxPlus carries the corrected Hopf solution. See
bin/not_in_paper/README.md.

    python bin/not_in_paper/maxplus_case3.py --dim 8
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _run import cli
from src.targets import MaxPlus

DIMS = (2, 4, 8)  # notebook used 8; runs above d=8 need explicit approval
M_VECTORS = 4
SEED = 0  # the notebook drew p_true unseeded; fixed here for reproducibility


def config(dim):
    P = np.random.default_rng(SEED).uniform(-1, 1, (M_VECTORS, dim))
    return MaxPlus(P, gamma=None)


if __name__ == "__main__":
    cli("maxplus_case3", config, DIMS)
