"""Max-plus prior J(x) = max_i {<p_i, x> - 0.5||p_i||^2}.   NOT IN main.pdf.

Ported from legacy/old_notebooks/exp_4_1_4_minplus_8D.ipynb. Same corrections and caveats as
maxplus_case3.py: the notebook's S(y,1) = max_i <p_i,y> violates S <= J at every
test point; the corrected Hopf solution lives in src.targets.MaxPlus. See
bin/not_in_paper/README.md.

    python bin/not_in_paper/maxplus_case4.py --dim 8
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _run import cli
from src.targets import MaxPlus

DIMS = (2, 4, 8)
M_VECTORS = 4
SEED = 0


def config(dim):
    P = np.random.default_rng(SEED).uniform(-1, 1, (M_VECTORS, dim))
    return MaxPlus(P, gamma=0.5 * np.sum(P * P, axis=1))


if __name__ == "__main__":
    cli("maxplus_case4", config, DIMS)
