"""Experiment package for the plug-and-play registration task (task document
``__tasks/task_pnp_registration.tex``).

Contents so far: Experiment 1, the mixture figure (Section 6.1 there) -- the
1-D two-Gaussian prior on which everything is exact, used to display the
information loss J vs f_reg = t * J_BVS and the necessity of the semiconvex
network class. Networks and the training loop live in ``src/`` (gradfit.py,
network.py); this package holds only what is specific to the experiment.

    mixture.py     the prior, denoiser, psi, f_reg, J_BVS -- exact
    readout.py     J_theta = G_theta - |y|^2/2 and its gradient
    experiment.py  data -> two fits -> metrics (entry point, --smoke)
    figures.py     the four-curve panel
"""
from . import paths  # noqa: F401  (puts the numerics ROOT on sys.path first)
from . import mixture, readout  # noqa: F401,E402

# experiment and figures are imported on demand (they pull torch/matplotlib
# and would trip runpy's double-import warning under `python -m`).
__all__ = ["experiment", "figures", "mixture", "paths", "readout"]
