"""The semiconvex readout: J_theta(y) = G_theta(y) - |y|^2/2 (eq:param of the
task document). G_theta is an input-convex network; the quadratic enters ONLY
here, at readout -- the trained object is an ordinary ICNN.

The chain rule is written down before it is coded (task document, Section 8):
no standardization is used in this experiment, so

    J_theta(y)      = G_theta(y) - |y|^2/2
    grad J_theta(y) = grad G_theta(y) - y

and the finite-difference gate in tests/test_readout.py checks exactly this.
Inputs may be flat (N,) arrays; the network sees (N, 1).
"""
import numpy as np

from src.gradfit import net_grad, net_value


def _col(y):
    return np.asarray(y, float).reshape(-1, 1)


def value(G, y):
    """J_theta(y) on flat 1-D points y: (N,) -> (N,)."""
    y2 = _col(y)
    return net_value(G, y2) - 0.5 * (y2 ** 2).sum(axis=1)


def grad(G, y):
    """grad J_theta(y) on flat 1-D points y: (N,) -> (N,)."""
    y2 = _col(y)
    return (net_grad(G, y2) - y2).ravel()
