"""Consolidated LPN / Hamilton-Jacobi experiment code (D2 convention).

One canonical Softplus network, corrected targets, mini-batch MSE training, and
two recovery methods. Per-experiment drivers live in ``bin/``.
"""
from .network import LPN, hidden_width
from . import targets, train, recovery, invert, plotting

__all__ = ["LPN", "hidden_width", "targets", "train", "recovery", "invert", "plotting"]
