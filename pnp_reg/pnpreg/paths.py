"""Every filesystem location this package uses, defined once.

Layout:

    numerics/                     ROOT
      src/                        shared networks + training (needs ROOT on sys.path)
      pnp_reg/          BASE
        pnpreg/                   PKG      -- this package
        results/figs/             FIGS     -- figures that travel (tracked)
        results/                  RESULTS  -- metrics.json (tracked)
"""
import os
import sys

PKG = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(PKG)                                  # pnp_reg/
ROOT = os.path.abspath(os.path.join(BASE, ".."))             # numerics/

RESULTS = os.path.join(BASE, "results")
FIGS = os.path.join(RESULTS, "figs")

# Experiment 2 externals: the verbatim PIRATE release (ext/PROVENANCE.md) and
# the tv_pm package (imported, never edited) for the calibration rows.
EXT_PIRATE = os.path.join(ROOT, "ext", "pirate")
PIRATE_CKPT_AWGN = os.path.join(EXT_PIRATE, "pretrained_model", "AWGN_denoiser", "OASIS.pth.tar")
PIRATE_CKPT_PLUS = os.path.join(EXT_PIRATE, "pretrained_model", "PIRATEplus", "OASIS.pth.tar")
PIRATE_FIELD = os.path.join(EXT_PIRATE, "data", "field.h5py")
TV_PM = os.path.join(ROOT, "tv_pm")

# src/ lives at ROOT and is imported as `src.gradfit` etc., so ROOT must be importable.
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# tvpm lives inside tv_pm/, which is not a package dir on ROOT's path.
if TV_PM not in sys.path:
    sys.path.insert(1, TV_PM)


def ensure_dirs():
    """Create the writable output directories."""
    for d in (RESULTS, FIGS):
        os.makedirs(d, exist_ok=True)
