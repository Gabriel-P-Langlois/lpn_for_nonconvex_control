"""Every filesystem location this package uses, defined once.

Before the package existed each module recomputed its own HERE/ROOT from
__file__ and pasted its own "results/figs" and "logs/ckpt" strings together.
Moving a file then silently changed where it read and wrote. Collecting the
locations here means a move breaks nothing and the layout is auditable in one
screen.

Layout:

    numerics/                     ROOT
      logs/ckpt/                  LOG_CKPT -- local training output (gitignored)
      src/                        the shared network code (needs ROOT on sys.path)
      tv_pm/            BASE
        tvpm/                     PKG      -- this package
        images/                   IMAGES   -- the source .mat images (tracked)
        data/                     DATA     -- cached sampler output
        results/figs/             FIGS     -- figures that travel
        results/ckpt/             RES_CKPT -- tracked checkpoint copies
"""
import os
import sys

PKG = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(PKG)                                  # tv_pm/
ROOT = os.path.abspath(os.path.join(BASE, ".."))             # numerics/

IMAGES = os.path.join(BASE, "images")
DATA = os.path.join(BASE, "data")
RESULTS = os.path.join(BASE, "results")
FIGS = os.path.join(RESULTS, "figs")

LOG_CKPT = os.path.join(ROOT, "logs", "ckpt")
RES_CKPT = os.path.join(RESULTS, "ckpt")
# Search order: local training output first, then the tracked copies that travel
# to a colleague (logs/ and *.pth are gitignored).
CKPT_DIRS = (LOG_CKPT, RES_CKPT)

# src/ lives at ROOT and is imported as `src.network`, so ROOT must be importable.
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def find_ckpt(name):
    """Absolute path of checkpoint `name`, searching CKPT_DIRS in order.

    Returns the LOG_CKPT path when the file is absent anywhere, so a caller can
    report a useful location or write there.
    """
    for d in CKPT_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(CKPT_DIRS[0], name)


def ensure_dirs():
    """Create the writable output directories. Read-only ones are not created."""
    for d in (DATA, FIGS, LOG_CKPT):
        os.makedirs(d, exist_ok=True)
