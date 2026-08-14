"""ONE font size for every figure in this repository.

Figures were carrying per-call `fontsize=` overrides between 6 and 13 pt, so a
panel title in one figure did not match a panel title in the next and a legend
could be half the size of the axis label beside it. The sizes are now set once,
here, through rcParams; plotting code sets no font size of its own.

Call `apply()` at import time in any module that draws. It is idempotent and
touches nothing but font sizes, so it cannot change the content of a figure.
"""
import matplotlib as mpl

FONT_SIZE = 11


def apply(size=FONT_SIZE):
    """Set every text element in every subsequent figure to `size` points."""
    mpl.rcParams.update({
        "font.size": size,
        "axes.titlesize": size,
        "axes.labelsize": size,
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        "legend.fontsize": size,
        "legend.title_fontsize": size,
        "figure.titlesize": size,
    })
