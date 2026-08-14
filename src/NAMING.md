# Naming of the prior-recovery methods

Adopted repo-wide. Prose, printed output and plot text use these names; **Python
identifiers and JSON metric keys were left unchanged on purpose**.

| old | prose / comments / docs | plots, legends, axis + panel titles |
|---|---|---|
| Route 1, LPN (Fang et al.) | LPN Iterative recovery | Iterative |
| Route 2, LPN (Ours) | One-shot recovery | One-shot |
| Direct (Ours) | Direct fit | Direct fit |

## Why identifiers were not renamed

`route1` / `route2` appear as function names (`recover_prior_route1`,
`route2_preimage`), as keys in every `logs/*_metrics.json` written to date
(`prior_rmse_route1`, `route1_diagnostic`, `route2_frac_query_uncovered`, ...), and in
`tests/test_diverged_route1.py`, which asserts on those keys. Renaming them would
invalidate the cached metrics that `notebooks/*.ipynb` read at cell 4 and break the
test suite, for no gain in the published output. If you do want the identifiers
renamed, it needs a migration for the logs, not a find-and-replace.

## Font sizes

`src/plotstyle.py` sets one size (11 pt) for every text element of every figure via
rcParams. Call `apply()` once at import in any module that draws; do not pass
`fontsize=` at call sites.
