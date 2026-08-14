# `notebooks/` — the synthetic prior-recovery study

One notebook per target family (`quadratic_l1`, `negl1`, `minplus`, `concave_quad`),
each a thin wrapper over `bin/_run.py`. They compare the two LPN recovery methods on
targets where a ground-truth prior exists.

## Naming

Renamed repo-wide; see `../src/NAMING.md` for the full table and the rationale.

| formerly | prose / comments | plot + legend | table column |
|---|---|---|---|
| Route 1 | LPN Iterative recovery | Iterative | `Iter …` |
| Route 2 | One-shot recovery | One-shot | `One-shot …` |
| Direct (Ours) | Direct fit | Direct fit | — |

Python identifiers and the `logs/*_metrics.json` keys these notebooks read
(`prior_rmse_route1`, `route2_median_prox_residual`, …) are **unchanged**, so the
cached metrics still load.

## Stored outputs carry the old labels

These notebooks ship **with** stored outputs, and those embedded figures and tables
were rendered before the rename. The notebook *source* is renamed; the *images
inside the stored outputs* still read "Route 1" / "Route 2" until the notebooks are
re-executed. `bin/plot.py` and `src/{plotting,diagnostics}.py` now emit the new
labels, so re-running cell 6 regenerates every figure correctly — budget roughly
45 min per dimension on CPU, and note that runs at d > 8 are opt-in by project
policy.

`tv_pm/recover_tv.ipynb` and `tv_pm/prior_routes/three_way_comparison.ipynb` ship
with outputs cleared, so they have no such staleness.
