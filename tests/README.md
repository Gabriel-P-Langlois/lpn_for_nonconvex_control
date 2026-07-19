# Regression tests

Plain scripts, no pytest. Run from `numerics/` with the project interpreter:

    ~/miniforge3/envs/lpn_env/bin/python tests/test_maxplus_bounds.py

Each asserts and prints; a non-zero exit means a claim broke.

| test | what it pins down |
|---|---|
| `test_maxplus_bounds.py` | `huber_S` equals the verified `QuadraticL1.hjsol_true`; `Gamma_K <= S` for grid/random/tangent slopes; tangency `Gamma_K(y_k) = S(y_k)` for `p_k = clip(y_k)`; the closed-form error matches the generic evaluation; the sandwich `Gamma_K <= S <= U_M` holds and the certificate `U_M - Gamma_K` dominates the true error. |
| `test_grid_closed_form.py` | The separable identity `S - Gamma_K = 0.5 * sum_{i:|y_i|<=1} dist(y_i, axis grid)^2`, and that a query with every `\|y_i\| > 1` has EXACTLY zero error on any tensor grid (the grid contains `p* = ` corner). This is the boundary/interior dichotomy that sets the exponent. |
| `test_diverged_route1.py` | When every Route-1 alpha diverges, `prior_rmse_route1` is `None`, not the min over the diverged runs (bug B12). Forces the branch by shrinking `preimage_bound` to 1e-3. Trains a real (smoke) network, so it takes ~1 min. |
| `test_chunked_conjugate.py` | `conjugate_samples` is bit-identical chunked vs unchunked, on a ragged split (bug B15). |

`test_diverged_route1.py` needs `SCRATCH` set to a writable directory:

    SCRATCH=/tmp/probe ~/miniforge3/envs/lpn_env/bin/python tests/test_diverged_route1.py
