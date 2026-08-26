"""T1 unit gate for the exact-prox ablation (experiments_plan.tex, sub:tests).

Plain script: assert + print, no pytest. Run from numerics/ with the project
interpreter:

    ~/miniforge3/envs/lpn_env/bin/python tests/test_exact_prox.py

Pins, for every family and no training:
  (A) the exact triples satisfy g(y_k) = <x_k, y_k> - psi(x_k) AND agree with the
      independent conjugate value g_exact(y_k) to ~1e-12 (correctness of grad_psi);
  (B) for the CONVEX families, g(y_k) = t*prior_true(y_k) + 0.5||y_k||^2; and for
      NegL1, J_BVS reconstructs from g and differs from the nonconvex prior J
      inside the hole;
  (C) grad_psi is the gradient of psi = cvx_true (central differences);
  (E) the relative-L2 reducer matches a hand computation.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.targets import QuadraticL1, NegL1, ConcaveQuad, Minplus
from src.exact_prox import (
    grad_psi, build_triples, g_exact, jbvs_exact, rel_l2,
)


def make_families(dim):
    mu1 = np.zeros(dim); mu1[0] = 1.0
    mu2 = np.ones(dim) / np.sqrt(dim)
    return {
        "quadratic_l1": (QuadraticL1(), True),
        "negl1": (NegL1(), False),          # nonconvex prior: g gives J_BVS != J
        "concave_quad": (ConcaveQuad(t=1.0), True),
        "minplus": (Minplus(mu1=mu1, mu2=mu2, sigma1=1.0, sigma2=1.0), True),
    }


def check_triples_and_maps(dim=3, n=4000):
    rng = np.random.default_rng(0)
    for name, (problem, convex) in make_families(dim).items():
        A = problem.train_halfwidth(4.0)
        x = rng.uniform(-A, A, (n, dim))
        y, g, xs = build_triples(problem, x)

        # (A) slope is exactly the sampled x, and g matches the independent
        #     conjugate value computed through preimage at y_k.
        assert np.array_equal(xs, x), f"{name}: slope must be the input x"
        fenchel = np.sum(x * y, axis=1) - problem.cvx_true(x)
        err_A1 = float(np.max(np.abs(g - fenchel)))
        gy = g_exact(problem, y)
        err_A2 = float(np.max(np.abs(g - gy)))
        assert err_A1 < 1e-10, f"{name}: Fenchel identity off by {err_A1:.2e}"
        assert err_A2 < 1e-8, f"{name}: g != g_exact(y) off by {err_A2:.2e}"

        # (B) convex families: g(y) = t*prior_true(y) + 0.5||y||^2.
        if convex:
            t = float(getattr(problem, "t", 1.0))
            closed = t * problem.prior_true(y) + 0.5 * np.sum(y * y, axis=1)
            err_B = float(np.max(np.abs(g - closed)))
            assert err_B < 1e-9, f"{name}: g != closed form off by {err_B:.2e}"
        else:
            # NegL1: recovered object is J_BVS, NOT the prior J. The two AGREE on
            # the samples y (|y_i| >= 1, outside the hole) but DIFFER inside the
            # hole (-1,1)^d, where the preimage collapses. Pin both facts.
            agree = float(np.mean(np.abs(jbvs_exact(problem, y) - problem.prior_true(y))))
            assert agree < 1e-9, f"{name}: J_BVS != J on samples, gap {agree:.2e}"
            zhole = rng.uniform(-0.9, 0.9, (n, dim))
            hole_gap = float(np.mean(np.abs(jbvs_exact(problem, zhole)
                                            - problem.prior_true(zhole))))
            assert hole_gap > 1e-3, f"{name}: expected J_BVS != J in hole, {hole_gap:.2e}"

        # (C) grad_psi IS the gradient of psi = cvx_true, by central differences
        #     at smooth points (kinks/ridges are measure zero, avoided by random
        #     x). This pins the forward map itself, independent of preimage.
        gp = grad_psi(problem, x)
        h = 1e-5
        fd = np.empty_like(x)
        for j in range(dim):
            ej = np.zeros(dim); ej[j] = h
            fd[:, j] = (problem.cvx_true(x + ej) - problem.cvx_true(x - ej)) / (2 * h)
        err_C = float(np.max(np.abs(gp - fd)))
        assert err_C < 1e-4, f"{name}: grad_psi != d(psi) off by {err_C:.2e}"

        print(f"  [{name}] A(fenchel {err_A1:.1e}, g_exact {err_A2:.1e}) "
              f"C(grad fd {err_C:.1e})  OK")


def check_reducer():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[1.5, 2.0], [3.0, 3.0]])
    diff = np.sqrt(0.5 ** 2 + 1.0 ** 2)
    denom = np.sqrt(1.5 ** 2 + 2.0 ** 2 + 3.0 ** 2 + 3.0 ** 2)
    assert abs(rel_l2(a, b) - diff / denom) < 1e-14, "rel_l2 mismatch"
    print(f"  [reducer] rel_l2 hand check OK ({rel_l2(a, b):.6f})")


if __name__ == "__main__":
    print("triples and forward map:")
    check_triples_and_maps()
    print("relative-L2 reducer:")
    check_reducer()
    print("\nALL PASS")
