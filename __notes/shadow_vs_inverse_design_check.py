"""The proximal shadow against the Esteve-Zuazua inverse-design projection.

Reference: Esteve and Zuazua, "The inverse problem for Hamilton-Jacobi
equations and semiconcave envelopes", SIAM J. Math. Anal. 52(6), 2020
(papers/2020_zuazua_inverseHJ.pdf). Their setting is the inviscid problem
with convex Hamiltonian; H(p) = |p|^2/2 is ours.

THE DICTIONARY (verified against their Theorems 2.1, 2.2 and 2.10;
derivation in note_experiments_summary.tex, Section 5, move 4):

  * target u_T = S(., t) = h/t, horizon T = t, h = t S the implicit
    regularizer's datum, psi = q - h, q = |.|^2/2.
  * Their reachability condition (Thm 2.2, eq. (6.11), quadratic H):
    D^2 u_T <= I/T, i.e. D^2 h <= I, i.e. PSI CONVEX -- exactly the
    proximal condition on the operator D = grad psi. The reachable set of
    the inviscid flow IS the proximal class.
  * Their projection (Thm 2.10): S_T^+ S_T^- u_T, the semiconcave envelope
    = smallest reachable target above u_T. In our variables: sup-convolution
    then Moreau envelope, which computes q - psi** (psi** = convex
    envelope). Operator level: grad psi**; conjugate side: grad psi*
    (biconjugation leaves the conjugate unchanged). Mu-free.
  * The proximal shadow: argmin over convex G of E_mu |grad G(D(z)) - z|^2;
    in one dimension the exact solution is isotonic regression of z on
    y = D(z) (weighted by mu through the sample).

WHAT THIS SCRIPT MEASURES (both checks print sup errors):

  1. Outside the fold region (where D is invertible), the fitted isotonic
     g equals grad psi* to sampling precision for BOTH operating measures:
     the shadow and the ES projection COINCIDE there, mu-independent.
  2. Inside the fold (the multivalued window, |y| < 0.102 here), they
     differ at order one and the shadow is mu-dependent: grad psi* jumps
     across the envelope's affine bridge; the isotonic fit mu-averages the
     branches (measured: sup gap 0.81 uniform, 1.06 skewed).

CONCLUSION. The shadow is not the inverse-design projection, but the two
agree exactly off the nonconvex fold and share the same fixed-point set
(the reachable = proximal class). Conjecture for the revision, not proved
here: the shadow solves a mu-weighted analogue of their obstacle problem.

Run: ~/miniforge3/envs/lpn_env/bin/python shadow_vs_inverse_design_check.py  (~30 s)
"""
import numpy as np
from scipy.optimize import isotonic_regression

A, B = 0.35, 2.0                       # psi'' = 1 - A B^2 cos(Bx) < 0 near 0


def psi(x):
    return x ** 2 / 2 + A * np.cos(B * x)


def D(x):
    return x - A * B * np.sin(B * x)   # psi', non-monotone on the fold


def main():
    xg = np.linspace(-5, 5, 400001)
    pv = psi(xg)
    yg = np.linspace(-2.0, 2.0, 801)
    gstar = np.array([xg[np.argmax(y0 * xg - pv)] for y0 in yg])  # grad psi*

    xc = np.arccos(1 / (A * B * B)) / B
    y_fold = abs(D(xc))
    print(f"fold: x_c = {xc:.3f}, 3-branch window |y| < {y_fold:.3f}")

    rng = np.random.default_rng(0)
    for tag, z in (("uniform mu", rng.uniform(-3, 3, 80000)),
                   ("skewed  mu", rng.normal(1.2, 0.9, 80000))):
        y = D(z)
        o = np.argsort(y)
        g = isotonic_regression(z[o]).x           # the 1-D shadow, exactly
        gi = np.interp(yg, y[o], g)
        d = np.abs(gi - gstar)
        out = d[np.abs(yg) > 3.5 * y_fold].max()
        ins = d[np.abs(yg) <= 3.5 * y_fold].max()
        print(f"{tag}: sup|g - grad psi*| off the fold {out:.4f} | on it {ins:.4f}")
        assert out < 1e-2, "shadow and ES projection must agree off the fold"


if __name__ == "__main__":
    main()
