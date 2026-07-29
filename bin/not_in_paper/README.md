# `bin/not_in_paper/` — experiments that appear NOWHERE in `main.pdf`

Everything in this directory is a real experiment with corrected mathematics,
but **none of it is reported in the paper**. Do not cite these numbers as if
they were. `main.pdf` §4 has exactly four subsections, and they map to the four
families in `bin/`:

| paper subsection | `bin/` config |
|---|---|
| Convex prior: the ℓ₁ norm | `quadratic_l1.py` |
| Non-convex: a min-plus algebra prior | `minplus.py` |
| Concave prior | `concave_quad.py` |
| Negative ℓ₁ norm | `negl1.py` |

## What is here

`maxplus_case3.py`, `maxplus_case4.py` — the max-plus / Hopf priors
J(x) = maxᵢ {⟨pᵢ,x⟩ − γᵢ}, ported from `legacy/old_notebooks/exp_4_1_3_minplus_8D.ipynb`
(γ = 0) and `legacy/old_notebooks/exp_4_1_4_minplus_8D.ipynb` (γᵢ = ½‖pᵢ‖²). Despite the
`minplus` in those filenames they are **max-plus**, unrelated to the paper's
min-plus mixture-of-quadratics.

## Two things to know before using them

**The notebook targets were wrong.** Both carried the wrong sign on the
`t·H(p)` term, giving `S(y,1) = maxᵢ{⟨pᵢ,y⟩ + ½‖pᵢ‖²}` (case 3) and
`maxᵢ⟨pᵢ,y⟩` (case 4). Both violate the Moreau bound `S ≤ J` at *every* test
point, so neither is a viscosity solution. `src.targets.MaxPlus` implements the
correct Hopf formula (a concave QP over the simplex), verified against a
brute-force grid Moreau envelope.

**The training scheme differs from D2.** The notebooks train with LPN's
*proximal matching* loss on a decreasing γ schedule (20 → 12.5 → 10), β = 10,
4–6 layers. These configs instead run the standard D2 pipeline (MSE regression
on ψ, β = 5, 2 layers), so they measure something different from what the
notebooks measured. That was a deliberate call; revisit if the proximal
matching scheme is ever the object of study.

## Why they may be worth keeping

The *vertex* restriction of the exact solution,
`maxᵢ{⟨pᵢ,y⟩ − γᵢ − ½t‖pᵢ‖²}`, is exactly the max-plus approximant `Γ_K` of
§3 — a lower bound on `S`, with a measurable gap (`MaxPlus.maxplus_approx`).
That makes these the natural substrate for **Phase 5**, the error-vs-K decay
experiment that referee R2.3 asked for.
