# Design — Experiments 1 and 2

Specification: `../../__tasks/task_pnp_registration.tex`, Section 6.1. This
file records the configuration, the decisions made during implementation, and
the observations of the first production run (2026-07-29). Single seed; the
run is an implementation test whose outcome is guaranteed by theory, so no
statistical claims rest on it.

## Configuration

- Prior: `p_φ = ½N(−2, ν) + ½N(2, ν)`, `ν = 0.05`; `ε = 1`, `t = σ²`;
  `σ ∈ {0.5, 0.25}`.
- Data per σ: `x_k ~ p_z` (N = 20 000 train, seed 1; 4 000 val, seed 2),
  `y_k = D(x_k)` exact.
- Networks: `LPN(in_dim=1, hidden=64, layers=2, beta=20)` for both fits
  (`hidden` must be explicit: the repo default of 256 is sized for d ≥ 2;
  `beta = 20` is a user decision, 2026-07-29, matching the value the TV
  experiment measured as best — the protocol default is 5).
- Training: `src.gradfit.train_grad`, batch 512, 20 000 steps, lr 1e-3 ×0.1 at
  50%/75%, best-validation checkpoint, `wclip` each step; both fits use the
  same loop and differ only in the target array (fit (a): `x_k`, so
  `G_θ ≈ ψ*`; fit (b): `x_k − y_k`, fitting the regularizer directly).
  The 20 000-step budget is a user decision (2026-07-29, "we'll increase it
  later"); it deviates from the repository protocol's 250 000 steps, as does
  the width (64 vs. the protocol's fixed 256), and both are recorded here as
  deviations rather than defaults.
- Exact reference: `f_reg = ψ* − ½y²` with `ψ*(y) = x*y − ψ(x*)` at
  `x* = D⁻¹(y)` (vectorized bisection; `D = ∇ψ` is strictly increasing).
  NOT by a bounded grid supremum — see decision 5.

## Decisions

1. **Metrics and figure are restricted to the sampled range of `D`** (the
   [0.5%, 99.5%] quantiles of the training `y`; about `[−2.2, 2.2]` at
   σ = 0.5). The regularizer is pinned on `range(D)` and merely bounded below
   off it (`thm:prior_optimality`), and with `x ~ p_z` the sampled range is
   compact: at σ = 0.5, reaching `y = 4` would require `x ≈ 14`, which `p_z`
   never produces. Scoring on a fixed `[−4, 4]` window measured extrapolation
   the theory does not promise, and read 68% where the pinned region reads
   8.9%.
2. **Curvature is measured on a decimated stencil** (spacing ≈ 0.02). The
   networks evaluate in float32, and a second difference amplifies value noise
   by `1/dx²`; at the native grid spacing (5e-4) the metric read quantization
   noise (−22 for a provably 1-semiconvex fit). The exact dip (width ~0.4) is
   fully resolved at 0.02.
3. **No standardization.** The 1-D inputs are well scaled; `Units` stays
   available in `src/gradfit.py` but is not used here.
4. **The finite-difference gate in `tests/test_readout.py` uses `eps = 1e-2`,
   tolerance 1e-3.** The helpers evaluate in float32, so the FD roundoff floor
   is ~1e-7/eps; the bug the gate exists to catch (a wrong linear term in the
   chain rule) would appear as an O(1) error.
5. **The conjugate must not be a bounded grid supremum** (bug found in audit,
   2026-07-29). `ψ*(y) = sup_x{xy − ψ(x)}` is attained at `x* = D⁻¹(y)`, which
   leaves any affordable grid quickly: `x* ≈ 26` for `y = 6` at `σ = 0.5`. A
   grid-truncated conjugate is affine beyond the truncation point, so the
   truncated `f_reg` carries curvature exactly −1 there — fabricating the very
   feature the experiment displays. The first version of `mixture.py` and of
   `__tasks/semiconvex_check.py` had this bug; it did not affect the
   experiment's metrics (whose window is interior) but corrupted the task
   document's Section 3.4/4 tables (min curvature −1.0000 → true −0.911 at
   σ = 0.5, −0.728 at σ = 1; information-loss 30.0%/77.1% → 20.8%/23.8%).
   Both are fixed (exact inverse via bisection) and gated by
   `tests/test_mixture.py` checks 5, 6, and 8.

## Observations (single seed; β = 20 run of 2026-07-29)

- σ = 0.5: rel. L2 (a) 1.2% / (b) 101%; held-out prox residual (a) 0.2% /
  (b) 100%; min curvature (a) −0.97 / (b) −0.00, exact −0.91.
- σ = 0.25: rel. L2 (a) 47% / (b) 101%; residual (a) 0.2% / (b) 83%.
- **β moves the two regimes in opposite directions** (comparison against the
  earlier β = 5 run, same everything else). Where the data covers the window
  (σ = 0.5), sharpening the activation improved the fit sevenfold (8.9% →
  1.2%). Where the operating distribution leaves a gap (σ = 0.25), it made
  the recovery *worse* (24% → 47%): in the unsampled stretch the network's
  shape is pure inductive bias, and a sharper Softplus permits a more
  arbitrary placement of the gradient transition across the gap, so the
  integrated value offset between the two data islands grows. The residual is
  blind to the difference (0.2–0.3% in all four cases) because it lives on
  the data.
- **The class carries the region the data cannot.** Even in the unsampled
  gap, fit (a) bends downward with curvature near −1, because the readout's
  `−½y²` supplies exactly that wherever the convex `G_θ` is near-affine —
  which is the true behavior of `f_reg` there. The failure mode is a value
  offset and an asymmetry, never a wrong sign of curvature.
- The convex control's certificate reads 83–100%: the failure is visible
  without ground truth, which is what makes the certificate usable when no
  exact reference exists.

## Observations — uniform-coverage variant (2026-07-29)

Run with `--sampling uniform`: `x_k` uniform on `[−A, A]`,
`A = D⁻¹(QUERY_A)`, `QUERY_A = 2.5` — the manuscript's training-box
construction, so `y = D(x)` covers the query window by design. Same N, steps,
network.

- Recovery becomes essentially exact at BOTH noise levels: rel. L2 (a)
  0.1% (σ = 0.5) and 0.04% (σ = 0.25), residuals ≤ 0.1%. The 47% failure of
  the p_z run at σ = 0.25 was therefore entirely a coverage effect, as the
  budget diagnostic already indicated (5× steps moved it 47% → 44%).
- **The median residual can mask a localized failure.** Under uniform
  sampling at σ = 0.5, the convex control's median residual is 2.2% while its
  value error is 108%: the region where the target gradient decreases (the
  bump, where a convex fit must fail) holds fewer than half the evaluation
  points, and the median sits in the well-fit majority. Report a high
  quantile (or the spatially resolved residual) alongside the median wherever
  the certificate is the only score.
- Figure: `results/figs/experiment1_mixture_uniform.{png,pdf}`; metrics:
  `results/metrics_uniform.json`. The two samplings answer different
  questions — p_z: what the denoiser's operating distribution supports;
  uniform: recovery of `f_reg` as an object — and the pair displays the
  coverage effect explicitly.

## Decisions — Experiment 2 unblocked (2026-07-30)

The two reading questions of the task document's Step 1c, answered as dated
decisions:

1. **PIRATE's weights ARE released** — official repo `wustl-cig/PIRATE-code`
   (MIT), shallow-cloned verbatim at `../ext/pirate/` (commit `4901b71`,
   2024-03-19; provenance in `../ext/PROVENANCE.md`). But at ONE configuration
   `sigma = 1`, not the ten this project assumed, plus the PIRATE+ network.
   The PIRATE+ checkpoint holds the SAME 12 DnCNN tensors under a
   `PIRATE.dnn.` key prefix — an exactly matched differential comparison.
   Route A is viable for the Jacobian probe; Route B stays primary for
   recovery (the release ships one example field, no training corpus).
2. **Scale arithmetic**: their code adds noise `sigma**2 * randn`, so the
   noise STANDARD DEVIATION is sigma squared — `sigma_eff = 1` at their
   `sigma = 1` — against a field of std 1.88 (release field, shape
   (1, 3, 80, 96, 112), ~2.6e6 components). Hence `t = sigma_eff^2 = 1` at
   `eps = 1`, curvature floor `-1/t = -1` in field units, noise at ~53% of
   the field scale: the information-loss statement is a headline, not a
   footnote.

Facts for the probe implementation, verified against their code: the DnCNN
predicts the NOISE, so `D(z) = z - dnn(z)` and the probe target is
`grad D = I - grad dnn`; 6 convolutions, 453,059 parameters, ELU (smooth),
no batch norm. Both checkpoints load in `lpn_env` (`h5py` installed
2026-07-30). Measured: forward pass 5.0 s CPU / 1.3 s MPS; one training step
of their script 29.7 s CPU / 4.5 s MPS (so a sigma-ladder retrain, if ever
wanted, is ~30 min per sigma on MPS and must be labeled as OUR retraining).

**Open choice (pending):** the tv_pm posterior-mean calibration row cannot
use the Gibbs sampler (not autograd-able). Candidates: (a) the exact n=2
quadrature denoiser (`tvpm/quadrature.py`), Jacobian by high-accuracy finite
differences, all three conditions hold by theorem; (b) the trained tv_pm
convex-ICNN prox via implicit differentiation,
`grad u_hat = (I + hess J_theta(u_hat))^{-1}`, all three by architecture.

## Design — Experiment 2, the Jacobian probe (2026-07-30)

Specification: task document Section 6.2 (revised 2026-07-30) and Section
5.2 for the estimators. Modules: `pnpreg/probe.py` (estimators),
`pnpreg/probe_targets.py` (operators), `pnpreg/probe_run.py` (driver).

### Configuration (production)

- Probes: 16 per point for ρ (calibration rows: max(16, 64) — the operators
  are tiny, extra probes are free and make the F2 z-score gate reliable);
  bootstrap B = 2000 for the SE of ρ.
- Lanczos: k = 30, 3 starts, full reorthogonalization (classical
  Gram–Schmidt twice), basis on CPU, every inner product in float64,
  tridiagonal by LAPACK; Ritz residuals reported with every estimate.
- Test points: 8 per PIRATE net, `z_i = field + N(0, I)` with CPU generators
  seeded `14 + i` (their operating distribution: `train_denoiser.py` adds
  noise of std `sigma**2 = 1`). Calibration: 8 mixture points (d = 64,
  σ = 0.5, seeds 11+i, point 0 has a coordinate pinned at z = 0), 16
  quadrature points (d = 2, σ = t = 20/256, seeds from 12), 8 ICNN-prox
  points (noisy cameraman patches, σ = 20/256, seed 13), float32 repeat 4
  points, floor 2 points.
- Devices: calibration float64 CPU; PIRATE rows float32 (MPS production,
  CPU cross-check). Probes and test inputs are CPU-seeded → bitwise
  identical across devices.

### Decisions (numbered, dated 2026-07-30)

1. **Three calibration rows, not one.** The pending choice recorded above
   (quadrature vs ICNN prox) was resolved by the user as BOTH, plus the
   coordinatewise mixture row — the mixture is the only row with a designed
   condition-3 VIOLATION, so the calibration covers both a pass and a fail
   of each condition.
2. **The tv_pm posterior-mean sampler is probed through surrogates.**
   MCMC is not autograd-able; the quadrature row (exact, n = 2, all
   conditions by theorem) and the ICNN-prox row (n = 64, all conditions by
   architecture) stand in, as decided 2026-07-30.
3. **ICNN row: float64 primary, float32 sensitivity.** The checkpoint
   (`tv_pm_64D_fc_m8000_w256_L2_b20.0_s250000_sig0.07812_t0.07812`) is cast;
   the prox is re-solved in float64 (L-BFGS, tolerance_grad 1e-12, residual
   gated < 1e-7); ∇D = (I+H)^{-1} with H the dense autograd Hessian (exact
   reference by eigendecomposition); the probed operator solves (I+H)w = v
   by CG with Hessian-vector products in the jvp slot and by dense Cholesky
   in the vjp slot — two distinct arithmetic paths for one
   symmetric matrix, so ρ's b_j reads their discrepancy (~1e-13 measured).
   The float32 repeat (gates 1e-3) is the precision bridge to the PIRATE
   rows, cf. the float32 rationale in `tests/test_readout.py`.
4. **The noise floor is a symmetric-by-construction surrogate plus
   identity_max — a two-path autodiff floor is IMPOSSIBLE here (measured).**
   Forward-mode jvp, reverse-over-reverse, and reverse-over-forward
   compositions of the same product agree BITWISE on CPU and MPS, float32
   and float64: autodiff transposes conv kernels exactly, so no composition
   yields an arithmetically independent second path. (The plan-stage
   observation that forward-over-reverse HVPs fail on MPS — an ELU
   double-backward forward-mode gap — is moot for the shipped design, which
   needs first-order products only.) The floor row is therefore
   `g(z) = z − Cᵀ elu(Cz + b)` with (C, b) the released first conv layer:
   Jacobian `I − Cᵀ diag(elu′) C`, exactly symmetric, same kernels, exact
   ρ = 0 — an end-to-end zero-test (measured 0.0 on CPU and MPS). The
   resolvable-asymmetry floor reported with the table is
   max(floor-row ρ, identity_max), where identity_max is the largest
   relative violation of ⟨v_i, J v_j⟩ = ⟨Jᵀ v_i, v_j⟩ over probe pairs (an
   identity for EVERY matrix, so it reads pure jvp-vs-vjp arithmetic noise;
   an error of that relative size in Jv − Jᵀv shifts ρ by about half of it).
   Kill-criterion rule: a PIRATE ρ prints as "≤ floor" unless it exceeds
   floor + 2 SE.
5. **Gates split deterministic vs statistical.** Deterministic quantities
   (ρ of an exactly symmetric operator, Lanczos vs dense eigenvalues,
   FD h-vs-2h) get tight absolute gates (1e-6..1e-12 in float64, 1e-3 in
   float32, always ≥ the Ritz residual bound); Monte Carlo quantities (F2)
   get z-score gates (< 4). A 1e-6 gate on a 16-probe Frobenius estimate
   would be statistically illiterate.
6. **Lanczos aggregation over starts**: min over starts for λ_min, max for
   λ_max (Lanczos converges to extremes from inside the spectrum), with the
   cross-start spread and per-start Ritz residuals in the JSON. The floor
   row's eigenvalue columns are dashes in the table: its spectrum is the
   surrogate's own, only its ρ is meaningful.
7. **Budget arithmetic** (measured on this machine, full field, MPS): jvp
   3.6 s, vjp 1.3 s after the one-time linearization (~1.5 GB retained
   graph, one point at a time, freed after use), S-matvec 4.9 s. Per point:
   ρ ≈ 16 × 4.9 ≈ 78 s, Lanczos ≈ 30 × 3 × 4.9 ≈ 440 s; 8 points × 2 nets
   ≈ 2.4 h. Trade rule if slower than modeled: drop starts 3 → 2, then
   points 8 → 6; never reduce k.
8. **In-run sanity for each PIRATE net**: one-probe jvp vs central-difference
   directional derivative (rel < 2e-2 in float32 — catches a broken
   torch.func composition, not float noise; the float64 crop version of the
   same check gates at 1e-5 in `tests/test_probe_targets.py`).

### Observations — production run (2026-07-30, MPS, probes 16, k 30, starts 2)

Gates: 191/191 passed. Asymmetry floor 1.1e-8 (the floor row read exactly 0
on both points; the floor is the largest jvp-vs-vjp bilinear inconsistency).
Full numbers: `results/probe_metrics.json`; figure:
`results/figs/experiment2_probe.{png,pdf}`.

- **PIRATE (the released AWGN denoiser, their σ = 1) fails all three
  conditions, decisively.** ρ = 0.4440–0.4443 across the 8 test points
  (SE ≤ 1.1e-4) — seven orders of magnitude above the floor; λ_min(S) ∈
  [−0.719, −0.697]; λ_max(S) ∈ [1.139, 1.144]; conditions 2 and 3 violated
  at 100% of test points, sizes 0.72 and 0.14 against eigenvalue
  tolerances ~2e-2 (Ritz) — the conclusions are far outside every error
  bar. The point-to-point stability (ρ varies by 3e-4) says this is a
  property of the network on its operating distribution, not of any
  particular input. Reading per the kill criteria: **the trained DnCNN is
  the gradient of nothing — no regularizer with prox = D exists, and the
  objective PIRATE's paper writes down does not exist as written for its
  own denoiser.** A Tier-2 fit to this D returns a proximal surrogate, and
  the held-out residual then measures exactly the distance of D from the
  proximal class.
- **PIRATE+ (DEQ fine-tuned, same 12 tensors) moves dramatically TOWARD
  the proximal class — the reverse of the differential hypothesis.**
  ρ = 0.01555–0.01556 (28× smaller than PIRATE, though still six orders
  above the floor, so strictly nonzero); λ_min(S) = 0.9456 ≥ 0 — condition
  2 PASSES at every point; λ_max(S) = 1.0532 — condition 3 fails, but by
  5% instead of 14%. The task document asked whether task fine-tuning
  moves the denoiser OFF the proximal set; the measurement answers that it
  moved it 28× closer to symmetric and into the PSD cone.
- **PIRATE+'s Jacobian is a ±5% perturbation of the identity**
  (spec(S) ⊂ [0.946, 1.053] at every point): the fine-tuned denoiser is a
  near-identity map on the operating distribution. Consistent with the
  role the DEQ training gives it — inside the iteration the residual
  enters through τ = 1e-7, so a well-adapted denoiser needs only a small,
  well-conditioned correction. This near-identity structure, not any
  explicit regularization, is presumably WHY it is so much closer to
  symmetric: the dnn-part of D = I − dnn is small.
- Run mechanics for the record: two earlier production attempts were
  jetsam-killed on this 16 GB machine — the MPS caching allocator grows
  ~2 GB per point unless `torch.mps.empty_cache()` is called between
  points (fix in `probe_run.run_point`); per-point resume files
  (`results/.probe_partial_*.json`, kept permanently) made the kills cost
  one in-flight point each. Per-point cost at starts = 2: 490–530 s
  (the starts 3 → 2 trade was invoked per decision 7; cross-start spreads
  in the JSON are ≤ 2e-3, so the third start was buying nothing). Total
  production wall clock ≈ 2.5 h across the three attempts.
- FD sanity: jvp vs central difference 3.2e-3 (PIRATE), 1.2e-5 (PIRATE+),
  both within the float32 gate; the float64 crop-level check in
  `tests/test_probe_targets.py` is at 4.4e-7.
- **CPU cross-check** (`probe_metrics_cpu_check.json`; one point per net,
  probes 4, k 10, starts 1, bitwise-identical inputs and probes): ρ agrees
  with MPS to 1.0e-4 (PIRATE: 0.4442 vs 0.4443) and 1e-5 (PIRATE+:
  0.01555 both). The reduced-k eigenvalue estimates (−0.612/1.083 and
  0.951/1.049) sit INSIDE the production values, as Lanczos guarantees
  (it converges to the extremes from inside), with Ritz residuals 7e-2 /
  5e-3 covering the gaps. Device effects are far below every conclusion.

## Design — Experiment 3.1, the affinity test and the closed-form quadratic (2026-07-30)

Motivation: Experiment 2 measured a PIRATE+ spectrum constant to ~1e-4
across test points. If the Jacobian itself is constant, the denoiser is
affine on the operating distribution and its implicit regularizer is a
QUADRATIC, computable in closed form with no training — the cheap precursor
that decides whether the full nonparametric recovery (task doc / note §6)
is needed. Modules: `pnpreg/affine.py`, `pnpreg/affine_run.py`,
`tests/test_affine.py`; Lanczos gained `return_ritz` (extreme Ritz vectors,
gated in `tests/test_probe.py` checks 8).

### Configuration (production)

- Affinity: 8 operating-distribution points (SAME seeds 14+i as Exp 2), 12
  pairs, err = ‖D(z_i)−D(z_j)−J(z̄)(z_i−z_j)‖/‖D(z_i)−D(z_j)‖ with one jvp
  at each midpoint; Jacobian-variation check on 2 pairs; float64 CPU repeat
  (4 points, 3 pairs). Constant term b from the SINGLE reference point
  (decision 3 below); cross-point b_i spread kept as a diagnostic.
- Self-check at a HELD-OUT input (seed 114): grad f_reg(y) = S^{-1}(y−b)−y
  must equal z−y at y = D(z); S^{-1} by CG on the matrix-free symmetric
  part (well-conditioned for PIRATE+, spec ⊂ [0.945, 1.054]; for the AWGN
  net S is indefinite, the closed form does not exist, and its self-check
  is reported only as the expected failure).
- Ritz vectors: one k=30 Lanczos per net with `return_ritz=True`; figure
  shows the favored pattern (λ_S > 1, negative f_reg curvature), the
  penalized pattern (λ_S < 1, positive curvature), and the b slice.

### Decisions (dated 2026-07-30)

1. **The dimensionless asymmetry that matters here is ρ of the RESIDUAL
   map, not of the denoiser.** The regularizer's gradient is R(z) = z−D(z),
   so the scale-free question "how far is the update from a gradient
   field?" is ρ(I−A) = ‖K‖_F/(2‖I−A‖_F). For a near-identity denoiser the
   identity part dominates ‖A‖ and makes ρ(A) misleadingly small: the
   smoke run measured ρ(A) = 0.012 against ρ_res ≈ 0.70 for PIRATE+ on the
   crop (norm ratio ‖I−A‖_F/‖A‖_F = 0.017). Exp 2's conditions 2 and 3
   (eigenvalue bounds on A) are unaffected — they are absolute statements —
   but Exp 2's "28× closer to symmetric" compares full-Jacobian ρ's and
   must be read with this in mind. `affine.residual_asymmetry` measures
   both; the self-check prediction uses ρ_res.
2. The self-check verdict rule: PIRATE+ passes if resid ≤ 3(affinity err
   median + ρ_res); the AWGN row is expected to fail (no closed form) and
   its verdict only asserts resid ≥ ρ_res/3. A False verdict blocks the
   write-up (affine_run exits 1).
3. **b from one reference point, not a cross-point average** (smoke-run
   fix): b_i = D(z_i) − J(z_ref)z_i mixes Jacobian variation along the
   LARGE field vector into the estimate; the average contaminated b at the
   18–30% level on the crop while pair-affinity read 1e-3. The spread of
   the b_i remains reported as exactly that diagnostic.
4. Affinity is claimed on the operating distribution only; `--scale-sweep`
   (off by default) probes s·field + noise at s ∈ {0.5, 1.5}.

### Observations — production run (2026-07-30, CPU, full field; `results/affine_metrics.json`)

Two earlier MPS attempts were abandoned for memory (the caching allocator
took the process to ~15 GB and, after a partial fix, a full-field float64
CPU stage took a rerun to 13 GB); per user instruction the code is CPU-only
and the float64 precision repeat runs on a crop. Production wall clock
78 min; peak transients ~2.5 GB.

- **PIRATE+ IS affine on the operating distribution: pair error 0.20%**
  (median 2.044e-3, max 2.048e-3 over 12 pairs; float64 crop repeat
  1.1e-3, so float32 arithmetic is not the limit), Jacobian variation
  0.8%. The closed-form quadratic account PASSES its falsifiable
  prediction: prox-identity residual at the held-out input 0.597 vs
  predicted 0.704 (= affinity error + rho_res) — the symmetric quadratic
  explains the residual map up to exactly the irreducible skew.
  Curvatures of f_reg: −0.0503 (favored direction) to +0.0574 (penalized),
  from spec(S) = [0.9457, 1.0530]. CG on S: 5 iterations.
- **The skew ceiling, quantified: rho_res = 0.702** (full field, vs
  rho_full = 0.0156; norm ratio ‖I−A‖_F/‖A‖_F = 0.0222). The residual map
  z − D(z) — the object that plays the regularizer's gradient — is 70%
  non-gradient in its own units, even though the map is affine to 0.2%.
  NO recovery in ANY function class can do better than this ceiling; the
  quadratic already attains it (0.597 ≤ 0.704). Exp 2's "28x closer to
  symmetric" (rho_full) is correct but must be read alongside rho_res:
  the near-identity part dilutes the full-Jacobian ratio.
- **The AWGN denoiser is NOT affine: pair error 0.50, Jacobian varying
  93% between operating points** (float64 crop repeat 0.27). Cross-point
  spectrum stability (Exp 2, ~2e-3) therefore does NOT imply a constant
  Jacobian. Its rho_res = 0.098 — in residual-relative units the AWGN
  update direction is only ~10% non-gradient (its residual map is 4.5x
  LARGER than the denoiser map, norm ratio 4.52), inverting the
  full-Jacobian ordering; but S indefinite (λ ∈ [−0.69, 1.14]) means no
  proximal reading exists regardless. Its self-check ran to the 30-iter
  CG cap (S indefinite; the expected failure, resid 26).
- **Figure** (`figs/experiment31_affine.{png,pdf}`): PIRATE+'s extreme
  Ritz patterns are coherent localized deformation structures and its
  constant term b̂ has smooth spatial organization (‖b̂‖ = 38 on a field
  of norm ~4e2, cross-point spread 0.42 — consistent with the 0.8%
  Jacobian variation amplified by ‖z‖); the AWGN counterparts are
  noise-like.
- **Consequences for Experiment 3** (decided by this run): the
  nonparametric fit is SUPERSEDED for PIRATE+ — its training data would
  contain no non-quadratic signal above 0.2%, and the recovery ceiling
  (rho_res) is already attained by the closed form. What survives: the
  plug-back test collapses to a linear solve (prox of a quadratic), and
  the nonparametric pipeline remains motivated only for the AWGN
  surrogate-distance question and for Route B. In the HJ reading, an
  affine denoiser means a GAUSSIAN implicit prior, and quadratics are
  closed under the viscous HJ semigroup: the entire σ-family of
  denoisers/regularizers follows from the one measured S by matrix
  algebra — the σ-ladder the release lacks exists analytically for
  PIRATE+.

## Design — Experiment 3.2, the bimodal field prior (2026-07-31)

The recovery experiment on our own exactly solvable model: the network
solves the BACKWARD Hamilton-Jacobi problem — its readout estimates
f_reg = t·J_BVS from forward-flow samples (denoiser evaluations) — and the
closed form is the reference. Modules: `pnpreg/bimodal.py` (exact model),
`pnpreg/bimodal_run.py` (driver), `tests/test_bimodal.py` (12 gates).

### Configuration (production)

- Model: n = 64 (8×8 patches), V = [u, V⊥] with u a centered Gaussian bump
  (width 1.5 px) and V⊥ from QR against the 2-D DCT basis. Prior: s = ⟨u,y⟩
  from the Experiment-1 mixture (±2, ν = 0.05, NONCONVEX direction);
  w_k ~ N(0, λ_k), λ_k = 4/k² (63 convex directions, J_BVS = J there).
  σ = 0.5 (t = 0.25) by default; `--sigmas` accepts more.
- Data: N = 40 000 train (seed 1), 8 000 val (seed 2), 4 000 held-out eval
  (seed 3); pairs (y = D(z), z) from the exact denoiser.
- Fits: LPN(in_dim=64, hidden=256, layers=2, beta=20), batch 512, 50 000
  steps via `src.gradfit.train_grad`. Fit (a): target ∇G(y_k) = z_k,
  readout J_θ = G_θ − q. Fit (b), convex control: plain ICNN, target
  ∇J_b(y_k) = z_k − y_k.

### Decisions (dated 2026-07-31)

1. **Annealing/semigroup testing DROPPED** (user decision): not needed for
   the backward-solution claim, and it carried most of the machinery (grid
   flows, two numerical conjugates, separability accounting). The
   experiment is recovery + control only.
2. **Economy budget**: 50 000 steps and beta = 20 (as Experiment 1), a
   recorded deviation from the 250k-step protocol. Measured cost 7.3
   ms/step → ~6 min per fit on CPU.
3. **Standardization ON** (`Units`): the mixture scale (±2) and the high
   perp modes (std √λ_63 ≈ 0.03) differ by two orders; targets are scaled
   per-coordinate.
4. **Scoring**: values compared after removing each side's mean (gradient
   training does not identify the additive constant); relative L2 of the
   CENTERED values on held-out y = D(z) (operating support — the
   Experiment-1 lesson). Curvature on a decimated stencil (0.05).
5. **Spectrum readout**: the fitted Hessian at held-out points, rotated to
   V-coordinates; diag[1:] vs the exact t/λ_k; the off-diagonal maximum is
   the fit's separability diagnostic; H_V[0,0] is the fitted u-curvature.
6. CPU only; peak memory well under 1 GB; checkpoints per fit in
   `results/ckpt/` (gitignored, regenerable in minutes).

### Observations — production run (2026-07-31, CPU; `results/bimodal_metrics.json`)

- **Fit (a), the semiconvex readout, recovers the exact backward solution.**
  Centered-value relative L2 3.20% on held-out operating points; u-slice
  minimum curvature −0.883 against the exact −0.911 (floor −1): the
  nonconvex dip between the modes is reproduced in shape and depth. The
  perp spectrum — 63 quadratic coefficients spanning four orders of
  magnitude, t/λ_k from 0.0625 to ~248 — is recovered to 0.40% relative
  L2. Residuals: median 2.5%, q95 4.1%.
- **Fit (b), the convex control, fails exactly where it must.** Overall
  relative L2 15.4%, and the failure is localized: its u-slice minimum
  curvature is 0.000 — a convex function bridges the dip with a flat
  segment — while its perp spectrum matches at 0.76%. The residual
  signature repeats the Experiment-1 lesson: median 4.5% but q95 16.5% —
  the failing region holds a minority of evaluation points, so the median
  understates it and the high quantile carries the information.
- The class separation is therefore 5× in overall value error and
  categorical in curvature (−0.883 vs 0.000 against an exact −0.911):
  the backward viscosity solution of a nonconvex prior is representable
  in the semiconvex class and not in the convex class, measured on the
  same data with the same architecture and budget.
- **The spectrum readout took three instruments to get right (2026-07-31;
  supersedes the first-run "k = 2, 3 dip" note and its 0.40% Hessian
  aggregate).** (i) The pointwise Hessian, averaged over 12 operating
  points, is fine in aggregate but its SOFT-mode entries are O(0.1)
  spatial-wiggle noise — at k = 1 it returned −0.009 for an exact 0.0625,
  and a log-axis plot drops the negative point, which is what made the
  first figure's low-k end unreadable. (ii) Value slices fail at the
  STIFF modes: over the operating range (std g_k√(λ_k+σ²), ~30× narrower
  than the prior's for high k) the quadratic value signal is ~2e-6,
  below float32 value noise; and slices through the origin read the
  network at s = 0, the unsampled inter-mode gap, where the fit is
  inductive bias (origin-anchored readouts gave fit (a) 5.6% — an
  off-support number, the Experiment-1 coverage lesson again).
  (iii) The instrument that works for every mode: the slope of the
  GRADIENT slice ⟨∇J_θ(y₀ + w v_k), v_k⟩ over the operating range,
  anchored at held-out operating points and averaged — signal =
  curvature × range ≈ 0.2–1.0 at all 63 modes, and the gradient is what
  training supervised. Final numbers: fit (a) 0.32% aggregate, 3.1%
  worst mode; fit (b) 1.13% / 4.2%. Hessian off-diagonal maxima: 0.77
  (a), 1.03 (b) against diagonal entries up to ~248 — the fits are close
  to separable, as the truth is.
- Rescoring is reproducible without retraining:
  `python -m pnpreg.bimodal_run --rescore` recomputes all metrics and the
  figure from the saved checkpoints in seconds.
- Wall clock: fit (a) 2479 s, fit (b) 461 s at the same 50 000 steps —
  the smoke-run rate (7.3 ms/step) matched fit (b); the fit (a) run
  shared the machine with other load, and the discrepancy is timing
  noise, not a model property. Peak memory well under 1 GB.
