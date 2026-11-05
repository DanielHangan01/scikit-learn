# Experiments J + K — combined results: when the fast mode works, at what scale, and what it buys

> **⚠ Written before Experiment K-optimized.** The *quality* results here
> (regimes, constant-k, strain correlations, mutually-exclusive levers) stand.
> The **speed** sections do not: every "deployable speedup" below is a
> model-based estimate derived from an `a + b·k` decomposition, and §8's
> epistemic-status discussion exists precisely because they could not be
> measured at the time. They have since been **measured directly** — and the
> "no win at near-cycle quality for cifar10_raw" claim is **overturned**. For
> anything speed-related use
> [EXPERIMENT_K_OPTIMIZED_RESULTS.md](EXPERIMENT_K_OPTIMIZED_RESULTS.md).

Combined conclusions of **Experiment J** (all 18 benchmark datasets,
N=400–5750, mechanism) and **Experiment K** (8 large sources, N up to 10,000,
wall-time). Design: [EXPERIMENT_J_K_SPEC.md](EXPERIMENT_J_K_SPEC.md).
Per-experiment detail: [EXPERIMENT_J_RESULTS.md](EXPERIMENT_J_RESULTS.md) ·
[EXPERIMENT_K_RESULTS.md](EXPERIMENT_K_RESULTS.md).

Evidence base: **26 datasets** (18 + 8, no overlap in tested instances),
**~6,100 SGDMDS fits**, every reported number a median of 5 paired seeds.
Two consistency ties between the sweeps: `coil20` re-run under K's runner
reproduced J's overheads exactly (1.079/1.035/1.018 at k=50/100/200), and
`fashion_mnist_full` (strain 0.369) matches J's 3000-point `fashion_mnist`
(0.378) — same regime, same behaviour, 3.3× the N.

---

## 1. The one-paragraph summary

Cutting the per-epoch budget from `N(N−1)/2` to `k·N` random pair updates
works — **when it works — with a fixed, N-independent k**, and which datasets
it works on is predicted almost perfectly by a cheap spectral measure of
2D-embeddability (`strain_top2`). Three regimes: **responsive** (strain ≳ 0.22:
k=200 lands within ~1–2% of full-cycle stress at every N tested, up to 10,000),
**hard-middle** (strain ~0.05–0.21: overhead plateaus +3–10% above cycle and
more budget barely helps), **degenerate** (strain < 0.05: matches cycle
trivially but a 2D embedding captures <5% of the structure anyway). The
wall-time payoff is governed by a *different* variable — the feature dimension
D — because the fast mode recomputes distances at O(D) per update while cycle
looks them up at O(1). **Strain sets how much budget you need; D sets what
each unit of budget costs.** The two multiply into the deliverable speedup,
which ranges from ~200× (low-D responsive) to ~1× (very-high-D at near-cycle
quality). Exact update-count reductions are certain arithmetic; wall-time
projections are estimates (§8).

---

## 2. The regime taxonomy — 26 datasets, one predictor

`strain_top2` (fraction of classical-MDS eigen-energy in the top 2 dims,
measured on an 800-point subsample — independent of the sweep) orders the
behaviour across **both** collections:

| Regime | strain_top2 | J datasets | K datasets | behaviour |
|---|---|---|---|---|
| **Responsive** | ≳ 0.22 | svhn, har, cifar10, fashion_mnist, seismic, coil20, fmd | feynman, california_housing, patchcamelyon, fashion_mnist_full, cifar10_raw | fixed k reaches ~+1–2% of cycle; budget is a powerful lever |
| **Hard-middle** | ~0.05–0.21 | spambase, bank, epileptic, secom, hiva, hatespeech | ag_news, tiny_imagenet, mnist_digits | plateaus +3–10% above cycle; budget has little leverage |
| **Degenerate** | < 0.05 | cnae9, sms, sentiment, imdb | — | overhead <+1% at any budget, but 2D captures <5% of variance → the embedding is poor regardless |

Quantitative strength of the link, measured two independent ways:

- **J (18 datasets, steepness `s` of the overhead-vs-k curve):**
  `Spearman(strain, s) = +0.75` (p<0.001); `pca_PR → s`: −0.74. Plus the
  N-controlled check at a common rung: `strain → overhead@k=25, N=600`:
  ρ=+0.72.
- **K (8 datasets, budget-to-target `k*`):**
  `Spearman(strain, k*(≤+5%)) = −1.00` (p<0.001) — a **flawless monotone
  ordering** at N=10,000 across four unrelated domains. The k* values:
  feynman 50 · california 84 · cifar10_raw 94 · fashion_full 98 ·
  patchcamelyon 107 · mnist_digits 239 · ag_news 352 · tiny_imagenet 380.
  Hard-middle needs ~7× the budget of the most responsive source, and the two
  lowest-strain sources **never reach +2%** at any tested k (≤400).

**Methodological refinement (K):** steepness `s` alone is ambiguous — a flat
curve means either "already at the floor at the smallest k" (feynman, s=0.31,
floor +1.0%) or "stuck above cycle" (ag_news, s=0.40, floor +4.3%). J's
classifier used `s`+`floor` jointly, which was necessary; **`k*` is the better
single metric** and is what any future auto-gating heuristic should use.

Boundary case worth remembering: `mnist_digits` (digit MNIST, strain 0.115) is
nominally hard-middle but its floor (+1.7% at k=400) sits close to responsive —
digit MNIST is genuinely more 2D-embeddable than text/image *embeddings* but
less than Fashion-MNIST. The regime boundary (~0.2) is a band, not a wall.

---

## 3. Constant-k: the budget is (near-)independent of N — and N-invariance is universal

**On responsive data, a fixed k holds quality as N grows.** Convergent
evidence:

- **J, reliable-p subset** (steep curve + ≥5 rungs): cifar10 p=0.16, coil20
  p=0.11, fashion p=0.06 — median **p ≈ 0.11** (`k* ∝ N^p`; a 10× larger
  dataset needs ~1.3× the k).
- **K, all five responsive sources at N=2000→10000**: k=200 overhead flat at
  ~+1–2% at every rung; fitted p = 0.00 (feynman) to 0.19 (patchcamelyon).
  Per-rung example (fashion_mnist_full, k=200): +1.2%, +1.0%, +1.2%, +1.1%.
- **J's collapse test**: pooled excess collapses on **absolute k** (updates
  per point), not on the coverage fraction `2k/N`, in 14/18 datasets — the
  mechanistic reason a fixed k can survive growing N.

**K's sharpening of the regime story:** N-invariance at fixed k is
**universal** — hard-middle datasets are *also* flat in N (ag_news k=200:
+7.7→+9.3% with no systematic drift). What the regime controls is the
**achievable floor**, not the N-scaling:

| | floor at max tested k | 
|---|---|
| responsive (K, k=400) | **+0.3% … +1.7%** |
| hard-middle (K, k=400) | **+4.3% … +4.8%** (mnist_digits +1.7%) |
| hard-middle (J, k=200, full N) | +3.3% … +9.5% (worst: epileptic +9.5%, hiva +8.8%) |

Practical operating points on responsive data: **k=200 for ~+1%** (safe
choice), **k=100 for ~+4–6%**, k=50 is noisy at large N (+10–22%, wide seed
spread) — the reliable knob starts at k=100. k=400 adds little over k=200
(+0.3–1.0% floors) at twice the cost.

---

## 4. The two speed levers are mutually exclusive (J)

The iteration penalty `stress(max_iter=15)/stress(30)` tracks 2D-embeddability
almost perfectly: `Spearman(strain, iter_penalty_peak) = +0.95` (still +0.92
without small-N outliers); `s → iter_penalty`: +0.83. Full-cycle's own 15-vs-30
ratio is ~1.00 everywhere — cycle is converged by epoch 15.

Meaning: **the datasets where the budget knob has leverage are exactly the
ones where cutting iterations hurts** (cifar10 peak 1.48×, fashion 1.32×), and
the datasets where `max_iter=15` is free (hiva, imdb ≈1.01×) are the ones where
the budget knob buys nothing. **Never advertise "cut budget AND cut
iterations."** On responsive data keep `max_iter=30`.

*(Scope: J only — K ran a single `max_iter=30` arm, so the lever interaction is
untested above N=5750. No reason to expect it to flip, but it is unverified.)*

---

## 5. Speed: D sets what each unit of budget costs (K)

The **exact arithmetic** (certain): at N=10,000, cycle performs
`N(N−1)/2 = 49,995,000` pair updates per epoch. The fast mode performs `k·N`:

| k | updates/epoch | **reduction vs cycle (exact)** |
|--:|--:|--:|
| 50 | 500,000 | **100.0×** |
| 100 | 1,000,000 | 50.0× |
| 200 | 2,000,000 | **25.0×** |
| 400 | 4,000,000 | 12.5× |

The **measured wall-time** win is far smaller than the update reduction, for
two separate, now-quantified reasons:

1. **The lazy O(D) per-update penalty** (inherent to the fast mode): budget
   recomputes each pair distance from features (O(D)); cycle reads a
   precomputed matrix entry (O(1)). Hence, at fixed k, speedup falls with D:
   `Spearman(deployable speedup, D) = −0.90` (p=0.037), and the ranking is
   k-invariant. Corroboration: **cycle's wall time is D-independent**
   (145–159 s at N=10k from D=3 to D=3072) — all D-dependence lives on the
   budget side.
2. **The O(N²) exact-stress scoring pass** (an artefact of our measurement,
   not of the method): both arms compute exact stress on the full N×N matrix;
   in the budget arm this k-independent cost (`a` in `budget_time = a + b·k`;
   `a/N² ≈ 360 ns`, constant across rungs — the O(N²) signature) dominates at
   low D. It caps *measured* speedup at 2.0–4.4× at N=10k, and at D=3072 even
   pushes measured below 1× at k≥200 while the underlying SGD work is smaller
   than cycle's.

Measured speedups at N=10,000 (direct measurements, trustworthy as stated):
k=50 → 4.4× (D=3,8) down to 2.0× (D=3072); k=200 → 4.1× down to 0.9×.

---

## 6. The deliverable: quality and speed at the *same* operating point

Quality and speed must always be quoted **at the same k**. At a fixed quality
target the two axes **multiply**: deliverable win = (how little budget you
need: `k*`, set by strain) × (how cheap each update is: ~1/D).

**Target ≤ +2% overhead, N=10,000** (K). "Est. speedup" is the model-based
deployable estimate — see §8 for its epistemic status; "update reduction" is
exact arithmetic:

| dataset | D | strain | k needed | overhead | update reduction | est. deployable speedup |
|---|--:|--:|--:|--:|--:|--:|
| feynman_iii_15_12 | 3 | 0.69 | 50 | +1.9% | **100×** | ~197× |
| california_housing | 8 | 0.65 | 200 | +1.7% | **25×** | ~49× |
| patchcamelyon | 512 | 0.31 | 200 | +1.2% | 25× | ~6× |
| fashion_mnist_full | 784 | 0.37 | 200 | +1.0% | 25× | ~4× |
| mnist_digits | 784 | 0.12 | 400 | +1.7% | 12.5× | ~2× |
| cifar10_raw | 3072 | 0.38 | 200 | +1.1% | 25× | **~1× (no win)** |
| ag_news | 384 | 0.08 | — | never | — | — |
| tiny_imagenet | 512 | 0.07 | — | never | — | — |

Readings:

- **Low-D responsive is the ideal target.** `california_housing` (real data,
  D=8) needs 25× fewer updates for +1.7% overhead, and each update is cheap —
  we *expect* a very large wall-time improvement (est. ~49×), though the exact
  multiplier awaits direct measurement (§8).
- **Very high D erases the win at near-cycle quality.** `cifar10_raw` needs
  k=200 for +1.1%, but each lazy update costs ~D=3072 distance flops — the
  25× update reduction is fully consumed by the per-update penalty (est. ~1×).
- **Hard-middle never reaches the target**: ag_news and tiny_imagenet cannot
  hit +2% at any tested budget; at a looser +5% they need k=352–400 for only a
  ~3–4× estimated win.

---

## 7. What the measurement itself revealed — sampled stress is the next build

Two independent walls, both caused by exact-stress scoring / the O(N²)
baseline machinery, both discovered the hard way:

- **RAM**: cycle materialises ~28 N² bytes (matrix + explicit pair list +
  per-pair arrays) — N=15,000 OOM-killed a 16 GB machine (measured peak model:
  ~8.1 GB); hence K's 10k cap.
- **Scoring**: `_safe_stress` is O(N²) and returns −1.0 above N=20,000 —
  stress is *unmeasurable* past 20k regardless of hardware, and below that the
  scoring pass contaminates every timing number (§5.2).

A **sampled-stress estimator** (unbiased random-pair estimate, O(sample)
memory) removes both at once: it uncaps N (≫20k), reveals the true wall-time
speedup, and — since J showed the full-cycle budget reproduces cycle exactly —
a high-budget random run can serve as the baseline where cycle itself is
infeasible. This is the top-priority build on the path to the upstream PR.

---

## 8. Epistemic status — what is measured vs what is estimated

**Exact (arithmetic or direct measurement, quotable without hedging):**

- All stress overheads, floors, k*, p, steepness, iteration penalties, and the
  correlations built on them (5-seed paired medians of exact stress).
- Update-count reductions `(N−1)/(2k)` — e.g. 25× fewer pair updates at
  k=200, N=10,000.
- Measured wall-time speedups (2.0–4.4× at N=10k, k=50) — real, but
  *conservative*: they include a scoring cost a deployment would not pay.
- Cycle-time D-independence; `a/N² ≈ const` (the O(N²) scoring signature).

**Estimated (model-based inference — quote with "expected/estimated"):**

- **Deployable speedups** (197×, 49×, …) come from the `budget_time = a + b·k`
  decomposition: they assume *all* k-independent time is scoring that a
  deployment skips, and *all* residual time is SGD work. If some fixed cost
  belongs to the fast mode itself (per-epoch bookkeeping), they are
  overestimates; conversely cycle also pays one-off costs (matrix + pair-list
  build) that the decomposition partly credits to it. At low D the intercept
  is ~97% of budget time, so the multiplier is highly fit-sensitive. **The
  defensible claim:** *"we process 25–100× fewer pairs and expect a very large
  wall-time improvement — order-of-magnitude 10–200× at low D — but the exact
  factor cannot be stated with certainty until the sampled-stress estimator
  allows a direct measurement."*
- The `~360 ns · N²` scoring-cost constant is machine-specific.

---

## 9. Combined conclusions (the J+K story in six statements)

1. **Budgeted uniform random sampling is a safe knob**: stress is monotone in
   budget everywhere, and the full-cycle budget reproduces cycle exactly
   (dispatcher sanity, J/I).
2. **Whether it has leverage is governed by 2D-embeddability**, measured
   cheaply by `strain_top2`: J ρ=+0.75 (steepness, 18 datasets), K ρ=−1.00
   (k*, 8 datasets at N=10k). Three regimes: responsive / hard-middle /
   degenerate; the boundary sits near strain ≈ 0.2 and is a band, not a wall.
3. **On responsive data the required budget is (near-)constant in N**
   (p ≈ 0.0–0.19 across J+K; absolute-k law wins 14/18) — validated from
   N=400 to N=10,000. Fixed k=200 → ~+1%; k=100 → ~+5%; k=50 unreliable at
   large N. **N-invariance itself is universal; the regime sets the floor**
   (responsive ~+1%, hard-middle +4–9%).
4. **Budget-cut and iteration-cut are mutually exclusive levers** (ρ=+0.95, J)
   — exactly the data where budget helps is where `max_iter=15` hurts.
5. **Speed is a separate axis, set by D** (ρ=−0.90, K): the lazy O(D)
   per-update cost means the deliverable win at fixed quality ranges from
   ~huge (D≤10: est. 50–200×) through modest (D~500–800: est. 4–8×) to nil
   (D≈3000 at near-cycle quality). Exact update reductions: 12.5–100×.
6. **The measurement apparatus is now the bottleneck** — the O(N²)
   exact-stress scoring caps measurable N at 20k, contaminates timings, and
   hides the low-D win. **Build the sampled-stress estimator next.**

**Upstream-PR guidance distilled:** expose `n_updates_per_epoch`; document the
regime dependence with `strain_top2` as the cheap self-diagnostic (potential
`fast_mode='auto'` gating on it, using `k*`-style targets rather than curve
steepness); recommend fixed k=100–200 on responsive data with `max_iter=30`;
warn that budget-cut and iteration-cut don't compose; note the speed win
depends on D and is largest for low-dimensional data; on low-strain data warn
the 2D embedding itself is questionable.

---

## 10. Caveats (merged)

- **Deployable speedups are estimates** (§8) — the single most important
  hedge when presenting these results.
- **Iteration-lever untested at scale** (K ran max_iter=30 only).
- **Small-N artifacts in J**: orl (N=400, 1 rung) unreliable throughout;
  svhn/har/seismic/sms (N<900) noisy for p and iteration penalty. Headline
  correlations survive dropping them.
- **Regime boundary thinly sampled**: strain ≈ 0.2 characterized by few
  datasets (bank/spambase/fmd in J; mnist_digits in K). A synthetic
  dialled-strain source would map it cleanly.
- **K's N capped at 10k by RAM**, 20k by `_safe_stress`; constant-k beyond
  10k is extrapolation until the sampled-stress estimator lands.
- **feynman is synthetic** (best case); `california_housing` is the convincing
  real low-D instance. `cifar10_raw` (pixels) ≠ J's `cifar10` (embeddings).
- **ρ=−1.00 is a perfect *rank* correlation on n=8** (chance ≈ 1/8! ≈ 2.5·10⁻⁵)
  — the ordering is exactly as predicted, but k* itself has interpolation
  error; do not read it as a deterministic functional law.
- The hf-collection sources are learned embeddings (text ones L2-normalised) —
  representative of embedding workloads, not of raw-feature workloads.
