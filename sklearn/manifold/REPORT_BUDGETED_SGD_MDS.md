# Budgeted Stochastic SGD-MDS: When Subsampling Works, at What Scale, and What It Buys

*Detailed report on the two benchmark studies: an 18-dataset mechanism study, and an 8-source large-N study run to N = 50,000*

---

## 1. Summary

We study a "fast mode" for SGD-based metric MDS in which the per-epoch update
budget is reduced from all N(N−1)/2 point pairs (full-cycle SGD) to k·N
uniformly random pairs, with k a small constant. Across 26 datasets and roughly
6,800 controlled fits, the findings are:

1. **Subsampling is safe but not universally effective.** Stress never degrades
   with more budget, and whether a small budget suffices is governed almost
   entirely by a single, cheaply computable property of the data — its
   2D-embeddability — with rank correlations of ρ = +0.75 (18 datasets) and
   ρ = −0.94 (8 datasets at N = 50,000) in two independent studies.
2. **Where subsampling works, the required budget is essentially independent
   of N.** A fixed k ≈ 200 keeps the embedding within ~1–2% of full-cycle
   stress from N = 2,000 to N = 50,000 on every responsive dataset, spanning
   feature dimensions D = 3 to 3,072. The fitted scaling exponent is
   p = 0.00–0.11.
3. **Embedding quality and wall-time gain are governed by different
   variables.** Quality depends on 2D-embeddability; the time saved per unit of
   budget depends on the feature dimension D (ρ = −0.99). At a fixed quality
   target the two multiply.
4. **The speedups are large and directly measured**: **7× to 1,070×** at
   ≈ +1% stress overhead, growing linearly in N. An earlier pass reported only
   2–4× because both arms evaluated exact stress *inside* the timed region at
   O(N²) cost; moving that evaluation outside the measurement (§3.5) revealed
   the true figures.

---

## 2. Background and question

SGD-MDS minimises raw metric stress

```
σ(X) = ½ · Σ_{i<j} ( ‖x_i − x_j‖ − δ_ij )²
```

by stochastic per-pair updates. The reference algorithm ("full cycle")
processes every pair once per epoch — O(N²) work per epoch, which dominates
runtime at scale. Earlier experiments established that restricting updates to
a *pivot* set is counterproductive, while **uniform random pair sampling at a
fixed budget of k·N updates per epoch** matches full-cycle quality surprisingly
often. The two studies reported here answer, systematically:

- **Study 1 (breadth, 18 datasets, N = 400–5,750):** *which* datasets tolerate
  budget reduction, and what property of the data predicts this;
- **Study 2 (scale, 8 large sources, N = 10,000–50,000):** whether the answer
  survives at an order of magnitude larger N, and what the reduction actually
  buys in wall-clock time.

---

## 3. Methods

### 3.1 Controlled comparison

Both studies use an identical, single-variable design. For each dataset,
subsample size N, and random seed:

- a **baseline fit**: full-cycle SGD-MDS (all N(N−1)/2 pairs per epoch,
  precomputed distance matrix);
- one **budgeted fit per k**: identical estimator, identical data, identical
  seed, but k·N random pair updates per epoch, with distances computed on the
  fly (O(N·D) memory instead of O(N²)).

All other settings are held fixed (2 output dimensions, learning rate 0.5,
hybrid annealing schedule, 30 epochs, single-threaded solver). Both arms are
scored with the same exact raw-stress metric over all pairs. The reported
quantity is the **stress overhead**

```
overhead(N, k) = stress_budgeted / stress_full-cycle ,
```

so 1.00 means the budgeted run matches full-cycle quality exactly.

### 3.2 Replication

Study 1 repeats every configuration with **5 independent seeds**; each seed
draws a fresh random subsample *and* re-seeds the optimiser, and within a seed
the baseline and budgeted runs share the same subsample, making the overhead a
paired ratio. Study 2 uses **one shared subsample per (dataset, N)** with a
single baseline fit and **5 optimiser seeds per budget** — justified by the
measured baseline seed-to-seed variation of 0.07–0.90%, against overheads of
+1–10%. All reported numbers are medians. Study 1 comprises 5,340 fits; Study 2,
630 fits over 31.7 hours of compute (plus an earlier 800-fit pass, §3.5).

### 3.3 Predictor of "embeddability": the strain spectrum

For each dataset we compute, on a fixed 800-point subsample (independent of
the benchmark sweep), the **top-2 eigenvalue fraction of the classical-MDS
Gram matrix**: double-centre the squared-distance matrix,
B = −½·J·D²·J, eigendecompose, and take

```
strain_top2 = (λ₁ + λ₂) / Σ_positive λ .
```

This is exactly the fraction of the data's geometric structure that a 2D
embedding can capture (the analogue of "explained variance of the first two
components", computed from distances). It costs one 800×800 eigensolve —
negligible — so it is a viable fit-time self-diagnostic. We also computed the
PCA participation ratio as a feature-space cross-check, and, following a
suggestion to test distance-norm concentration as an alternative predictor, we
measured the coefficient of variation of the pairwise-distance distribution:
it showed **no significant correlation** with budget-responsiveness (|ρ| ≤ 0.2,
p > 0.5), with clean counterexamples in both directions, whereas the spectral
measures remained strong — the signal is in the spectrum's shape, not in the
width of the distance histogram.

### 3.4 Datasets

**Study 1:** the full 18-dataset benchmark collection (images, text, tabular,
biomedical; N = 400–5,750, D = 24–1,617), each subsampled along an N-ladder
{400 … full N}.

**Study 2:** 8 large sources chosen to span feature dimension across three
orders of magnitude and both behavioural regimes, each subsampled to
N ∈ {10,000, 20,000, 35,000, 50,000}:

| source | N available | D | strain_top2 | character |
|---|--:|--:|--:|---|
| Feynman symbolic-regression manifold | 100,000 | 3 | 0.69 | synthetic, extreme low-D |
| California housing | 20,640 | 8 | 0.65 | real tabular, low-D |
| AG News (sentence embeddings) | 100,000 | 384 | 0.08 | text embeddings |
| PatchCamelyon (image embeddings) | 100,000 | 512 | 0.31 | histopathology |
| Tiny-ImageNet (image embeddings) | 100,000 | 512 | 0.07 | natural images |
| MNIST digits (raw pixels) | 70,000 | 784 | 0.12 | boundary case |
| Fashion-MNIST (raw pixels, full) | 70,000 | 784 | 0.37 | anchor to Study 1 |
| CIFAR-10 (raw pixels) | 50,000 | 3,072 | 0.38 | very high-D |

Budgets tested: k ∈ {10, 25, 50, 100, 200} (Study 1) and
k ∈ {50, 100, 200, 400} (Study 2). California housing is run at its true full
size (20,640); the others are subsampled to the ladder. The upper problem size
is capped at N = 50,000 by the memory footprint of the *baseline* — the
full-cycle distance matrix plus explicit pair list, ~36 N² bytes ≈ 84 GB at
N = 50,000 — not by the fast mode, which is O(N·D).

### 3.5 A measurement correction

Study 2 was run twice. In the first pass, both arms evaluated exact stress
*inside* the timed region. That evaluation is O(N²) and it dominated the
budgeted runs — up to ~97% of their wall time on low-dimensional data — so the
measured speedups (2–4×) reflected our instrumentation rather than the
algorithms, and the true figures could only be estimated by modelling.

In the second pass we disabled in-fit stress evaluation and scored each run
once afterwards, outside the timing, using a memory-chunked exact scorer.
Stress is purely observational — no part of the optimiser reads it — and we
verified that the resulting embeddings are **bit-identical**. Timing is now
pure fit time, so the measured speedup is the deployable speedup, with no
modelling involved. Scoring accounts for 4.1% of total wall clock and is
recorded separately.

To validate the change, N = 10,000 was retained as a shared rung between the
two passes: stress overheads agree to within 0.01 on all 8 datasets at every
budget k ≥ 100. All results below are from the corrected pass.

---

## 4. Results

### 4.1 A single property of the data predicts whether subsampling works

Datasets fall into three regimes, ordered by `strain_top2`:

| Regime | strain_top2 | behaviour |
|---|---|---|
| **Responsive** | ≳ 0.22 | budget is a powerful lever: fixed k reaches within ~1–2% of full-cycle stress |
| **Hard middle** | ~0.05–0.21 | overhead plateaus +3–10% above full cycle; additional budget barely helps |
| **Degenerate** | < 0.05 | matches full cycle at any budget — but 2D captures <5% of the structure, so the embedding itself is of limited value |

The strength of the link, measured two independent ways:

- **Study 1** (18 datasets): correlation between `strain_top2` and the
  steepness of the overhead-vs-k curve: **ρ = +0.75** (p < 0.001); confirmed
  by the PCA participation ratio (ρ = −0.74) and by an N-controlled comparison
  at a common subsample size (ρ = +0.72).
- **Study 2** (8 datasets, at each source's largest N): correlation between
  `strain_top2` and **k\*** — the budget needed to come within +5% of
  full-cycle stress: **ρ = −0.94** (p = 0.005). Sorting the sources by
  embeddability very nearly sorts them by required budget:

  | source | strain_top2 | k\* (budget for ≤ +5%) |
  |---|--:|--:|
  | Feynman manifold | 0.69 | 50 |
  | California housing | 0.65 | 103 |
  | CIFAR-10 raw | 0.38 | 116 |
  | Fashion-MNIST | 0.37 | 112 |
  | PatchCamelyon | 0.31 | 142 |
  | MNIST digits | 0.12 | 285 |
  | AG News | 0.08 | never reaches +5% |
  | Tiny-ImageNet | 0.07 | never reaches +5% |

  The ordering is monotone across four unrelated domains (synthetic, tabular,
  image, text) with a single inversion — CIFAR-10 raw and Fashion-MNIST, whose
  embeddability scores differ by 0.01 and are effectively tied. The most
  embeddable source needs ~6× less budget than the least, and the two
  least-embeddable sources never reach +5% at any tested budget.

A methodological note: curve *steepness* alone is an ambiguous summary — a flat
curve can mean "already optimal at the smallest budget" (the Feynman manifold,
at its floor already at k = 50) or "stuck above full-cycle quality" (AG News,
whose curve is flat but never descends near full cycle). The budget-to-target
quantity k\* disambiguates the two and is the metric we now prefer.

### 4.2 Where it works, the required budget is essentially independent of N

Modelling the required budget as k\* ∝ N^p:

- Study 1, on the datasets where p is statistically meaningful (steep curve,
  ≥5 ladder rungs): p = 0.06–0.16, median ≈ 0.11.
- Study 2, all responsive sources, N = 10,000 → 50,000: **p = 0.00–0.11**.

Directly, at fixed k = 200 across the ladder (10k / 20k / 35k / 50k):

| source | overhead across the ladder |
|---|---|
| Feynman manifold | +0.2%, +0.1%, +0.1%, +0.5% |
| Fashion-MNIST | +1.3%, +1.1%, +1.1%, +1.2% |
| CIFAR-10 raw | +1.1%, +1.1%, +1.2%, +1.3% |
| PatchCamelyon | +1.9%, +1.9%, +2.1%, +2.5% |

In practical terms: **growing the dataset 10× requires only ~1.3× the
per-point budget.** A pooled regression test further shows the overhead is a
function of absolute k (updates per point) rather than of the sampled
*fraction* of pairs, in 14 of 18 datasets — the mechanistic form of the same
statement.

Two sharpenings from Study 2: N-invariance at fixed k holds in the *hard
middle* as well — what that regime changes is the **floor** the overhead
plateaus at (+6–10% vs ~+1% for responsive data), not the N-scaling. And
practical operating points: k = 200 is the reliable "~1%" setting; k = 100
gives ~+5–7%; k = 50 is unreliable at scale (seed spreads as wide as
[+14%, +36%]).

### 4.3 The two speed levers do not compose

Halving the number of epochs (30 → 15) is free for full-cycle SGD (ratio
≈ 1.00 everywhere — it has converged by epoch 15). For budgeted runs, the
penalty for cutting epochs correlates with `strain_top2` at **ρ = +0.95**:
precisely the datasets where the budget lever works are those where cutting
epochs hurts (up to +48% stress), and vice versa. One may cut the budget *or*
the epochs, never both. (Established in Study 1; Study 2 ran a single
30-epoch arm, so this interaction is untested above N = 5,750.)

### 4.4 Wall-time: the gain per unit of budget is set by the feature dimension

The budgeted mode trades memory for computation: it avoids the O(N²) distance
matrix by recomputing each sampled pair's distance from the raw features — an
O(D) cost per update — whereas full-cycle looks the distance up in its
precomputed matrix. Dividing each fit's total time by its total update count
gives the per-update cost of each algorithm:

| | per-update cost |
|---|---|
| **full cycle** | **112.1 ns ± 2.6**, across D = 3 → 3,072 — i.e. independent of dimensionality |
| **budgeted** | **≈ 52 ns at D ≤ 8**, rising by ≈ 0.6 ns per dimension to 2,031 ns at D = 3,072 |

Two consequences.

**First, the speedup separates into two independent factors.** Writing it out,

```
speedup ≈ (N / 2k) · (cost_cycle / cost_budget(D)) ,
```

the first factor grows linearly in N and the second is a constant set by
dimensionality. D scales the whole curve up or down; it does not change the
fact that the curve grows with N. So high dimensionality *delays* the win
rather than capping it — CIFAR-10 raw, our hardest case, goes 1.4× → 2.8× →
5.1× → **7.1×** across the ladder at a constant +1.1–1.3% overhead. Our earlier
pass, which stopped at N = 10,000, wrongly concluded that this dataset gets no
useful speedup at near-cycle quality.

**Second, full-cycle's "free" lookup is not free.** At D ≤ 8 a budgeted update
costs less than half a full-cycle pair-update, because full-cycle pays
cache-hostile random access into a multi-gigabyte matrix plus a per-epoch
reshuffle of its pair, target and weight arrays (1.25 billion elements each at
N = 50,000), while the budgeted kernel touches only O(k·N) elements and
computes distances from D-dimensional vectors in registers. The two costs cross
at **D ≈ 60**; below that, the fast mode is *faster than the naive pair-count
bound predicts* (200–214% of it).

So the two axes separate cleanly:

> **2D-embeddability determines how much budget you need; the feature
> dimension determines what each unit of budget costs.**

### 4.5 The deliverable, at matched quality

Quality and speed must be quoted at the same operating point. The table fixes
the quality target at ≤ +2% overhead, at each source's largest N, and reports
the smallest tested budget reaching it, the exact reduction in pair updates per
epoch, and the **measured** wall-time gain.

| source | D | N | k needed | overhead | update reduction | **measured speedup** |
|---|--:|--:|--:|--:|--:|--:|
| Feynman manifold | 3 | 50,000 | 50 | +1.4% | 500× | **1070×** |
| California housing | 8 | 20,640 | 400 | +1.0% | 26× | 52× |
| PatchCamelyon | 512 | 50,000 | 400 | +0.7% | 62× | 12.6× |
| Fashion-MNIST | 784 | 50,000 | 200 | +1.2% | 125× | **19.6×** |
| CIFAR-10 raw | 3,072 | 50,000 | 200 | +1.3% | 125× | **7.1×** |
| MNIST digits | 784 | 50,000 | — | never reaches +2% (+2.5% at k = 400) | — | — |
| AG News | 384 | 50,000 | — | never reaches +2% | — | — |
| Tiny-ImageNet | 512 | 50,000 | — | never reaches +2% | — | — |

At a looser ≤ +5% target, California housing reaches +2.1% at k = 200 (103×),
PatchCamelyon +2.5% at k = 200 (25×), and MNIST digits +2.5% at k = 400 (9.4×);
AG News and Tiny-ImageNet still never qualify.

Readings: low-dimensional, well-embeddable data is the ideal case — two orders
of magnitude faster at indistinguishable quality. Every responsive source gains
substantially at N = 50,000, including the D = 3,072 case. The limitation is no
longer speed but the quality floor on poorly-embeddable data: AG News and
Tiny-ImageNet cannot reach near-cycle quality at any budget we tested, which is
a property of their geometry rather than of the method's efficiency.

---

## 5. Measurement status

**Directly measured** (no modelling): all stress overheads, floors, k\*,
scaling exponents and correlations (paired medians of exact stress); all
wall-time gains in §4.4 and §4.5, which compare pure fit time with stress
evaluation excluded from both arms; and the per-update costs. **Exact
arithmetic**: the update-count reductions (12.5×–500×).

**Residual caveats.** Study 2 uses a single baseline fit per (dataset, N), so
the overhead denominator carries no error bar — justified by the 0.07–0.90%
baseline variation measured in the first pass, but it means the reported error
spreads reflect budgeted-run variation only. Timings are wall-clock on a single
node; the budgeted-run seed-to-seed variation is 0.01–5.5%, so speedups should
be read as accurate to roughly ±5%, not to three significant figures.

---

## 6. Limitations

- **Problem size**: N ≤ 50,000, bounded by the *baseline's* memory footprint
  (~84 GB at that size). The fast mode itself is O(N·D) and is untested above
  50,000 only because there is nothing left to compare it against.
- **Epoch-budget interaction** untested above N = 5,750.
- **The regime boundary** (strain ≈ 0.2) is sampled by few datasets; a
  synthetic source with tunable embeddability would map it precisely.
- The strongest low-D result comes from a synthetic manifold; California
  housing (real, D = 8) is the corroborating real-data case.
- Several large sources are pre-computed neural embeddings (with text
  embeddings L2-normalised), representative of embedding workloads rather
  than raw-feature workloads.
- The crossover at D ≈ 60 (§4.4) is interpolated across a gap in our
  dimensional coverage (we have D = 8 and D = 384, nothing between), so it is
  an order-of-magnitude estimate.

---

## 7. Next steps

1. **Extend beyond N = 50,000.** The obstacle is now purely the reference
   algorithm: full-cycle needs ~36 N² bytes (≈ 335 GB at N = 100,000). Since a
   full-budget random run reproduces full-cycle exactly, it can serve as the
   reference where full-cycle is infeasible, decoupling the ceiling from the
   baseline's memory.
2. **Automatic gating**: `strain_top2` costs one small eigensolve at fit time,
   so the estimator can self-diagnose the regime and choose full-cycle vs
   budgeted mode (and a default k) — a candidate `fast_mode='auto'` API,
   gated on the budget-to-target quantity k\* rather than curve steepness.
3. **Boundary mapping** with synthetic tunable-strain data; a dimensional sweep
   between D = 8 and D = 384 to pin the cost crossover; and validation of the
   budget knob under non-metric (isotonic) MDS, which is so far untested.
4. Upstream documentation: recommend fixed k = 100–200 with the full epoch
   count on responsive data; warn that budget and epoch reductions do not
   compose; note that the wall-time gain depends on dimensionality and grows
   with N; and flag low-strain data as cases where a 2D embedding is itself
   questionable.
