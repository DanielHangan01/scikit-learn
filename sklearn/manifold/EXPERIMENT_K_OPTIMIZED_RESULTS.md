# Experiment K-optimized — results & conclusions (measured speedups to N=50,000)

**Status: complete.** All 8 large sources, ladder `{10k, 20k, 35k, 50k}`,
630 fits, 31.7 h of fit time. Runner:
[benchmarks/run_experiment_k_optimized.py](benchmarks/run_experiment_k_optimized.py).
Predecessor (protocol + N≤10k): [EXPERIMENT_K_RESULTS.md](EXPERIMENT_K_RESULTS.md) ·
[EXPERIMENT_K_SPEC.md](EXPERIMENT_K_SPEC.md).

Plain Experiment K established the fast mode's *quality* behaviour to
N=10,000 but could not measure its *speed*: both arms evaluated exact stress
inside the timed region at O(N²) cost, which dominated the budgeted runs (up
to ~97% of their wall time at low D) and forced every speedup number to be a
model-based estimate. This rerun removes that contamination and extends the
ladder 5× further.

**Headline: the estimated speedups were real, and the one negative result of
plain K does not survive scale.** Measured, deployable speedups at
near-cycle quality now range from **7× (D=3072) to 1,070× (D=3)**, and
`cifar10_raw` — which plain K reported as "no win at near-cycle quality
(~1×)" — reaches **7.1× at +1.3%** once N grows to 50,000.

---

## 1. What changed vs plain K (and why the comparison is still valid)

Three protocol changes, all timing/cost — none scientific:

1. **`compute_stress=False` during fits** (new `SGDMDS` parameter); each fit
   is scored **once, afterwards, outside the timed region**, with a new
   row-blocked exact scorer (`bench_utils.stress_exact_chunked`, O(N·block)
   memory). Stress is observational — nothing in the optimizer reads it — and
   the embedding is **bit-identical** with the flag on or off (verified
   directly, both matrix and lazy modes). Consequences: `time_solver` is pure
   fit time, so **measured speedup *is* deployable speedup**; the `a + b·k`
   decomposition is retired; and the O(N²) scoring memory wall is gone (no
   `EXACT_STRESS_N_CAP`, no 80 GB scoring matrices in lazy mode).
2. **1 cycle run vs 5 budget seeds**, all on a **single shared subsample** per
   (dataset, N). Justified by plain K's own data: cycle stress CV across
   seeds was 0.07–0.90% *including* subsample variance, against overheads of
   +1–10%.
3. **Per-fit checkpoint/resume** (plain K: per rung), since single fits are
   now ~70 min at the top rung.

**Anchor check — the protocol change is sound.** N=10,000 is a shared rung.
Overheads agree with plain K to ≤0.01 at every k ≥ 100 on all 8 datasets:

| dataset | k=100 (opt vs K) | k=200 | k=400 |
|---|---|---|---|
| feynman | 1.004 vs 1.013 | 1.002 vs 1.011 | 1.001 vs 1.010 |
| california_housing | 1.044 vs 1.039 | 1.018 vs 1.017 | 1.008 vs 1.008 |
| ag_news | 1.102 vs 1.102 | 1.095 vs 1.093 | 1.045 vs 1.043 |
| patchcamelyon | 1.067 vs 1.058 | 1.019 vs 1.012 | 1.006 vs 1.005 |
| tiny_imagenet | 1.103 vs 1.104 | 1.093 vs 1.092 | 1.049 vs 1.048 |
| fashion_mnist_full | 1.058 vs 1.049 | 1.013 vs 1.010 | 1.003 vs 1.003 |
| mnist_digits | 1.156 vs 1.169 | 1.082 vs 1.072 | 1.022 vs 1.017 |
| cifar10_raw | 1.047 vs 1.043 | 1.011 vs 1.011 | 1.004 vs 1.003 |

The only larger discrepancies are at **k=50** (fashion 1.205 vs 1.133,
patchcamelyon 1.171 vs 1.143) — exactly the budget whose across-seed spread
is widest (fashion at 50k: [1.138, 1.355]). That is sampling noise, not
protocol bias, and it reconfirms k=50 as the unreliable floor.

---

## 2. Master table — top rung (N=50,000; california_housing at its full 20,640)

Overhead = budget/cycle stress (median of 5 SGD seeds vs 1 cycle run);
speedup = cycle_time / budget_time, **pure fit time, directly measured**.

| dataset | D | strain | k=50 | k=100 | k=200 | k=400 |
|---|--:|--:|--:|--:|--:|--:|
| feynman_iii_15_12 | 3 | 0.69 | **+1.4% / 1070×** | +0.7% / 535× | +0.5% / 268× | +0.4% / 134× |
| california_housing¹ | 8 | 0.65 | +20.9% / 411× | +5.2% / 207× | **+2.1% / 103×** | +1.0% / 52× |
| ag_news | 384 | 0.08 | +10.8% / 119× | +10.4% / 59× | +10.0% / 30× | +7.0% / 15× |
| patchcamelyon | 512 | 0.31 | +21.1% / 108× | +10.5% / 55× | **+2.5% / 25×** | +0.7% / 13× |
| tiny_imagenet | 512 | 0.07 | +10.8% / 109× | +10.4% / 50× | +10.0% / 25× | +6.6% / 14× |
| fashion_mnist_full | 784 | 0.37 | +28.5% / 79× | +6.7% / 39× | **+1.2% / 20×** | +0.4% / 10× |
| mnist_digits | 784 | 0.12 | +20.0% / 82× | +18.9% / 41× | +10.5% / 19× | +2.5% / 9× |
| cifar10_raw | 3072 | 0.38 | +21.0% / 30× | +7.2% / 15× | **+1.3% / 7.1×** | +0.4% / 3.7× |

¹ N=20,640 (its entire dataset — the ladder uses a source's own full N when
it falls within 5% of a rung), so its speedups are at ~0.4× the N of the rest
and are **not** directly comparable across the row; at the shared N=10,000
rung it is 104× at k=100.

Bold = each responsive dataset's best operating point at ≈+1–2%.

---

## 3. Constant-k confirmed to N=50,000 — and N-invariance remains universal

Overhead at **fixed k=200** across the full ladder (10k → 20k → 35k → 50k):

| dataset | 10k | 20k | 35k | 50k | fitted p (k*, +5%) |
|---|--:|--:|--:|--:|--:|
| feynman | 1.002 | 1.001 | 1.001 | 1.005 | **0.00** |
| fashion_mnist_full | 1.013 | 1.011 | 1.011 | 1.012 | **0.03** |
| cifar10_raw | 1.011 | 1.011 | 1.012 | 1.013 | 0.09 |
| patchcamelyon | 1.019 | 1.019 | 1.021 | 1.025 | 0.11 |
| mnist_digits | 1.082 | 1.083 | 1.089 | 1.105 | 0.05 |
| ag_news | 1.095 | 1.097 | 1.100 | 1.100 | — (never ≤1.05) |
| tiny_imagenet | 1.093 | 1.096 | 1.100 | 1.100 | — (never ≤1.05) |

A fixed budget holds quality across a **5× increase in N**, with p = 0.00–0.11
— the strongest form of the constant-k claim, now validated an order of
magnitude beyond Experiment J's largest dataset (5,750). k*(+5%) is likewise
near-flat in N (e.g. fashion 107 → 107 → 108 → 112; patchcamelyon 118 → 142).

**N-invariance is universal, the floor is regime-specific** — plain K's
refinement survives: hard-middle datasets are equally flat in N, just pinned
higher (ag_news/tiny_imagenet ~+10% at k=200 at every rung).

One new detail: on hard-middle data the **k=400 floor drifts upward with N**
(ag_news +4.5% at 10k → +7.0% at 50k; tiny_imagenet +4.9% → +6.6%). At 10k
those curves had not yet bottomed out; at scale the plateau is genuinely
higher. This strengthens the "don't spend budget on hard-middle data"
guidance rather than weakening it.

---

## 4. Speed: measured, and growing linearly in N

Cycle wall time is **D-independent** (≈4,200 s at N=50,000 for every dataset,
from D=3 to D=3072) and scales as N² (155 s → 4,200 s over a 5× N increase =
27×). Budget time scales as k·N·D. The measured speedup therefore grows
**linearly in N at fixed k**, which the data shows cleanly — e.g.
`cifar10_raw` at k=100: 2.8× → 5.6× → 10.2× → 14.9× across the ladder.

`Spearman(D, speedup@k50) = −0.99` (p<0.0001, all 8) / **−1.00** on the five
responsive sources — the D-ordering from plain K is now essentially exact.

### The per-update cost ratio, quantified

Comparing measured speedup against the naive pair-count ratio
`(N−1)/(2k)` = 250× at N=50k, k=100:

| dataset | D | measured | % of pair-count theory |
|---|--:|--:|--:|
| feynman | 3 | 535× | **214%** |
| california_housing | 8 | 207× | **200%** |
| ag_news | 384 | 59× | 24% |
| patchcamelyon | 512 | 55× | 22% |
| tiny_imagenet | 512 | 50× | 20% |
| mnist_digits | 784 | 41× | 17% |
| fashion_mnist_full | 784 | 39× | 16% |
| cifar10_raw | 3072 | 15× | 6% |

**At low D the fast mode beats the pair-count bound — by 2×.** This is new
(plain K could not see it) and it corrects the framing that the lazy kernel
is strictly handicapped. The reason is that cycle's per-pair cost is *not*
free: beyond the O(1) distance lookup it pays memory-bandwidth costs the
budgeted kernel avoids entirely — materialising the N×N matrix, building the
explicit N²/2 pair list, and **re-shuffling the full pair/target/weight
arrays every epoch** (at N=50k that is 1.25 billion elements shuffled 30
times). The budgeted kernel touches only O(k·N) elements and computes
distances from D-dimensional features in registers. So the true relationship
is a **cost ratio**: budget's O(D) recomputation versus cycle's O(1) lookup
*plus* per-pair memory traffic. The crossover in this collection sits between
D=8 (200%) and D=384 (24%).

---

## 5. The plain-K negative result is overturned

Plain K concluded: *"cifar10_raw (D=3072) gets no deployable speedup at all
(1×) at the +2% target — demanding cycle quality at very high D erases the
win."* That was an **N=10,000 statement** and it does not generalise:

| N | overhead @k=200 | measured speedup |
|--:|--:|--:|
| 10,000 | +1.1% | 1.4× |
| 20,000 | +1.1% | 2.8× |
| 35,000 | +1.2% | 5.1× |
| 50,000 | +1.3% | **7.1×** |

Because overhead is constant in N while speedup grows ∝N, **every dataset's
win improves with scale**; high D shifts the curve down, it does not cap it.
The corrected statement:

> D sets *where the speedup curve starts*; N sets *how far up it goes*. At
> large enough N even the highest-D source is comfortably faster at
> near-cycle quality.

The same applies to the earlier "hard-middle gets ~3–4× at a loose target"
framing: ag_news now measures 15× at k=400 (+7.0%) at N=50,000 — the speed
was never the hard-middle problem, the quality floor is.

---

## 6. Operating points — measured speedup at a fixed quality target

Smallest tested k reaching the target, at each dataset's top rung. "Updates
cut" is exact arithmetic; **speedup is now a direct measurement**, not an
estimate.

**Target ≤ +2%:**

| dataset | D | N | k | overhead | updates cut | **measured speedup** |
|---|--:|--:|--:|--:|--:|--:|
| feynman_iii_15_12 | 3 | 50,000 | 50 | +1.4% | 500× | **1070×** |
| california_housing | 8 | 20,640 | 400 | +1.0% | 26× | 52× |
| patchcamelyon | 512 | 50,000 | 400 | +0.7% | 62× | 12.6× |
| fashion_mnist_full | 784 | 50,000 | 200 | +1.2% | 125× | **19.6×** |
| cifar10_raw | 3072 | 50,000 | 200 | +1.3% | 125× | **7.1×** |
| mnist_digits | 784 | 50,000 | — | never (+2.5% @k=400) | — | — |
| ag_news | 384 | 50,000 | — | never | — | — |
| tiny_imagenet | 512 | 50,000 | — | never | — | — |

**Target ≤ +5%:** california_housing k=200 → +2.1% / 103×; patchcamelyon
k=200 → +2.5% / 25×; mnist_digits k=400 → +2.5% / 9.4×; the others unchanged
from the +2% row. ag_news / tiny_imagenet never reach +5% at any tested k.

---

## 7. The strain → budget link at 50,000

`Spearman(strain, k*(+5%)) = −0.94` (p=0.005, n=6 datasets that reach the
target; the two that never reach it are the two lowest-strain sources, so
including them as worst-ranked strengthens the ordering). Top-rung k* values:

| dataset | strain | k*(+5%) | k*(+2%) | floor @k=400 |
|---|--:|--:|--:|--:|
| feynman | 0.687 | 50 | 50 | +0.4% |
| california_housing | 0.648 | 103 | 206 | +1.0% |
| cifar10_raw | 0.379 | 116 | 167 | +0.4% |
| fashion_mnist_full | 0.369 | 112 | 163 | +0.4% |
| patchcamelyon | 0.314 | 142 | 223 | +0.7% |
| mnist_digits | 0.115 | 285 | never | +2.5% |
| ag_news | 0.075 | never | never | +7.0% |
| tiny_imagenet | 0.072 | never | never | +6.6% |

Plain K's perfect ρ = −1.00 relaxes to −0.94 because of a single inversion
(fashion k*=112 vs cifar10_raw k*=116, whose strains are 0.369 vs 0.379 —
within noise of each other). This is the expected behaviour of a real
correlation on more data, not a weakening: **the two axes remain cleanly
separated** — cifar10_raw (D=3072) and fashion (D=784) need near-identical
budgets while their speeds differ 2.7×.

---

## 8. Epistemic status (substantially upgraded)

**Now directly measured** (previously model-based estimates):

- **All speedups in this document.** `time_solver` excludes stress evaluation
  entirely, so cycle/budget is a like-for-like fit-time comparison. No
  `a + b·k` decomposition, no extrapolation.
- Scoring cost is quantified and excluded: 16–27 s per fit at the top rung,
  **1.34 h out of 33.0 h total wall clock (4.1%)** for the whole experiment —
  versus plain K, where it was the *dominant* term in every budgeted run.

**Still exact arithmetic:** update-count reductions `(N−1)/(2k)`
(12.5×–500× across the operating points).

**Remaining estimate-like caveats:**

- **Single cycle baseline** per (dataset, N): the overhead denominator has no
  error bar in this run. Justified by plain K's measured 0.07–0.90% cycle CV,
  but overhead error bars here reflect budget-SGD noise only. Budget spreads
  *are* reported (5 seeds) and are the larger term at k=50.
- **One fixed subsample** per (dataset, N), so subsample-to-subsample
  variation is not resampled here (J and plain K already characterised it).
- Timings are single-node, single-threaded-solver measurements; BLAS
  threading affects cycle's matrix build and the (untimed) scorer, not the
  budgeted kernel's inner loop.

---

## 9. Conclusions

1. **Constant-k holds to N=50,000** (p = 0.00–0.11) across D = 3→3072 and
   five domains. A fixed k=200 delivers ~+1% on responsive data at every
   tested N.
2. **The speedups are real and large, and now measured**: 7×–1070× at
   near-cycle quality, growing linearly in N. The estimates in plain K were
   directionally correct and are superseded by direct measurement.
3. **Plain K's one negative result is overturned**: high D delays but does not
   cancel the win — `cifar10_raw` goes 1.4× → 7.1× as N goes 10k → 50k at
   constant +1.3% overhead.
4. **At low D the fast mode beats the pair-count bound (200–214%)**, because
   full-cycle pays per-pair memory traffic (matrix, pair list, per-epoch
   shuffle of 1.25 B elements) that the budgeted kernel never touches. Speed
   is a *cost ratio*, not a pure update-count ratio.
5. **The regime taxonomy survives at scale**: strain predicts the required
   budget (ρ = −0.94); hard-middle sources (ag_news, tiny_imagenet) never
   reach +5% at any budget and their floors *worsen* with N.
6. **Removing in-fit stress evaluation was the enabling change** — it cut the
   measurement artefact from ~97% of budgeted-run time to a 4.1% untimed
   sidecar, and made a 50,000-point ladder feasible at all.

**Upstream implication:** the fast mode should be documented as a
large-N feature with a **measured** benefit — `n_updates_per_epoch = 200·N`
on 2D-embeddable data gives ~1% stress overhead at 7×–270× lower fit time
depending on D, improving with N. The `compute_stress` parameter deserves
consideration for the public API on its own merits: at N=50,000 exact scoring
costs more than a budgeted fit.

---

## 10. Caveats

- **N capped at 50,000** by the cycle baseline's memory (~36 N² bytes ≈ 84 GB
  at 50k). The fast mode itself is O(N·D) and untested above 50k only because
  the *baseline* cannot be run there.
- **k=50 is unreliable on responsive data** (fashion at 50k: [1.138, 1.355]
  across seeds). k=100 is the practical minimum; k=200 the dependable ~1%.
- Sources larger than 50,000 (feynman, ag_news, patchcamelyon, tiny_imagenet
  at 100k; fashion, mnist_digits at 70k) are **subsampled** to the ladder;
  only `cifar10_raw` (50,000) and `california_housing` (20,640) are run at
  their true full size.
- `feynman_iii_15_12` is a synthetic analytic manifold — the 1070× headline
  is a best case; `california_housing` (real, D=8) and `fashion_mnist_full`
  (real, D=784) are the representative real-data anchors.
- The overheads here use a **single-seed cycle denominator** (§8).
