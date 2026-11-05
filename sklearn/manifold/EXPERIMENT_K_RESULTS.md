# Experiment K — results & conclusions (large-N budget scaling)

> **⚠ Partly superseded by
> [EXPERIMENT_K_OPTIMIZED_RESULTS.md](EXPERIMENT_K_OPTIMIZED_RESULTS.md)**
> (same 8 sources, ladder extended to N=50,000, stress evaluation moved
> outside the timed region). Still authoritative for: the protocol, the
> N≤10k quality numbers, and the regime analysis. **Superseded**: (a) every
> speedup figure here — the "measured" ones are contaminated by O(N²)
> in-fit scoring and the "deployable" ones were model-based estimates; the
> optimized rerun measures them directly (7×–1070×); (b) §4's conclusion
> that `cifar10_raw` gets **no win (~1×)** at near-cycle quality — that is
> an N=10,000 artefact: at N=50,000 it measures **7.1× at +1.3%**, because
> speedup grows ∝N at constant overhead.

**Status: 8 large sources complete** (N up to 10,000, subsampled from sources
up to N=100k). Runner:
[benchmarks/run_experiment_k.py](benchmarks/run_experiment_k.py) · plots:
[benchmarks/plot_experiment_k.py](benchmarks/plot_experiment_k.py) ·
design/hypotheses: [EXPERIMENT_K_SPEC.md](EXPERIMENT_K_SPEC.md).

Experiment K stress-tests the budgeted-random fast mode at **larger N than
I / J** (J's largest was epileptic, N=5750) and **isolates the wall-time
payoff**. It subsamples each large source up an N-ladder (intrinsic
dimensionality fixed, only N varies) and measures both stress overhead and
wall-time speedup at each N. Two questions: (1) does the responsive-regime
constant-k prediction survive at scale, and (2) what governs the actual
speedup?

---

## 1. Conditions

- **8 sources**, chosen to span feature dimension D (3 → 3072) and regime.
  `fashion_mnist_full` is the full 70k Fashion-MNIST (OpenML); the other 7 are
  wired from bulk collections via
  [benchmarks/prep_shortlist_datasets.py](benchmarks/prep_shortlist_datasets.py)
  (symlinked into `datasets/<name>/`). `strain_top2` is on standardized data
  (the pipeline standardizes on load).
- **N-ladder** `{2000, 4000, 7000, 10000}`, capped at 10k by **RAM** (the cycle
  baseline's ~28 N² bytes OOMs a 16 GB box at 15k), not by the method.
- **Budget grid** `k ∈ {50, 100, 200, 400}` (`n_updates_per_epoch = k·N`).
- **Baseline vs fast**: cycle = `dissimilarity='euclidean'`; fast =
  `dissimilarity='lazy'`, `sampling_strategy='random'`. Identical estimator
  settings (`max_iter=30`, lr=0.5, hybrid scheduler), same raw-stress metric.
- **Replication**: 5 seeds, paired within seed, median reported. Runner
  checkpoints/resumes per rung.

---

## 2. Master table (@ N=10,000, max_iter=30, median of 5 seeds)

Each cell pairs **stress overhead / deployable speedup at the same budget k** —
quality and speed must be read at one operating point, never mixed across
budgets. Overhead = budgeted/cycle stress; **deployable** speedup removes the
O(N²) exact-stress scoring cost (§5). `reg` = responsive (strain≥0.22) /
hard-middle. Sorted by D.

| dataset | D | strain | reg | k=50 | k=100 | k=200 | k=400 |
|---|--:|--:|:--:|--:|--:|--:|--:|
| feynman_iii_15_12 | 3 | 0.69 | R | **+1.9% / 197×** | +1.3% / 99× | +1.1% / 49× | +1.0% / 25× |
| california_housing | 8 | 0.65 | R | +10.5% / 198× | +3.9% / 99× | **+1.7% / 49×** | +0.8% / 25× |
| ag_news | 384 | 0.08 | H | +10.6% / 31× | +10.2% / 15× | +9.3% / 8× | +4.3% / 4× |
| patchcamelyon | 512 | 0.31 | R | +14.3% / 25× | +5.8% / 12× | **+1.2% / 6×** | +0.5% / 3× |
| tiny_imagenet | 512 | 0.07 | H | +10.7% / 24× | +10.4% / 12× | +9.2% / 6× | +4.8% / 3× |
| fashion_mnist_full | 784 | 0.37 | R | +13.3% / 16× | +4.9% / 8× | **+1.0% / 4×** | +0.3% / 2× |
| mnist_digits | 784 | 0.12 | H | +19.8% / 16× | +16.9% / 8× | +7.2% / 4× | +1.7% / 2× |
| cifar10_raw | 3072 | 0.38 | R | +21.8% / 5× | +4.3% / 2× | **+1.1% / 1×** | +0.3% / 1× |

Bold = each responsive dataset's best operating point at ≈+1–2% quality (§2b).

**Measured** speedup (i.e. including the O(N²) scoring pass — what a timed run
actually shows today):

| dataset | k=50 | k=100 | k=200 | k=400 |
|---|--:|--:|--:|--:|
| feynman_iii_15_12 | 4.4× | 4.3× | 4.1× | 3.8× |
| california_housing | 4.4× | 4.3× | 4.1× | 3.8× |
| ag_news | 3.8× | 3.3× | 2.8× | 2.0× |
| patchcamelyon | 3.7× | 3.2× | 2.6× | 1.8× |
| tiny_imagenet | 3.6× | 3.1× | 2.5× | 1.8× |
| fashion_mnist_full | 3.2× | 2.7× | 2.0× | 1.3× |
| mnist_digits | 3.3× | 2.7× | 2.0× | 1.3× |
| cifar10_raw | 2.0× | 1.4× | 0.9× | 0.5× |

## 2b. Operating points — speedup at a *fixed quality target*

The deliverable number: the smallest k reaching the target, and the speedup
*there*. This is the honest "what do I actually get" view.

**Target ≤ +2% overhead** (near-cycle quality):

| dataset | D | strain | k needed | overhead | **deployable speedup** |
|---|--:|--:|--:|--:|--:|
| feynman_iii_15_12 | 3 | 0.69 | 50 | +1.9% | **197×** |
| california_housing | 8 | 0.65 | 200 | +1.7% | **49×** |
| patchcamelyon | 512 | 0.31 | 200 | +1.2% | 6× |
| fashion_mnist_full | 784 | 0.37 | 200 | +1.0% | 4× |
| mnist_digits | 784 | 0.12 | 400 | +1.7% | 2× |
| cifar10_raw | 3072 | 0.38 | 200 | +1.1% | **1×** |
| ag_news | 384 | 0.08 | — | never | n/a |
| tiny_imagenet | 512 | 0.07 | — | never | n/a |

**Target ≤ +5% overhead** (looser):

| dataset | k needed | overhead | deployable speedup |
|---|--:|--:|--:|
| feynman_iii_15_12 | 50 | +1.9% | 197× |
| california_housing | 100 | +3.9% | 99× |
| fashion_mnist_full | 100 | +4.9% | 8× |
| patchcamelyon | 200 | +1.2% | 6× |
| ag_news | 400 | +4.3% | 4× |
| tiny_imagenet | 400 | +4.8% | 3× |
| cifar10_raw | 100 | +4.3% | 2× |
| mnist_digits | 400 | +1.7% | 2× |

**Two axes multiply into the deliverable speedup.** At a fixed *quality target*,
the win is set by (a) *how little budget you need* — `k*`, governed by
`strain_top2`/regime — times (b) *how cheap each update is* — governed by 1/D.
`feynman` wins on both (needs only k=50 **and** D=3 → 197×); `california_housing`
is equally low-D but less responsive, so it needs k=200 → 49×. At the +2% target
`cifar10_raw` (D=3072) gets **no deployable speedup at all (1×)** — demanding
cycle quality at very high D erases the win.

---

## 3. H-K1 — constant-k on responsive data holds at scale ✓

All five responsive sources hold **k=200 overhead flat across N** (all
~+1–2%, no upward drift from N=2000 to 10000) with **k=400 floors of
+0.3–1.7%**. Fitted exponents `p ≈ 0.0–0.19`. The constant-k claim now holds
across **D = 3 → 3072** and five very different domains (synthetic physics,
tabular, histopathology, clothing, natural images). feynman (D=3) is nearly
budget-insensitive (`p=0.00`, +1% at every k).

---

## 4. H-K3 — speed tracks D, not quality: confirmed (ρ = −0.90) — the headline

**At a fixed budget k**, speedup among responsive datasets falls monotonically
with D across a 1000× range (shown at k=50; see below — the ranking is
k-invariant):

| dataset | D | strain | cycle time | deployable speedup @k=50 |
|---|--:|--:|--:|--:|
| feynman_iii_15_12 | 3 | 0.69 | 145.6s | **197×** |
| california_housing | 8 | 0.65 | 147.3s | **198×** |
| patchcamelyon | 512 | 0.31 | 148.7s | 25× |
| fashion_mnist_full | 784 | 0.37 | 144.9s | 16× |
| cifar10_raw | 3072 | 0.38 | 158.7s | **5×** |

- `Spearman(deployable_speedup, D) = −0.90` (p=0.037) — and **identical at
  k = 50, 100, 200, 400**. Because `deployable(k) = cycle_time/(b·k)`, changing
  k rescales every dataset by the same 1/k factor, so the ranking (and ρ) cannot
  depend on the budget chosen. The k=50 row is therefore not a cherry-pick.
- `Spearman(deployable_speedup, strain) = +0.50` (p=0.39, **not** significant —
  a confound: the two low-D sets happen to be high-strain).

**Scope of this claim (important).** This is speed *per unit budget*. It does
**not** say a dataset achieves 197× *at good quality* — that requires pairing k
with the quality it delivers (§2b). At a fixed *quality target*, both axes
matter: `k*` (set by strain) × per-update cost (set by D).

**Mechanism, airtight.** The lazy budget kernel recomputes each pair distance
O(D); cycle looks it up O(1) from its prebuilt matrix. So per update, budget
is ~D× more expensive — the price of the O(N·D) memory that lets it exceed
cycle's O(N²) ceiling. Corroboration: **cycle time is D-independent**
(145–159s from D=3 to 3072), because cycle is dominated by its N²/2 pair
*count* + O(N²) scoring, not the O(N²·D) matrix build. All the D-dependence
of speedup therefore lives on the lazy-budget side. Plot:
`results/experiment_k/_plots_combined/speedup_vs_D.{png,pdf}`.

**Consequence — quality and speed are governed by different things:** quality at
a given k by `strain_top2` (regime), speed per unit budget by D. The ideal
fast-mode target is **low-D + responsive**, and the *deliverable* win at
near-cycle quality (§2b) is **197× for feynman (D=3, k=50)** and **49× for
`california_housing` (real, D=8, k=200)** — vs **1× for `cifar10_raw`
(D=3072)**, where demanding +1% quality erases the speedup entirely.

---

## 5. The measurement now hides the speedup — sampled stress is necessary

`SGDMDS._safe_stress` builds the full N×N matrix purely to *measure* stress
(fit `budget_time = a + b·k`; the k-independent `a` scales exactly as N²). A
deployment using sampled stress skips it. The gap this opens is largest
exactly where the fast mode is best — **low D**:

- `california_housing`: **measured 4.4× vs deployable ~200×** — the O(N²)
  scoring is ~97% of the measured budget time, hiding a ~40× understatement.
- `cifar10_raw` (D=3072): measured speedup even drops **below 1×** at k≥200
  (the fast mode looks *slower* than cycle) while deployable is still 5× — the
  scoring pass entirely masks the win.

This moves the sampled-stress estimator from "nice next step" to **necessary**:
it (a) reveals the true speedup and (b) unlocks N>20k (where `_safe_stress`
returns −1.0).

---

## 5b. The strain → budget link, confirmed at ρ = −1.00 (the cleanest result here)

J established that intrinsic dimensionality (`strain_top2`) governs
budget-responsiveness, using the overhead-vs-k **steepness** `s` across 18
mostly-small datasets (ρ=+0.75). K re-tests that link at **N=10,000**, on **8
independent sources**, across a **1000× span of D** — and it holds essentially
perfectly, with a better metric.

**`k*` = the budget needed to reach a fixed quality target** (log-log
interpolated), at N=10,000:

| dataset | strain | **k\*(+5%)** | k\*(+2%) | floor @k=400 | steepness s |
|---|--:|--:|--:|--:|--:|
| feynman_iii_15_12 | 0.687 | **50** | 50 | +1.0% | 0.31 |
| california_housing | 0.648 | 84 | 171 | +0.8% | 1.25 |
| cifar10_raw | 0.379 | 94 | 147 | +0.3% | 2.00 |
| fashion_mnist_full | 0.369 | 98 | 148 | +0.3% | 1.83 |
| patchcamelyon | 0.314 | 107 | 160 | +0.5% | 1.65 |
| mnist_digits | 0.115 | 239 | 370 | +1.7% | 1.19 |
| ag_news | 0.075 | 352 | **never** | +4.3% | 0.40 |
| tiny_imagenet | 0.072 | **380** | **never** | +4.8% | 0.37 |

- **`Spearman(strain, k*(+5%)) = −1.00` (p<0.001)** — a *flawless* monotone
  ordering across all 8 datasets and four unrelated domains (synthetic physics,
  tabular, histopathology, natural images, text). Hard-middle needs **~7× more
  budget** than the most responsive source (380 vs 50); the two lowest-strain
  sets **never reach +2%** at any tested budget.
- `Spearman(strain, floor@k400) = −0.60` (p=0.12) — same direction, weaker:
  responsive floors at +0.3–1.0%, hard-middle at +4.3–4.8%.

### Steepness is ambiguous — prefer `k*` (methodological refinement)

`Spearman(strain, s) = +0.19` (p=0.65) — apparently *no* link. Excluding
feynman it jumps to **+0.79 (p=0.036)**. The reason: **a flat curve has two
opposite meanings.**

- `feynman`: s=0.31 (flat) but floor **+1.0%** — flat because it is *already at
  the floor at k=50*. Flat for a **good** reason.
- `ag_news`: s=0.40 (flat) but floor **+4.3%** — flat because it is *stuck above
  cycle*. Flat for a **bad** reason.

Steepness conflates "nothing left to gain" with "nothing to be gained". J's
classifier already guarded against this by using **both** `s` and `floor`; K
shows why that was necessary, and that **`k*` is the better single metric** — it
is unambiguous (feynman's k*=50 correctly ranks it best) and it is the number a
user actually wants ("what budget do I set?"). Recommend `k*` over `s` in any
future analysis or auto-gating heuristic.

### The two axes are cleanly separated

The strain→budget link is **independent of D**: `cifar10_raw` (D=3072) and
`california_housing` (D=8) have near-identical `k*` (94 vs 84) because their
*strains* are similar-ish and their *dimensions* are not. Sharpest statement of
the K result:

> **`strain_top2` sets the budget you need; D sets what each unit of budget
> costs.**

Plot: `results/experiment_k/_plots_combined/kstar_vs_strain.{png,pdf}`.

## 6. H-K4 — hard-middle: confirmed, with a sharper statement of the regime

| regime | k=200 overhead flat in N? | floor at k=400 |
|---|---|---|
| responsive (feynman…cifar10_raw) | yes (~+1%) | **+0.3–1.7%** |
| hard-middle (ag_news, tiny_imagenet) | yes (~+8–9%) | **+4.3–4.8%** |
| mnist_digits (boundary, strain 0.12) | yes (~+7%) | +1.7% |

The refinement: **"constant-k" (overhead invariant to N at fixed k) is
*universal* — it holds for hard-middle too.** What the regime controls is not
the N-scaling but the **achievable floor**: responsive bottoms out at ~+1%,
hard-middle plateaus at ~+4–5% and never reaches cycle quality at any practical
k. This is sharper than J's "budget barely helps."

`mnist_digits` (digit MNIST) is the instructive boundary case: strain 0.12
(nominally hard-middle) but its floor (+1.7% at k=400) is far closer to
responsive than ag_news/tiny_imagenet (+4–5%) — digit MNIST is genuinely more
2D-embeddable than text/imagenet CLIP embeddings, though less than Fashion
(0.37).

---

## 7. Conclusions

1. **Constant-k is confirmed at scale and across domains** (D=3→3072, five
   responsive sources): a fixed k (≈200 for +1%) holds quality flat as N grows.
2. **Speed and quality are governed by different things, and both set the
   deliverable win.** At a fixed budget, speedup ← D (lazy O(D) per-update cost,
   ρ=−0.90, k-invariant); quality at that budget ← `strain_top2`. At a fixed
   *quality target* the two multiply: `k*`(strain) × 1/D. Deliverable speedup at
   ≈+1–2% quality spans **197× (D=3, k=50)** → **49× (D=8, k=200)** → **4–6×
   (D=512–784)** → **1× (D=3072)**. The PR should recommend the fast mode most
   strongly for **low-D responsive** data, and note it buys nothing at
   near-cycle quality once D is very large.
3. **The measurement is the bottleneck.** Exact-stress scoring is O(N²): it
   understates the deployable speedup by up to ~40× (low D) and is why N>20k is
   unmeasurable. The **sampled-stress estimator is now the top-priority build**.
4. **Regime controls the floor, not the N-scaling.** N-invariance at fixed k is
   universal; responsive floors at ~+1%, hard-middle at ~+4–9%.

---

## 8. Caveats / open items

- **N capped at 10k by RAM**, not by the fast mode. 15k/20k need ≥16 GB free;
  the sampled-stress estimator removes the O(N²) baseline/scoring wall.
- **Deployable speedup is an estimate** (SGD-work term of the `a + b·k` fit);
  at low D the O(N²) intercept dominates, so the exact multiplier (≈200×) is
  fit-sensitive, though the order of magnitude is robust (cycle does ~1000× more
  raw work than budget SGD at D=3).
- **feynman is synthetic** (an analytic physics manifold) — its ~200×/constant-k
  result is a clean best case; `california_housing` (real, D=8, ~198×) is the
  convincing real-data instance.
- `cifar10_raw` is **raw pixels** (D=3072); the separate `cifar10` (D=1024) is
  embeddings — different inputs.
- **k=50 is noisy** on responsive high-D data; the reliable knob starts at
  k=100 (k=200 for ~+1%).

Full per-N/per-k tables and the timing decomposition are reproducible from
`results/experiment_k/<name>/results.csv`; plots in
`results/experiment_k/_plots_combined/` (`speedup_vs_D`, `combined_overhead_vs_k`,
`combined_overhead_vs_N_fixed_k`, `combined_kstar_vs_N`).
