# Experiment J — Results & conclusions

Budget-scaling behaviour of the fast mode (`sampling_strategy='random'`,
`n_updates_per_epoch = k·N`) across **all 18 datasets**, and its mechanistic
link to intrinsic dimensionality. Generalises Experiment I-scaling
(single-source N-ladder) to the full collection and adds a per-dataset
intrinsic-dimensionality measurement so the scaling behaviour can be
*explained*, not just reported.

Runner: [benchmarks/run_experiment_j.py](benchmarks/run_experiment_j.py) ·
analysis/plots: [benchmarks/plot_experiment_j.py](benchmarks/plot_experiment_j.py) ·
spec + hypotheses: [EXPERIMENT_J_SPEC.md](EXPERIMENT_J_SPEC.md) ·
master table: `results/experiment_j/_plots_combined/master_table.csv`.

---

## 1. What was run

Per dataset: subsampled to a geometric N-ladder (shared rungs
`{400,600,900,1350,2000,3000,4000,5000}` ≤ full N + full N), **5 random
seeds** per point, `max_iter ∈ {15, 30}`. At each `(N, seed)`: SGD-cycle
baseline + SGD-random at `k·N` for `k ∈ {10,25,50,100,200}` (filtered to
`k < (N−1)/2`). Overhead`(N,k) = median_seed(random_stress / cycle_stress)`.
Every reported number is a **median over 5 randomized repeats** (fresh
subsample + fresh SGD seed each). Intrinsic dimensionality measured once per
dataset on a fixed 800-point subsample.

**Definitions used below**
- `strain_top2` — fraction of classical-MDS (double-centred squared-distance)
  eigen-energy in the top 2 dims = "how well the data embeds in 2D." High =
  easy. The primary intrinsic-dimensionality measure.
- `pca_PR` — PCA participation ratio `(Σλ)²/Σλ²` = effective number of feature
  dimensions. Inversely related to `strain_top2`.
- `s` (steepness) — fit of `overhead − 1 ∝ k^(−s)`: how fast overhead falls
  as budget rises. High = budget is a powerful lever.
- `floor` — min overhead reached at the largest tested budget (k=200) at full
  N. Near 1 = budget reaches cycle; >1.05 = saturates above cycle.
- `p` — exponent of `k* ∝ N^p` (budget needed to hold a target overhead vs N).

---

## 2. Master table (sorted by 2D-embeddability)

| dataset | N | D | strain_top2 | pca_PR | rungs | s | floor | p(1.10) | iter_pen_peak | regime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| imdb | 3250 | 700 | 0.015 | 355.2 | 7 | 0.67 | 1.003 | -0.00 | 1.01 | degenerate |
| sentiment | 2748 | 200 | 0.024 | 156.8 | 6 | 0.41 | 1.007 | -0.00 | 1.01 | degenerate |
| sms | 835 | 500 | 0.026 | 275.0 | 3 | 0.62 | 1.003 | 0.00 | 1.01 | degenerate |
| cnae9 | 1080 | 856 | 0.036 | 198.4 | 4 | 0.49 | 1.005 | -0.00 | 1.01 | degenerate |
| hatespeech | 3221 | 100 | 0.045 | 84.7 | 7 | 0.14 | 1.038 | -0.00 | 1.01 | hard-middle |
| hiva | 3076 | 1617 | 0.085 | 93.9 | 6 | 0.07 | 1.088 | 0.89¹ | 1.01 | hard-middle |
| secom | 1567 | 590 | 0.107 | 61.0 | 5 | 0.27 | 1.044 | 0.76¹ | 1.04 | hard-middle |
| epileptic | 5750 | 178 | 0.139 | 26.9 | 9 | 0.33 | 1.095 | 0.48 | 1.06 | hard-middle |
| bank | 2059 | 63 | 0.143 | 31.3 | 5 | 0.75 | 1.033 | 0.34 | 1.09 | hard-middle |
| spambase | 4601 | 57 | 0.208 | 23.6 | 8 | 0.50 | 1.064 | 0.26 | 1.11 | hard-middle |
| fmd | 997 | 1536 | 0.221 | 23.5 | 4 | 1.12 | 1.014 | 0.16 | 1.20 | responsive |
| coil20 | 1440 | 400 | 0.342 | 13.2 | 5 | 1.20 | 1.018 | 0.11 | 1.25 | responsive |
| seismic | 646 | 24 | 0.359 | 9.0 | 3 | 1.49 | 1.013 | 0.33 | 1.23 | responsive |
| fashion_mnist | 3000 | 784 | 0.378 | 11.6 | 6 | 1.45 | 1.014 | 0.06 | 1.32 | responsive |
| cifar10 | 3250 | 1024 | 0.402 | 9.2 | 7 | 1.65 | 1.007 | 0.16 | 1.48 | responsive |
| har | 735 | 561 | 0.573 | 3.7 | 3 | 0.80 | 1.033 | 1.45¹ | 1.86 | responsive |
| svhn | 732 | 1024 | 0.631 | 3.0 | 3 | 1.12 | 1.011 | 0.15 | 2.61 | responsive |
| orl | 400 | 396 | 0.901 | 1.3 | 1 | 2.48 | 1.057 | —² | 7.52 | (artifact) |

¹ `p` unreliable — flat/near-target curve makes `k*` unstable (see §5).
² orl: single N rung (full N=400), `p` undefined; a small-N artifact.

Regime labels here are the *corrected* taxonomy (§4), not the runner's
mechanical `classify()` (which is confused by the N-confounded `floor_fullN`
and mislabels some rows, e.g. bank/har as "intermediate", orl as
"saturating").

---

## 3. Headline: the two-regime hypothesis was wrong; it's better described by two axes that both rise with 2D-embeddability

The clean mechanistic result is about **steepness `s`**, not the exponent `p`:

- **Steepness is set by intrinsic dimensionality.** `strain_top2 → s`:
  Spearman **ρ = +0.75** (p<0.001); `pca_PR → s`: **ρ = −0.74**. More
  2D-embeddable data ⇒ steeper overhead-vs-`k` curve ⇒ budget is a more
  powerful lever. (Plot: `steepness_vs_intrinsic_dim`.)
- **But the overhead gap at a fixed small budget also *grows* with
  2D-embeddability.** At a controlled N=600, `strain → overhead@k=25`:
  **ρ = +0.72**. Counterintuitive: the more 2D structure a dataset has, the
  *worse* a small random sample does relative to cycle — because there is
  more resolvable structure for full coverage to exploit that a subsample
  misses.

These offset, which is why "high strain = well-behaved" fails as stated.
Resolution: high-strain datasets have **big but recoverable** gaps (steep
curve closes them fast); low-strain datasets have **small** gaps (nothing to
resolve). See the two-dataset contrast in
`../benchmarks/results/experiment_i_scaling/_plots_combined/combined_overhead_vs_k.png`
(fashion_mnist steep fan vs hiva flat band).

---

## 4. The corrected taxonomy — three regimes

Overhead above SGD-cycle at each dataset's full N (median of 5 seeds,
max_iter=30):

| Regime | strain_top2 | examples | @k=50 | @k=100 | @k=200 | verdict |
|---|---|---|---|---|---|---|
| **Responsive** | high (0.22–0.63) | cifar10, fashion, coil20, fmd, svhn | +3…+18% | +2…+4% | **+1…+2%** | fast mode reaches cycle at fixed k; keep max_iter=30 |
| **Hard middle** | mid (0.09–0.21) | epileptic, hiva, spambase, secom, bank | +9…+25% | +8…+12% | **+3…+10%** | plateaus above cycle; budget barely helps |
| **Degenerate** | very low (<0.05) | imdb, sentiment, sms, cnae9 | <+1% | <+1% | **<+1%** | matches cycle trivially, but 2D captures <4% variance → embedding poor regardless |

Concrete per-dataset numbers (full N, max_iter=30):

- **Responsive** — cifar10 +17.7%/+2.6%/+0.7% at k=50/100/200; fashion
  +17.7%/+4.1%/+1.4%; coil20 +7.9%/+3.5%/+1.8%; fmd +3.0%/+2.0%/+1.4%.
  The knob has real leverage — but note **k=50 is not enough at large N**
  (cifar/fashion still +17.7% at k=50); use k≈100–200 to be within a few %.
- **Hard middle** — hiva +9.8%/+9.5%/+8.8% (barely moves: +11.2% even at
  k=10); epileptic +13.0%/+11.2%/+9.5%; spambase +17.7%/+12.4%/+6.4%;
  secom +9.3%/+8.3%/+4.4%. This is where the fast mode genuinely
  underperforms — added budget does not close the gap within k≤200.
- **Degenerate** — imdb +0.9%/+0.5%/+0.3%; sentiment +1.1%/+0.8%/+0.7%;
  cnae9 +0.8%/+0.6%/+0.5%. Overhead ≈ 1 at any budget, but `strain_top2`
  ≤ 0.045 means a 2D embedding is a poor summary of these datasets.

**"A few percent" pinned down:** at the largest tested budget (k=200) the
hard-middle floor is **+3.3% to +9.5%** above cycle, worst on **hiva (+8.8%)**
and **epileptic (+9.5%)**; responsive datasets reach **+0.7% to +1.8%**.

---

## 5. Why `p` is an unreliable summary

`k* ∝ N^p` is only well-defined where the overhead-vs-k curve actually
crosses the target and there are enough N rungs. That excludes two of the
three regimes: degenerate datasets (`overhead ≈ 1` everywhere → `p ≈ 0`
trivially) and hard-middle/saturating datasets (flat curve → `k*` swings
wildly). So the `p`-based hypotheses came out muddied — `strain → p` was weak
and wrong-signed (ρ=+0.46), and `s → p` was null (ρ=−0.04). Only the
**responsive datasets with ≥5 rungs and a steep curve give a trustworthy p**:

| dataset | N | strain | rungs | s | p(1.10) |
|---|---:|---:|---:|---:|---:|
| cifar10 | 3250 | 0.402 | 7 | 1.65 | 0.16 |
| coil20 | 1440 | 0.342 | 5 | 1.20 | 0.11 |
| fashion_mnist | 3000 | 0.378 | 6 | 1.45 | 0.06 |

**Median reliable `p` = 0.11** → for the regime that matters, the budget is
**near-constant in N** (10× the data needs ≈ 10^0.11 ≈ 1.3× the k). This is
the generalization of I-scaling's `fashion p≈0.06`. The confounded
cross-dataset "p ≈ 0.7" estimate from before I-scaling is retracted.

---

## 6. The cleanest result — the two speed levers are mutually exclusive

The iteration penalty (`stress(max_iter=15)/stress(30)`, worst over mid-k)
tracks 2D-embeddability almost perfectly:

- `strain_top2 → iter_penalty_peak`: **ρ = +0.95** (p<0.001; still **+0.92**
  dropping the small-N outliers orl/svhn/har).
- `s → iter_penalty_peak`: **ρ = +0.83**.
- Cycle's own 15-vs-30 ratio is ~1.00 everywhere (max +2%, orl aside) —
  confirming full-cycle SGD is converged by iteration 15.

Interpretation: **the datasets where the budget knob has leverage (steep,
high-strain) are exactly the ones where cutting iterations hurts** (cifar10
peak 1.48×, fashion 1.32×, svhn 2.61×), and the datasets where cutting
iterations is free (hiva, imdb, cnae9: ~1.01×) are the ones where the budget
knob does nothing. **You essentially never get to cut both.** On responsive
datasets keep `max_iter=30`; on hard-middle/degenerate ones `max_iter=15` is
free but the budget knob adds no value. (Plot: `iter_penalty_vs_intrinsic_dim`.)

---

## 7. Which invariant controls overhead

The collapse test (pooled `log(excess)` vs `log k` vs `log f=2k/N`) favours
the **absolute-`k` law over the coverage-fraction law in 14/18 datasets**.
The 4 exceptions (hiva, har, secom, orl) are flat-curve / small-N cases where
neither law fits well (`R²` low). Takeaway: **think in `k` (updates per point
≈ 2k, independent of N), not in sampled fraction `2k/N`.** This is why a fixed
`k` mostly holds as datasets grow — consistent with the near-constant `p`.

---

## 8. Conclusions & upstream implications

1. **Whether the fast mode works is governed by intrinsic dimensionality**
   (`strain_top2`), not N alone. Three regimes: responsive (works, fixed k),
   hard-middle (saturates a few % above cycle), degenerate (matches cycle but
   the embedding is poor anyway).
2. **For the regime that matters, the budget is near-constant in N** (p≈0.11).
   Recommend a fixed `k≈100–200` for within-a-few-% quality; `k≈50` suffices
   at N ≲ 1500.
3. **Budget-cut and iteration-cut do not compose** — never advertise both.
4. **`strain_top2` is cheap to compute at fit time** (one 800-point
   double-centred-distance SVD), so the estimator could auto-gate fast-vs-cycle
   and even pick `k`. This is a concrete future API direction — a
   `fast_mode='auto'` that inspects 2D-embeddability up front.
5. **Doc guidance for the PR**: the fast mode is the recommended path on
   2D-embeddable data; on low-embeddability data warn that the embedding
   itself is questionable; on the moderate-intrinsic-dim band, prefer full
   cycle or accept the ~5–10% plateau.

## 9. Caveats

- **Small-N artifacts.** orl (N=400, 1 rung), svhn/har/seismic/sms (N<900):
  `p` and `iter_penalty` are noisy there; orl's iter penalty (7.5×) and
  `s=2.48` are not to be trusted. The strong correlations survive dropping
  them.
- **Regime boundaries are two datasets thin at the edges.** The transition
  between hard-middle and responsive (`strain_top2 ≈ 0.2`) is characterized
  by only bank/spambase/fmd; a synthetic source with dialled intrinsic
  dimensionality (via `lift_to_high_dim`) would map the boundary cleanly —
  a natural follow-up.
- **The `classify()` labels in the raw runner output are N-confounded** (they
  use `floor_fullN`); the taxonomy in §2/§4 is the corrected one.
