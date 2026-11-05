# Experiment J — Budget-scaling of k across all 18 datasets

## 1. Premise

Experiment I-scaling (two sources, subsampled to an N-ladder) showed the
budget multiplier needed to hold a fixed stress overhead scales as
`k* ∝ N^p`, and that the exponent is **bimodal**: `fashion_mnist` gave
`p ≈ 0` (a fixed k holds quality at any N), while `hiva` *saturated* (the
overhead-vs-k curve is nearly flat, so the knob has no leverage). I argued
the regime is set by **intrinsic dimensionality** but never measured it.

Experiment J generalises I-scaling to the **full 18-dataset collection** and
adds the missing mechanistic measurement: per-dataset intrinsic
dimensionality, correlated against the scaling exponent `p` and the
overhead-vs-k steepness `s`. The goal is to turn the two-regime story into a
*predictive* one — ideally, "given (N, intrinsic dim), here is the k you
need," with no sweep required.

Runner: [benchmarks/run_experiment_j.py](benchmarks/run_experiment_j.py) ·
analysis/plots: [benchmarks/plot_experiment_j.py](benchmarks/plot_experiment_j.py).

**Results & conclusions (complete): [EXPERIMENT_J_RESULTS.md](EXPERIMENT_J_RESULTS.md).**

## 2. Design (removes the size/difficulty confound)

Per dataset:

- **Intrinsic dimensionality** on a fixed seeded subsample of 800 points
  (size fixed so the estimate is N-independent and comparable across
  datasets):
  - `pca_participation_ratio` `= (Σλ)²/Σλ²` — smooth effective feature dim.
  - `pca_n90`, `pca_n95` — PCA components for 90% / 95% variance.
  - **`strain_top2_frac`** — fraction of the classical-MDS (double-centred
    squared-distance) eigen-energy in the top 2 dimensions. This is exactly
    "how much of the structure a 2D embedding can capture" — the quantity
    hypothesised to drive the regime. High = easy/2D-embeddable.
  - `strain_participation_ratio` — effective MDS dimensionality.

- **Subsample N-ladder** at shared absolute rungs
  `{400, 600, 900, 1350, 2000, 3000, 4000, 5000}` ≤ full N, plus the full N
  as top rung. Shared rungs let any common N be compared across datasets.
  At each `(N, seed)`: SGD-cycle baseline + SGD-random at `n_updates = k·N`
  for `k ∈ {10, 25, 50, 100, 200}` (filtered to `k < (N−1)/2`).
  `max_iter ∈ {15, 30}`, 5 seeds.
  `overhead(N,k) = median_seed(random_stress / cycle_stress)` (computed per
  `max_iter`; the scaling analysis uses `max_iter=30`).

Datasets with full N ≥ ~1500 (≥4 rungs) yield a per-dataset exponent `p`;
smaller datasets (orl, seismic, har, sms, svhn, fmd, cnae9) still contribute
their overhead-vs-k curve, steepness, floor, and intrinsic dim to the
cross-dataset analysis.

## 3. Derived quantities (in the analyzer)

- `k*(N)` — log-log interpolated budget hitting overhead targets
  `{1.05, 1.10, 1.15}`.
- `p` — slope of `log k*` vs `log N` (per target; needs ≥3 rungs).
- `s` — steepness of `excess ∝ k^-s`, fit at full N and at a common rung
  (N=900) for a controlled cross-dataset comparison.
- `floor` — minimum overhead reached at the largest tested k (full N): does
  the curve bottom out above 1?
- **collapse test** — pooled across all `(N,k)`, does `log(excess)` fit
  better against `log k` (→ constant-k / updates-per-point law) or against
  `log f` with `f = 2k/(N−1)` (→ linear-k / coverage law)? Per-dataset
  diagnostic of *which bounding law applies*.
- **iteration penalty** — `stress(max_iter=15) / stress(max_iter=30)` per
  `(N, k)`. For full-cycle this is ~1 (15 iters already converge); for
  budgeted random, Experiment I found a *mid-k hump* (worst at k≈25–50).
  J measures whether that hump's size correlates with regime / intrinsic
  dimensionality.
- `regime` ∈ {constant-k, intermediate, saturating} from `s` and `floor`.

## 4. Hypotheses

- **H-J1 (mechanism).** `p` decreases, and `s` increases, with
  `strain_top2_frac` (and decreases with PCA participation ratio). I.e.
  more 2D-embeddable data → steeper overhead-vs-k → constant-k regime;
  high intrinsic dim → flat curve → saturating regime.
- **H-J2 (inverse link).** `p` is inversely related to `s`: a steep curve
  means a small k-bump cancels the N-drift, so k* is flat.
- **H-J3 (which law).** For low-intrinsic-dim datasets the overhead collapses
  on absolute k (const-k law wins, `R²_k > R²_f`); for high-intrinsic-dim
  datasets it collapses on coverage f (linear-k law wins).
- **H-J4 (predictability).** `p` (or the right k for a target N) is
  predictable from `(N, strain_top2_frac)` well enough to replace the sweep
  with a heuristic.
- **H-J5 (iteration interaction).** The budget × iteration penalty (the
  mid-k `stress(15)/stress(30)` hump) is largest in the constant-k /
  steep-curve regime — i.e. exactly the datasets where the budget knob has
  leverage are the ones where you also must not cut iterations. Saturating
  datasets show ~no penalty (already plateaued at 15).

## 5. Outputs

Per dataset: `overhead_vs_k`, `kstar_vs_N` (with fitted p), in
`results/experiment_j/<dataset>/` and combined grids in `_plots_combined/`.

Per dataset also: `iter_penalty_vs_k` — `stress(15)/stress(30)` vs k, one
line per N (the mid-k hump).

Cross-dataset (`_plots_combined/`):
- `master_table.csv` — every scalar per dataset (N, D, intrinsic dim, p at
  each target, s, floor, collapse R², mid-k iteration penalty, regime).
- `p_vs_intrinsic_dim`, `steepness_vs_intrinsic_dim`, `p_vs_pca_dim`,
  `p_vs_steepness`, `regime_map`, `collapse_test`,
  `iter_penalty_vs_intrinsic_dim` — the mechanistic scatters (each annotated
  with a Spearman ρ).

## 6. Decision tree

- **H-J1 confirmed (clean p-vs-strain trend):** publish a heuristic mapping
  intrinsic dim → recommended k scaling in the upstream docs. Fast mode is
  *quantitatively* gated, not hand-waved.
- **Regimes are sharp (bimodal, not continuous):** ship a binary rule —
  compute `strain_top2_frac` cheaply at fit time; if above a threshold use a
  fixed k, else fall back to cycle.
- **No correlation:** the regime is governed by something other than
  intrinsic dimensionality; inspect outliers (high-D-but-constant-k or
  low-D-but-saturating) for the real driver.

## 7. Cost note

The expensive runs are SGD-cycle at the largest N rung for high-D datasets
(hiva D=1617, fmd D=1536, cifar10/svhn D=1024; cost ~ N²·D per epoch).
`MAX_ITER_VALUES = [15, 30]` (both arms, for H-J5), which roughly doubles the
cost — the full sweep is many hours / overnight. The runner is **resumable**
(skip-if-exists per dataset) — run high-D datasets individually, or trim
`BASE_LADDER` / `N_REPEATS` / drop `15` for a faster first pass.
