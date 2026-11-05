# Experiments J + K — combined spec: budget scaling of the fast mode across data (J) and scale (K)

This document unifies the designs of **Experiment J** (breadth: all 18
benchmark datasets, mechanism) and **Experiment K** (scale: 8 large sources up
to N=10,000, wall-time). The two experiments run the *same core protocol* on
different dataset collections with different emphases; together they
characterise the budgeted-random fast mode (`sampling_strategy='random'`,
`n_updates_per_epoch = k·N`) along **three axes: data character (strain), data
size (N), and feature dimension (D)**.

The original per-experiment documents remain authoritative for their details:
[EXPERIMENT_J_SPEC.md](EXPERIMENT_J_SPEC.md) ·
[EXPERIMENT_J_RESULTS.md](EXPERIMENT_J_RESULTS.md) ·
[EXPERIMENT_K_SPEC.md](EXPERIMENT_K_SPEC.md) ·
[EXPERIMENT_K_RESULTS.md](EXPERIMENT_K_RESULTS.md).
Combined conclusions: [EXPERIMENT_J_K_RESULTS.md](EXPERIMENT_J_K_RESULTS.md).

---

## 1. The question

Full-cycle SGD-MDS updates all `N(N−1)/2` pairs per epoch — O(N²). The fast
mode does `k·N` uniformly random pair updates per epoch — O(k·N). The joint
J+K question:

> **When, and by how much, can the per-epoch budget be cut from N(N−1)/2 to
> k·N without materially hurting the embedding — and what does the cut buy in
> wall time?**

J answers the *when* (which datasets respond, and why — intrinsic
dimensionality). K answers the *at scale* and *how much time* (does the answer
survive N=10,000, and what governs the speedup — feature dimension D).

## 2. Shared protocol (identical in both experiments)

Both arms run the same `SGDMDS` estimator on the same data, minimising the same
raw stress `σ = ½·Σ_{i<j}(‖x_i−x_j‖ − δ_ij)²`, differing **only** in the
per-epoch sampling budget:

| Parameter | Value (both arms, both experiments) |
|---|---|
| `n_components` | 2 |
| `metric` | True |
| `n_init` | 1 |
| `learning_rate_init` | 0.5 |
| `scheduler` | `"hybrid"`, `scheduler_switch_ratio=0.5` |
| `epsilon` | 0.001 |
| `n_jobs` | 1 (single-threaded; wall-times comparable) |

- **Baseline (cycle):** `dissimilarity='euclidean'` — full N×N distance matrix,
  deterministic pass over all `N(N−1)/2` pairs per epoch.
- **Fast (budget):** `dissimilarity='lazy'`, `sampling_strategy='random'`,
  `n_updates_per_epoch = k·N` — O(N·D) memory, distances recomputed per update.
- **Metric:** the estimator's own `stress_` (exact raw stress over all pairs)
  for both arms → the overhead ratio is apples-to-apples.

**Paired, randomized replication** (the significance argument):

- `N_REPEATS = 5` seeds (`SEED_BASE=42`, seeds 42–46).
- Per `(N, seed)`: a fresh seeded row-subsample
  (`RandomState(seed).choice(N_full, N, replace=False)`); the **same subsample
  and the same `random_state`** are given to the cycle baseline and to every
  budget k → the overhead is a *paired* ratio, cancelling subsample and init
  variance.
- `overhead(N, k) = median over seeds of (budget_stress / cycle_stress)`,
  computed per `max_iter`.

**Intrinsic-dimensionality measurement** (identical code in both:
`run_experiment_j.intrinsic_dim_measures`, imported by K's runner): computed
once per dataset on a fixed seeded 800-point subsample (`ID_SEED=0`) of the
**standardized** data (`load_local_dataset` applies `StandardScaler`):

- `strain_top2_frac` — fraction of classical-MDS eigen-energy in the top 2
  dims: build `B = −½·J·D²·J` (double-centred squared-distance matrix),
  eigendecompose, `(λ1+λ2)/Σλ_positive`. = "2D-embeddability". **The primary
  regime predictor.** Cheap (~one 800² eigensolve) and independent of the
  performance sweep, so the strain↔behaviour correlations are not circular.
- `pca_participation_ratio` `(Σλ)²/Σλ²`, `pca_n90/n95` — feature-side
  effective dimensionality (secondary/confirmatory).

Both runners are **skip-if-exists / resumable**; K additionally checkpoints
`results.csv` after every completed rung and resumes per-rung.

## 3. Where J and K differ

| | **J** (breadth / mechanism) | **K** (scale / wall-time) |
|---|---|---|
| datasets | all 18 benchmark datasets (N=400–5750, D=24–1617) | 8 large sources (N_full=20,640–100,000, D=3–3072), subsampled |
| N-ladder | `{400,600,900,1350,2000,3000,4000,5000}` ≤ full N, + full N | `{2000,4000,7000,10000}` (10k-capped by RAM, §7) |
| budget grid k | `{10, 25, 50, 100, 200}` (filtered `k < (N−1)/2`) | `{50, 100, 200, 400}` (same filter) |
| `max_iter` arms | **{15, 30}** — probes the budget × iteration interaction (H-J5) | **{30}** only (iteration lever not re-tested at scale) |
| timing analysis | none (stress only) | **yes** — measured speedup + deployable decomposition (§5) |
| total fits | 5,340 | 800 |
| emphasis | which datasets respond & *why* (strain, regimes, collapse law, iteration penalty) | does it survive N=10k; what governs speed (D); operating points |

The k-grids overlap at `{50, 100, 200}`, and `coil20` was re-run under K's
runner as a consistency check (reproduced J's numbers exactly:
k50=1.079×/k100=1.035×/k200=1.018×).

## 4. Dataset inventory (26 sources total)

### J — the 18-dataset collection (at each dataset's own full N)

`strain` = `strain_top2_frac`, standardized data. Regimes (corrected taxonomy,
J results §4): **R** responsive (strain ≥ ~0.22), **H** hard-middle
(~0.05–0.21), **D°** degenerate (< ~0.05).

| dataset | N | D | strain | pca_PR | regime |
|---|--:|--:|--:|--:|:--:|
| orl | 400 | 396 | 0.901 | 1.3 | (small-N artifact) |
| svhn | 732 | 1024 | 0.631 | 3.0 | R |
| har | 735 | 561 | 0.573 | 3.7 | R |
| cifar10 (embeddings) | 3250 | 1024 | 0.402 | 9.2 | R |
| fashion_mnist (3k subset) | 3000 | 784 | 0.378 | 11.6 | R |
| seismic | 646 | 24 | 0.359 | 9.0 | R |
| coil20 | 1440 | 400 | 0.342 | 13.2 | R |
| fmd | 997 | 1536 | 0.221 | 23.5 | R |
| spambase | 4601 | 57 | 0.208 | 23.6 | H |
| bank | 2059 | 63 | 0.143 | 31.3 | H |
| epileptic | 5750 | 178 | 0.139 | 26.9 | H |
| secom | 1567 | 590 | 0.107 | 61.0 | H |
| hiva | 3076 | 1617 | 0.085 | 93.9 | H |
| hatespeech | 3221 | 100 | 0.045 | 84.7 | H |
| cnae9 | 1080 | 856 | 0.036 | 198.4 | D° |
| sms | 835 | 500 | 0.026 | 275.0 | D° |
| sentiment | 2748 | 200 | 0.024 | 156.8 | D° |
| imdb | 3250 | 700 | 0.015 | 355.2 | D° |

### K — the 8 large sources (all tested on the shared ladder up to N=10,000)

| dataset | N_full | D | strain | pca_PR | reg | role |
|---|--:|--:|--:|--:|:--:|---|
| feynman_iii_15_12 | 100,000 | 3 | 0.687 | 3.0 | R | D-axis: extreme low-D (synthetic physics) |
| california_housing | 20,640 | 8 | 0.648 | 3.8 | R | D-axis: low-D, **real** |
| ag_news | 100,000 | 384 | 0.075 | 102.4 | H | falsification: text embeddings |
| patchcamelyon | 100,000 | 512 | 0.314 | 13.9 | R | D-axis: mid-D histopathology |
| tiny_imagenet | 100,000 | 512 | 0.072 | 107.3 | H | falsification: image embeddings |
| mnist_digits | 70,000 | 784 | 0.115 | 55.1 | H | boundary case (digit MNIST) |
| fashion_mnist_full | 70,000 | 784 | 0.369 | 12.1 | R | D-axis anchor; ties K to J's fashion subset |
| cifar10_raw | 50,000 | 3072 | 0.379 | 10.6 | R | D-axis: very high-D **raw pixels** |

Provenance and preparation: see K spec §2b (OpenML fetch for
`fashion_mnist_full`; relative symlinks into the bulk collections for the other
seven, via `prep_shortlist_datasets.py`). Caveats to carry: the `hf_*` sources
are pre-computed learned embeddings (text ones L2-normalised); `cifar10_raw`
(pixels, D=3072) ≠ J's `cifar10` (embeddings, D=1024); `feynman` is synthetic.

## 5. Derived quantities

Quality (both experiments):

- **overhead**`(N,k)` — `median_seed(budget_stress / cycle_stress)`; 1.00 =
  matches full-cycle.
- **`s` (steepness)** — fit `overhead − 1 ∝ k^(−s)` at a fixed N. High = budget
  is a powerful lever. **Known ambiguity (K finding): a flat curve can mean
  "already at the floor" (good) or "stuck above cycle" (bad) — never interpret
  `s` without `floor`.**
- **`floor`** — min overhead at the largest tested k. The regime-controlled
  quantity.
- **`k*(target)`** — log-log-interpolated smallest k reaching a target overhead
  (e.g. ≤1.05×). **The preferred single metric** (unambiguous; it is the number
  a user sets).
- **`p`** — slope of `log k*` vs `log N` (`k* ∝ N^p`): p=0 → constant budget
  per point at any N; p=1 → budget ∝ N (no asymptotic win). Reliable **only**
  where the curve actually crosses the target with ≥3 N rungs (J results §5).
- **iteration penalty** (J only) — `stress(max_iter=15)/stress(max_iter=30)`
  per (N, k); probes whether the budget and iteration levers compose.
- **collapse test** (J only) — pooled `log(excess)` regressed on `log k`
  (absolute-k law) vs `log 2k/(N−1)` (coverage-fraction law); which R² wins.

Speed (K only):

- **measured speedup** — `median_seed(cycle_time / budget_time)`. A direct
  measurement, but **contaminated** by the O(N²) exact-stress scoring pass
  (`SGDMDS._safe_stress` builds the full N×N matrix in both arms purely to
  report `stress_`).
- **deployable speedup** — fit `budget_time = a + b·k` across the k grid at
  fixed N; `a` is the k-independent scoring cost (verified `a/N² ≈ const ≈
  360 ns·N²` across all rungs — the O(N²) signature), `b·k` the SGD work;
  deployable = `cycle_time / (b·k)`. **This is a model-based estimate, not a
  measurement** — see the epistemic-status note in
  [EXPERIMENT_J_K_RESULTS.md](EXPERIMENT_J_K_RESULTS.md) §8. The *exact*
  companion quantity is the update-count ratio `(N−1)/(2k)`, which is
  arithmetic, not inference.

## 6. Hypotheses (with outcome pointers)

From J ([spec](EXPERIMENT_J_SPEC.md)):

- **H-J1** strain → steeper curve / smaller p. → *Partially confirmed;
  reframed*: strain → steepness ρ=+0.75, but `p` unreliable outside the
  responsive regime (J results §3, §5).
- **H-J2** steep curve ⇒ flat k*. → *Confirmed on the reliable subset.*
- **H-J3** which law (absolute-k vs coverage) → *absolute-k wins 14/18* (§7).
- **H-J4** predictability of k from (N, strain) → *usable heuristic; see
  combined results §3.*
- **H-J5** budget×iteration interaction largest on responsive data →
  *Confirmed, ρ=+0.95 — the two levers are mutually exclusive* (J results §6).

From K ([spec](EXPERIMENT_K_SPEC.md)):

- **H-K1** constant-k survives N=10k on responsive data → *Confirmed* (all 5
  responsive sources, p≈0.0–0.19).
- **H-K2** speedup grows with N → *Confirmed* (measured 1.2×→3.2× over
  N=2000→10000 at k=50).
- **H-K3** speed tracks D, not strain → *Confirmed*, Spearman(speedup, D) =
  −0.90, k-invariant; cycle time D-independent.
- **H-K4** regime behaviour generalises to large sources → *Confirmed with
  refinement*: N-invariance at fixed k is universal; the regime controls the
  achievable **floor** (responsive ~+1%, hard-middle +4–9%).

## 7. Outputs & where everything lives

| artefact | J | K |
|---|---|---|
| raw rows | `results/experiment_j/<ds>/results.csv` + `meta.json` | `results/experiment_k/<ds>/results.csv` + `meta.json` |
| per-dataset plots | `overhead_vs_k`, `kstar_vs_N`, `iter_penalty_vs_k` | `overhead_vs_k`, `kstar_vs_N`, `overhead_vs_N_at_fixed_k` |
| cross-dataset | `master_table.csv`, `steepness_vs_intrinsic_dim`, `iter_penalty_vs_intrinsic_dim`, `collapse_test`, `regime_map`, … | `speedup_vs_D`, `kstar_vs_strain`, combined grids |
| runner | `run_experiment_j.py` | `run_experiment_k.py` |
| plotter | `plot_experiment_j.py` | `plot_experiment_k.py` |

The two headline cross-dataset figures of the combined story:

- `results/experiment_k/_plots_combined/kstar_vs_strain` — **strain sets how
  much budget you need** (ρ=−1.00 over K's 8; generalises J's ρ=+0.75).
- `results/experiment_k/_plots_combined/speedup_vs_D` — **D sets what each
  unit of budget costs** (ρ=−0.90; deployable vs measured gap = the O(N²)
  scoring contamination).

## 8. Cost / memory constraints (why the ladders stop where they do)

The **cycle baseline** is the binding constraint, twice over:

1. **RAM**: euclidean mode materialises the N×N matrix **plus** the explicit
   N²/2 pair list and per-pair target/weight arrays — ~28 N² bytes peak.
   N=10,000 → ~3.6 GB (safe on 16 GB); N=15,000 → ~8.1 GB (OOM-killed);
   N=20,000 → ~14 GB. Hence K's 10k cap on a 16 GB machine.
2. **Scoring**: `_safe_stress` is O(N²) and hard-returns −1.0 above N=20,000
   (`STRESS_CAP`) — above that, overhead is unmeasurable *regardless* of RAM.

Both constraints are removed by the planned **sampled-stress estimator** (top
priority after K): an unbiased random-pair estimate of stress with O(sample)
memory would (a) uncap N and (b) stop the scoring pass from contaminating
timing measurements.
