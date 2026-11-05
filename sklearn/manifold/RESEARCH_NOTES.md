# Research notes

Reference companion to [CLAUDE.md](../../CLAUDE.md): what the project has found,
which experiment found it, and where the detail lives. Open work is in
[STATUS.md](STATUS.md). The `EXPERIMENT_*_RESULTS.md` docs stay authoritative;
this file only summarises them.

**Speed figures below predate the `noexcept` fix and come from different
machines.** Stress figures don't have this problem: budgeted fits reproduced
across machines to 8.6e-12.

---

## 1. Findings

### 1.1 The primitive: budgeted uniform random sampling

`sampling_strategy='random'` + `n_updates_per_epoch` (lazy mode only):

- `"auto"` → `N(N−1)/2` (historical default; full-cycle count, random order).
- integer → exactly that many random pair draws per epoch. Set it to `k·N`.

Setting an int with `sampling_strategy != 'random'` raises `ValueError`.

### 1.2 Recommended settings

- **k = 200** → ~+1% stress vs full cycle on responsive data. The dependable
  choice.
- **k = 100** → ~+5%. Practical minimum.
- **k = 50** → unreliable at large N (seed spreads up to [+14%, +36%]). Avoid.
- Keep **`max_iter=30`** on responsive data (see 1.5).
- Gate on N: below N≈1500 per-fit overhead dominates and the knob is not faster.

### 1.3 When it works: the regime theory (Experiments I-scaling, J)

Governed by **`strain_top2`** = fraction of classical-MDS (double-centred
squared-distance) eigen-energy in the top 2 dims = "how 2D-embeddable the data
is". Measured on a fixed seeded 800-point subsample; cheap enough to compute at
fit time. All values are on standardized data.

| Regime | `strain_top2` | Behaviour | Examples |
|---|---|---|---|
| **Responsive** | ≳ 0.22 | Fixed k reaches ~+1–2% of cycle | cifar10, fashion_mnist, coil20, fmd, svhn, patchcamelyon, feynman, california_housing |
| **Hard middle** | ~0.05–0.21 | Plateaus +3–10% above cycle; budget barely helps | hiva, epileptic, spambase, secom, bank, ag_news, tiny_imagenet, mnist_digits |
| **Degenerate** | < 0.05 | Matches cycle trivially, but 2D captures <5% of structure so the embedding is poor regardless | imdb, sentiment, sms, cnae9 |

Evidence: `strain_top2` → curve steepness ρ = **+0.75** (18 datasets, J);
`strain_top2` → required budget `k*` ρ = **−0.94** (8 datasets at N=50k,
K-optimized). PCA participation ratio confirms (ρ = −0.74).

**Use `k*` (budget to hit a quality target), not curve steepness**, as the
summary metric. Steepness is ambiguous: a flat curve means either "already at
the floor" or "stuck above cycle" (K-optimized §5b).

### 1.4 What it buys: speed (K-optimized)

**Two independent axes:** `strain_top2` sets *how much budget you need*; **D**
sets *what each unit of budget costs*. At a fixed quality target they multiply.

Per-update costs, measured:

- **cycle: ~112 ns/pair, D-independent** (± 2.6 ns across D=3→3072): an O(1)
  matrix lookup, but paying cache-hostile random access plus a per-epoch
  reshuffle of the whole pair/target/weight arrays (1.25 B elements at N=50k).
- **budget: ~52 ns at D≤8, rising ~0.6 ns per dimension** to 2031 ns at D=3072.
- **Crossover ≈ D=60** (bracketed by measured D=8 and D=384; interpolated).
  Below it, a budgeted update is *cheaper* than a cycle pair-update and the
  fast mode **beats the pair-count bound** (200–214% of it).

Measured speedup at N=50,000, k=200, ≈+1% overhead: **267× (D=3) · 25× (D=512)
· 20× (D=784) · 7.1× (D=3072)**. Speedup grows **∝N** at fixed k, so high D
delays the win rather than capping it. Quoting k=50's speed next to k=200's
quality was a real error made earlier in this project.

Most of the 52 ns intercept is the GIL overhead (§1.8). Once that is fixed, the
laptop numbers predict an intercept near 17 ns, a crossover near D≈100 and about
3× larger low-D speedups. These are predictions, not measurements.

### 1.5 The two speed levers do not compose

`max_iter` 30→15 penalty is rank-correlated with budget-responsiveness at
**ρ = +0.95**: exactly the datasets where the budget knob works are those where
cutting iterations hurts (cifar10 +48%, fashion +32%). Full-cycle SGD is itself
converged by epoch 15 (ratio ~1.00).

### 1.6 The spectral baseline: Pivot MDS (Experiment L)

Standalone Pivot MDS (Brandes & Pich 2007) on all 26 datasets, as a non-SGD
reference point. **Budgeted SGD wins on quality 26/26** (Pivot MDS 1.2–18× cycle
stress, median 3.34× on the 18-set / 1.71× on the 8 large; SGD k=200 is
1.00–1.10×), even giving Pivot MDS its best k, best selection strategy, and its
stress-optimal rescale. **Pivot MDS is 7–51× faster.** A genuine Pareto
trade-off, not a knockout.

- **Mechanism:** Pivot MDS is not cheaper per unit work. It does the same
  O(k·N·D) distance work, *once* instead of once per epoch. Its dominant cost is
  **pivot selection** (33–95% of total, strictly sequential, linear in k), not
  the SVD (0.1–1.4 s regardless of D). Predicts ≈30× at mid-D, observed.
- **`strain_top2` does NOT predict spectral quality.** It governs *sampling*
  behaviour, while Pivot MDS's gap is a *strain-vs-stress* question. ag_news
  (strain 0.075) is Pivot MDS's best case; california_housing (0.648) its worst.
- **Rescaling is mandatory** for the comparison: Pivot MDS's optimal α is
  4.4–8.0, SGD's is 0.99992–1.00034 (verified). Rescale the spectral embedding,
  never SGD.

### 1.7 Scaling law

`k* ∝ N^p` with **p = 0.00–0.11** across J (N=400–5750) and K-optimized
(N=10,000–50,000). A 10× larger dataset needs ~1.3× the budget per point. J's
collapse test (14/18 datasets) favours the absolute-`k` law over the
coverage-fraction `2k/N` law: **think in updates-per-point, not sampled
fraction**.

N-invariance at fixed k is **universal** (hard-middle datasets are equally flat
in N); what the regime controls is the achievable **floor**.

### 1.8 Point blocking and SQuaD-MDS (Experiments M, N)

- **SQuaD-MDS** reaches 1.04× (small suite) / 1.10× (large suite) cycle stress at
  7–14× less time than k=200. It fails on heavy-tailed data (mnist_digits,
  california_housing). Its best configs sat at the top of the tested `n_iter`
  range, so these are bounds until the extended ladder runs.
- **Blocking** with sequential updates costs no stress and is 3–4.5× faster at
  N=50k. At k=200 it has 7% lower stress than SQuaD-MDS's best at 2.7× its time
  (11.2× with the original kernel).
- **GIL finding:** `_lazy_sgd_step` isn't `noexcept`, costing ~35 ns per update:
  3.3× on a small k=200 fit, with a bit-identical embedding. Blocking alone gives
  ~1.4× at low D and 3.1–3.7× at high D; quoting blocked times against the
  original kernel credits blocking with the fix.
- **Open hypothesis:** SQuaD-MDS's PCA init may explain why it needs fewer pairs.

---

## 2. Experiments and documents

Runners are `benchmarks/run_experiment_<x>.py`. Raw results are in
`benchmarks/results/experiment_<x>/<dataset>/{results.csv, meta.json}`;
`results/README.md` maps each paper element to its data.

| Exp | Compared | Status / doc |
|---|---|---|
| A | SGD-MDS vs SMACOF: quality and runtime | Paper §4.1 |
| B | Runtime vs N: euclidean, lazy, SMACOF | Paper Fig. 3 |
| C | Non-metric MDS quality | Not cited |
| D | SGD-cycle vs pivot-euclidean at fractional k | Not cited |
| E | Pivot scaling vs N | |
| F | Pivot k-sweep; motivated G | Pivot history (§4) |
| G | PCA pivot selection vs raw MaxMin: PCA-10 wins by 10–35%, still 5–11× worse than cycle | Pivot history (§4) |
| H | Diagnosis: ruled out LR, confirmed coverage, pivots actively hurt at equal budget | `EXPERIMENT_H_RESULTS.md` |
| I | Introduced `n_updates_per_epoch`: k ∈ {10,25,50,100,200} × max_iter ∈ {15,30}, 7 datasets | `EXPERIMENT_I_SPEC.md` |
| I-scaling | Single-source N-ladder → `k* ∝ N^p` (fashion p≈0.06; hiva saturates) | `EXPERIMENT_I_SPEC.md` §7 |
| J | All 18 datasets + intrinsic-dim measurement → regime theory; iteration-lever interaction; master table | `EXPERIMENT_J_RESULTS.md` |
| K | 8 large sources, D=3→3072, N≤10k. Quality stands; speed superseded, including its "cifar10_raw gets no win" claim | `EXPERIMENT_K_RESULTS.md` |
| **K-opt** | Same protocol, in-fit scoring removed, single-cycle baseline, per-fit resume, N to 50,000. 630 fits, 31.7 h. Current authority on speed | `EXPERIMENT_K_OPTIMIZED_RESULTS.md` |
| L | Spectral Pivot MDS on all 26 datasets, `--suite j18` / `--suite k8`. 2,400 fits | `EXPERIMENT_L_RESULTS.md` |
| M | SQuaD-MDS on all 26 datasets, `--suite j18` / `--suite k8` | `EXPERIMENT_M_SPEC.md`, `EXPERIMENT_M_RESULTS.md` |
| N | Point blocking: m ∈ {2,4,8,16} × random/partition × sequential/sum/mean at equal pair budget. Microbenchmark: `bench_blocked_kernel.py` | `EXPERIMENT_N_SPEC.md`, `EXPERIMENT_N_RESULTS.md` |

Other documents:

- `EXPERIMENT_J_SPEC.md`, `EXPERIMENT_K_SPEC.md` (includes the dataset inventory
  and provenance), `PIVOT_DIAGNOSIS_PLAN.md` (historical).
- `EXPERIMENT_J_K_SPEC.md`, `EXPERIMENT_J_K_RESULTS.md`: unified J+K narrative.
  Written before K-optimized, so its speedups are the old estimates.
- `REPORT_BUDGETED_SGD_MDS.md`: self-contained supervisor report, current as of
  K-optimized.
- `PAPER_REVISION_PLAN.md`: paper edits. Parts are outdated; see `STATUS.md` §2.
- `EXPERIMENT_O_P_SPEC.md`: proposed reporting runs for the paper. Not yet run.

---

## 3. The paper

In `SGD for MDS/` at the repo root (untracked), with `figures/` and
`arxiv_collector.py`; each paper folder has its own `references.bib`.

- `workshop/camera-ready.tex`: *"Bridging Graph Drawing and Dimensionality
  Reduction with Stochastic Stress Optimization"* (GDxDR workshop, published).
- `pacificvis/main.tex`: *"Improving Multidimensional Scaling with Stochastic
  Stress Optimization"*, the extended version, in progress. Its structure is due
  to change to the three-step storyline in CLAUDE.md §2.

### What the workshop paper claims

- SGD-MDS vs scikit-learn SMACOF on the Espadoto et al. 18-dataset collection.
- Convergence in **15–30 epochs** vs SMACOF's hundreds of iterations.
- Better final stress in **14 of 18** datasets; more consistent across seeds
  than SMACOF (which is init-sensitive).
- IMDB is the notable failure case, attributed to **high norm concentration**.
- Precomputed mode ~2× faster than SMACOF at N=3000; **lazy-mode speedup
  diminishes as D grows**.
- Hardware: Intel i7 (20 cores), 16 GB RAM, Pop!_OS 22.04.

Its limitations section named scalability, graph-drawing stress approximation and
a limited benchmark. All three are now addressed: budgeted sampling to N=50,000,
the `k·N` fast mode, and 18 → 26 datasets at D=3→3072.

### New material since the workshop paper

Already in `pacificvis/main.tex`:

1. The budgeted fast mode, `k·N` updates per epoch (§1.1).
2. The negative result: pivots are *actively* worse than uniform random at equal
   budget (§4). The analogy fails because SGD has no spectral generalization step.
3. `strain_top2` as a predictor of when subsampling works (§1.3).
4. The constant-k scaling law (§1.7).
5. Measured speedups, 7×–1070× at ≈+1% overhead, growing ∝N (§1.4).
6. The per-update cost model, which explains the workshop paper's "lazy speedup
   diminishes as D increases" (§1.4).
7. A direct test of the IMDB norm-concentration hypothesis: distance concentration
   has **no** correlation with budget-responsiveness (|ρ| ≤ 0.2, p > 0.5) while
   the spectral measures do (ρ = +0.75). Concentration is real there, but it isn't
   what governs subsampling.
8. The Pivot MDS baseline across all 26 datasets (§1.6).

Not yet in it:

9. The budget × iteration interaction (§1.5).
10. Point-blocked SGD and the SQuaD-MDS comparison (§1.8). SQuaD-MDS appears only
    as related work so far.

---

## 4. Pivot SGD-MDS: investigation closed

**Do not restart this line without reading `EXPERIMENT_H_RESULTS.md`.**

> **Not to be confused with Experiment L.** This section is about *pivot sampling
> inside SGD* (restricting gradient updates to (i, pivot) pairs): closed, and a
> failure. Experiment L is the *standalone spectral* Pivot MDS of Brandes & Pich:
> a legitimate published algorithm, run as a baseline, and it works (just 1.2–18×
> worse on stress than budgeted SGD). Same paper as inspiration, entirely
> different use.

**Goal:** cut per-epoch cost to O(N·k) by restricting updates to (i, pivot) pairs,
borrowing from Pivot MDS / Landmark MDS (Brandes & Pich 2007).

**Verdict: pivots do not work, and they are worse than the trivial alternative.**
At a *fixed update budget* `k·N`, pure random pair sampling lands within 1–14% of
full-cycle stress on every dataset, while pure pivot sampling is **5–9× worse**.
Pivots are not a useful inductive bias for SGD; they are an actively bad sampling
distribution.

The arc:

- **F** (`pivot_strategy="maxmin"`): closing the gap needs k ≈ tens of percent of
  N, not a small constant. At k=200 (14% of N) coil20's stress is still ~5× cycle.
- **G** (`pivot_strategy="maxmin_pca"`, `pivot_pca_dim=10`): PCA-space pivot
  selection is consistently better (10–35% at equal k), so the distance
  concentration hypothesis was partly right, but the gap to cycle stays 5–11× at
  small k. **Selection quality was never the bottleneck.**
- **H** (diagnosis): two candidate causes, opposite fixes. *H2 mechanical/LR*:
  **ruled out** (LR sweep flat; best-LR pivot still 5–9× cycle). *H1 structural
  coverage*: **confirmed** (hybrid α sweep monotonic from 5–9× at α=1 to
  1.01–1.14× at α=0). The decisive observation is that α=0 does the *same number
  of updates* as pure pivot, just on random pairs.
- **Why the analogy failed:** Brandes & Pich get small-k power from a *global
  spectral solve*, not from anchor structure. SGD has no such generalization
  mechanism, so pivot-restricted gradients leave non-pivot geometry unenforced.

**Kept in the tree** for reproducibility of F/G/H, **not** for the PR:
`sampling_strategy ∈ {'pivot','hybrid'}`, `pivot_strategy`, `pivot_pca_dim`,
`hybrid_alpha`, and `_pivot_mds.py`. `'hybrid'` reuses both lazy kernels
back-to-back; no third kernel was needed.

**One trap worth knowing:** classical-MDS-family embeddings fit inner products,
not distances, so they land at a stress-suboptimal *scale*. Always apply the
closed-form rescale `α = ⟨D_emb,Δ⟩/⟨D_emb,D_emb⟩` before comparing their stress
against SGD (`run_pivot_mds_benchmark` does). Skipping this once overstated the
spectral baseline's gap by up to ~6×. Scale-corrected, PivotMDS-spectral at k=50
is 1.4–3.4× cycle on 5 of 6 datasets and matches full ClassicalMDS, so the
published small-k claim replicates; the pivots were never the problem.

---

## 5. Benchmark tooling

### `bench_utils.py`

- `load_local_dataset(name)`: loads from `datasets/` or generates synthetic
  (`s_curve`, `swiss_roll`, `torus`); applies `StandardScaler`.
- `run_sgd_benchmark`, `run_random_budget_benchmark`, `run_pivot_benchmark`,
  `run_hybrid_benchmark`, `run_pivot_mds_benchmark`, `run_squad_mds_benchmark`:
  return `{"embedding", "stress", "history", "time", "n_iter", ...}`, with
  `history` as `[(cumulative_time, stress), ...]`.
- `run_pivot_mds_benchmark` applies the stress-optimal rescale and times pivot
  selection plus spectral solve; selection dominates (§1.6).
- `stress_exact_chunked(X, embedding, block_size=2048)`: exact O(N·block) scorer,
  matching `_full_stress` to <1e-10. For SGD (α is 0.9999–1.0003).
- `stress_and_optimal_scale_chunked(X, embedding, ...)`: same pass, also returns
  `(stress_scaled, stress_raw, alpha)`. For classical-MDS-family embeddings, whose
  α is 4.4–8.0. Derived from three accumulable scalars, so one blocked pass gives
  everything.
- `stress_and_optimal_scale_chunked_multi(X, embeddings, ...)`: batched over many
  embeddings of the same X. Input-distance blocks are computed once and reused,
  so scoring m embeddings costs ~1/m of m separate calls when D ≫ n_components.
- `per_point_stress`.

`_safe_stress` also skips above the module-level `EXACT_STRESS_N_CAP` (default
20,000; a safety guard the K runners raise). BLAS multithreading affects only the
matrix build and the external scorer, together <1% of a cycle fit when threaded.

### Plotting K-optimized

`plot_experiment_k.py` takes `--exp <results-dirname>` (default `experiment_k`):

```bash
python plot_experiment_k.py --exp experiment_k_optimized
```

`--exp experiment_k_optimized` also flips the module-level `OPTIMIZED` flag, which
**retires the `a + b·k` "deployable" fit** (it exists only to strip plain K's
in-fit O(N²) scoring; applying it to K-opt would double-count) and adds three
plots the paper uses: `speedup_vs_N`, `overhead_vs_N_k200`, and
`budget_summary_k200` (the two paired, the figure in the paper). Cross-dataset
plots use `k_ref=200` under K-opt and `k_ref=50` under plain K.

**`results/experiment_k/_plots_combined/speedup_vs_D.pdf` is contaminated**
(in-fit O(N²) scoring inside the timed region). Never use it or any other plain-K
speed figure. Quality figures from K and J are fine.

### Large-N

- Cycle memory is ~36·N² bytes (matrix, explicit pair list, per-pair arrays,
  shuffle copies). N=10k ≈ 3.4 GB, 20k ≈ 13 GB, 50k ≈ 84 GB, 100k ≈ 335 GB.
  Calibrated to 20k, **extrapolated above**: verify peak RSS on a new node before
  committing a big allocation.
- `MAX_MEM_GB` in the K runners skips rungs over budget. A memory-skipped run
  deliberately writes no `meta.json`, so re-running on a bigger node resumes.
- K-optimized checkpoints per fit; plain K per rung.

### Datasets

- `prep_fashion_mnist_full.py`: the one true downloader (OpenML → 70k
  Fashion-MNIST).
- `prep_shortlist_datasets.py`: **symlinks only**; wires 7 names into the bulk
  collections (`hf_datasets_p1/p2`, `pmlb_datasets`, `scanpy_datasets`,
  `sklearn_datasets`). Those collections have **no provenance record** and can't
  be regenerated by any script in the repo.
- `orl` is defective in the published file: 360 of 400 rows are all-zero, in both
  features and label, leaving 41 distinct points.

### Plot conventions (D, F, G, H)

3-column grids, per-plot legends, IQR bands around medians, cycle baseline as a
dashed line with `ax.axhspan` (not `fill_between`, which bakes in linear-scale
xlim before `set_xscale('log')`). Titles carry `(N=…)` and `D=`. Pivot palette:
`maxmin` orange `#ff7f0e`, `pca10` blue `#1f77b4`. Two convergence views:
`convergence_by_k_absolute` and `..._relative` (each replicate divided by its own
initial stress before aggregation). K adds two cross-dataset figures:
`speedup_vs_D` and `kstar_vs_strain`, one per axis of §1.4.

---

## 6. Upstream PR: remaining work

- Export `SGDMDS` from `sklearn/manifold/__init__.py`.
- Module-level `smacof`-equivalent function (optional).
- Changelog + `doc/` narrative entry, including `n_updates_per_epoch` and
  `compute_stress`.
- Validate the budget knob under **non-metric** (`metric=False`) MDS. Never
  tested; all of I/J/K is metric.

**Doc guidance should state:** the regime dependence (`strain_top2` as a cheap
self-diagnostic), that budget-cut and iteration-cut don't compose, that the speed
win depends on D and grows with N, and that low-strain data yields a poor 2D
embedding regardless of solver.

**Candidate future API:** `fast_mode='auto'`: compute `strain_top2` on an 800-point
subsample at fit time, gate cycle-vs-budget and pick k. Gate on a `k*`-style
target, not curve steepness (§1.3).
