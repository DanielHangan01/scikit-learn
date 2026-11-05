# Experiment K — large-N stress-test of the budgeted-random fast mode

## 1. Premise

Experiment J established that on **responsive** (2D-embeddable) data the budget
`k*` needed to hold a fixed stress overhead is near-constant in N (`k* ~ N^p`,
p≈0.1), and that intrinsic dimensionality (`strain_top2_frac`) predicts which
regime a dataset is in. But J's largest dataset was `epileptic` (N=5750), and
every N in J came from a *different* dataset — so the constant-k claim was
never checked at genuinely large N on a single fixed source, and the *wall-time*
payoff at scale was never isolated (only the stress overhead was).

Experiment K closes both gaps. It takes **one large source**, subsamples it up
an N-ladder well beyond J's range (intrinsic dimensionality fixed, only N
varies), and measures both stress overhead **and** wall-time speedup at each N.
The questions:

1. Does the responsive-regime constant-k result (`p≈0`) survive at large N, or
   does a fixed k eventually fall behind cycle as N grows?
2. What is the *actual* wall-clock speedup of the fast mode at scale, and how
   does it grow with N?

Runner: [benchmarks/run_experiment_k.py](benchmarks/run_experiment_k.py) ·
plots: [benchmarks/plot_experiment_k.py](benchmarks/plot_experiment_k.py).

**Results & conclusions: [EXPERIMENT_K_RESULTS.md](EXPERIMENT_K_RESULTS.md).**

## 2. Design (single-source N-ladder, same paired design as I-scaling / J)

Per source:

- **Intrinsic-dimensionality scalars** on a fixed seeded 800-point subsample
  (reused verbatim from `run_experiment_j.intrinsic_dim_measures`:
  `strain_top2_frac`, `pca_participation_ratio`, ...), so each large source is
  placed in the same responsive / hard-middle / degenerate taxonomy.
- **N-ladder** — bridges from J's range up past its largest dataset (5,750)
  toward each source's own full N (up to 100,000). Rungs are filtered
  per-source to `<= full N` and `<= STRESS_CAP` (see §7); the *reachable* top
  rung on any given run is additionally gated by `MAX_MEM_GB` (§7), since the
  cycle baseline is the memory bottleneck, not `STRESS_CAP` itself. The
  original results in §2 were produced with the ladder capped at 10k for a
  16 GB machine — the runner now defaults to a taller ladder for cluster use,
  configurable back down for smaller machines.
- At each `(N, seed)`: draw a seeded row-subsample of size N (identical subset
  used for the cycle baseline and every budget k, so overhead is a **paired**
  ratio); run
  - **SGD-cycle** baseline: `dissimilarity='euclidean'` (deterministic pass
    over all N(N−1)/2 pairs) — the overhead denominator and speed reference;
  - **SGD-random** at `n_updates_per_epoch = k·N` for each
    `k ∈ {50, 100, 200, 400}` (`dissimilarity='lazy'`), filtered to
    `k < (N−1)/2`.
- `max_iter=30` (the scaling reference; append 15 to also probe the iteration
  lever at scale, at ~2× the expensive-baseline cost). 5 seeds.

Both arms minimise the identical raw-stress objective and report the estimator's
own `stress_`; the only difference between them is the per-epoch sample budget,
so the overhead ratio and the timing comparison are apples-to-apples.

**Multiple sources.** `DATASETS` in the runner is a plain list — add any large
source (into `datasets/<name>/`, pooling any train/test split, fixed-seed
subsample). Shared ladder rungs make the per-source exponent `p` and overhead
curves directly comparable.

## 2b. Dataset inventory (the 8 sources actually run)

Chosen to span **feature dimension D (3 → 3072)** and **regime**, so the two
axes (quality ← `strain_top2`, speed ← D) can be separated. All intrinsic-dim
scalars are measured by the runner on a fixed seeded 800-point subsample of the
**standardized** data (`load_local_dataset` applies `StandardScaler`), and are
recorded in `results/experiment_k/<name>/meta.json`.

`N_full` = source size (each is subsampled to the `{2000, 4000, 7000, 10000}`
ladder at run time). `strain` = `strain_top2_frac` (2D-embeddability; higher =
more responsive). `pca_PR` = PCA participation ratio (effective feature dim).
`n90` = PCA components for 90% variance. Regime: **R** = responsive
(strain ≥ 0.22), **H** = hard-middle.

| dataset | N_full | D | strain | pca_PR | n90 | reg | role |
|---|--:|--:|--:|--:|--:|:--:|---|
| `feynman_iii_15_12` | 100,000 | 3 | 0.687 | 3.0 | 3 | R | D-axis: extreme low-D (synthetic physics manifold) |
| `california_housing` | 20,640 | 8 | 0.648 | 3.8 | 4 | R | D-axis: low-D, **real** data |
| `ag_news` | 100,000 | 384 | 0.075 | 102.4 | 162 | H | falsification: text embeddings |
| `patchcamelyon` | 100,000 | 512 | 0.314 | 13.9 | 60 | R | D-axis: mid-D histopathology |
| `tiny_imagenet` | 100,000 | 512 | 0.072 | 107.3 | 197 | H | falsification: image embeddings |
| `mnist_digits` | 70,000 | 784 | 0.115 | 55.1 | 115 | H | falsification: digit MNIST (boundary case) |
| `fashion_mnist_full` | 70,000 | 784 | 0.369 | 12.1 | 93 | R | D-axis anchor (first source run) |
| `cifar10_raw` | 50,000 | 3072 | 0.379 | 10.6 | 84 | R | D-axis: very high-D **raw pixels** |

### Provenance

`fashion_mnist_full` is built from OpenML ("Fashion-MNIST" v1, train+test
pooled, `/255`) by
[benchmarks/prep_fashion_mnist_full.py](benchmarks/prep_fashion_mnist_full.py).
The other seven are wired from the bulk collections as **relative symlinks**
(`datasets/<name>/X.npy` → the collection file) by
[benchmarks/prep_shortlist_datasets.py](benchmarks/prep_shortlist_datasets.py) —
no data duplicated:

| dataset | source file |
|---|---|
| `feynman_iii_15_12` | `pmlb_datasets/pmlb_feynman_III_15_12.npy` |
| `california_housing` | `sklearn_datasets/sklearn_california_housing.npy` |
| `ag_news` | `hf_datasets_p1/hf_text_fancyzhx__ag_news.npy` |
| `patchcamelyon` | `hf_datasets_p2/hf_image_1aurent__PatchCamelyon.npy` |
| `tiny_imagenet` | `hf_datasets_p2/hf_image_zh-plus__tiny-imagenet.npy` |
| `mnist_digits` | `pmlb_datasets/pmlb_mnist.npy` |
| `cifar10_raw` | `hf_datasets_p1/hf_image_uoft-cs__cifar10_raw.npy` |

Notes worth carrying: the `hf_*` sets are **pre-computed learned embeddings**
(sentence-transformer / CLIP), not raw features — the text ones (D=384) are
additionally **L2-normalised** (points on a hypersphere); `cifar10_raw`
(D=3072) *is* raw pixels and is distinct from the older `cifar10` (D=1024,
embeddings) used in J. `feynman_iii_15_12` is a synthetic analytic manifold —
a clean best case, so `california_housing` is the convincing real-data
low-D instance.

## 3. Derived quantities (in the analyzer / summary)

- `overhead(N, k) = median_seed(random_stress / cycle_stress)`.
- `k*(N)` — log-log interpolated budget hitting overhead targets
  `{1.05, 1.10, 1.15}`; `p` = slope of `log k*` vs `log N`.
- **speedup**`(N, k) = median_seed(cycle_time / budget_time)` — the measured
  wall-time win.
- **deployable speedup** — fit `budget_time = a + b·k` per N; the k-independent
  term `a` is the O(N²) exact-stress **scoring** cost (`SGDMDS._safe_stress`
  builds the full N×N matrix purely to measure stress — a deployment using
  sampled stress skips it). Attributing only `b·k` to the fast mode gives the
  speedup a real deployment would see. (`a/N² ≈ const` is the diagnostic that
  the fixed cost is indeed the O(N²) scoring, not per-fit overhead.)

## 4. Hypotheses

- **H-K1 (constant-k at scale).** On responsive data, overhead at fixed k stays
  flat as N grows (`p≈0`); a fixed k=100–200 holds within a few % / ~1% of
  cycle at every N up to 10k.
- **H-K2 (speedup grows with N).** Cycle is O(N²) per epoch, budget O(k·N), so
  `cycle_time/budget_time` increases with N.
- **H-K3 (speed and quality are different axes).** The wall-time speedup is far
  below the naive pair-count ratio `(N−1)/2k`, for two reasons a deployment
  cares about differently: (a) the lazy kernel recomputes each pair distance
  O(D), vs O(1) matrix lookup in cycle — so **speed tracks D**; (b) exact-stress
  scoring is O(N²) and inflates every timed run. Quality, meanwhile, tracks
  `strain_top2`, not D. Prediction: lower-D sources get larger speedups at the
  same N / quality.
- **H-K4 (generality).** H-K1 and the D-vs-speed split hold across large sources
  spanning D and regime (a low-D and a high-D large dataset; a hard-middle large
  dataset should *not* reach constant-k, mirroring hiva in J).

## 5. Outputs

Per source (`results/experiment_k/<name>/`): `results.csv` (raw stress + time
rows), `meta.json` (N, D, ladder, k grid, intrinsic-dim scalars). Plots:
- `overhead_vs_k` — overhead vs budget, one line per N (explicit k ticks).
- `kstar_vs_N` — k*(N) with fitted p and p=0.5 / p=1 reference slopes.
- `overhead_vs_N_at_fixed_k` — the constant-k "money" plot: fixed k vs N; flat
  line ⇒ a fixed budget holds quality as N grows.

Combined grids in `_plots_combined/` (titles auto-annotate `strain_top2`):
`combined_overhead_vs_k`, `combined_overhead_vs_N_fixed_k`,
`combined_kstar_vs_N`.

Cross-dataset — **one figure per axis** (these carry the headline results):
- `speedup_vs_D` — deployable *and* measured speedup vs D, coloured by regime.
  **D sets what each unit of budget costs.** The measured/deployable gap shows
  the O(N²) scoring contamination.
- `kstar_vs_strain` — `k*` (budget needed for a quality target) vs
  `strain_top2`. **Strain sets how much budget you need.** Datasets that never
  reach the target are drawn as open markers at the top edge.

## 6. Decision tree

- **H-K1 confirmed:** the upstream fast-mode docs recommend a fixed k
  (≈100–200) for responsive data with no N-dependent scaling — the strongest
  form of the constant-k claim.
- **H-K3 confirmed:** the O(N²) exact-stress scoring is the measured-speedup
  bottleneck *and* the reason N>20k is unmeasurable → build the **sampled-stress
  estimator** (reveals the true speedup, unlocks N≫20k). Document speed as a
  function of D and N, separately from quality (`strain_top2`).
- **A large source breaks H-K1 (fixed k drifts up with N):** the constant-k
  regime is bounded in N after all — inspect whether it correlates with
  `strain_top2` dropping, and re-gate the PR recommendation.

## 7. Cost / memory note

*(Original 16 GB workstation results — §2 — used `N_LADDER` capped at 10,000
and `STRESS_CAP=20,000`. The runner has since been updated for cluster-scale
runs; this section describes the current, more general behaviour. Numbers
below are recalibrated but the underlying mechanism is unchanged.)*

The **cycle baseline is the memory bottleneck**, not the fast mode. In
euclidean mode it materialises the N×N distance matrix, the explicit N²/2
pair list, per-pair target/weight arrays, and transient buffers during pair
construction and scoring — **~36 N² bytes peak** (calibrated against observed
OOM behaviour up to N=20,000; *extrapolated* above that — large-N numpy
internals, e.g. `np.triu_indices`, may add further transient overhead not
exercised by the calibration run):

| N | est. peak | | N | est. peak |
|---:|---:|---|---:|---:|
| 10,000 | ~3.4 GB | | 35,000 | ~41 GB |
| 20,000 | ~13.4 GB | | 50,000 | ~84 GB |
| — | — | | 70,000 | ~164 GB |
| — | — | | 100,000 | ~335 GB |

On a 16 GB workstation the safe ceiling is ~10k. On a cluster node, raise
`N_LADDER` and `STRESS_CAP` to the target dataset's own full N (the runner's
`DATASETS` list documents each source's full N, up to 100,000) and set
`MAX_MEM_GB` to the node's usable RAM — the runner then **estimates peak
memory per rung and skips** (rather than attempts and OOM-crashes) any rung
that would exceed it, printing a clear message. A memory-skipped run
deliberately does **not** write `meta.json`, so re-running later — e.g. after
moving to a bigger node — resumes exactly where it left off (completed rungs
are read back from `results.csv`; only the skipped rungs are retried).

Second, independent ceiling: `SGDMDS._safe_stress` (the estimator's own exact
stress evaluator) skips scoring — returns −1.0, making overhead undefined —
above a configurable `EXACT_STRESS_N_CAP` (default 20,000, a conservative
safety guard rather than a technical limit). The runner raises this
automatically at start-up (`_configure_stress_cap()`) to match its own
`STRESS_CAP`, so it is not a separate thing to configure by hand — but note
it *is* a second, independent O(N²) cost from the memory table above (see the
runner's module docstring), so raising `STRESS_CAP` alone does not remove the
RAM ceiling.

**Recommended practice on a new/unfamiliar node**: run one dataset up to an
intermediate rung first (e.g. 35k) and watch peak RSS before committing the
full ladder to a large batch allocation, since the ~36 N² coefficient is
extrapolated past N=20,000.

The runner **checkpoints `results.csv` after every completed rung and resumes
per-rung** (writes `meta.json` only when every configured rung — memory-skips
included — has been attempted), so a kill loses at most the rung in flight —
important because the large-N baselines are the slow, kill-prone part. Add
high-D / large sources one at a time; run the top rung when the machine is
otherwise idle.
