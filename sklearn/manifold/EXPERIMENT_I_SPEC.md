# Experiment I — Budget knob on `sampling_strategy='random'`

## 1. Premise

Experiment H closed the pivot SGD-MDS investigation with a decisive surprise: at fixed
update budget `k·N` per epoch, **pure random pair sampling (hybrid α=0) lands within
1–14% of SGD-cycle stress on every dataset**, while pure pivot sampling at the *same*
total update count is 5–9× worse. On coil20 (N=1440, k=50) that is 72,000 updates per
epoch — about **14× less work than full cycle** — at within 7% of cycle quality.

The implication: the right primitive for a "fast mode" in SGD-MDS is **budgeted uniform
random pair sampling**, not pivots. Today `sampling_strategy='random'` always performs
`N(N−1)/2` random pair draws per epoch — the same total update count as cycle, just
sampled instead of enumerated. That hides the speedup behind a strategy name.

Experiment I exposes the per-epoch budget as a first-class user-facing knob
(`n_updates_per_epoch`) and characterises the **stress-vs-budget Pareto curve** across
the H datasets plus one larger-N dataset (`epileptic`, N=5750), at two iteration
budgets (`max_iter ∈ {15, 30}`).

If the curve looks clean, `n_updates_per_epoch` becomes the recommended fast-mode API
in the upstream PR, and the pivot/hybrid machinery is formally demoted to "research
artefact, not in the public surface."

---

## 2. Assumptions

- **The H finding generalises within the budget range we test.** H established that
  `k·N` random updates ≈ cycle quality at a single point (k=50). We assume that the
  same near-equivalence holds across the broader sweep `k ∈ {10, 25, 50, 100, 200}`
  on the same six datasets, and that the qualitative pattern carries to `epileptic`
  (N=5750), which is ~1.8× larger than the largest H dataset.
- **Halving `max_iter` from 30 to 15 either halves wall time at minor quality cost,
  or re-exposes a budget-quality gap.** Both outcomes are informative. The H runs
  all used `max_iter=30`; we do not yet know whether the optimiser is in
  diminishing-returns territory at iteration 15 or still actively converging.
- **At the full-cycle budget `n_updates_per_epoch = N(N−1)/2`, `random` matches
  `cycle` stress up to seed noise.** This is the sanity check that the API edit
  (replacing a hardcoded constant with a resolved value) did not introduce a bug.
- **Wall time is roughly linear in `n_updates_per_epoch` at fixed `max_iter`.** The
  Cython kernel does a constant amount of work per update, so the time-vs-budget
  plot is the cheap sanity gate before believing the stress-vs-budget plot.
- **Pivot/hybrid runs are not re-collected here.** Experiment H already produced
  matched-budget pivot and hybrid numbers at k=50; the H-I3 hypothesis (Pareto
  comparison) reads those off the H CSVs rather than re-running them.

---

## 3. Hypotheses

- **H-I1 — monotonicity.** Median stress is non-increasing in
  `n_updates_per_epoch` across all six budgets, on every dataset, at both
  `max_iter` values. (A non-monotone curve would imply seed noise dominates the
  budget effect at some k — useful to know.)
- **H-I2 — iteration composability.** At k=50, `max_iter=15` lands within ~2× of
  `max_iter=30`'s stress on every dataset. If true, the budget knob and the
  iteration count compose cleanly and we can recommend either as an independent
  speedup lever. If false, the spec doc for the upstream API needs to couple them
  (e.g. a minimum `max_iter` heuristic).
- **H-I3 — Pareto dominance.** The (wall time, stress) curve for `random` at
  varying budget strictly dominates the same curve for pivot and hybrid at matched
  wall time (read off H's results at k=50). I.e. at any time budget the user has,
  budgeted random beats budgeted pivot.

---

## 4. Sweep matrix

- **Datasets** (7): `fashion_mnist, coil20, fmd, hiva, cifar10, cnae9, epileptic`.
  First six match H exactly; `epileptic` (N=5750) extends the N range to probe
  scaling.
- **Budgets** (6): `n_updates_per_epoch = k·N` for `k ∈ {10, 25, 50, 100, 200}`,
  plus the full-cycle budget `n_updates_per_epoch = N·(N−1)//2`. The full-cycle
  point is the sanity anchor: it must match SGD-cycle within seed noise.
- **`max_iter`** (2): `{15, 30}`. H used 30.
- **Seeds**: 5 (matching H: `SEED_BASE = 42, range(5)`).
- **Baselines per dataset**: SGD-cycle at both `max_iter ∈ {15, 30}` (5 seeds each).
- **Run count per dataset**: `6 budgets × 2 max_iter × 5 seeds + 2 max_iter × 5
  seeds (cycle) = 70 runs`. Across 7 datasets: **490 runs total**. Tractable in a
  day on a single machine given H ran in ~2h.
- **Sampler config**: lazy mode (`dissimilarity='lazy'`), `weighting='uniform'`,
  `learning_rate_init=0.5` (matching H baselines), `scheduler='exponential'`,
  `switch_ratio=0.5`, `epsilon=0.001`.

---

## 5. Success criteria

- **Reproduces H.** At `k=50, max_iter=30`, the `random@k·N` stress on each of
  the six H datasets matches (within seed noise) the H result for `hybrid α=0.0`
  at the same k (i.e. within 1–14% of SGD-cycle).
- **Sanity at full cycle.** At `n_updates_per_epoch = N(N−1)/2`, the `random`
  stress is statistically indistinguishable from `SGD-cycle` at the matched
  `max_iter`. This validates the dispatcher edit in `_run_lazy`.
- **Monotonic budget effect.** Stress decreases (or stays flat) as
  `n_updates_per_epoch` grows, at every dataset and every `max_iter`.
- **Pareto dominance.** On the `stress_vs_time` plot, the `random` curve sits
  below (or on) the cycle point, and below any matched-time pivot/hybrid point
  from H.

---

## 6. Decision tree for next steps

- **Curve is clean (monotone, Pareto-dominant, max_iter=15 within ~2× of 30):**
  ship `n_updates_per_epoch` as the recommended fast-mode API in the upstream PR.
  Default stays `"auto"` (≡ `N(N−1)/2`) for backward compatibility; documentation
  recommends `k·N` for `k ∈ {25, 50, 100}` as the typical fast-mode setting.
  Demote `sampling_strategy ∈ {'pivot', 'hybrid'}` and the pivot helpers to
  internal/research.
- **`max_iter=15` shows a large quality drop (>2× the 30-iter stress) at low
  budget:** the knobs do not compose cleanly. Recommend coupling the budget knob
  to a minimum `max_iter` heuristic (e.g. `max_iter >= 30` when
  `n_updates_per_epoch < N²/4`) in both the docstring and the spec doc.
- **`epileptic` (N=5750) shows the speed/quality advantage *growing* with N:**
  strong empirical case for promoting the API and possibly raising the default
  budget recommendation for large N (e.g. `k=25` for N > 3000).
- **Stress at full-cycle budget does not match SGD-cycle:** the dispatcher edit
  introduced a bug. Stop, debug `_run_lazy`, do not interpret any other I results
  until the sanity point passes.
- **Monotonicity violated at some k:** seed noise likely dominates. Increase
  `N_REPEATS` to 10 for the affected dataset/budget and re-run that slice.

---

## 7. Follow-up: budget scaling law (Experiment I-scaling)

### Question

Experiment I showed that at *fixed* k the stress overhead grows with N. But
the seven datasets confound dataset *size* with dataset *difficulty*, so I
cannot read the scaling law off them directly. The follow-up answers: **to
hold overhead fixed, how must k grow with N?**

```
k*  ~  N^p          p = 0  -> constant k suffices (budget ~ N, maximal win)
                    p = 1  -> linear k needed     (budget ~ N^2, no win)
```

### Two bounding hypotheses

- **Coverage-fraction law (p=1).** If overhead is controlled by the fraction
  of pairs sampled per epoch, `f = 2k/(N−1)`, then holding overhead fixed
  means holding `f` fixed → `k ∝ N`. Budget `= kN ∝ N²` — the speedup
  vanishes asymptotically.
- **Updates-per-point law (p=0).** Each point receives `2k` gradient touches
  per epoch regardless of N, so if that is the controller, `k` can stay
  constant and overhead is N-independent. Budget `∝ N`.

Reality is expected to lie between, with the exponent governed by the
**intrinsic dimensionality** of the data: genuinely low-dim structure → p
near 0; high effective dimension → p near 1.

### Design (removes the size/difficulty confound)

Single source dataset, subsampled to a geometric N ladder
(`{400, 600, 900, 1350, 2000, 3000}`, ratio ~1.5), so intrinsic
dimensionality is held fixed and only N varies. For each `(N, seed)`: one
seeded row-subsample, an SGD-cycle baseline (the overhead denominator), and
SGD-random at `n_updates_per_epoch = k·N` for `k ∈ {10, 25, 50, 100, 200}`
(filtered to `k < (N−1)/2` so the budget never exceeds cycle). `max_iter=30`,
5 seeds. Overhead`(N, k) = median_seed(random_stress / cycle_stress)`; `k*`
is the log-log interpolated budget reaching a target overhead; `p` is the
slope of `log k*` vs `log N`.

Run on two sources of differing intrinsic dimensionality — `fashion_mnist`
(moderate; steep overhead-vs-k in I) and `hiva` (high-D; nearly floored in
I) — to test whether p tracks intrinsic dimensionality.

Runner: [benchmarks/run_experiment_i_scaling.py](benchmarks/run_experiment_i_scaling.py) ·
plots: [benchmarks/plot_experiment_i_scaling.py](benchmarks/plot_experiment_i_scaling.py)
(`overhead_vs_k`, `kstar_vs_N` with the fitted exponent, plus combined grids).

### What each outcome means

- **p ≈ 0.5–0.8 (expected):** sublinear. For 10× the data you need ~3–7× the
  budget, not 10×. Budget `~ N^{1+p}` stays subquadratic → the fast mode keeps
  a net speedup `~ N^{1−p}` that grows with N. Recommend scaling k like
  `√N … N^0.8` in the docs rather than a flat k.
- **p ≈ 1 on a source:** that source's intrinsic dimension is too high for the
  budget knob to scale — overhead is floored by the cycle-vs-achievable gap,
  not coverage. Flag that the fast mode is N-gated *and* difficulty-gated.
- **p differs markedly between sources:** confirms the exponent is set by
  intrinsic dimensionality; the doc recommendation should mention that
  high-D / high-intrinsic-dim data needs k closer to linear.

### Results (complete)

The two sources land in **opposite regimes** — the controlled experiment
revises the earlier cross-dataset guess of "p ≈ 0.7" (which was an artifact
of mixing regimes plus the size/difficulty confound).

**`fashion_mnist` (D=784) — compressible regime → constant k.**
The overhead-vs-k curve is *steep* (2.0× at k=10 → ~1.0× at k=200), and the
budget to hold a fixed overhead is **flat in N**:

| target | k* at N=400 → 3000 | fitted p |
|---|---|---|
| ≤1.05× | 95, 76, 83, 74, 96, 91 | **0.03** |
| ≤1.10× | 64, 48, 55, 50, 62, 66 | **0.06** |
| ≤1.15× | 51, 37, 42, 38, 48, 54 | **0.07** |

p ≈ 0 → a *fixed* k≈50–65 holds 1.10× overhead across the whole ladder.
(At fixed k=50 the overhead drifts up only mildly, 1.13× → 1.18×, and the
steep curve means a tiny k bump recovers the target — hence flat k*.) This
is the "updates-per-point" law winning, and it is the **best case for the
fast mode**: budget `= kN ∝ N`, speedup `∝ N`.

**`hiva` (D=1617) — incompressible regime → the knob saturates.**
The overhead-vs-k curve is *shallow* (only 1.11× → 1.03× over a 20× budget
increase) and the overhead is **floored** at ~1.05–1.11×, with the floor
itself rising mildly with N. Consequences:

- target ≤1.15×: reached at k=10 for every N → p ≈ 0 (trivially).
- target ≤1.10×: the floor brushes the threshold, so k* is unstable and
  reads p ≈ 0.89 — but this is an *artifact of a flat curve crossing the
  target*, not evidence that coverage needs linear k. A few-percent change
  in the floor swings k* wildly (10 → 49 → 39, non-monotone).
- target ≤1.05×: unreachable at any tested k for most N.

For this kind of data the budget knob has **weak leverage**: overhead is set
by the irreducible cycle-vs-achievable gap (the same 2D-embeddability ceiling
that gave hiva 3.35× even for the spectral baseline in H), not by coverage.

### Conclusion

The scaling exponent is **not a single number** — it is governed by how
*steep* the overhead-vs-k curve is, which is set by the data's intrinsic
dimensionality:

- **Compressible / low-intrinsic-dim data:** p ≈ 0. Pick a fixed k≈50–100
  and it holds across N. Fast mode keeps a near-linear speedup.
- **Incompressible / high-intrinsic-dim data:** the knob saturates at an
  overhead floor that budget cannot reduce. k is the wrong lever; the limit
  is the achievable-stress gap, not coverage.

So the practical recommendation is *not* "scale k like N^0.7." It is: a fixed
moderate k (≈50–100) suffices for the datasets where the fast mode is worth
using at all; on data where overhead won't come down, no budget setting helps
and the user should fall back to full cycle (or accept the floor). The
earlier "k ∝ N^0.5–0.85" reading was the blur of these two regimes seen
through the size/difficulty confound.

Runner/plots produced: `results/experiment_i_scaling/{fashion_mnist,hiva}/`
(`overhead_vs_k`, `kstar_vs_N`) and `_plots_combined/`.
