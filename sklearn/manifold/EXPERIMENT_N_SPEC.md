# Experiment N — point-blocked sampling for budgeted SGD-MDS

**Status:** implemented and tested; not yet run. Runner:
`benchmarks/run_experiment_n.py`. Kernel microbenchmark:
`benchmarks/bench_blocked_kernel.py`.

---

## 1. Why

SQuaD-MDS runs 7–14× faster than budgeted SGD at k=200 (Experiment M, k8,
clean timing). Two causes were identified:

- **Memory reads per pair.** SQuaD-MDS reads 4 feature rows per quartet and
  gets 6 pairs from them, so 0.67·D values per pair. We read 2 rows per pair,
  so 2·D.
- **Pair count.** Its default 1,000 iterations process 1,500·N pairs. k=200
  processes 6,000·N.

The first cause does not depend on SQuaD's objective. **Point-blocking** takes
only that part: draw *m* points, compute all m(m−1)/2 feature-space distances
from one read of their rows, then apply our ordinary raw-stress step to each
pair.

What stays the same:

- the objective (raw stress)
- the per-pair update
- the budget, always counted in **pairs**. `k·N` means the same work for
  every m.

Row reads per pair fall to 2D/(m−1):

| m | pairs per block | row reads per pair |
|--:|--:|--:|
| 2 (today) | 1 | 2.00·D |
| 4 (as in SQuaD) | 6 | 0.67·D |
| 8 | 28 | 0.29·D |
| 16 | 120 | 0.13·D |

## 2. A finding made while building it: the original kernel pays a GIL tax

`_lazy_sgd_step` is a `cdef inline void` function that is not declared
`noexcept`. Under Cython 3, every call to it is followed by
`__Pyx_ErrOccurredWithGIL()`, which **acquires and releases the GIL once per
pair update**. Cython prints a performance hint about this on every build.

- **Affected:** the random-pair and pivot lazy kernels.
- **Not affected:** the matrix/cycle kernel (no inner function call) and the
  SQuaD-MDS kernel (checked: no such calls).

The blocked kernel's helpers are `noexcept`. At m=2 with random blocks it runs
**the same algorithm with a bit-identical embedding**, and:

| setting | legacy kernel | blocked kernel, m=2 | speedup |
|---|--:|--:|--:|
| microbenchmark, N=5,000, D=3 | 50.1 ns/pair | 14.5 ns/pair | 3.5× |
| microbenchmark, N=5,000, D=64 | 62.9 ns/pair | 34.6 ns/pair | 1.8× |
| full `SGDMDS` fit, k=200, N=3,000, D=8 | 0.91 s | 0.28 s | 3.3× |

**Consequences** (none resolved yet; see STATUS.md):

- The K-optimized cost model's budget intercept (~52 ns/update) is mostly this
  overhead, not sampling or distance work.
- Experiment M compared a SQuaD-MDS kernel without the overhead against an SGD
  kernel with it. At low D that tilted the comparison against SGD.
- Fixing the original kernel is one keyword (`noexcept`). It changes no
  results, but it changes the timing of every earlier budget run, so it has
  been deliberately left alone. `SGDMDS` defaults still use it.

## 3. What was implemented

| File | Change |
|---|---|
| `_sgd_mds_cython.pyx` | New `run_sgd_epoch_lazy_blocked`, its `noexcept` helpers, and a `draw_blocks` test hook. **The original kernels are untouched.** |
| `_sgd_mds.py` | New parameters `block_size=2`, `block_sampling='random'`, `block_update='sequential'` (experimental; `sampling_strategy='random'` only). The module flag `FORCE_BLOCKED_KERNEL` sends plain m=2 through the new kernel. |
| `benchmarks/bench_utils.py` | `run_random_budget_benchmark` passes the three parameters and `force_blocked_kernel` through. The defaults change nothing. |
| `tests/test_blocked_sgdmds.py` | 95 tests, listed below. |
| `tests/data_blocked_golden.npz` | Embeddings recorded from commit 7c4301d, *before* any change, covering budget (uniform, inverse, non-metric), random-auto, cycle, pivot lazy and euclidean, and hybrid. All must reproduce bitwise. |

What the tests check:

- **Golden fixtures:** every sampling mode reproduces bit-for-bit.
- **m=2 equivalence, kernel level:** all three update modes are bit-identical
  to the legacy kernel, across weighting × metric/non-metric.
- **m=2 equivalence, estimator level:** the same holds through `SGDMDS`, and
  under `FORCE_BLOCKED_KERNEL`.
- **Distance paths:** the fused m=4 pass is bit-identical to the generic one.
- **Sampling:** random blocks follow the legacy xorshift stream; partition
  blocks are disjoint and complete; the pair marginals are uniform (χ²).
- **Update rules:** exact (`assert_array_equal`) against Python oracles for
  every mode at m ∈ {3, 4, 6, 8}, with duplicates in random blocks and a
  partial final block. m ∈ {3, 4, 6, 8} covers every distance path.
- **Invariants:** the centroid is conserved and the budget is exact in pairs.
- **Validation:** invalid parameter combinations raise errors.

### Design details

- **Random blocks** are drawn *with replacement*: m independent
  `xorshift32 % N` draws.
  - Every pair of slots is therefore exactly a legacy (i, j) draw, including
    i == j, which is skipped but counted, as legacy does.
  - At m=2 the two draws *are* legacy's i and j. That is what makes m=2
    bit-identical.
- **Partition blocks:** an in-kernel Fisher–Yates shuffle of an `int32`
  buffer.
  - Each epoch starts with a fresh shuffle.
  - When fewer than m points remain, the kernel reshuffles, and the N mod m
    leftover points sit out, as in SQuaD-MDS.
  - The shuffle runs in the kernel rather than in NumPy (unlike our SQuaD
    port). That keeps one call per epoch and adds no draws from the
    estimator's Python `rng`. Its cost is O(N) per partition against
    O(N·(m−1)/2·D) of work; the microbenchmark measures it.
- **Pair order** within a block is fixed and lexicographic. Slots are
  exchangeable, so this introduces no bias.
- **Update modes:**
  - `sequential`: each step is applied immediately. This is our rule.
  - `sum`: all steps are computed from the block's starting positions and
    their per-point sum is applied once. This is SQuaD's rule, a gradient
    step on the block's stress.
  - `mean`: `sum` divided by the number of pairs each point was in (m−1 on a
    full block).
  - At m=2 all three are identical.
  - Risk: `sum` loses the μ ≤ 1 no-overshoot guarantee, because a point's
    m−1 moves add up. `mean` keeps it stable but is effectively a learning
    rate cut by 1/(m−1).
- **Distance paths:**
  - m=4 uses a fused single pass: 4 loads per feature feeding 6 running sums,
    like the SQuaD-MDS kernel. This keeps the comparison fair.
  - Every other m uses pairs outer and features inner, taking **four pairs at
    a time with independent sums**. One sum per pair is a serial chain of D
    dependent additions. In the first dry run this made m=8/16 *slower* than
    m=4 (0.31 s vs 0.20 s on coil20), purely from code generation. With
    chunking all three run at 0.20 s.
  - Every path sums each pair over f in order, so all of them are
    bit-identical.

## 4. Hypotheses

1. **Speed.** Time per pair falls with m, roughly as row reads fall
   (1/(m−1) relative to blocked m=2), wherever the fit is memory-bound
   (large N·D). It flattens once distance arithmetic or per-pair update
   overhead dominates. Expect the gain to grow with D.
2. **Quality at equal pair count.** Pairs in a block are correlated: in a
   block of 8, each point is in 7 of the 28 pairs. So stress at a fixed k
   should rise with m.
   - The question is by how much, and whether the rise follows the regime
     theory (`strain_top2`).
   - If it stays within ~1% up to m=8, blocking is close to free speed.
3. **Update rule.** sequential ≥ sum on stability. `sum` may diverge at
   m ≥ 8 early in the fit (the coil20 dry run at m=16 random: 8.6× legacy
   stress). `mean` is stable but slower to converge. Expected quality order:
   sequential ≳ sum > mean at m ≥ 8.
4. **Sampling.** partition ≈ random on the median, with slightly lower spread
   across seeds.

**Primary metric:** time to reach a stress target, e.g. ≤ 1.01× the legacy
k=200 stress. It folds H1 and H2 into one number. Read it off each arm's k
curve, the same way K-optimized derives k\*.

Every speed claim is quoted **twice**:

- against **legacy**: what we'd tell a user, fix plus blocking;
- against **blocked m=2**: what blocking alone buys.

## 5. Protocol

- **Suites:** the same data and seeds as the reference experiments, so the
  legacy arm reproduces them. This is checked on every fit
  (`repro_rel_err`, `matches_legacy`).
  - **j18:** 18 datasets at full N, rows permuted by `RandomState(seed)`
    exactly as J does, seeds 42–46. (M did not permute; that makes no
    difference for SQuaD's statistics.)
  - **k8:** K-optimized's `SUBSAMPLE_SEED=42` subsample at N=10,000 and at the
    largest rung ≤ 50,000 (california_housing: 20,640). Seeds 42–44.
- **SGD settings:** max_iter=30, lr=0.5, hybrid scheduler, switch 0.5,
  eps 0.001. Identical to J and K-optimized.
- **Arms:**
  - j18 runs 21: the two m=2 random-sequential baselines (legacy and blocked
    kernel), m=2 partition, and the full cross m ∈ {4, 8, 16} × {random,
    partition} × {sequential, sum, mean}.
  - k8 runs 7 by default: the baselines plus m ∈ {2, 4, 8} × {random,
    partition} × sequential. Widen it with `--update`/`--m` after j18.
- **k grids:** j18 {25, 50, 100, 200, 400}, filtered by J's `valid_k`
  (k < (N−1)/2): orl has no 200 or 400, and seismic, svhn and har have no 400.
  k8 {100, 200, 400}. k=400 rows have no J reference to reproduce, since J
  stopped at 200; K-optimized does have k=400, so k8 rows do.
- **Timing:** `time_solver` is the full `fit_transform` wall time with
  `compute_stress=False`.
- **Scoring:** outside the timed region, batched per X, in chunks of 16.
  - `stress` is the stress-optimal rescale; `stress_raw` and `scale_alpha`
    are kept too.
  - A non-finite embedding is recorded as `stress=inf, diverged=True`.
- **Checkpointing:** per fit, keyed on (N, k, m, sampling, update, kernel,
  seed).
- **Machine:** time ratios are only meaningful within one run on one quiet
  machine. The legacy arm is re-run here rather than joined from K-optimized,
  which ran on a different VM.

## 6. Running it

```bash
cd sklearn/manifold/benchmarks
python bench_blocked_kernel.py                 # ~10 min: ns/pair by m, D, switch
python run_experiment_n.py --list
python run_experiment_n.py --suite j18         # est. ~4–7 h
python run_experiment_n.py --suite k8          # est. ~14–22 h (lean default)
python run_experiment_n.py --suite k8 --m 2,4,8,16 --update sequential,sum,mean  # widen after j18
```

The estimates come from the K-optimized cost model for the legacy arm, with
the other arms assumed to run at 0.4–0.7× legacy time. After widening the
grid, delete a dataset's `meta.json` to resume it; only the new arms run.

## 7. Outputs

- `results/experiment_n_{j18,k8}/<dataset>/results.csv`: one row per fit.
  Columns: arm, m, sampling, update, kernel, k, pairs_per_epoch,
  blocks_per_epoch, seed, time_solver, time_scoring, stress, stress_raw,
  scale_alpha, diverged, matches_legacy, repro_rel_err.
- `meta.json`: the reproduction summary and intrinsic-dimension measures.
- `results/bench_blocked_kernel/results.csv`.

The console summary, for each (N, k) and arm, shows:

- stress / legacy
- stress / cycle
- **fix**: legacy time ÷ arm time
- **block**: blocked-m2 time ÷ arm time
- on k8, the best SQuaD-MDS point from Experiment M
