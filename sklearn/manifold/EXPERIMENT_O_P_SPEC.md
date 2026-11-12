# Experiments O, P and Q — the paper's reporting runs

**Status:** design decisions settled (§13); cluster details pending. The algorithm
freeze and the harness (§10) are implemented and smoke-tested; nothing has run at
full size. Runners don't exist yet; `run_experiment_o.py`,
`run_experiment_p.py` and `run_experiment_q.py` are the proposed names.

**Prerequisite reading:** [CLAUDE.md](../../CLAUDE.md) §2 (paper storyline),
[EXPERIMENT_N_RESULTS.md](EXPERIMENT_N_RESULTS.md),
[EXPERIMENT_M_RESULTS.md](EXPERIMENT_M_RESULTS.md).

---

## 1. Why

The paper needs every time and stress number to be comparable. Today they aren't:

- K-optimized ran on a VM, M and N on a laptop, and A on an unrecorded machine.
- J's timings include in-fit O(N²) scoring, and paper Table 2 inherits them
  despite its caption.
- The original lazy kernel pays a GIL round-trip per update.
- Nothing has per-epoch stress history at large N.

So instead of re-timing the old experiments, the paper's numbers come from new
ones: **one per dataset suite** (O, P), plus a runtime-vs-N sweep (Q), each running
all its algorithms on the same inputs from one frozen commit. The aim is to gather
all the data first and choose the figures afterwards. The paper's three steps are then slices of the
same data:

| Paper step | Comparison | Source |
|---|---|---|
| 1 | SMACOF vs full SGDMDS | O |
| 2 | Full vs budgeted SGDMDS | O, P |
| 3 | Budgeted vs point-blocked SGDMDS, plus adapted SQuaD-MDS | O, P |

Q adds runtime scaling for all three steps. Stress-only findings from earlier
experiments are reused (§8).

---

## 2. Algorithms

| Name | Configuration |
|---|---|
| **Full SGDMDS** | `SGDMDS()`: the defaults are the golden config (`sampling_strategy='cycle'`, 30 epochs, `lr=0.5`, hybrid scheduler, switch 0.5, `epsilon=0.001`, `n_init=1`) |
| **Budgeted** | Same, `sampling_strategy='random'`, `n_updates_per_epoch=k·N`, `block_size=2` |
| **Point-blocked** | Same, `block_size=16`. Budget counted in pairs |
| **Adapted SQuaD-MDS** | Experiment M's grid (§3), which includes the authors' default |
| **SMACOF** | See §2.1 |

SGDMDS init is random. PCA init is deferred (§3).

### 2.1 SMACOF configuration

**Base:** `research/sgd-mds` is rebased on upstream `main` at 1.10.dev0
(`3c5e5effc`, 2026-11-04). SMACOF's code there matches release 1.9.1 apart from a
docs line, so the paper can cite 1.9.1. Re-check this before submission, because
1.10 may have shipped by then.

| Arm | Settings | Why |
|---|---|---|
| Default | `init='random'`, `n_init=1` | The 1.9.1 behaviour, and like-for-like with one SGD run. Pass `init` explicitly: the default still warns |
| Classical init | `init='classical_mds'` | Planned default for 1.10, and the one maintainers will compare against |

All arms use `n_init=1`, `eps=1e-6`, `max_iter=300`,
`metric_mds=True`, `metric='euclidean'`, `normalized_stress=False`, `n_jobs=1`, on
features.

**Experiment A's SMACOF is not the default one.** `run_smacof_benchmark` sets
`eps=1e-12`, so every run in A used all 300 iterations (its `n_init=1` with random
init does match today's default). Replaying sklearn's
stopping rule at `eps=1e-6` over A's saved histories (17 datasets, medians over
20 seeds, denominator from the saved embedding, so approximate) gives:

| | Forced 300 iterations | Default `eps=1e-6` |
|---|--:|--:|
| SMACOF time ÷ SGD time, median | 4.62× | 3.88× |
| Range | 2.84–6.57× | 0.76–6.36× |
| Datasets where SMACOF is faster | 0 | 1 (imdb) |
| SMACOF stress ÷ SGD stress, median | 1.080 | 1.080 |
| Datasets with lower SGD stress | 17 of 17 | 17 of 17 |

The paper's claim survives, but with less margin on time, and it should be
measured at the real default.

**Known limit, accepted:** 13 of 17 datasets still hit the 300-iteration cap at
`eps=1e-6`, so on most datasets SGD's lower stress is partly SMACOF being
unconverged. A longer SMACOF run was considered and left out, so the comparison is
against SMACOF as shipped, at its default budget. The paper should say so.

---

## 3. Settled before the runs

| | Question | Outcome |
|---|---|---|
| T1 | Budget k | k=200 is the headline; the ladder stays for trade-off plots |
| T2 | Block size m | m=16, sequential updates, random blocks |
| T3 | PCA init for SGDMDS | **Deferred.** See below |
| T4 | SQuaD-MDS settings | No separate tuning stage: O and P run Experiment M's grid as configured in `run_experiment_m.py`, and each dataset's best setting is picked from those results afterwards |

**Deferred: PCA init for SGDMDS.** Worth testing later: {random, PCA} × {full,
budgeted, blocked} on the small suite and 3 large sources at N=10k, 5 seeds, about
3–5 h, and it needs a small init option. The reasons to test it stand: SQuaD-MDS
gets a PCA init, sklearn's SMACOF is moving to a classical init, and N's leading
explanation for SQuaD-MDS's per-pair efficiency is its init. If it's tested after
the freeze and helps, the affected runs have to be repeated.

---

## 4. Experiment O: small suite, all five algorithms

**Data:** 17 datasets (J's 18 minus `orl`, which has 360 all-zero rows) at full N, one fixed X per dataset
shared by every method and seed. Unlike J, rows aren't permuted per seed, so input
distances can be computed once per dataset. O therefore doesn't reproduce J
bitwise; §11 covers the reproduction check.

**Seeds:** 10 (42–51).

| Arm | Grid |
|---|---|
| SMACOF | The two arms of §2.1 |
| Full SGDMDS | Golden config |
| Budgeted | k ∈ {50, 100, 200, 400}, with k < (N−1)/2 |
| Blocked | Same k |
| SQuaD-MDS | M's j18 grid: `n_iter` ∈ {100, …, 8000} × lr ∈ {150, 550, 1500} × exaggeration × init × step cap, 252 configs |

About 45,000 fits, of which ~43,000 are SQuaD-MDS. Budgeting isn't expected to be faster below N≈1,500. The paper
should show that rather than hide it.

---

## 5. Experiment P: large suite, without SMACOF

**Data:** the 8 K-optimized sources on the ladder {10k, 20k, 35k, 50k}
(california_housing: {10k, 20,640}), using K-optimized's fixed subsample
(`SUBSAMPLE_SEED=42`), so the budget arm should reproduce K-optimized's stress.

| Arm | Grid | Seeds |
|---|---|---|
| Full SGDMDS | Golden config; ~84 GB at 50k | 1 (3 at N=10k, for an error bar) |
| Budgeted | k ∈ {100, 200, 400} | 5 |
| Blocked | Same k | 5 |
| SQuaD-MDS | M's k8 grid: `n_iter` ∈ {100, …, 4000} × exaggeration, lr 550, PCA init, step cap 4; 12 configs | 5 |

SMACOF runs only on the small suite (decided 2026-11-12): at these sizes it is far
too slow, and the large suite is about SGDMDS's speedups and SQuaD-MDS. (The
pilot's california_housing still has 12 SMACOF fits at N=10k from before; the
analysis ignores them.) About 2,750 fits.

Rungs above 50k are out of scope: there's no full-sweep baseline there.

---

## 5a. Experiment Q: runtime vs N

Replaces Experiment B with the same data: `s_curve` (D=100), `swiss_roll` (D=300)
and `torus` (D=500) at N ∈ {500, 1000, …, 5000}, 10 repeats.

Arms: the SMACOF default arm, full SGDMDS, and budgeted and blocked at k=200. Same
measurement protocol as O. 1,200 fits.

B's lazy arm (random pairs at the full pair budget, computed on the fly) is off
by default (decided 2026-11-12): no one would choose it, and the cost of
computing distances on the fly already comes from the microbenchmark. It stays
runnable with `--random-auto`, into a separate results root.

---

## 6. Kernel microbenchmark

Extend `bench_blocked_kernel.py` to give the cost model for both optimisations:
ns per pair against D for the full-sweep kernel, the budget kernel and blocked
m=16, plus SQuaD-MDS's per-pair cost. Keep N=50,000 and the D grid
{3, 8, 64, 384, 784, 3072}. The full-sweep arm needs its own memory plan: at N=50k
its arrays alone are tens of GB. Under 1 h.

---

## 7. Measurement protocol

### 7.1 Timing

- `time_fit` is wall time from features to embedding, on one thread: the matrix
  build, PCA or classical init and scheduler setup are all inside it.
- Every fit runs under `threadpool_limits(1)`. Scoring runs outside it, with the
  job's other cores.
- Stress stays outside the timed region unless the algorithm needs it:

| Algorithm | Needs stress? | Consequence |
|---|---|---|
| SMACOF | Yes: stopping rule ([_mds.py:193-195](_mds.py#L193-L195)) | Stays in the timing. It reuses distances already computed for the iteration |
| SGDMDS (all forms) | No | `compute_stress=False`; snapshots through `_epoch_hook` |
| SQuaD-MDS | No | Snapshots through `squad_mds(callback=)` |

### 7.2 Stress history

SMACOF already has it: this fork's `_mds.py` records `(elapsed, stress)` every
iteration and exposes `stress_history_`. That is the exact stress SMACOF computes
anyway, so it costs nothing and needs no instrumented copy.

SGDMDS and SQuaD-MDS **save a copy of the embedding** at each snapshot and score
it after the fit. A copy costs O(N), so timing is barely touched; the hook times
itself and `time_fit` subtracts that.

| Algorithm | Snapshots |
|---|---|
| SGDMDS | Initial embedding, then every epoch |
| SQuaD-MDS | Initial embedding, then ~20 log-spaced iterations |

**Small suite: exact.** Input distances are computed once per dataset and reused
for every snapshot, seed and method (≤265 MB at N=5,750). Each snapshot then costs
only 2-D distances, ~0.2 s at worst.

**Large suite: fixed-pair estimate, then exact at the end.**

- Draw M = 10⁷ uniform pairs (i<j) once per (dataset, N), and compute their input
  distances δ once. That takes under a minute even at D=3,072, and storing them
  costs ~160 MB.
- For each snapshot, accumulate over those pairs `S_ee = Σd²`, `S_ed = Σdδ` and
  `S_dd = Σδ²`. With T = N(N−1)/2:
  - raw stress ≈ (T/M)(S_ee − 2S_ed + S_dd)
  - scale-optimal stress ≈ (T/M)(S_dd − S_ed²/S_ee)
- Each snapshot costs ~0.1 s, independent of D.
- All methods share the same pairs, so sampling errors largely cancel when
  comparing curves.
- **Validation:** every fit also gets an exact final score from the row-blocked
  scorer, and records the estimate's relative error at the last snapshot. Heavy
  tails are the risk (california_housing, mnist_digits). If any error exceeds
  0.5%, raise M.

All **table values are exact**. The estimate only draws curves.

Measured on three sources at N=10k (the spread follows exactly from the variance of
the squared residuals over all pairs): at M = 10⁷ the estimate's standard error is
0.05–0.13% for a trained embedding, and at most 0.47% for the random initial one
(california_housing, which is heavy-tailed). Building the sample takes 1 s at D=8
and 137 s at D=3,072, once per (dataset, N); each snapshot then takes 0.27 s.

### 7.3 Stress metric

Each fit reports exact raw stress, the stress-optimal scale α, and stress at that
scale. Tables use scale-optimal stress for every method; for SGDMDS and SMACOF α≈1,
so it barely differs from raw. Curves use raw stress for SMACOF vs SGDMDS, and
scale-optimal stress wherever SQuaD-MDS appears.

### 7.4 Two kinds of figure

The SGD schedule depends on `max_iter`, and SQuaD-MDS's decay and exaggeration
depend on `n_iter`. Epoch 15 of a 30-epoch run isn't a 15-epoch run. So:

- **Convergence curves** show the history within one run.
- **Trade-off plots** use final values across the k and `n_iter` ladders.

### 7.5 Summary metrics

- Ratio to full SGDMDS, same dataset and N.
- Median and IQR over seeds; Wilcoxon signed-rank across datasets for SGD vs SMACOF.
- **Time to target,** from the histories: time to reach SMACOF's final stress
  (step 1), and time to reach 1.01× full SGDMDS's stress (steps 2–3). This compares
  methods fairly when their configurations aren't equivalent.
- Every speedup is paired with its stress overhead at the same k.

### 7.6 Saved per fit

`results.csv` per dataset: arm, config, seed, `time_fit`, `time_scoring`, final
stress (raw, scaled, α), estimate error, iterations run. History goes in a sidecar
file. `meta.json` holds the machine fingerprint: host, CPU model, job ID, git commit
and dirty flag, BLAS library and architecture, library versions, peak memory, and
dataset checksums. Final embeddings are saved for seed 42, for embedding figures.

---

## 8. Reused without rerunning

These are stress-only claims, and stress reproduces across machines to 8.6e-12 for
budgeted fits. The upstream rebase didn't disturb them: all 26 scaled datasets, and
exact fit outputs for full, budgeted and blocked SGDMDS, SQuaD-MDS, Pivot MDS and
SMACOF on coil20, are bit-identical between 1.8.dev0 and 1.10.dev0.

| Claim | Source |
|---|---|
| Regime theory (`strain_top2`), when budgeting works | J, K-opt |
| k\* ∝ N^p scaling law | I-scaling, K-opt |
| Pivots fail, as "we also tried" | H, F, G |
| Pivot MDS stress baseline | L |
| Partition, sum and mean rejected | N |
| Budget and epoch cuts don't compose | J |

J and L ran `bank` at 2,000 of 2,059 points and `hiva` at 3,000 of 3,076. That is
accepted: O covers both at full N, and each old result is compared with a baseline
at the same reduced size. Footnote it if the paper quotes either as full size.

**From these sources, quote stress only.** In particular, "Pivot MDS is 7–51×
faster" is a timing claim and doesn't carry over. If the paper wants it, add Pivot
MDS as an arm of O and P (about 1 core-hour).

Superseded for the paper: A → O; K-optimized speed → P; J-based Table 2 → O and P;
M and N comparisons → O and P; B → Q.

---

## 9. Paper mapping

Provisional: figures get chosen once the data is in.

| Section | Figures and tables | Source |
|---|---|---|
| 1. SMACOF vs full SGDMDS | Stress and time table; convergence curves; time to reach SMACOF's stress; seed spread; runtime vs N | O, Q |
| 2. Budgeting | Overhead and speedup vs N at k=200; small-suite results including where it isn't faster; regime theory; cost model | O, P, §6, J |
| 3. Point blocking | Trade-off plot of full, budgeted, blocked and SQuaD-MDS; convergence curves; cost per pair vs D | O, P, §6 |

---

## 10. Implementation

Nothing below is started. It all stays in `sklearn/manifold/`.

**Algorithm freeze: done, not yet tagged.**

1. `noexcept` on `_pair_update` (was `_lazy_sgd_step`), `xorshift32` and
   `_interp_isotonic`. No `__Pyx_ErrOccurredWithGIL` remains in the generated C.
2. Research code removed from the estimator: pivot, hybrid, partition, sum and mean
   updates, `FORCE_BLOCKED_KERNEL`. Pivot selection moved to `_pivot_mds.py`.
3. **The random kernel stays,** contrary to the original plan. After the fix it is
   bit-identical to the blocked kernel at m=2 but up to 35% faster at low D, so
   removing it would have slowed the budgeted arm and flattered blocking.
4. Parameter names follow `MDS`: `metric_mds`, `metric`, `metric_params`.
   `sampling_strategy` replaces `dissimilarity='lazy'`, since `'cycle'` always
   stores the matrix and `'random'` never does. The proposed `precompute_distances`
   was unnecessary. Unused `tol`, `batch_size` and `n_jobs` are gone.
5. Defaults are the golden configuration, and `block_size` defaults to 16.
6. `stress_history_` and all in-fit scoring are gone. Private hooks replace them:
   `SGDMDS._epoch_hook` and `squad_mds(callback=)`.
7. Verified: 12 golden embeddings, 26 scaled datasets and 7 exact fits are
   bit-identical to the code before the freeze, and `check_estimator` passes on both
   strategies.

**Harness: done, in `benchmarks/`.**

- `paper_harness.py`: timed fits for SGDMDS, SMACOF and SQuaD-MDS under
  `threadpool_limits(1)`; the snapshot `Recorder`, which removes its own time;
  `ExactScorer` (input distances cached once per X) and `PairSampleScorer`; batched
  exact final scores; `ResultStore` (per-fit resume, `results.csv`,
  `histories.jsonl`, seed-42 embeddings); and the machine fingerprint in
  `meta.json`.
- `run_experiment_o.py`, `run_experiment_p.py`, `run_experiment_q.py`: one process
  per dataset. `--quick` and `--seeds` shrink the grid for smoke tests; P also takes
  `--ladder` and `--score-pairs`.
- `compare_results.py`: the reproduction checks of §11.
- `bench_blocked_kernel.py`: gains the `squad` and `cycle` arms of §6.
- `bench_utils.py`: thread limits inside every wrapper's timed region, and
  `run_smacof_benchmark` without deprecated arguments. Its `eps` stays 1e-12 by
  default, so A and B still mean what they meant.
- **Plotters for the new tree are not written:** figures are chosen after the data.

Results go to `SGDMDS_RESULTS_ROOT`, default `benchmarks/results_paper` (ignored by
git), and the reference for comparisons to `SGDMDS_REFERENCE_ROOT`, default
`benchmarks/results`.

**Not for the PR:** this fork's `stress_history_` patch to `_mds.py` changes
`smacof`'s return values. Keep it in the research branch.

---

## 11. Integrity checks

Before the campaign (all checked on the laptop):

- Full manifold tests pass, including bit-identity.
- Thread limits hold: BLAS reports one thread inside a fit, recorded in `meta.json`.
- Snapshots don't change the embedding (`test_epoch_hook`), and the hook costs
  about 0.1 ms per fit.
- The exact scorer matches `stress_exact_chunked` to 1e-15, and the pair estimate's
  error is as stated in §7.2.
- Smoke runs of O, P and Q resume per fit, and O's SQuaD-MDS reproduces
  Experiment M to 9e-15.

After it:

- P's budget arm reproduces K-optimized's stress, for the same (N, k, seed).
- P's full arm vs K-optimized (seed 42). This is the first cross-machine check
  through BLAS.
- The original kernel's stress in O vs a few J-convention fits run for this purpose.
- The microbenchmark reproduces the laptop's ordering: random fastest at low D, blocked fastest at high D.
- Estimator error across all P fits.

---

## 12. Run order and compute

### Safeguards: no run is in vain

| Risk | Protection |
|---|---|
| Other jobs slow a fit | Contention can only change a fit's time, never its result. A canary runs **between** fits at most every 15 min (`canary.csv`, and the latest value in every row). `retime.py` reruns selected fits (`--slow F` by canary, or `--config RE`), checks each computes exactly what the original did, and writes only new times to `retimed.csv` |
| Job killed (10 h limit, node failure) | Per-fit resume; P scores at least every 15 min of fitting |
| Code or libraries change mid-campaign | Each dataset's results are tied to the provenance of their first run (git commit, library versions, BLAS kernel, CPU model); a runner refuses to resume under a different one |
| Running uncommitted code | Refused unless `SGDMDS_ALLOW_MIXED=1` (development smoke tests only) |
| A bug or a bad setting | `cluster/pilot.sh` first: O on coil20 and epileptic, P on california_housing, Q on torus, at full size. Its results are kept; the full launch skips what it finished |

**Pilot (2026-11-12, coil20, epileptic, california_housing, torus):** all 5,950
fits complete, no divergences, one BLAS thread inside fits, snapshot overhead at
most 3 ms; pair-estimate error median 0.11%, max 0.30%. Reproduction: P budget vs
K-optimized 1e-13, **P full vs K-optimized 3e-14** (so K-optimized ran on this
AVX-512 hardware), SQuaD-MDS vs M k8 2.5e-12, vs M j18 2e-14 median (0.2% max,
since M j18 ran on the laptop). Two changes followed: the full sweep's peak memory
is ~49 bytes per N² (20.9 GB at N=20,640), so P requests 200 GB; and arms now run
seed by seed (`seed_major`), so the arms being compared are timed over the same
stretches of a job. The canary's D=8 value ranged 11.6–19.4 ns, likely boost
clock varying with node load (D=784 varied only 1.2×).

**Calibration (2026-11-11, exclusive bare-metal node):** 8, 16 and 32 copies of
the kernel benchmark pinned to consecutive cores ran 1.35×, 1.40× and 1.38×
slower at the median (worst 1.9–2.5×). The flat curve points at the 8-core
chiplets: each held 8 memory-bound copies in every case. Hence 8 cores per job
(`-c 8`, roughly one chiplet), and at most 4 O, 4 P and 2 Q jobs at once (~80
cores, ~30% of the two nodes). Adjust a running array with
`scontrol update JobId=<array job> ArrayTaskThrottle=N`. The microbenchmark's
`cycle` arm (3.6 ns per pair) excludes the per-epoch reshuffle of all pairs, so
the full sweep's real per-pair cost must come from P's full-sweep fits.


| Step | What | Where | Est. core-h | Peak RAM |
|---|---|---|--:|---|
| 1 | Harness (§10) | Laptop | — | — |
| 2 | Setup: rsync, build, checksums, smoke test | Cluster | — | — |
| 3 | Microbenchmark and contention calibration | Cluster | ~1 | 3 GB+ (§6) |
| 4 | O | Cluster | 25–40 | ~5 GB |
| 5 | P | Cluster | ~60 | **~100 GB** at 50k full; ~16 GB otherwise |
| 6 | Q | Cluster | 10–15 | ~2 GB |
| 7 | Checks, plots, results docs | Laptop | — | — |

Estimates scale recorded fit and scoring times on the old machines; treat them as
rough, and allow up to double on slower server cores. O's largest part is
SQuaD-MDS's grid (~10 h, plus up to ~7 h of exact snapshot scoring at 0.07 s per
snapshot at N=5,750), then SMACOF (0.12 h per seed for
all datasets at 300 iterations, measured in A with unpinned BLAS, so one thread will
be slower by an unknown factor). P's largest parts are the full-sweep baseline
(~15 h), budgeted (~15 h) and SQuaD-MDS (~7 h), plus ~5 h of estimated
snapshot scoring and the batched exact final scores.

Wall-clock is about 1–2 days if jobs run in parallel per dataset. The longest single
job is likely cifar10_raw in P.

Commands, from `benchmarks/`, one process per dataset:

```bash
export SGDMDS_RESULTS_ROOT=$PWD/results_paper   # the default
export SGDMDS_MAX_MEM_GB=<node RAM>              # P only: skip full sweeps that won't fit
python bench_blocked_kernel.py
python run_experiment_o.py <dataset>             # --list shows the datasets
python run_experiment_p.py <dataset>
python run_experiment_q.py <dataset>
python compare_results.py                        # after O and P
```

Smoke test on a new machine (minutes, into a scratch root):

```bash
SGDMDS_RESULTS_ROOT=/tmp/smoke python run_experiment_o.py seismic --seeds 1 --quick
SGDMDS_RESULTS_ROOT=/tmp/smoke python run_experiment_q.py torus --n 500 --seeds 1
SGDMDS_RESULTS_ROOT=/tmp/smoke python run_experiment_p.py california_housing \
    --ladder 2000 --seeds 1 --quick --score-pairs 1000000
SGDMDS_RESULTS_ROOT=/tmp/smoke python bench_blocked_kernel.py --quick
```

Job scripts wait for the cluster details (§13).

### Cluster practicalities

- **Frozen checkout:** a dedicated clone at the tag, never edited during the
  campaign. The editable install rebuilds on import, so an edit would silently
  change the kernel mid-run. Smoke-test concurrent imports, since many jobs would
  share one build directory.
- **Data:** rsync `datasets/` (~29 GB) and `results/` (the reference tree, 253 MB).
  Compare checksums on both sides with `find -L`, because some datasets are symlinks.
- **Cores:** one CPU model throughout, `--cpu-bind=cores`-style pinning, and one
  process per physical core.
- **Scoring threads:** ~4 cores per job, so scoring gets threads without touching
  neighbours' cores.
- **Contention:** the budget kernel runs at memory bandwidth, so neighbouring jobs
  can slow each other. Calibrate jobs per node so the slowdown stays under 3%. Exact
  isolation matters less than one machine: the claims are ratios of 2.7× and up.
- **Memory:** pin the 50k full fit to one NUMA node if one has the RAM. The 36·N²
  estimate is extrapolated above N=20k.
- **BLAS at one thread:** the matrix build gets slower, by ~3.5% of a full fit at
  D=3072 by rough estimate. That slightly favours budgeted speedups; state it.

---

## 13. Decisions

**Open:** choose SQuaD-MDS's setting per dataset on O's seeds 42–46 and report it on
47–51 (see §14, point 1). No code change needed.


Settled 2026-11-05:

1. **SMACOF arms:** the default and `classical_mds`, both at `max_iter=300`.
   **The comparison is random-init SMACOF vs random-init SGDMDS** (decided
   2026-11-12): same starting point, so it compares the algorithms. The
   classical-init arm still runs, as data for a later study of classical init for
   both methods; it isn't part of the main comparison.
2. **orl:** dropped from O.
3. **PCA init:** deferred (§3).
4. **SQuaD-MDS:** no tuning stage; rerun M's grid inside O and P.
5. **Rungs above 50k:** skipped.
6. **Seeds:** 10 for O; 5 for P, with full SGDMDS at 1 (3 at N=10k).
7. **Figures:** gather the data first. B's runtime-vs-N data stays, as Q.

### The COMA cluster (docs.coma-cluster.de, read 2026-11-11)

- **Slurm, partition `compute`** (from `sinfo`, 2026-11-11, which differs from the
  docs): an H100 node (Intel Xeon Gold 6548N); two bare-metal H200 nodes
  `gpu-nvidia-h200-2` and `-3` (2× AMD EPYC 9555 Turin, 256 hardware threads,
  2.3 TB each); and three KVM virtual machines `gpu-nvidia-h200-1-studvm-[1-3]`
  (56 vCPUs, 440–500 GB each) carved out of `gpu-nvidia-h200-1`, which is drained.
  **All runs use the two bare-metal H200 nodes**
  (`--exclude=gpu-nvidia-h100,gpu-nvidia-h200-1-studvm-[1-3]`): one CPU model, and
  no hypervisor or hidden VM neighbours in the timings. Other nodes sit in
  `develop`, which needs a ticket.
- **CPU limits are not enforced,** and the nodes are shared with GPU jobs. Every
  run therefore records a canary timing at start and end (`meta.json`,
  `canary_ns_per_pair`); a job with a slow canary ran on a contended node.
- `--exclusive` is allowed for benchmarks: used for the microbenchmark and the
  contention calibration only (`cluster/calibrate.sbatch`). On a bare-metal node it
  also blocks 8 H200 GPUs, so run it at a quiet time.
- **Storage:** home (`/storage/home/<user>`, snapshotted daily and backed up
  off-site to LRZ) holds the repo (`~/scikit-learn`), Miniforge and the results
  (`~/sgd-mds-results`); `~/nobackup` (snapshotted, no off-site backup) holds the
  datasets (`manifold-datasets`, linked as `benchmarks/datasets`) and the laptop's
  results for comparison (`manifold-results-reference`). Both are NFS on every node.
- **Software:** Spack modules; Python environments via Miniforge (not Anaconda's
  installer) or venv. Our `environment.yml` uses conda-forge only.
- **Login node:** no heavy work, and processes die at logout; build inside an
  allocation (`cluster/setup.sh`).
- **Access:** eduVPN (MWN split tunnel), then `ssh login.coma-cluster.de`.
- **Paper acknowledgement:** "Experiments presented in this work were carried out
  using the CIT-TUM-HN cluster at TUM Campus Heilbronn."

Scripts in `benchmarks/cluster/`: `setup.sh` (one-time), `env.sh`, and the jobs
`smoke`, `calibrate`, `o`, `p` and `q` (`.sbatch`). Submit from that folder, in this
order: smoke, calibrate, then O, P and Q with the array throttles (`%8`) set from
the calibration, e.g. `./submit.sh p.sbatch 4`.

**Bitwise reproducibility is per instruction set.** The H200 nodes have AVX-512
(AMD Zen 5); the laptop doesn't. OpenBLAS (Cooperlake vs Haswell kernels) and
NumPy's `exp` then round the last digit differently, and SGD amplifies that: the
full sweep (distances from BLAS) gives a different, equally good embedding, and
some learning-rate schedules differ by one ulp. Budgeted and blocked fits on
identical inputs stay bit-identical. Consequences: all reporting runs happen on the
cluster, so they are mutually consistent; comparisons with the laptop or VM
results are statistical for anything touching BLAS; and the golden tests use
CPU-independent inputs (pdist distances, 10-epoch schedules).

**Job length:** every job is capped at 10 hours (Daniel's practice on COMA).
Runners resume per fit and P scores at least every 15 minutes of fitting, so a job
that hits the limit loses at most ~15 minutes plus the fit in flight; the longest
single fit (the full sweep at N=50k) takes about 70 minutes. `cluster/submit.sh`
queues several 10-hour runs one after another. Memory is no constraint (nodes have
2.3 TB). The calibration takes ~20–40 minutes of one exclusive node; it is
requested for one hour.

---

## 14. Known asymmetries

The comparison is fair where it matters: every method sees the same input, runs
on one thread, pays its own setup (distance matrix, PCA or classical init) inside
its time, is scored the same way, and avoids stress it doesn't need inside the
timed region. Budgeted and blocked runs use the same pair budget. These
asymmetries remain, and the paper should state them:

1. **SQuaD-MDS gets far more tuning:** 252 settings in O, best per dataset picked
   afterwards, against one fixed SGDMDS configuration. Picking the best of 252 on
   the same seeds also flatters it slightly. Proposed fix: select on seeds 42–46,
   report on 47–51 (O only; P has 5 seeds and 12 settings).
2. **Initialisation differs:** SQuaD-MDS starts from PCA, SMACOF's classical arm
   from classical MDS, SGDMDS at random (PCA init is deferred, §3). This favours the
   other methods.
3. **SMACOF stops at 300 iterations,** at its default budget, and hits the cap on
   13 of 17 datasets. Part of SGD's stress advantage is SMACOF being unconverged.
4. **P's full-sweep baseline has one seed above N=10k,** so P's overhead ratios rest
   on a single run per (dataset, N); the three seeds at N=10k show the noise.
5. **SMACOF's curves are raw stress only** (from its own record). Fine against
   SGDMDS, where α≈1; compare SQuaD-MDS curves at the best scale.

Unknown until the cluster calibration: how much jobs sharing a node slow each other.

