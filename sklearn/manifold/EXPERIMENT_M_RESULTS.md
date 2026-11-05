# Experiment M — results & conclusions (SQuaD-MDS stress baseline, all 26 datasets)

**Status: complete.** Both suites, 5 seeds each, **17,800 fits** total (16,200 on
j18, 1,600 on k8). Runner:
[benchmarks/run_experiment_m.py](benchmarks/run_experiment_m.py) ·
Implementation: [_squad_mds.py](_squad_mds.py) + [_squad_mds_cython.pyx](_squad_mds_cython.pyx) ·
Spec: [EXPERIMENT_M_SPEC.md](EXPERIMENT_M_SPEC.md) ·
Validation: [benchmarks/validate_squad_mds.py](benchmarks/validate_squad_mds.py)

A *stochastic quartet* baseline for the whole benchmark collection: SQuaD-MDS
(Lambert et al., 2022) partitions the points into N/4 random disjoint quartets
each iteration and follows the gradient of a cost over each quartet's six
pairwise distances. O(N·D) per iteration, no distance matrix.

**Headline: the paper's "relaxation" claim is too strong as written.** SQuaD-MDS
reaches **1.04× full-cycle stress (median, j18)** and **1.08× (median, k8)** —
close enough that "optimizes a relaxation of stress rather than stress itself"
misleads about the *outcome* even though it is accurate about the *objective*.
The claim needs softening. But there is a real finding underneath it: **the
method at its published default is 1.53× median, and almost the whole gap is one
heuristic (`exaggerate_d`), not the relaxation.**

---

## 1. Conditions

| | Suite A (`j18`) | Suite B (`k8`) |
|---|---|---|
| datasets | the 18-dataset collection | the 8 large sources |
| N | each dataset's **full N** (400–5,750) | ladder {10k, 20k, 35k, 50k, 70k, 100k} |
| subsample | none (full N) | K-optimized's fixed subsample (seed 42) |
| SGD reference | Experiment J | Experiment K-optimized |
| seeds | 5 | 5 |
| grid | 180 configs | 8 configs |
| fits | 16,200 | 1,600 |

Grid: `n_iter` × `lr` × `exaggerate_d` × `init` × `max_step_frac`. j18 carries
the full cross (5 × 3 × 2 × 2 × 3); k8 runs lean (4 × 1 × 2 × 1 × 1) at the
reference learning rate, since each fit is ~100× more expensive.

### What the method actually optimizes (verified, not assumed)

Per quartet, **both** the high-dimensional and the embedding distances are
divided by their own sum before the squared error is taken:

```
delta_hat_p = delta_p / sum(delta)      d_hat_p = d_p / sum(d)
L_Q = sum over the 6 pairs of (delta_hat_p - d_hat_p)^2
```

The cost is therefore **scale-free** — invariant to a global rescaling of the
embedding — and the method imposes no scale constraint by design (paper §2.2).
Confirmed two ways: by deriving the gradient by hand and matching the authors'
kernel term-for-term, and by finite differences of that loss.

### Validation evidence (spec §5)

| check | result |
|---|---|
| **End-to-end vs the authors' released code** | **5e-15** relative over 120 iterations, with and without exaggeration |
| Gradient vs their scalar kernel | 2e-16 relative |
| Gradient vs finite differences of the stated loss | 6e-9 relative |
| **RNX AUC vs SMACOF** (coil20, the paper's own comparison) | 0.389 vs 0.423 = **0.921 ratio**; paper claims "often similar" |
| Loss decrease (fixed held-out quartets) | monotone, final/initial 0.669 |
| O(N) per iteration | log-log slope **1.040** for N ≥ 16,000 |
| Degenerate check | distance correlation 0.820 (feynman), 0.963 (california_housing) |

The reference-equivalence check is the load-bearing one — it plays the role
Experiment L's "Pivot MDS reduces to ClassicalMDS at k=N to 3e-14" played there,
and is pinned in [tests/test_squad_mds.py](tests/test_squad_mds.py) with the
authors' kernel transcribed verbatim as an oracle.

### Three methodological requirements

1. **The embedding has no meaningful global scale** and must be rescaled before
   its stress means anything. We apply the closed-form optimum
   `alpha = <D_emb,D>/<D_emb,D_emb>`. Measured α spans **0.11–3.47** across
   datasets, so this is a large correction. The spec's subtler worry — that
   per-quartet normalisation might defeat a single global α — **does not
   materialise**: α is tightly clustered within each dataset (spread < 0.01
   across all 5 seeds and every config on k8).
2. **Timing covers the same thing on both sides.** Both run from raw array to
   finished embedding with stress evaluation excluded. SQuaD-MDS's PCA
   initialisation is *inside* its timed region (the authors' own `main.py` puts
   it outside), so our timings are if anything pessimistic for SQuaD-MDS.
3. **The kernel is compiled on both sides.** SGD-MDS's hot loop is Cython, so a
   NumPy SQuaD-MDS would have compared kernels rather than algorithms. We wrote
   [_squad_mds_cython.pyx](_squad_mds_cython.pyx), which fuses the whole quartet
   pass; it is 5.3–8.7× faster than the NumPy path and agrees with it to 1.9e-15.

---

## 2. Quality

### Suite A — the 18 datasets (full N)

Ratios to full-cycle SGD stress at the same (dataset, N, seed). "SQuaD best" is
the best over all 180 configs — generous, matching how Experiment L gave Pivot
MDS its best k and best strategy.

| dataset | D | N | 2D-emb | **SQuaD best** | authors' default | SGD k=200 |
|---|--:|--:|--:|--:|--:|--:|
| seismic | 24 | 646 | 35.9% | 1.05× | 1.81× | 1.01× |
| spambase | 57 | 4,601 | 20.8% | **1.19×** | 1.83× | 1.06× |
| bank ¹ | 63 | 2,059 | 14.3% | — | — | — |
| hatespeech | 100 | 3,221 | 4.5% | 1.03× | 1.18× | 1.04× |
| epileptic | 178 | 5,750 | 13.9% | 1.16× | 1.62× | 1.10× |
| sentiment | 200 | 2,748 | 2.4% | 1.02× | 1.14× | 1.01× |
| orl ² | 396 | 400 | 90.1% | **16.03×** | 439.79× | 1.06× |
| coil20 | 400 | 1,440 | 34.2% | 1.06× | 1.25× | 1.02× |
| sms | 500 | 835 | 2.6% | 1.02× | 1.77× | 1.00× |
| har | 561 | 735 | 57.3% | 1.08× | 1.59× | 1.03× |
| secom | 590 | 1,567 | 10.7% | 1.11× | 1.80× | 1.04× |
| imdb | 700 | 3,250 | 1.5% | 1.04× | 1.05× | 1.00× |
| fashion_mnist | 784 | 3,000 | 37.8% | 1.06× | 1.44× | 1.01× |
| cnae9 | 856 | 1,080 | 3.6% | 1.02× | 2.87× | 1.00× |
| svhn | 1024 | 732 | 63.1% | **0.98×** | 1.31× | 1.01× |
| cifar10 | 1024 | 3,250 | 40.2% | 1.03× | 1.17× | 1.01× |
| fmd | 1536 | 997 | 22.1% | 1.03× | 1.17× | 1.01× |
| hiva ¹ | 1617 | 3,076 | 8.5% | — | — | — |

¹ **No reference.** Experiment M runs these at the corrected full N (2,059 /
3,076); Experiment J is still at its old truncated rung (2,000 / 3,000). Their
absolute stress is recorded — bank 2.990e+07, hiva 1.812e+09 — but no ratio is
available until J is re-run. Stress scales with N², so approximating across a 3%
N gap would introduce ~6% error, the same size as the effects being measured.

² **orl is a broken dataset, not a method failure.** 360 of its 400 rows are
identical, which produces ~65 fully-collapsed quartets per iteration and an
O(lr/S) gradient blow-up. See §6.

**Excluding orl: SQuaD best 0.98–1.19× (median 1.04×) vs SGD k=200 1.00–1.10×
(median 1.01×). SQuaD-MDS wins on 2 of 16.**

For scale: Pivot MDS on this same suite (Experiment L) was **1.38–12.42×, median
3.34×**. SQuaD-MDS is in an entirely different class as a stress baseline.

### Suite B — the 8 large sources

| dataset | D | N=10k | N=20k | N=35k | N=50k | SGD k=200 @50k |
|---|--:|--:|--:|--:|--:|--:|
| feynman_iii_15_12 | 3 | 1.09× | 1.05× | 1.06× | 1.11× | 1.00× |
| california_housing | 8 | 1.27× | **4.60×** ³ | — | — | 1.02× |
| ag_news | 384 | **1.06×** | **1.06×** | **1.07×** | **1.07×** | 1.10× |
| patchcamelyon | 512 | 1.08× | 1.08× | 1.08× | 1.08× | 1.02× |
| tiny_imagenet | 512 | **1.08×** | **1.08×** | **1.08×** | **1.08×** | 1.10× |
| fashion_mnist_full | 784 | 1.14× | 1.11× | 1.12× | 1.11× | 1.01× |
| mnist_digits | 784 | 1.69× | 1.55× | 1.56× | 1.52× | 1.10× |
| cifar10_raw | 3072 | 1.05× | 1.05× | 1.05× | 1.05× | 1.01× |

³ at N=20,640 (full N). Bold = SQuaD-MDS beats SGD k=200.

**SQuaD best 1.05–4.60× (median 1.08×) vs SGD k=200 1.00–1.10× (median 1.02×).
SQuaD-MDS wins on 8 of 30 cells.** The 70k and 100k rungs ran but have no cycle
denominator (see §7), so they contribute absolute stress and timings only.

**Quality is flat in N.** ag_news, patchcamelyon, tiny_imagenet and cifar10_raw
are constant to two decimals across a 5× range of N. Whatever SQuaD-MDS's gap
is on a dataset, it is a property of the dataset, not of its size.

---

## 3. Speed — 7–14× faster than budgeted SGD, both sides clean

Suite B only: Experiment K-optimized sets `compute_stress=False` and Experiment
M scores outside the timed region, so this is the one apples-to-apples
comparison available. (Experiment J does **not** set that flag, so Suite A's
SGD times include in-fit O(N²) scoring and are inflated — see §7.)

| dataset | D | N | SQuaD | SGD k=200 | cycle | vs k=200 | vs cycle |
|---|--:|--:|--:|--:|--:|--:|--:|
| feynman_iii_15_12 | 3 | 50,000 | 2.3 s | 15.6 s | 4,183 s | 6.9× | 1,876× |
| california_housing | 8 | 20,640 | 0.9 s | 6.6 s | 687 s | 7.0× | 732× |
| ag_news | 384 | 50,000 | 12.8 s | 140.2 s | 4,151 s | 11.0× | 323× |
| tiny_imagenet | 512 | 50,000 | 15.9 s | 167.2 s | 4,155 s | 10.5× | 269× |
| patchcamelyon | 512 | 50,000 | 13.3 s | 167.8 s | 4,230 s | 12.6× | 315× |
| fashion_mnist_full | 784 | 50,000 | 18.7 s | 215.9 s | 4,233 s | 11.6× | 228× |
| mnist_digits | 784 | 50,000 | 16.2 s | 226.8 s | 4,289 s | 14.0× | 264× |
| cifar10_raw | 3072 | 50,000 | 44.0 s | 609.2 s | 4,354 s | 13.8× | 99× |

**6.9–14.2× faster than SGD k=200 across all 30 cells (median 11.1×), and
faster on 30 of 30.**

### The mechanism, and why 11× is exactly what it should be

Both kernels do **identical arithmetic per pair** (2·D flops) and differ only in
memory traffic:

- **SGD-MDS** draws i and j independently, loads two rows, uses them for **one**
  pair → **2·D loads per pair**.
- **SQuaD-MDS** loads four rows and extracts **six** pairs → **0.667·D loads per
  pair**.

That is a **3× traffic advantage**, and it is pure cache blocking: *m* points
contain m(m−1)/2 pairs, so loads per pair go as 2D/(m−1). On top of that,
SQuaD-MDS at n_iter=1000 does 1500·N pair-constraints against SGD k=200's
6000·N — **4× fewer pairs**. 3 × 4 = **12×**, against the measured median 11.1×.

So the honest statement is:

> SQuaD-MDS is not faster because its objective is cheaper. It does the same
> per-pair work, but extracts 3× more pairs from each byte of data it moves,
> and runs 4× fewer of them.

Measured per-pair cost confirms this: 30 / 90 / 182 ns at D = 24 / 400 / 784,
against budgeted SGD's measured `52 + 0.6·D` = 66 / 292 / 522 ns.

**The quartet's speed advantage is separable from its objective.** Nothing about
the 3× requires the relative-distance normalisation — it requires *grouping*.
See §8.

---

## 4. Configuration — dependence, and whether it can be predicted

### 4.1 How much does configuration matter?

A great deal in the raw, and almost nothing once two binary flags are set
correctly.

| scope | median spread | worst |
|---|--:|--:|
| full 180-config grid | **3.7×** | 13.3× (epileptic) |
| fixing `exaggerate_d=False`, `init=pca`, `n_iter ≥ 500` | **1.28×** | 6.6× (cnae9) |
| additionally fixing `lr=1500`, `n_iter=2000` | **1.007×** regret | 1.28× (seismic) |

So the dependence is real but *structured*. It is not a method that needs
per-dataset tuning; it is a method with two settings that must not be wrong.

### 4.2 The axes, ranked

| axis | right answer | wins on | cost of the wrong answer | data-dependent? |
|---|---|--:|--:|---|
| `exaggerate_d` | **False** | 15 of 16 | 1.21× vs 1.04× median | **no** — universal |
| `init` | **pca** | 15 of 16 | 1.39× vs 1.05× median | **no** — universal |
| `lr` | 1500 (see 4.3) | — | up to 6× on 5 of 15 | **yes** |
| `n_iter` | as high as affordable | — | monotone | no — a budget knob |
| `max_step_frac` | any | — | inert | no |

`n_iter` is **not a tuning knob**: quality improves monotonically with it on
every dataset, and every best config in both suites sat at the grid ceiling
(§7). It buys quality at exactly linear cost — the Pareto dial, not a choice.

### 4.3 Learning rate is the only genuinely data-dependent knob, and it is bimodal

At `n_iter=2000, exaggerate_d=False, init=pca`, the spread across
lr ∈ {150, 550, 1500} has a **median of 1.03×** — irrelevant on 10 of 15
datasets. On the other five it dominates everything else:

| dataset | lr=150 | lr=550 | lr=1500 | spread |
|---|--:|--:|--:|--:|
| cnae9 | 6.05× | 1.91× | **1.19×** | **5.10×** |
| sms | 2.45× | 1.10× | **1.02×** | 2.40× |
| secom | 2.24× | 1.30× | **1.11×** | 2.01× |
| epileptic | 2.09× | 1.18× | **1.16×** | 1.79× |
| spambase | 2.12× | 1.23× | **1.19×** | 1.79× |
| *(the other 10)* | — | — | — | 1.01–1.28× |

Every sensitive dataset wants the *highest* learning rate tested, which raises
the possibility that 1500 is itself a ceiling rather than an optimum — untested.

### 4.4 Recommended universal configuration

`n_iter=2000, lr=1500, exaggerate_d=False, init='pca'`, regret measured against
each dataset's own best over all 180 configs:

| configuration | median regret | 90th pct | worst |
|---|--:|--:|--:|
| **i2000, lr1500, noexa, pca** | **1.007×** | **1.104×** | **1.276×** (seismic) |
| i2000, lr550, noexa, pca | 1.000× | 1.133× | 1.880× (cnae9) |
| i2000, lr150, noexa, pca | 1.017× | 2.246× | 5.947× (cnae9) |
| i1000, lr550, noexa, pca | 1.016× | 1.334× | 2.092× (cnae9) |
| **i1000, lr550, exa, pca** (authors' default) | **1.390×** | 1.733× | 2.823× (cnae9) |

lr=1500 is preferred over lr=550 on **worst case** despite a marginally worse
median: it never fails badly, whereas lr=550 collapses on cnae9. Switching the
two flags away from the authors' default is worth ~39% of stress on a typical
dataset.

### 4.5 Can the right configuration be predicted from the data? Mostly no — and mostly it need not be

Every scalar recorded in `meta.json` (N, D, `strain_top2`,
`pca_participation_ratio`, `strain_participation_ratio`, `pca_n90`, `pca_n95`)
tested by Spearman correlation against three targets:

| target | best predictor | ρ | p | verdict |
|---|---|--:|--:|---|
| achievable stress ratio | N (45 cells) | +0.42 | 0.005 | **artifact** — see below |
| achievable stress ratio | N (15 j18 datasets) | +0.43 | 0.106 | not significant |
| which `lr` wins | `pca_participation_ratio` | +0.54 | 0.038 | fails correction |
| how much `lr` matters | `strain_top2` | −0.44 | 0.101 | not significant |

**The N correlation does not survive scrutiny.** It is between-dataset, not
within: at fixed dataset the ratio is flat across a 5× range of N, and
*improves* on mnist_digits.

| dataset | 10k | 20k | 35k | 50k | |
|---|--:|--:|--:|--:|---|
| ag_news | 1.06× | 1.06× | 1.07× | 1.07× | flat |
| patchcamelyon | 1.08× | 1.08× | 1.08× | 1.08× | flat |
| tiny_imagenet | 1.08× | 1.08× | 1.08× | 1.08× | flat |
| cifar10_raw | 1.05× | 1.05× | 1.05× | 1.05× | flat |
| fashion_mnist_full | 1.14× | 1.11× | 1.12× | 1.11× | flat |
| mnist_digits | 1.69× | 1.55× | 1.56× | 1.52× | **improves** |

So ρ = +0.42 across cells is only "the larger sources in this collection happen
to be harder for unrelated reasons". **Dataset size carries no information**
about how well SQuaD-MDS will do on it.

**The `pca_participation_ratio` → `lr` signal is a hypothesis, not a finding.**
p = 0.038 with n = 15 and seven predictors tested; a Bonferroni threshold would
be p < 0.007. Note also that `pca_participation_ratio` and
`strain_participation_ratio` return *identical* ρ and p on this collection —
they are rank-equivalent here, so they are not independent evidence. The
direction (higher effective dimensionality → higher learning rate) is at least
mechanistically plausible and deserves a targeted test, but must not be quoted
as a result.

**`strain_top2` carries no information about SQuaD-MDS** (ρ = +0.17, p = 0.54),
matching §7's finding and Experiment L's for Pivot MDS.

**The practical consequence is mild**, because the two axes that matter most
have universal answers and the third rarely matters. Fix the flags, set
lr = 1500, spend `n_iter`: within 1.01× of per-dataset-optimal typically,
1.28× worst case. If the last few percent matter on a specific dataset, the lr
ranking is stable across `n_iter` (verified at 500 / 1000 / 2000), so a 3-point
lr sweep at `n_iter=250` picks the winner cheaply before the real budget is
spent.

### 4.6 Where the published default's 1.53× goes

**`exaggerate_d` is the whole story.** It uses *squared* high-dimensional
distances for the first 60% of iterations, which after sum-normalisation
stretches a quartet's far/near target ratio from 6× to 36×. It is **absent from
the paper** — a code-only feature that `main.py` turns on by default — and it is
tuned for RNX, the metric the paper reports. Measured on coil20 the two metrics
move in opposite directions:

| | RNX AUC | stress/cycle |
|---|--:|--:|
| exaggeration on (authors' default) | **0.389** | 1.25× |
| exaggeration off | 0.331 | **1.07×** |

Two exceptions worth noting: on **ag_news and tiny_imagenet** (k8's two
low-`strain_top2` sources) exaggeration *wins*, and they are also the two
datasets where SQuaD-MDS beats budgeted SGD.

**The trust region is provably inert.** `max_step_frac` ∈ {∞, 4.0, 1.0} gives
identical results to three decimals on every dataset except orl, with a measured
clip rate of 0.000% on clean data. It exists only to stop the duplicate-point
blow-up (§6) and can be reported as not affecting any other number.

**Seed stability is excellent**: median spread (max/min over 5 seeds at each
dataset's best config) is **1.003×**, worst 1.014×.

---

## 5. What determines the gap — outlier sensitivity

Two k8 datasets do markedly worse than the rest: `mnist_digits` (1.52×) and
`california_housing` (4.60×). `mnist_digits` is the sharper puzzle, because
`fashion_mnist_full` has **identical D (784) and identical N (70,000)** and
scores 1.11×.

### 5.1 It is not the lean k8 grid

k8 tested only `lr=550` and `n_iter ≤ 1000`, so under-tuning was the obvious
suspect. A targeted sweep at N=10,000 rules it out:

| lr | i1000 | i2000 | i4000 | | fashion (i1000 / i4000) |
|--:|--:|--:|--:|---|--:|
| 550 | 1.69× | 1.57× | 1.47× | | 1.14× / 1.09× |
| 1500 | 1.68× | 1.57× | 1.47× | | 1.14× / 1.10× |
| 3000 | 1.70× | 1.59× | 1.49× | | 1.16× / 1.10× |
| 6000 | 1.73× | 1.62× | 1.51× | | 1.19× / 1.11× |

Learning rate is **irrelevant across a 10× range**, and quadrupling `n_iter`
buys only 13%. The gap is intrinsic to the dataset.

### 5.2 The error is concentrated in the longest distances

Squared stress error decomposed by input-distance decile (N=4,000, both
embeddings α-rescaled):

| decile | mnist: share of SQuaD error | SQuaD/SGD | fashion: SQuaD/SGD |
|---|--:|--:|--:|
| 1–9 (d ≤ 46) | 51.6% | 1.02–1.23× | 0.97–1.10× |
| **10** (d ≤ 566) | **48.4%** | **4.52×** | 1.18× |

**Half of SQuaD-MDS's total error on mnist_digits sits in the top distance
decile, at 4.5× SGD's error there.** Deciles 1–9 are essentially fine. The
decile boundaries tell the story directly: mnist jumps from 46.3 at decile 9 to
566.1 at decile 10 (12×), while fashion goes 50.4 → 197.6 (4×). mnist_digits
carries a small population of extreme outlier points.

Cross-checked with Fashion-MNIST's class labels: SQuaD-MDS is **better** than
SGD within-class (0.92×) and **worse** between-class (1.08×) — the error is on
long pairs, not short ones.

> This **overturns the natural hypothesis**, which was that MNIST's ten
> well-separated digit clusters would cause short within-cluster distances to
> collapse. The measurement says the opposite. Recorded here because the wrong
> explanation is the more intuitive one.

### 5.3 Mechanism: the per-quartet normalisation weights each quartet by 1/S²

The quartet loss is `sum_p (delta_p/sum(delta) - d_p/sum(d))^2`. An absolute
error `eps` in one distance contributes about `(eps/S)^2`, where `S = sum(d)` is
that quartet's own distance sum — whereas raw stress contributes `eps^2`.

So SQuaD-MDS implicitly applies a **per-quartet weight of 1/S²**, making its
objective a relative-error criterion (Sammon-like) rather than raw stress. When
all distances are comparable, `S` is near-constant across quartets, the weight
is a constant, and the two objectives nearly coincide. When the distance
distribution is heavy-tailed, quartets containing an outlier have a very large
`S`, so **all of their errors are down-weighted — precisely the pairs raw stress
cares about most**, since stress counts absolute error.

The method is therefore structurally blind to exactly the pairs that dominate
the metric it is being scored on. This is the same mechanism behind
`california_housing`, whose farthest point sits 59× the median distance from the
centroid (Experiment L §5) and which also broke Pivot MDS at 18.07×.

### 5.4 A predictor computable before fitting

`p99.9(pairwise distance) / median`, on a subsample — one `pdist`, seconds:

| statistic | ρ vs SQuaD gap | p | n |
|---|--:|--:|--:|
| **p99.9 / median** | **+0.53** | **0.0085** | 23 |
| coefficient of variation | +0.46 | 0.0283 | 23 |

The tail ratio beats CV, and unlike every statistic in §4.5 this survives
correction for the handful of predictors tested. It works best as a
**one-sided threshold**:

| | n | median gap | **max gap** |
|---|--:|--:|--:|
| p99.9/median ≤ 4 | 17 | 1.05× | **1.11×** |
| p99.9/median > 4 | 6 | 1.17× | 4.60× |

**All four datasets with a gap above 1.15× rank 18, 21, 22 and 23 out of 23** on
this statistic — spambase 4.77, epileptic 8.14, california_housing 8.69,
mnist_digits 10.40 (the highest in the collection).

The implication is asymmetric and should be stated that way: **a light tail
guarantees SQuaD-MDS does well** (held on 17 of 17), while a heavy tail is
*necessary but not sufficient* for it to do badly — seismic (7.73 → 1.05×) and
cnae9 (3.60 → 1.02×) are false positives.

This is a more useful diagnostic than anything in §4.5: it does not predict the
best *configuration*, but it predicts whether the *method* suits the data at
all, and it is cheap enough to run as a pre-flight check.

---

## 6. orl — a dataset defect, not a method failure

Unguarded SQuaD-MDS reaches **439× full-cycle stress** on orl. The cause is
fully diagnosed and is a property of the data:

**orl's 400 rows contain only 41 unique values — one of them repeated 360
times.** The per-quartet gradient carries a factor `2(d_hat − delta_hat)/S`
where S is the sum of the quartet's six *embedding* distances. Four
high-dimensionally identical points start at the same embedding coordinate;
because each copy lands in a different quartet each iteration, they drift apart
by ~1e-9 within a few iterations, after which S ≈ 2e-8 while the six unit
vectors are perfectly well-formed — giving |u| ≤ 2/S ≈ 9e7, a step of ~1e10, and
those points leave the embedding permanently. On orl the first such event occurs
at **iteration 3**.

Whether it happens at all is governed by the multiplicity of the most repeated
row. Expected collapsed quartets per iteration across the collection:

| dataset | most repeated row | E[collapsed quartets/iter] |
|---|--:|--:|
| **orl** | **360 of 400** | **65.5** |
| sentiment | 315 of 2,748 | 0.117 |
| hatespeech | 165 of 3,221 | 0.008 |
| everything else | — | ~0 |

So this is **one pathological dataset**, not a general hazard. The
`max_step_frac` trust region caps each point's per-iteration displacement at a
multiple of the embedding's RMS radius — a change of step *length* only, which
cannot move the optimum — and brings orl to 16.03×. Note the released reference
implementation would do *worse*: it divides by `sum(Dhd_quartet)` unguarded,
producing NaN on a fully-degenerate quartet.

**This finding is not specific to Experiment M.** orl's 90% duplication affects
its numbers in Experiments J, K and L too, and is worth checking wherever orl
appears.

---

## 7. Caveats

- **Both suites are ladder-limited; the numbers are pessimistic for
  SQuaD-MDS.** *Every* best config in j18 sits at `n_iter=2000` and *every* best
  in k8 at `n_iter=1000` — the grid ceilings. Quality was still improving at the
  top: k8's i500→i1000 gains were −1.7% (cifar10_raw), −3.2%
  (fashion_mnist_full), −7.2% (california_housing), −9.1% (mnist_digits); j18's
  cnae9 was at 2.05× and still falling. A higher ladder would improve every
  SQuaD-MDS quality number — **and proportionally reduce the speedup**, since
  cost is exactly linear in `n_iter`.
- **Suite A's timings are contaminated on the SGD side.** Experiment J does not
  pass `compute_stress=False`, so its `time_solver` includes in-fit O(N²)
  scoring. The j18 time ratios (median 8.0×) are therefore **inflated by an
  unmeasured amount**, worst on low-D/high-N datasets where scoring dominates
  (spambase, D=57, N=4,601 is ~74% scoring by the cost model). **Use Suite B's
  numbers for any speed claim.** This also affects Experiment L's Suite A
  speedups, which were computed the same way.
- **bank and hiva have no reference** until Experiment J is re-run at the
  corrected full N.
- **Possible machine mismatch in Suite B timings.** K-optimized's cycle times
  (~4,200 s at 50k) match the cluster run recorded in
  [EXPERIMENT_K_OPTIMIZED_RESULTS.md](EXPERIMENT_K_OPTIMIZED_RESULTS.md). If
  Experiment M ran on different hardware, the ratios mix two machines; Experiment
  L measured those two as differing ~1.07× on the SGD workload.
- **svhn's 0.98× is a tie, not a win.** Cycle spans [18.93, 19.81]M across seeds,
  SQuaD [18.97, 19.10]M. SQuaD's median is lower and its variance far tighter,
  but the ranges overlap. Report as "matches full-cycle SGD with lower seed
  variance."
- **The 70k and 100k rungs have no cycle denominator.** The cycle baseline needs
  ~36·N² bytes (164 GB at 70k, 335 GB at 100k) so `CYCLE_MAX_N = 50_000` skips
  it there. Those rungs contribute timings and absolute stress only.
- **`strain_top2` does not predict SQuaD-MDS's gap** — ρ = +0.17, p = 0.54 over
  15 datasets. The same conclusion Experiment L reached for Pivot MDS: the
  regime theory is specific to *pair subsampling inside SGD*, not a general
  property of approximate MDS.

---

## 8. Conclusions

1. **The "relaxation" claim must be softened.** SQuaD-MDS reaches 1.04× (j18)
   and 1.08× (k8) of full-cycle stress at its best configuration. Saying it
   "optimizes a relaxation of stress rather than stress itself" is accurate
   about the objective but implies a quality gap that does not exist. The
   accurate framing: *a scale-free per-quartet objective costs a few percent of
   raw stress, not a qualitative difference.*
2. **But the published default does cost 1.53× (median).** The distinction
   between "the method as shipped" and "the method tuned for stress" is real and
   worth stating — and the gap is almost entirely `exaggerate_d`, an
   RNX-oriented heuristic absent from the paper, not the relaxation itself.
3. **Budgeted SGD still wins on quality** — 1.01–1.02× median against 1.04–1.08×,
   on 14 of 16 (j18) and 22 of 30 (k8) — but the margin is small, and
   SQuaD-MDS is **7–14× faster**. This is a genuine Pareto trade-off, much
   tighter than Pivot MDS's (which lost by 3.34× median).
4. **SQuaD-MDS wins exactly where our fast mode is weakest.** On ag_news and
   tiny_imagenet — the hard-middle, low-`strain_top2` sources where budgeted SGD
   plateaus at +10% regardless of k — SQuaD-MDS reaches 1.06–1.08%, between SGD
   k=200 and k=400, at ~1/11th the wall time.
5. **The best configuration is not predictable from the data — but barely needs
   to be.** The full grid spans 3.7× (median), yet two binary flags —
   `exaggerate_d=False`, `init='pca'` — carry almost all of it, and both have the
   *same* answer on 15 of 16 datasets. Learning rate is the only genuinely
   data-dependent knob; it is irrelevant on 10 of 15 datasets and decisive (up to
   6×) on the other five, and no cheap statistic predicts which — N, D,
   `strain_top2` and the participation ratios all fail, with dataset size in
   particular carrying **no** information (flat within-dataset across a 5× range
   of N). Fortunately little prediction is needed: a single universal setting
   (`n_iter=2000, lr=1500, exaggerate_d=False, init='pca'`) lands within
   **1.007× median / 1.276× worst** of per-dataset-optimal, against 1.390× for
   the authors' published default. Practical guidance: don't tune it per
   dataset, just don't ship it with exaggeration on.
6. **Whether the method suits the data at all *is* predictable, and the property
   is outlier sensitivity (§5).** SQuaD-MDS's per-quartet normalisation weights
   each quartet by 1/S², so quartets containing an outlier — the ones with the
   largest distance sum — have all their errors down-weighted, precisely the
   pairs raw stress cares about most. Measured directly: on `mnist_digits`, 48%
   of SQuaD-MDS's total error sits in the top distance decile at 4.5× SGD's error
   there, while deciles 1–9 are within 1.02–1.23×. A single cheap statistic,
   `p99.9(pairwise distance) / median`, correlates at **ρ = +0.53, p = 0.0085**
   (n=23) — the only predictor in this experiment that survives correction — and
   works as a one-sided screen: **≤ 4 guarantees a gap under 1.11× (17 of 17)**,
   while all four datasets above 1.15× rank in the top six. Note this *overturned*
   the intuitive hypothesis, which was that clustered data would make short
   within-cluster distances collapse; the error is on the longest distances, not
   the shortest.
7. **The speed advantage is cache blocking, not the objective.** Both methods do
   2·D flops per pair; SQuaD-MDS loads 0.667·D per pair against SGD's 2·D
   because four points yield six pairs. That 3×, times 4× fewer pairs, predicts
   12× against a measured 11.1×. **This is transferable to our own method** —
   sampling mini-batches of m points and using all m(m−1)/2 pairs would give
   SGD-MDS the same benefit, with loads per pair of 2D/(m−1) (m=8 → 2.3× better
   than SQuaD-MDS, m=16 → 5×) while keeping raw stress as the objective. Whether
   the correlated sampling costs statistical efficiency is untested and is the
   obvious follow-up.

### Framing for the paper

SQuaD-MDS is designed for neighbourhood preservation and is evaluated in its own
paper on RNX curves. Scoring it on stress measures it against an objective it
does not optimize — the same caveat §5 of our paper already records for Pivot
MDS. The result to claim is **"this is the stress SQuaD-MDS reaches"**, not
"SQuaD-MDS is worse". Our own RNX validation makes the point concrete: at the
setting where SQuaD-MDS is best on stress it is *worse* on RNX (0.331) than at
its default (0.389). The two methods target different things, and a reader
should come away understanding that rather than a ranking.
