# Paper results: the three steps

Results from Experiments O and P, grouped the way the paper argues:

1. Full SGDMDS improves on SMACOF, on the small suite.
2. Budgeted SGDMDS improves on full SGDMDS, on the large suite.
3. Blocked SGDMDS improves on both, on the large suite.

Each step gives the claim, its tables, and the limits the paper must state with
it. The full analysis is in [EXPERIMENT_O_P_Q_RESULTS.md](EXPERIMENT_O_P_Q_RESULTS.md);
per-setting tables for one dataset are in
[EXPERIMENT_O_P_DATASET_TABLES.md](EXPERIMENT_O_P_DATASET_TABLES.md).

## Setup that applies to every table

- **Suites:**
  - Small suite (O): 17 real datasets at full size, N = 646–5,750, seeds
    42–51.
  - Large suite (P): 8 sources at N = 10k, 20k, 35k and 50k (california_housing
    stops at its full 20,640). Budgeted and blocked use seeds 42–46. Full
    SGDMDS uses seed 42 only, or 42–44 at 10k, because one full sweep at 50k
    needs ~120 GB and ~73 minutes.
- **Algorithms:**
  - SMACOF is scikit-learn's `MDS` as shipped: metric, random init, `n_init=1`,
    `max_iter=300`, `eps=1e-6`.
  - SGDMDS uses the golden configuration: 30 epochs, lr 0.5, hybrid schedule,
    random init.
  - Budgeted SGDMDS draws k·N random pairs per epoch; blocked SGDMDS draws the
    same budget as blocks of 16 points.
- **Base:** "÷ full" and "speedup" are against full SGDMDS's median stress and
  median time on the same dataset and N.
- **Stress** is raw stress Σ_{i<j}(d_ij − δ_ij)² at the optimal scale. Stress-1
  is Kruskal's normalised stress, sqrt(stress / Σ δ_ij²).
- **Time** is single-thread wall time on one CPU model (AMD EPYC 9555), from the
  input features to the embedding. Stress scoring is excluded, except for
  SMACOF, whose stopping rule needs it.
- **Summaries** are medians over seeds; the IQRs are in the analysis CSVs.
  Speedups within 15% count as comparable (≈); "(slower)" means more than 15%
  slower.

## Step 1: full SGDMDS vs SMACOF (small suite)

**Claim.** On all 17 datasets, full SGDMDS reaches lower stress than SMACOF
(median 0.937× SMACOF's). It is faster on 16 (median 2.3×), and reaches SMACOF's
final stress a median 6.2× sooner than SMACOF finishes. Wilcoxon signed-rank
over the 17 datasets: p = 1.5·10⁻⁵ for stress and 2.9·10⁻⁴ for time.

**Table 1. Small suite: full SGDMDS vs SMACOF**

| dataset | N | D | stress-1 SMACOF | stress-1 full | full ÷ SMACOF stress | SMACOF iterations | SMACOF time | full time | speedup | to SMACOF's stress |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| seismic | 646 | 24 | 0.230 | 0.215 | 0.877 | 300 (10/10 at cap) | 0.53 s | 0.17 s | 3.19× | 12.46× |
| svhn | 732 | 1,024 | 0.197 | 0.187 | 0.901 | 174 (0/10 at cap) | 0.43 s | 0.21 s | 2.05× | 6.21× |
| har | 735 | 561 | 0.214 | 0.205 | 0.916 | 300 (8/10 at cap) | 0.71 s | 0.21 s | 3.40× | 9.15× |
| sms | 835 | 500 | 0.346 | 0.346 | 0.998 | 106 (0/10 at cap) | 0.36 s | 0.29 s | 1.25× | 2.03× |
| fmd | 997 | 1,536 | 0.333 | 0.332 | 0.991 | 300 (8/10 at cap) | 1.65 s | 0.46 s | 3.60× | 5.91× |
| cnae9 | 1,080 | 856 | 0.271 | 0.270 | 0.997 | 166 (0/10 at cap) | 1.01 s | 0.5 s | 2.01× | 3.28× |
| coil20 | 1,440 | 400 | 0.303 | 0.293 | 0.937 | 300 (10/10 at cap) | 4.57 s | 1.73 s | 2.64× | 7.54× |
| secom | 1,567 | 590 | 0.315 | 0.311 | 0.971 | 300 (10/10 at cap) | 4.49 s | 1.28 s | 3.51× | 7.31× |
| bank | 2,059 | 63 | 0.333 | 0.327 | 0.966 | 300 (10/10 at cap) | 10.3 s | 2.66 s | 3.88× | 8.61× |
| sentiment | 2,748 | 200 | 0.355 | 0.355 | 0.996 | 300 (10/10 at cap) | 18.4 s | 6.77 s | 2.72× | 4.75× |
| fashion_mnist | 3,000 | 784 | 0.268 | 0.256 | 0.911 | 300 (10/10 at cap) | 21.6 s | 8.74 s | 2.47× | 9.17× |
| hiva | 3,076 | 1,617 | 0.347 | 0.334 | 0.925 | 300 (10/10 at cap) | 22.9 s | 10.2 s | 2.24× | 6.50× |
| hatespeech | 3,221 | 100 | 0.351 | 0.346 | 0.973 | 300 (10/10 at cap) | 25.2 s | 11 s | 2.30× | 4.56× |
| imdb | 3,250 | 700 | 0.403 | 0.403 | 1.000 | 64 (0/10 at cap) | 5.63 s | 11.9 s | 0.47× (slower) | 0.73× (slower) |
| cifar10 | 3,250 | 1,024 | 0.280 | 0.269 | 0.922 | 300 (10/10 at cap) | 25.7 s | 12.6 s | 2.04× | 6.84× |
| spambase | 4,601 | 57 | 0.247 | 0.237 | 0.922 | 300 (10/10 at cap) | 48.9 s | 29 s | 1.69× | 5.54× |
| epileptic | 5,750 | 178 | 0.195 | 0.185 | 0.904 | 300 (10/10 at cap) | 78.3 s | 55 s | 1.42× | 5.50× |
| **median** |  |  |  |  | **0.937** |  |  |  | **2.30×** | **6.21×** |

"to SMACOF's stress" is SMACOF's time divided by the time full SGDMDS takes to
reach SMACOF's median final stress.

**The paper must also state:**

- **Where SMACOF converges early, the gap closes.** SMACOF stops before its cap
  on svhn, sms, cnae9 and imdb. On sms, cnae9 and imdb the two then differ by at
  most 0.3%, and on imdb SMACOF is faster (0.47×). On svhn SMACOF converges to a
  minimum 10% worse.
- **The margin shrinks with N** (Experiment Q, three synthetic datasets at
  N = 500–5,000).
  - On swiss_roll and torus, SMACOF converges in 115–145 iterations and becomes
    the faster of the two from N=3,500 and 4,000, at equal stress.
  - On s_curve it uses all 300 iterations and is still 1.6× slower at 5,000.
  - The reason is cost per pair: a full-sweep epoch costs 26 ns per pair at
    N≤1,100 but ~115 ns from N≈5,750, about 95% of it the per-epoch shuffle of
    the pair arrays. That is ~3 SMACOF iterations at small N and ~7 at large N.
- **The comparison is random init for both methods.** With classical init,
  SMACOF has lower stress than full SGDMDS on 10 of 17 datasets. scikit-learn
  1.10 switches `MDS`'s default to classical init
  ([_mds.py:476-477](_mds.py#L476-L477)), so "scikit-learn's default" changes
  meaning. The classical- or PCA-init study for SGDMDS (spec §3) is needed to
  make the step 1 claim against 1.10's default.
- **coil20's full-sweep time is ~1.35× too slow** (it ran on the busiest node).
  Its stress result stands.

## Step 2: budgeted vs full SGDMDS (large suite)

**Claim.** At k=200, budgeted SGDMDS is faster than full SGDMDS on all 30
(source, N) rungs. The median speedup over the 8 sources grows from 7.6× at 10k
to 28.9× at 50k, while the median stress overhead stays at 1.019–1.025×.

**Table 2. Large suite: budgeted ÷ full at k=200 (stress · speedup)**

| dataset | D | 10k | 20k | 35k | 50k | full time at top N |
|---|--:|--:|--:|--:|--:|--:|
| feynman_iii_15_12 | 3 | 0.991 · 275× | 1.001 · 580× | 1.001 · 925× | 1.005 · 1,329× | 4,334 s |
| california_housing | 8 | 1.019 · 270× | 1.021 · 518× | — | — | 767 s |
| ag_news | 384 | 1.095 · 15.07× | 1.097 · 15.28× | 1.100 · 25.11× | 1.100 · 35.57× | 4,431 s |
| patchcamelyon | 512 | 1.019 · 7.46× | 1.018 · 12.26× | 1.021 · 20.42× | 1.025 · 28.86× | 4,307 s |
| tiny_imagenet | 512 | 1.093 · 7.70× | 1.096 · 12.83× | 1.100 · 21.87× | 1.100 · 29.05× | 4,362 s |
| fashion_mnist_full | 784 | 1.013 · 4.78× | 1.011 · 8.83× | 1.011 · 15.20× | 1.012 · 22.05× | 4,351 s |
| mnist_digits | 784 | 1.082 · 4.57× | 1.083 · 8.76× | 1.089 · 15.57× | 1.105 · 24.73× | 4,583 s |
| cifar10_raw | 3,072 | 1.011 · 1.45× | 1.011 · 2.94× | 1.012 · 5.18× | 1.013 · 7.57× | 4,366 s |

**Table 3. Large suite: budgeted ÷ full per k (median over sources: stress · speedup, speedup range)**

| arm | k | 10k | 20k | 35k | 50k |
|---|--:|--:|--:|--:|--:|
| budget | 100 | 1.063 · 15.1× (2.9×–531×) | 1.101 · 25.2× (5.9×–1,136×) | 1.099 · 40.8× (10.3×–1,841×) | 1.104 · 57.8× (15.2×–2,637×) |
| budget | 200 | 1.019 · 7.6× (1.5×–275×) | 1.019 · 12.5× (2.9×–580×) | 1.021 · 20.4× (5.2×–925×) | 1.025 · 28.9× (7.6×–1,329×) |
| budget | 400 | 1.007 · 3.8× (0.7×–140×) | 1.008 · 6.3× (1.5×–292×) | 1.006 · 10.2× (2.6×–467×) | 1.007 · 14.5× (3.8×–668×) |

**The paper must also state:**

- **Pair every speedup with its overhead at the same k.** Overhead falls with k
  (1.06–1.10 at k=100, 1.019–1.025 at k=200, 1.006–1.008 at k=400) while the
  speedup halves with each doubling of k.
- **Three sources pay ~10% at k=200 at every N:** ag_news, tiny_imagenet and
  mnist_digits (1.08–1.11). The other five pay at most 2.5%.
- **The speedup has two factors:** the pair saving (N−1)/(2k), times the ratio
  of cost per pair.
  - At D ≤ 8 a budgeted pair costs ~11 ns against the full sweep's ~115 ns, so
    the 270–1,329× there are about 10× the pair saving. That second factor comes
    from the full sweep's shuffle (step 1 note).
  - At high D a budgeted pair costs more than a full-sweep pair (0.6–1.1 ns per
    dimension once memory-bound), so cifar10_raw (D=3,072) gains only 1.45–7.6×.
- **Budgeting doesn't pay at small N with high D.** It is slower than full on 7
  of 17 small-suite datasets at k=200, all with D ≥ 500, because
  each sampled pair computes its distance over all D dimensions. Hence step 2
  uses the large suite.
- **Full SGDMDS has one seed above 10k.** At 10k, the 3 full seeds differ by at
  most 0.04%, except feynman_iii_15_12 (4.4%); its ratios below 1 are that noise.
- **Don't advertise cutting k and epochs together:** the two levers don't
  compose.

## Step 3: blocked SGDMDS vs budgeted and full (large suite)

**Claim.**

- At the same k, blocked SGDMDS matches budgeted's stress: median ratio 0.997,
  never more than 2.5% worse, and up to 4% better (mnist_digits).
- At D ≥ 384 it is 3.2–4.5× faster than budgeted at 50k.
- Against full at k=200, it gives a median 120× at 50k, at 1.018× stress.
- It is the fastest way to reach within 1–2% of the full sweep's stress.

**Table 4. Large suite: blocked ÷ full at k=200 (stress · speedup)**

| dataset | D | 10k | 20k | 35k | 50k | full time at top N |
|---|--:|--:|--:|--:|--:|--:|
| feynman_iii_15_12 | 3 | 1.002 · 222× | 1.002 · 468× | 0.991 · 815× | 1.005 · 1,179× | 4,334 s |
| california_housing | 8 | 1.019 · 229× | 1.021 · 487× | — | — | 767 s |
| ag_news | 384 | 1.080 · 34.19× | 1.079 · 63.15× | 1.090 · 110× | 1.091 · 159× | 4,431 s |
| patchcamelyon | 512 | 1.016 · 25.00× | 1.013 · 49.70× | 1.016 · 85.08× | 1.018 · 121× | 4,307 s |
| tiny_imagenet | 512 | 1.073 · 25.06× | 1.089 · 49.32× | 1.087 · 88.25× | 1.095 · 120× | 4,362 s |
| fashion_mnist_full | 784 | 1.009 · 16.86× | 1.009 · 34.54× | 1.011 · 60.56× | 1.012 · 86.29× | 4,351 s |
| mnist_digits | 784 | 1.043 · 16.88× | 1.057 · 34.43× | 1.069 · 59.90× | 1.069 · 92.26× | 4,583 s |
| cifar10_raw | 3,072 | 1.009 · 4.81× | 1.010 · 9.69× | 1.010 · 16.94× | 1.011 · 24.59× | 4,366 s |

**Table 5. Large suite: blocked ÷ full per k (median over sources: stress · speedup, speedup range)**

| arm | k | 10k | 20k | 35k | 50k |
|---|--:|--:|--:|--:|--:|
| blocked | 100 | 1.048 · 50.0× (9.6×–445×) | 1.045 · 98.8× (19.4×–959×) | 1.096 · 170× (33.9×–1,615×) | 1.092 · 243× (49.0×–2,345×) |
| blocked | 200 | 1.018 · 25.0× (4.8×–229×) | 1.017 · 49.5× (9.7×–487×) | 1.016 · 85.1× (16.9×–815×) | 1.018 · 120× (24.6×–1,179×) |
| blocked | 400 | 1.008 · 12.5× (2.4×–116×) | 1.007 · 24.8× (4.8×–245×) | 1.005 · 42.6× (8.5×–409×) | 1.005 · 59.0× (12.3×–591×) |

**Table 6. Each source at its largest N: all arms at k=200 (stress ÷ full · speedup over full)**

| dataset | N | D | full time | budgeted | blocked | blocked ÷ budgeted stress | blocked over budgeted | SQuaD-MDS best |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| feynman_iii_15_12 | 50k | 3 | 4,334 s | 1.005 · 1,329× | 1.005 · 1,179× | 0.997 | ≈0.89× | 1.072 · 491× |
| california_housing | 20,640 | 8 | 767 s | 1.021 · 518× | 1.021 · 487× | 1.001 | ≈0.94× | 4.199 · 201× |
| ag_news | 50k | 384 | 4,431 s | 1.100 · 35.57× | 1.091 · 159× | 0.991 | 4.46× | 1.055 · 94.28× |
| patchcamelyon | 50k | 512 | 4,307 s | 1.025 · 28.86× | 1.018 · 121× | 0.993 | 4.21× | 1.054 · 80.42× |
| tiny_imagenet | 50k | 512 | 4,362 s | 1.100 · 29.05× | 1.095 · 120× | 0.995 | 4.21× | 1.059 · 72.74× |
| fashion_mnist_full | 50k | 784 | 4,351 s | 1.012 · 22.05× | 1.012 · 86.29× | 1.000 | 3.92× | 1.074 · 64.95× |
| mnist_digits | 50k | 784 | 4,583 s | 1.105 · 24.73× | 1.069 · 92.26× | 0.958 | 3.73× | 1.326 · 71.70× |
| cifar10_raw | 50k | 3,072 | 4,366 s | 1.013 · 7.57× | 1.011 · 24.59× | 0.998 | 3.22× | 1.032 · 24.61× |

"blocked over budgeted" is budgeted's time ÷ blocked's time. SQuaD-MDS best is
its lowest-stress setting of 12 (`n_iter` ≤ 4,000, lr 550, PCA init, step cap
4, exaggeration off or on), compared at its optimal scale.

**Table 7. Each source at its largest N: speedup over the full run to reach 1.01× and 1.05× full's final stress**

| dataset | N | 1.01× full | 1.01× budget | 1.01× blocked | 1.01× squad | 1.05× full | 1.05× budget | 1.05× blocked | 1.05× squad |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| feynman_iii_15_12 | 50k | 1.79× | 3,596× (k100) | 3,213× (k100) | — | 2.53× | 6,296× (k100) | 5,647× (k100) | — |
| california_housing | 20,640 | 1.78× | 345× (k400) | 389× (k400) | — | 2.52× | 1,021× (k200) | 1,538× (k100) | — |
| ag_news | 50k | 1.78× | — | — | — | 2.49× | — | 144× (k400) | — |
| patchcamelyon | 50k | 1.77× | 22.39× (k400) | 99.59× (k400) | — | 2.49× | 56.63× (k200) | 276× (k200) | — |
| tiny_imagenet | 50k | 1.78× | — | — | — | 2.38× | — | 94.14× (k400) | — |
| fashion_mnist_full | 50k | 1.77× | 18.94× (k400) | 75.10× (k400) | — | 2.50× | 50.38× (k200) | 210× (k200) | — |
| mnist_digits | 50k | 1.76× | — | — | — | 2.42× | 23.61× (k400) | 103× (k400) | — |
| cifar10_raw | 50k | 1.75× | 6.38× (k400) | 20.85× (k400) | — | 2.44× | 17.21× (k200) | 62.35× (k200) | 125× |

Each cell is the fastest k (or SQuaD-MDS setting) whose seeds mostly reach the
target. "—" means none does. "full" is the full run reaching the target before
it ends.

**The paper must also state:**

- **At D ≤ 8, blocking costs 6–19%** (0.81–0.94× of budgeted's speed). The
  plain random kernel is faster there, which is why both kernels are kept.
- **Credit SQuaD-MDS as the source of the blocking idea, and don't claim to beat
  it.** It optimises a scale-free per-quartet objective, and its authors evaluate
  it on R_NX, not stress.
  - Across the 30 rungs it reaches 1.01× full's stress on none, and 1.05× on 7.
    At 1.10× it is the fastest arm on 17 of 30.
  - Blocked SGDMDS is fastest at 1.01× on 13 rungs and at 1.05× on 21.
  - So blocked SGDMDS owns the accurate end of the trade-off, and SQuaD-MDS the
    cheap end.
- **SQuaD-MDS's quality is a bound.** Its best setting is the top of its
  `n_iter` ladder on 29 of 30 rungs, and the last doubling still gained 1.0%
  (median), up to 6.7% on mnist_digits.
- **The time-to-target cells are not shorter runs.** A run that reaches 1.01× at
  about 56% of its time was still on a 30-epoch schedule; a 17-epoch run would end
  elsewhere.
- **P's histories are estimated from 10⁷ pairs.** Final stresses are exact, and
  the estimate is within 0.47% (0.03% median).

## Small suite: full vs budgeted vs blocked at every k

Steps 2 and 3 use the large suite. This table shows why, and can go in an
appendix: on the small suite the cheaper arms often aren't cheaper.

**Table 8. Small suite: budgeted and blocked ÷ full per k (stress · speedup)**

| dataset | N | D | full time | arm | k=50 | k=100 | k=200 | k=400 |
|---|--:|--:|--:|---|--:|--:|--:|--:|
| seismic | 646 | 24 | 0.17 s | budgeted | 1.154 · 7.86× | 1.029 · 4.02× | 1.006 · 2.04× | — |
| seismic | 646 | 24 | 0.17 s | blocked | 1.121 · 8.52× | 1.029 · 4.37× | 1.003 · 2.21× | — |
| svhn | 732 | 1,024 | 0.21 s | budgeted | 1.098 · 0.40× (slower) | 1.074 · 0.20× (slower) | 1.025 · 0.10× (slower) | — |
| svhn | 732 | 1,024 | 0.21 s | blocked | 1.101 · ≈0.95× | 1.063 · 0.48× (slower) | 1.032 · 0.24× (slower) | — |
| har | 735 | 561 | 0.21 s | budgeted | 1.129 · 0.73× (slower) | 1.089 · 0.37× (slower) | 1.049 · 0.19× (slower) | — |
| har | 735 | 561 | 0.21 s | blocked | 1.114 · 1.60× | 1.081 · 0.82× (slower) | 1.045 · 0.42× (slower) | — |
| sms | 835 | 500 | 0.29 s | budgeted | 1.007 · ≈1.00× | 1.004 · 0.51× (slower) | 1.003 · 0.25× (slower) | 1.002 · 0.13× (slower) |
| sms | 835 | 500 | 0.29 s | blocked | 1.007 · 2.17× | 1.004 · ≈1.11× | 1.003 · 0.56× (slower) | 1.002 · 0.28× (slower) |
| fmd | 997 | 1,536 | 0.46 s | budgeted | 1.034 · 0.43× (slower) | 1.022 · 0.22× (slower) | 1.017 · 0.11× (slower) | 1.012 · 0.05× (slower) |
| fmd | 997 | 1,536 | 0.46 s | blocked | 1.028 · ≈1.06× | 1.020 · 0.53× (slower) | 1.015 · 0.27× (slower) | 1.005 · 0.13× (slower) |
| cnae9 | 1,080 | 856 | 0.5 s | budgeted | 1.008 · 0.79× (slower) | 1.006 · 0.40× (slower) | 1.005 · 0.20× (slower) | 1.004 · 0.10× (slower) |
| cnae9 | 1,080 | 856 | 0.5 s | blocked | 1.008 · 1.84× | 1.006 · ≈0.93× | 1.004 · 0.47× (slower) | 1.003 · 0.23× (slower) |
| coil20 | 1,440 | 400 | 1.73 s | budgeted | 1.068 · 3.99× | 1.033 · 2.02× | 1.015 · ≈1.01× | 1.007 · 0.51× (slower) |
| coil20 | 1,440 | 400 | 1.73 s | blocked | 1.063 · 8.42× | 1.030 · 4.32× | 1.015 · 2.19× | 1.007 · ≈1.10× |
| secom | 1,567 | 590 | 1.28 s | budgeted | 1.095 · 2.02× | 1.088 · ≈1.02× | 1.051 · 0.51× (slower) | 1.017 · 0.26× (slower) |
| secom | 1,567 | 590 | 1.28 s | blocked | 1.092 · 4.57× | 1.078 · 2.31× | 1.029 · 1.16× | 1.012 · 0.58× (slower) |
| bank | 2,059 | 63 | 2.66 s | budgeted | 1.129 · 24.38× | 1.067 · 13.16× | 1.039 · 6.86× | 1.014 · 3.51× |
| bank | 2,059 | 63 | 2.66 s | blocked | 1.100 · 31.98× | 1.054 · 17.57× | 1.028 · 9.25× | 1.013 · 4.75× |
| sentiment | 2,748 | 200 | 6.77 s | budgeted | 1.011 · 17.90× | 1.009 · 9.12× | 1.007 · 4.61× | 1.006 · 2.32× |
| sentiment | 2,748 | 200 | 6.77 s | blocked | 1.011 · 33.69× | 1.008 · 17.28× | 1.007 · 8.75× | 1.006 · 4.41× |
| fashion_mnist | 3,000 | 784 | 8.74 s | budgeted | 1.107 · 5.43× | 1.034 · 2.72× | 1.008 · 1.36× | 1.003 · 0.68× (slower) |
| fashion_mnist | 3,000 | 784 | 8.74 s | blocked | 1.138 · 12.71× | 1.047 · 6.39× | 1.011 · 3.20× | 1.003 · 1.60× |
| hiva | 3,076 | 1,617 | 10.2 s | budgeted | 1.101 · 2.28× | 1.097 · ≈1.14× | 1.089 · 0.57× (slower) | 1.056 · 0.29× (slower) |
| hiva | 3,076 | 1,617 | 10.2 s | blocked | 1.100 · 7.07× | 1.093 · 3.55× | 1.081 · 1.78× | 1.038 · ≈0.89× |
| hatespeech | 3,221 | 100 | 11 s | budgeted | 1.045 · 45.46× | 1.041 · 23.56× | 1.037 · 11.99× | 1.033 · 6.05× |
| hatespeech | 3,221 | 100 | 11 s | blocked | 1.044 · 72.33× | 1.040 · 37.71× | 1.035 · 19.27× | 1.028 · 9.74× |
| imdb | 3,250 | 700 | 11.9 s | budgeted | 1.009 · 7.58× | 1.006 · 3.80× | 1.003 · 1.90× | 1.001 · ≈0.95× |
| imdb | 3,250 | 700 | 11.9 s | blocked | 1.008 · 17.63× | 1.005 · 8.87× | 1.003 · 4.45× | 1.001 · 2.23× |
| cifar10 | 3,250 | 1,024 | 12.6 s | budgeted | 1.155 · 5.39× | 1.029 · 2.70× | 1.009 · 1.35× | 1.003 · 0.68× (slower) |
| cifar10 | 3,250 | 1,024 | 12.6 s | blocked | 1.108 · 13.01× | 1.025 · 6.54× | 1.008 · 3.28× | 1.003 · 1.64× |
| spambase | 4,601 | 57 | 29 s | budgeted | 1.241 · 124× | 1.131 · 64.70× | 1.062 · 32.90× | 1.034 · 16.66× |
| spambase | 4,601 | 57 | 29 s | blocked | 1.218 · 179× | 1.105 · 93.73× | 1.053 · 48.03× | 1.030 · 24.33× |
| epileptic | 5,750 | 178 | 55 s | budgeted | 1.133 · 79.63× | 1.113 · 40.39× | 1.093 · 20.32× | 1.065 · 10.21× |
| epileptic | 5,750 | 178 | 55 s | blocked | 1.127 · 136× | 1.111 · 69.16× | 1.090 · 34.92× | 1.059 · 17.52× |

k=400 is missing for seismic, svhn and har, because k must be below (N−1)/2.
coil20's full time is ~1.35× too slow, so its speedups are overstated.

**Summary over the 17 datasets** (median stress ÷ full · median speedup;
faster / comparable / slower):

| arm | k=50 | k=100 | k=200 | k=400 (14 datasets) |
|---|--:|--:|--:|--:|
| budgeted | 1.098 · 5.39× (12 / 1 / 4) | 1.034 · 2.70× (10 / 2 / 5) | 1.017 · 1.35× (9 / 1 / 7) | 1.010 · 0.68× (5 / 1 / 8) |
| blocked | 1.100 · 8.52× (15 / 2 / 0) | 1.040 · 4.37× (12 / 2 / 3) | 1.015 · 2.21× (12 / 0 / 5) | 1.007 · 1.62× (8 / 2 / 4) |

What the table shows:

- **The stress cost of a given k is about the same as on the large suite.**
  Roughly 1.10× at k=50, 1.03–1.04× at k=100, 1.015× at k=200 and 1.01× at
  k=400. A smaller k doesn't cost less stress on a small dataset.
- **The time saving is small at small N.** k=200 already covers 7–62% of all
  pairs per epoch here, against 0.8–4% on the large suite.
- **At high D and N ≤ 1,100, budgeted is slower than full at every k** (svhn,
  har, fmd, cnae9; sms is ≈1.00× at k=50).
  - Each sampled pair computes its distance over all D dimensions, while the
    full sweep reads it from the precomputed matrix.
  - Blocked roughly halves that cost per pair: it is faster at k=50 on har,
    sms and cnae9, but still slower from k=100 or k=200.
- **Blocked is faster than budgeted at the same k on every small dataset**,
  1.08× at D=24 up to 3.1× at D=1,617, at the same stress (median ratio 0.999).
- **At low D the cheap arms win even here:** bank, hatespeech, spambase and
  epileptic gain 9–48× at k=200 (blocked). But their stress cost is 3–9%,
  above the median.

## Figures that go with this

From the results doc's §8; none are made yet.

| step | figure |
|---|---|
| 1 | F1: convergence curves for SMACOF vs full on four small datasets (fashion_mnist, epileptic, svhn, imdb). F2: per-dataset stress ratio against speedup. F3: runtime against N from Q, with SMACOF's crossover. |
| 2 | F4: speedup against overhead, one point per (source, N, k). F5: speedup at k=200 against N, coloured by D, with the cost model. F6: cost per pair against N and D. |
| 3 | F7: stress–time trade-off per source at 50k: full, budgeted and blocked over k, SQuaD-MDS over `n_iter`. F8: convergence curves at one 50k rung. F9: which arm reaches each target first. |

## Reproduce

```bash
cd sklearn/manifold/benchmarks
python analyze_experiment_o_p_q.py   # writes results_paper/analysis/
python paper_tables.py               # every table above, ~1 min (stress-1 loads the datasets)
```
