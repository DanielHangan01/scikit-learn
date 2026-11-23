# Experiments O, P and Q — results

**Status:** analysis complete (2026-11-18). Figures not made yet; §8 proposes
them. Campaign: 2026-11-12 to 11-17 on COMA's two bare-metal H200 nodes (AMD EPYC
9555), commit `dc57d0d19` (pilot datasets `5eeb9e04a`), one thread per fit.
Design and protocol: [EXPERIMENT_O_P_SPEC.md](EXPERIMENT_O_P_SPEC.md).

**Data:** `benchmarks/results_paper/experiment_{o,p,q}/` (git-ignored).
**Analysis:** [benchmarks/analyze_experiment_o_p_q.py](benchmarks/analyze_experiment_o_p_q.py)
recomputes every number here in about 10 s. It writes 34 CSV tables, plus
`tables.md` with tables T1–T16 below, to `results_paper/analysis/`. Its functions
return DataFrames, so the plotters can import it.

## Headlines

**Step 1: SMACOF (random init) vs full SGDMDS.**

- **Small suite (O):** full SGDMDS has lower stress on all 17 datasets (median
  0.937× SMACOF's, range 0.877–1.000) and is faster on 16 (median 2.30×,
  1.25–3.88×). imdb is the exception (0.47×): SMACOF converges there in 64
  iterations.
- **Time to SMACOF's stress:** full SGDMDS reaches SMACOF's final stress a median
  6.2× sooner than SMACOF finishes. SMACOF reaches full SGDMDS's stress in none of
  its 170 fits.
- **Runtime vs N (Q) limits the claim.** On swiss_roll and torus, SMACOF converges
  in 115–145 iterations, matches full SGDMDS's stress to within 0.1%, and is the
  faster of the two from N=3,500 and N=4,000.
- **Why the margin shrinks with N:** a full-SGDMDS epoch costs 26 ns per pair at
  N≤1,100 but 115 ns from N≈5,750. That cost is almost all the per-epoch shuffle of
  the pair arrays, not the SGD kernel (§1.4).

**Step 2: budgeted vs full SGDMDS.**

- **Large suite (P):** budgeted is faster on all 30 rungs at k≤200. At k=200 the
  median is 15× (1.45–1,329×) for 1.020× stress (0.991–1.105). The speedup grows
  with N; the overhead barely changes.
- **Small suite (O) at k=200:** faster on 9 of 17 datasets, comparable on 1,
  slower on 7. Q puts the break-even at N=1,000, 2,000 and 2,500 for D=100, 300
  and 500.
- **Cost model:** speedup = pair saving (N−1)/(2k) × cost-per-pair ratio. The
  ratio ranges from ~10 at D≤8 (11 against 115 ns) to 0.06 at D=3,072. So low-D
  speedups are 10× the pair saving, and small N at high D loses.

**Step 3: blocked vs budgeted vs full SGDMDS, plus SQuaD-MDS.**

- **Blocked vs budgeted at the same k:** blocked matches budgeted's stress (median
  ratio 0.999 on O, 0.997 on P). It is 1.1–3.1× faster on O and 2.3–4.5× on P at
  D≥384, but 6–19% slower at D≤8.
- **Who reaches a stress target first:** at 1.01× full stress it is the full
  sweep itself or blocked SGDMDS (budgeted on 5 P rungs). SQuaD-MDS reaches
  1.01× on only 2 of 17 O datasets and none of P's 30 rungs. At 1.10× SQuaD-MDS
  is first on 15 of 17 and 17 of 30. Blocked SGDMDS owns the accurate end of the
  trade-off; SQuaD-MDS owns the cheap end.
- **SQuaD-MDS's quality is a bound.** Its best setting is the top of its
  `n_iter` ladder on 16 of 17 O datasets and 29 of 30 P rungs. The last doubling
  still gained 0.5% (median, O) and 1.0% (P), and up to 6.7% on mnist_digits.
  Choosing on seeds 42–46 and reporting on 47–51 moves its stress by at most
  0.002.

**Classical-init SMACOF (later study):** below every full-SGDMDS seed on 9 of 17
datasets and within 0.3% on 6 more. It is worse on coil20 (+0.7%) and epileptic
(+1.3%).

**Needs a decision (§7).** The full sweep's shuffle decides both the SMACOF
comparison at large N and the size of low-D budget speedups. Step 1's claim needs
scoping. coil20's full-sweep time is inflated.

---

## 0. Data and conventions

| | O | P | Q |
|---|--:|--:|--:|
| datasets × N | 17 at full N (646–5,750) | 8 sources × {10k, 20k, 35k, 50k}: 30 rungs | 3 × {500, …, 5,000}: 30 |
| fits analysed | 44,650 | 2,746 | 1,200 |
| seeds | 42–51 | 42–46; full arm 42 only (42–44 at 10k) | 85–94 |
| divergences | 0 | 0 | 0 |

**Removed:** pilot fits outside the design were deleted from the results tree on
2026-11-19. Copies are in `~/research-backups/sgd-mds/removed_pilot_rows/`, and
the copy on COMA was not touched.

- P's SMACOF fits: california_housing at 10k, 3 seeds × 2 inits, so 6 fits,
  not 12. `smacof_inits` was also dropped from that folder's `meta.json`.
- Q's 100 `random-auto-m2` fits on torus.

**Integrity:**

- The stress-optimal scale α stays within 0.002 of 1 for SMACOF and every
  SGDMDS arm, so raw and scaled stress agree for them. SQuaD-MDS's α spans
  0.08–3.65.
- P's pair estimate is off by 0.03% at the median and 0.47% at most at the last
  snapshot. Every table value is exact.
- SQuaD-MDS's step cap fired on at most 0.8% of point-steps in O.
- Seed spread is small next to the effects. In step 2, the stress-ratio IQR is
  0.003 wide at the median (0.15 at worst, seismic at k=50). The speedup IQR is
  within ±1% at the median and ±4% at worst.

**Conventions** (also in the script's docstring):

- **Summaries:** medians and IQRs over seeds. Stress is scale-optimal (spec
  §7.3). The exception is step 1's time to target, which uses raw stress because
  SMACOF records only raw.
- **"÷ full":** divides by the full arm's median at the same dataset and N. In P
  above 10k, that median is one fit.
- **15% rule:** a time ratio between 1/1.15 and 1.15 is "comparable", shown as ≈
  in the tables. "(slower)" marks ratios below 1/1.15.
- **Time to target:** the first crossing in a fit's history, interpolated
  linearly between the two snapshots around it. Without interpolation, SGD arms
  would read ~4% later and SQuaD-MDS ~15% later (medians), since SQuaD-MDS's
  snapshots are log-spaced. A config counts only if more than half its seeds
  reach the target. P's targets come from the full arm's *estimated* final
  stress, on the same 10⁷ pairs as the histories.
- **SQuaD-MDS selection:** per dataset, the setting with the lowest median stress
  on seeds 42–46, reported on 47–51 (O). P chooses and reports on its 5 seeds.
  For time to target, the setting that reaches the target soonest is chosen the
  same way. SGDMDS arms use every seed and have at most 4 settings.

**Timing caveats:**

- **coil20's full sweep is ~1.35× too slow next to coil20's other arms.**
  - coil20 ran in the pilot, arm by arm rather than seed by seed, under the
    campaign's slowest canary (787 ns at D=784, against ~650 typical).
  - Compared with secom, at similar N, coil20's SMACOF and budgeted times are
    ~1.1–1.2× slow, but its full sweep is 1.6× slow. It costs 54.6 ns per pair,
    against 33–34 ns at N≈1,500 in secom and Q.
  - Its pair arrays are about L3-sized at this N, which plausibly makes the full
    sweep the arm most exposed to neighbouring jobs.
  - Consequence: coil20's budgeted speedups are overstated. k=200 reads ≈1.01×
    but is likely ~0.75×. Its step 1 verdict doesn't change.
- **epileptic** also ran arm by arm in the pilot. Its per-pair costs match the
  other datasets, so no correction is needed.
- **feynman_iii_15_12's full arm** varies by 4.4% in stress across its 3 seeds
  at 10k; every other source varies by at most 0.04%. Its P ratios above 10k rest
  on one full fit and carry a few % of uncertainty. Its budgeted ratio of 0.991
  at 10k is that noise.

---

## 1. Step 1: SMACOF (random init) vs full SGDMDS

### 1.1 Small suite (O)

**T1. Step 1, O: SMACOF (random init) vs full SGDMDS**

| dataset | N | D | SMACOF iters (at cap) | SMACOF stress IQR | full ÷ SMACOF stress | SMACOF time | full time | SMACOF ÷ full time |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| seismic | 646 | 24 | 300 (10/10) | 9.1% | 0.877 [0.875–0.880] | 0.53 s [0.53–0.53 s] | 0.17 s [0.17–0.17 s] | 3.19× |
| svhn | 732 | 1024 | 173 (0/10) | 2.6% | 0.901 [0.897–0.913] | 0.43 s [0.33–0.56 s] | 0.21 s [0.21–0.21 s] | 2.05× |
| har | 735 | 561 | 300 (8/10) | 2.2% | 0.916 [0.914–0.925] | 0.71 s [0.71–0.71 s] | 0.21 s [0.21–0.21 s] | 3.40× |
| sms | 835 | 500 | 106 (0/10) | 0.0% | 0.998 [0.998–0.998] | 0.36 s [0.33–0.4 s] | 0.29 s [0.29–0.29 s] | 1.25× |
| fmd | 997 | 1536 | 300 (8/10) | 0.1% | 0.991 [0.988–0.995] | 1.65 s [1.64–1.66 s] | 0.46 s [0.46–0.46 s] | 3.60× |
| cnae9 | 1080 | 856 | 166 (0/10) | 0.2% | 0.997 [0.997–0.998] | 1.01 s [0.71–1.11 s] | 0.5 s [0.5–0.5 s] | 2.01× |
| coil20 | 1440 | 400 | 300 (10/10) | 3.9% | 0.937 [0.935–0.940] | 4.57 s [4.56–4.59 s] | 1.73 s [1.71–1.76 s] | 2.64× |
| secom | 1567 | 590 | 300 (10/10) | 1.7% | 0.971 [0.970–0.971] | 4.49 s [4.47–4.51 s] | 1.28 s [1.28–1.29 s] | 3.51× |
| bank | 2059 | 63 | 300 (10/10) | 0.6% | 0.966 [0.964–0.967] | 10.3 s [10.3–10.4 s] | 2.66 s [2.64–2.73 s] | 3.88× |
| sentiment | 2748 | 200 | 300 (10/10) | 0.0% | 0.996 [0.996–0.996] | 18.4 s [18.3–18.4 s] | 6.77 s [6.63–6.91 s] | 2.72× |
| fashion_mnist | 3000 | 784 | 300 (10/10) | 2.9% | 0.911 [0.911–0.911] | 21.6 s [21.4–21.6 s] | 8.74 s [8.69–8.8 s] | 2.47× |
| hiva | 3076 | 1617 | 300 (10/10) | 0.5% | 0.925 [0.925–0.926] | 22.9 s [22.6–23 s] | 10.2 s [10.1–10.3 s] | 2.24× |
| hatespeech | 3221 | 100 | 300 (10/10) | 0.1% | 0.973 [0.972–0.973] | 25.2 s [25.1–25.3 s] | 11 s [10.9–11 s] | 2.30× |
| imdb | 3250 | 700 | 64 (0/10) | 0.0% | 1.000 [1.000–1.000] | 5.63 s [5.54–5.68 s] | 11.9 s [11.7–12.1 s] | 0.47× (slower) |
| cifar10 | 3250 | 1024 | 300 (10/10) | 6.6% | 0.922 [0.921–0.922] | 25.7 s [25.6–25.9 s] | 12.6 s [12.4–12.8 s] | 2.04× |
| spambase | 4601 | 57 | 300 (10/10) | 0.7% | 0.922 [0.921–0.923] | 48.9 s [48.6–49.1 s] | 29 s [28.8–29.4 s] | 1.69× |
| epileptic | 5750 | 178 | 300 (10/10) | 0.2% | 0.904 [0.904–0.905] | 78.3 s [78.1–78.5 s] | 55 s [54.9–55.2 s] | 1.42× |

- **Stress:** full SGDMDS is lower on all 17 datasets, and every full fit is
  below every SMACOF fit on 16 (fmd's distributions overlap). Wilcoxon
  signed-rank over the 17 dataset medians: p = 1.5e-5.
- **Time:** full SGDMDS is faster on 16 datasets under the 15% rule and slower on
  imdb. Wilcoxon p = 2.9e-4.
- **Most of the stress gap is SMACOF's iteration cap** (spec §14.3). SMACOF hits
  its 300-iteration cap on 13 datasets, on every seed for 11 of them. It
  converges on four (svhn, sms, cnae9 and imdb, in 64–173 iterations), and there
  SGD's advantage shrinks to ≤0.3%. svhn is the exception: SMACOF converges to a
  minimum 10% worse.

### 1.2 Time to reach SMACOF's final stress (O)

**T2. Step 1, O: time to reach SMACOF's final stress**

| dataset | N | SMACOF time | full: time to SMACOF's stress | seeds reaching | SMACOF time ÷ that | same for budget k200 | same for blocked k200 |
|---|--:|--:|--:|--:|--:|--:|--:|
| seismic | 646 | 0.53 s | 0.043 s [0.041–0.044 s] | 10/10 | 12.46× | 29.73× (10/10) | 37.37× (9/10) |
| svhn | 732 | 0.43 s | 0.069 s [0.067–0.077 s] | 10/10 | 6.21× | 0.71× (10/10) | 1.72× (10/10) |
| har | 735 | 0.71 s | 0.078 s [0.076–0.079 s] | 10/10 | 9.15× | 1.45× (10/10) | 3.61× (10/10) |
| sms | 835 | 0.36 s | 0.18 s [0.18–0.18 s] | 10/10 | 2.03× | — (0/10) | — (1/10) |
| fmd | 997 | 1.65 s | 0.28 s [0.27–0.28 s] | 9/10 | 5.91× | — (0/10) | — (0/10) |
| cnae9 | 1080 | 1.01 s | 0.31 s [0.31–0.31 s] | 10/10 | 3.28× | — (0/10) | — (0/10) |
| coil20 | 1440 | 4.57 s | 0.61 s [0.59–0.63 s] | 10/10 | 7.54× | 6.92× (10/10) | 16.59× (10/10) |
| secom | 1567 | 4.49 s | 0.61 s [0.61–0.62 s] | 10/10 | 7.31× | — (0/10) | — (5/10) |
| bank | 2059 | 10.3 s | 1.2 s [1.19–1.25 s] | 10/10 | 8.61× | — (2/10) | 58.88× (9/10) |
| sentiment | 2748 | 18.4 s | 3.88 s [3.77–3.96 s] | 10/10 | 4.75× | — (0/10) | — (0/10) |
| fashion_mnist | 3000 | 21.6 s | 2.35 s [2.33–2.39 s] | 10/10 | 9.17× | 11.98× (10/10) | 31.50× (10/10) |
| hiva | 3076 | 22.9 s | 3.52 s [3.41–3.6 s] | 10/10 | 6.50× | — (1/10) | — (5/10) |
| hatespeech | 3221 | 25.2 s | 5.52 s [5.48–5.6 s] | 10/10 | 4.56× | — (0/10) | — (0/10) |
| imdb | 3250 | 5.63 s | 7.66 s [7.6–7.77 s] | 10/10 | 0.73× | — (0/10) | — (0/10) |
| cifar10 | 3250 | 25.7 s | 3.76 s [3.7–3.79 s] | 10/10 | 6.84× | 8.93× (10/10) | 25.24× (10/10) |
| spambase | 4601 | 48.9 s | 8.82 s [8.76–8.93 s] | 10/10 | 5.54× | 103× (10/10) | 169× (10/10) |
| epileptic | 5750 | 78.3 s | 14.3 s [14.1–14.4 s] | 10/10 | 5.50× | 52.48× (10/10) | 91.12× (10/10) |

- **Full SGDMDS** reaches SMACOF's median final stress in 169 of 170 fits. It
  gets there a median 6.2× sooner than SMACOF finishes (2.0–12.5×); only imdb is
  slower (0.73×).
- **SMACOF** never reaches full SGDMDS's median final stress (0 of 170 fits).
- **Budgeted and blocked at k=200** reach SMACOF's stress on 8 and 9 datasets,
  up to 169× sooner. Elsewhere their own overhead is larger than SMACOF's
  deficit.

### 1.3 Runtime vs N (Q)

Each cell: SMACOF's time ÷ full SGDMDS's time (>1 means SGDMDS is faster) ·
SMACOF's median iterations · SMACOF's stress ÷ full SGDMDS's stress.

**T3. Step 1, Q: SMACOF ÷ full time · SMACOF iterations · SMACOF ÷ full stress**

| N | s_curve (D=100) | swiss_roll (D=300) | torus (D=500) |
|--:|--:|--:|--:|
| 500 | 2.98× · 300 · 1.002 | 2.98× · 300 · 0.999 | 2.20× · 219 · 0.999 |
| 1000 | 3.61× · 300 · 1.003 | 3.27× · 287 · 0.999 | 2.09× · 167 · 0.999 |
| 1500 | 3.46× · 300 · 1.004 | 1.89× · 165 · 1.000 | 1.76× · 153 · 1.000 |
| 2000 | 3.66× · 300 · 1.004 | 1.75× · 143 · 1.000 | 1.80× · 146 · 1.000 |
| 2500 | 3.46× · 300 · 1.005 | 1.54× · 132 · 1.001 | 1.69× · 144 · 1.000 |
| 3000 | 2.45× · 300 · 1.005 | ≈1.05× · 126 · 1.001 | 1.17× · 141 · 1.000 |
| 3500 | 2.07× · 300 · 1.005 | 0.85× (slower) · 122 · 1.000 | ≈0.94× · 140 · 1.000 |
| 4000 | 1.88× · 300 · 1.005 | 0.76× (slower) · 122 · 1.001 | 0.83× (slower) · 137 · 1.000 |
| 4500 | 1.68× · 300 · 1.006 | 0.66× (slower) · 118 · 1.001 | 0.74× (slower) · 136 · 1.000 |
| 5000 | 1.61× · 300 · 1.006 | 0.63× (slower) · 116 · 1.000 | 0.71× (slower) · 134 · 1.000 |

**T3s. Step 1, Q: log-log slope of median time against N (N ≥ 1,000)**

| dataset | smacof-random | full | budget-k200 | blocked-k200-m16 |
|---|--:|--:|--:|--:|
| s_curve | 2.25 | 2.83 | 1.06 | 0.98 |
| swiss_roll | 1.81 | 2.84 | 1.01 | 0.99 |
| torus | 2.13 | 2.87 | 0.99 | 1.00 |

- **Growth rates:** full SGDMDS's time grows as N^2.83–2.87, SMACOF's as
  N^1.81–2.25, and budgeted and blocked as N^0.98–1.06.
- **swiss_roll and torus:** SMACOF overtakes full SGDMDS from N=3,500 and
  N=4,000 respectively, after one comparable rung. On these two it stops at
  115–145 iterations and matches full SGDMDS's stress to within 0.1%. It reaches
  full SGDMDS's median final stress in 19 and 42 of 100 fits.
- **s_curve:** SMACOF uses all 300 iterations, ends 0.2–0.6% worse, and is
  still slower at N=5,000 (1.6×, and falling).
- **Time to SMACOF's stress (`step1_q_ttt.csv`):** 2.9–6.3× sooner on s_curve.
  On swiss_roll and torus it falls to ≈1.0× by N=5,000. At N≤1,000 (swiss_roll)
  and N≤2,000 (torus), SGDMDS doesn't reach SMACOF's stress on most seeds,
  because SMACOF's stress is equal there or slightly lower.

### 1.4 Why the margin shrinks with N

**T4. Cost per pair: full sweep (per epoch) and SMACOF (per iteration), by N**

| N | fits | full sweep, ns per pair per epoch | SMACOF, ns per pair per iteration |
|--:|--:|--:|--:|
| ≤1,100 | 120 | 25–29 | 8–11 |
| 1,400–2,100 | 90 | 33–55 | 12–16 |
| 2,500–2,750 | 40 | 45–59 | 16–16 |
| 3,000–3,250 | 80 | 64–78 | 16–17 |
| 3,500–5,000 | 130 | 77–97 | 15–17 |
| 5,750 | 10 | 110–110 | 16–16 |
| 10k | 24 | 109–116 | — |
| 20k | 7 | 113–115 | — |
| 35k | 8 | 112–120 | — |
| 50k | 7 | 114–121 | — |

- **Full SGDMDS:**
  - An epoch costs 25–29 ns per pair up to N≈1,100 and 110–121 ns from
    N≈5,750, whatever D is. The rise starts around where the pair arrays and
    their shuffled copies (~50 bytes per pair) outgrow one 8-core chiplet's 32 MB
    L3.
  - Setup (distance matrix, pair arrays) is under 2% of a full fit at N≥10k:
    0.4% at D=3 and 2.0% at D=3,072. So ~98% of the time is the 30 epochs.
- **The shuffle dominates.**
  - The SGD kernel costs 3.6 ns per pair in the microbenchmark (spec §12).
  - For metric MDS, the only other per-epoch work in
    [`_run_cycle`](_sgd_mds.py) is `shuffle(pairs, targets, weights)`: a
    permutation of all N(N−1)/2 pairs and three gathers.
  - By elimination the shuffle takes ~110 of the ~115 ns, about 95% of a large
    full fit.
- **SMACOF** costs 8–11 ns per pair per iteration at N≤1,100 and 15–17 ns from
  N≈2,000; its work is BLAS-friendly.
- **Result:** one full-SGDMDS epoch costs ~3 SMACOF iterations at small N and ~7
  at large N. SGDMDS's 30 epochs therefore beat SMACOF only when SMACOF needs
  more than ~90 iterations (small N) or ~210 (large N). Every verdict in T1 and
  T3 follows from this ratio and SMACOF's iteration count.

---

## 2. Step 2: budgeted vs full SGDMDS

Each cell: budgeted stress ÷ full stress · full time ÷ budgeted time, at the same
dataset, N and k.

### 2.1 Small suite (O)

**T5. Step 2, O: budgeted ÷ full (stress overhead · speedup)**

| dataset | N | D | k=50 | k=100 | k=200 | k=400 |
|---|--:|--:|--:|--:|--:|--:|
| seismic | 646 | 24 | 1.154 · 7.86× | 1.029 · 4.02× | 1.006 · 2.04× | — |
| svhn | 732 | 1024 | 1.098 · 0.40× (slower) | 1.074 · 0.20× (slower) | 1.025 · 0.10× (slower) | — |
| har | 735 | 561 | 1.129 · 0.73× (slower) | 1.089 · 0.37× (slower) | 1.049 · 0.19× (slower) | — |
| sms | 835 | 500 | 1.007 · ≈1.00× | 1.004 · 0.51× (slower) | 1.003 · 0.25× (slower) | 1.002 · 0.13× (slower) |
| fmd | 997 | 1536 | 1.034 · 0.43× (slower) | 1.022 · 0.22× (slower) | 1.017 · 0.11× (slower) | 1.012 · 0.05× (slower) |
| cnae9 | 1080 | 856 | 1.008 · 0.79× (slower) | 1.006 · 0.40× (slower) | 1.005 · 0.20× (slower) | 1.004 · 0.10× (slower) |
| coil20 | 1440 | 400 | 1.068 · 3.99× | 1.033 · 2.02× | 1.015 · ≈1.01× | 1.007 · 0.51× (slower) |
| secom | 1567 | 590 | 1.095 · 2.02× | 1.088 · ≈1.02× | 1.051 · 0.51× (slower) | 1.017 · 0.26× (slower) |
| bank | 2059 | 63 | 1.129 · 24.38× | 1.067 · 13.16× | 1.039 · 6.86× | 1.014 · 3.51× |
| sentiment | 2748 | 200 | 1.011 · 17.90× | 1.009 · 9.12× | 1.007 · 4.61× | 1.006 · 2.32× |
| fashion_mnist | 3000 | 784 | 1.107 · 5.43× | 1.034 · 2.72× | 1.008 · 1.36× | 1.003 · 0.68× (slower) |
| hiva | 3076 | 1617 | 1.101 · 2.28× | 1.097 · ≈1.14× | 1.089 · 0.57× (slower) | 1.056 · 0.29× (slower) |
| hatespeech | 3221 | 100 | 1.045 · 45.46× | 1.041 · 23.56× | 1.037 · 11.99× | 1.033 · 6.05× |
| cifar10 | 3250 | 1024 | 1.155 · 5.39× | 1.029 · 2.70× | 1.009 · 1.35× | 1.003 · 0.68× (slower) |
| imdb | 3250 | 700 | 1.009 · 7.58× | 1.006 · 3.80× | 1.003 · 1.90× | 1.001 · ≈0.95× |
| spambase | 4601 | 57 | 1.241 · 124× | 1.131 · 64.70× | 1.062 · 32.90× | 1.034 · 16.66× |
| epileptic | 5750 | 178 | 1.133 · 79.63× | 1.113 · 40.39× | 1.093 · 20.32× | 1.065 · 10.21× |

**T5s. Step 2, O: budgeted and blocked ÷ full, summary per k**

| arm | k | rungs | overhead: median (range) | speedup: median (range) | faster / ≈ / slower |
|--:|--:|--:|--:|--:|--:|
| blocked | 50 | 17 | 1.100 (1.007–1.218) | 8.52× (0.95×–179×) | 15 / 2 / 0 |
| blocked | 100 | 17 | 1.040 (1.004–1.111) | 4.37× (0.48×–93.73×) | 12 / 2 / 3 |
| blocked | 200 | 17 | 1.015 (1.003–1.090) | 2.21× (0.24×–48.03×) | 12 / 0 / 5 |
| blocked | 400 | 14 | 1.007 (1.001–1.059) | 1.62× (0.13×–24.33×) | 8 / 2 / 4 |
| budget | 50 | 17 | 1.098 (1.007–1.241) | 5.39× (0.40×–124×) | 12 / 1 / 4 |
| budget | 100 | 17 | 1.034 (1.004–1.131) | 2.70× (0.20×–64.70×) | 10 / 2 / 5 |
| budget | 200 | 17 | 1.017 (1.003–1.093) | 1.35× (0.10×–32.90×) | 9 / 1 / 7 |
| budget | 400 | 14 | 1.010 (1.001–1.065) | 0.68× (0.05×–16.66×) | 5 / 1 / 8 |

### 2.2 Large suite (P)

**T6. Step 2, P: budgeted ÷ full (stress overhead · speedup)**

| dataset | N | D | k=100 | k=200 | k=400 |
|---|--:|--:|--:|--:|--:|
| ag_news | 10000 | 384 | 1.102 · 30.12× | 1.095 · 15.07× | 1.045 · 7.54× |
| ag_news | 20000 | 384 | 1.103 · 30.35× | 1.097 · 15.28× | 1.054 · 7.61× |
| ag_news | 35000 | 384 | 1.104 · 49.77× | 1.100 · 25.11× | 1.071 · 12.55× |
| ag_news | 50000 | 384 | 1.104 · 70.55× | 1.100 · 35.57× | 1.070 · 17.71× |
| california_housing | 10000 | 8 | 1.045 · 514× | 1.019 · 270× | 1.009 · 138× |
| california_housing | 20640 | 8 | 1.052 · 1,015× | 1.021 · 518× | 1.010 · 262× |
| cifar10_raw | 10000 | 3072 | 1.047 · 2.91× | 1.011 · 1.45× | 1.004 · 0.72× (slower) |
| cifar10_raw | 20000 | 3072 | 1.106 · 5.87× | 1.011 · 2.94× | 1.004 · 1.47× |
| cifar10_raw | 35000 | 3072 | 1.066 · 10.33× | 1.012 · 5.18× | 1.004 · 2.60× |
| cifar10_raw | 50000 | 3072 | 1.072 · 15.16× | 1.013 · 7.57× | 1.004 · 3.78× |
| fashion_mnist_full | 10000 | 784 | 1.058 · 9.43× | 1.013 · 4.78× | 1.003 · 2.38× |
| fashion_mnist_full | 20000 | 784 | 1.060 · 18.15× | 1.011 · 8.83× | 1.003 · 4.47× |
| fashion_mnist_full | 35000 | 784 | 1.061 · 30.56× | 1.011 · 15.20× | 1.003 · 7.65× |
| fashion_mnist_full | 50000 | 784 | 1.067 · 44.38× | 1.012 · 22.05× | 1.004 · 11.04× |
| feynman_iii_15_12 | 10000 | 3 | 0.994 · 531× | 0.991 · 275× | 0.990 · 140× |
| feynman_iii_15_12 | 20000 | 3 | 1.004 · 1,136× | 1.001 · 580× | 1.001 · 292× |
| feynman_iii_15_12 | 35000 | 3 | 1.004 · 1,841× | 1.001 · 925× | 1.001 · 467× |
| feynman_iii_15_12 | 50000 | 3 | 1.007 · 2,637× | 1.005 · 1,329× | 1.004 · 668× |
| mnist_digits | 10000 | 784 | 1.156 · 9.16× | 1.082 · 4.57× | 1.022 · 2.31× |
| mnist_digits | 20000 | 784 | 1.174 · 17.58× | 1.083 · 8.76× | 1.031 · 4.43× |
| mnist_digits | 35000 | 784 | 1.187 · 31.01× | 1.089 · 15.57× | 1.028 · 7.73× |
| mnist_digits | 50000 | 784 | 1.189 · 49.43× | 1.105 · 24.73× | 1.025 · 12.37× |
| patchcamelyon | 10000 | 512 | 1.067 · 15.08× | 1.019 · 7.46× | 1.006 · 3.75× |
| patchcamelyon | 20000 | 512 | 1.099 · 24.47× | 1.018 · 12.26× | 1.006 · 6.19× |
| patchcamelyon | 35000 | 512 | 1.099 · 40.76× | 1.021 · 20.42× | 1.006 · 10.20× |
| patchcamelyon | 50000 | 512 | 1.105 · 58.19× | 1.025 · 28.86× | 1.007 · 14.54× |
| tiny_imagenet | 10000 | 512 | 1.103 · 15.22× | 1.093 · 7.70× | 1.049 · 3.81× |
| tiny_imagenet | 20000 | 512 | 1.104 · 25.88× | 1.096 · 12.83× | 1.055 · 6.47× |
| tiny_imagenet | 35000 | 512 | 1.104 · 44.41× | 1.100 · 21.87× | 1.060 · 11.13× |
| tiny_imagenet | 50000 | 512 | 1.104 · 57.81× | 1.100 · 29.05× | 1.066 · 14.51× |

**T6s. Step 2, P: budgeted and blocked ÷ full, summary per k**

| arm | k | rungs | overhead: median (range) | speedup: median (range) | faster / ≈ / slower |
|--:|--:|--:|--:|--:|--:|
| blocked | 100 | 30 | 1.056 (0.994–1.176) | 124× (9.62×–2,345×) | 30 / 0 / 0 |
| blocked | 200 | 30 | 1.017 (0.991–1.095) | 61.86× (4.81×–1,179×) | 30 / 0 / 0 |
| blocked | 400 | 30 | 1.005 (0.990–1.045) | 30.91× (2.41×–591×) | 30 / 0 / 0 |
| budget | 100 | 30 | 1.099 (0.994–1.189) | 30.79× (2.91×–2,637×) | 30 / 0 / 0 |
| budget | 200 | 30 | 1.020 (0.991–1.105) | 15.42× (1.45×–1,329×) | 30 / 0 / 0 |
| budget | 400 | 30 | 1.006 (0.990–1.071) | 7.69× (0.72×–668×) | 29 / 0 / 1 |

- **Every rung is faster at k≤200.** The one rung where k=400 isn't is
  cifar10_raw at 10k (0.72×).
- **N dependence:** at k=200, the overhead rises only slightly with N (median
  1.019 at 10k, 1.025 at 50k). The speedup grows roughly in proportion to N:
  median 7.6× at 10k, 12.3× at 20k, 20.4× at 35k and 28.9× at 50k.
- **Hard sources:** ag_news, tiny_imagenet and mnist_digits keep ≥8% overhead
  at k=200 and 2–7% at k=400. These are the datasets the regime theory (J,
  K-optimized; `strain_top2`) marks as hard to budget. That claim is reused,
  not re-derived.

### 2.3 Break-even N (Q)

**T7. Step 2, Q: speedup over full SGDMDS at k=200**

| N | D=100 budget | D=100 blocked | D=300 budget | D=300 blocked | D=500 budget | D=500 blocked |
|--:|--:|--:|--:|--:|--:|--:|
| 500 | 0.80× (slower) | ≈1.13× | 0.28× (slower) | 0.55× (slower) | 0.17× (slower) | 0.36× (slower) |
| 1000 | 1.66× | 2.41× | 0.55× (slower) | ≈1.12× | 0.32× (slower) | 0.71× (slower) |
| 1500 | 2.85× | 4.30× | ≈0.95× | 1.94× | 0.57× (slower) | 1.29× |
| 2000 | 3.84× | 5.95× | 1.31× | 2.69× | 0.79× (slower) | 1.81× |
| 2500 | 6.07× | 9.81× | 2.13× | 4.40× | 1.27× | 2.89× |
| 3000 | 10.20× | 16.65× | 3.56× | 7.41× | 2.18× | 4.91× |
| 3500 | 14.04× | 23.03× | 5.02× | 10.37× | 3.17× | 7.11× |
| 4000 | 17.67× | 28.90× | 6.34× | 13.15× | 3.96× | 8.88× |
| 4500 | 21.15× | 35.00× | 7.63× | 15.85× | 4.79× | 10.74× |
| 5000 | 24.40× | 40.28× | 8.87× | 18.46× | 5.31× | 12.09× |

- **Break-even at k=200:** budgeted is faster from N=1,000 (D=100), 2,000
  (D=300) and 2,500 (D=500). Blocked is faster from 1,000, 1,500 and 1,500.
- **Overhead:** 1.005–1.012 at every N (`vs_full_q.csv`).

### 2.4 Where budgeting isn't faster, and why

With setup neglected (under 2% at large N), t_full ≈ 30·c_full(N)·N(N−1)/2 and
t_budget ≈ 30·c_budget(D, N)·kN. So:

> speedup ≈ (N−1)/(2k) × c_full(N) / c_budget(D, N)

With the measured per-pair costs, this matches the measured speedups to within
4% (svhn 0.10×, secom 0.49×, hiva 0.56×, fashion_mnist 1.34×, spambase 32.5×,
feynman at 50k 1,325×). That mainly confirms that setup doesn't matter; what
matters is how each cost depends on N and D:

**T8. Cost per pair: budgeted (m=2) and blocked (m=16), by D**

| dataset | N | D | budget ns/pair | blocked ns/pair | budget ÷ blocked |
|---|--:|--:|--:|--:|--:|
| feynman_iii_15_12 (P) | 50000 | 3 | 11 | 12 | 0.89 |
| california_housing (P) | 20640 | 8 | 12 | 13 | 0.94 |
| seismic (O) | 646 | 24 | 21 | 19 | 1.09 |
| spambase (O) | 4601 | 57 | 32 | 22 | 1.45 |
| hatespeech (O) | 3221 | 100 | 48 | 30 | 1.61 |
| epileptic (O) | 5750 | 178 | 79 | 46 | 1.71 |
| swiss_roll (Q) | 5000 | 300 | 136 | 65 | 2.08 |
| ag_news (P) | 50000 | 384 | 417 | 93 | 4.49 |
| coil20 (O) | 1440 | 400 | 198 | 92 | 2.15 |
| torus (Q) | 5000 | 500 | 230 | 101 | 2.28 |
| patchcamelyon (P) | 50000 | 512 | 494 | 118 | 4.18 |
| tiny_imagenet (P) | 50000 | 512 | 501 | 121 | 4.15 |
| fashion_mnist (O) | 3000 | 784 | 356 | 152 | 2.35 |
| fashion_mnist_full (P) | 50000 | 784 | 657 | 168 | 3.91 |
| mnist_digits (P) | 50000 | 784 | 618 | 166 | 3.73 |
| cifar10 (O) | 3250 | 1024 | 478 | 197 | 2.42 |
| fmd (O) | 997 | 1536 | 709 | 288 | 2.47 |
| hiva (O) | 3076 | 1617 | 969 | 312 | 3.10 |
| cifar10_raw (P) | 50000 | 3072 | 1922 | 593 | 3.24 |

- **c_full(N):** see T4, from 26 to 115 ns.
- **c_budget:** ≈ 10 + 0.45·D ns while X fits in cache (O and Q, within ~10%).
  At N≥20k and D≥384 it becomes memory-bound at 0.6–1.1 ns per dimension: 356 ns
  on fashion_mnist at N=3,000 against 657 ns at N=50k. hiva is already
  above the cache-resident line in O (X is 40 MB, more than one chiplet's L3).
- **c_blocked:** ≈ 10 + 0.19·D ns at any N.
- **Where budgeting isn't faster:**
  - N≤1,100 with D≥500, at any k: svhn, har, fmd and cnae9 (sms is ≈1.00× at
    k=50). At k=200 the pair saving is only 1.8–2.7×, but a budgeted pair costs
    8–24× a full-sweep pair.
  - N=1,400–3,250 with D≳400, at high k: secom and hiva from k=200, and coil20,
    fashion_mnist and cifar10 at k=400. imdb is comparable at k=400.
  - In P, only cifar10_raw at 10k with k=400.
- **Low D is the mirror image.** A budgeted pair costs 11 ns there, against the
  full sweep's 115. So feynman's 1,329× at k=200 and N=50k is 125× from the pair
  saving times 10.6× from cost per pair. That second factor is the shuffle
  (§1.4, §7).

---

## 3. Step 3: blocked vs budgeted vs full SGDMDS, plus SQuaD-MDS

### 3.1 Blocked vs budgeted at the same k

T9 shows k=200; the CSV `blocked_{o,p}.csv` has every k.

**T9. Step 3: blocked vs budgeted at k=200, paired by seed**

| dataset | N | D | blocked ÷ budget stress | budget ÷ blocked time |
|---|--:|--:|--:|--:|
| seismic (O) | 646 | 24 | 0.996 [0.990–1.001] | ≈1.08× |
| spambase (O) | 4601 | 57 | 0.990 [0.988–0.995] | 1.46× |
| bank (O) | 2059 | 63 | 0.989 [0.986–0.995] | 1.35× |
| hatespeech (O) | 3221 | 100 | 0.998 [0.998–0.999] | 1.61× |
| epileptic (O) | 5750 | 178 | 0.997 [0.995–0.998] | 1.72× |
| sentiment (O) | 2748 | 200 | 1.000 [1.000–1.000] | 1.90× |
| coil20 (O) | 1440 | 400 | 0.999 [0.996–1.001] | 2.16× |
| sms (O) | 835 | 500 | 1.000 [1.000–1.000] | 2.21× |
| har (O) | 735 | 561 | 1.004 [0.987–1.017] | 2.24× |
| secom (O) | 1567 | 590 | 0.980 [0.972–0.987] | 2.28× |
| imdb (O) | 3250 | 700 | 1.000 [1.000–1.000] | 2.34× |
| fashion_mnist (O) | 3000 | 784 | 1.002 [0.999–1.004] | 2.35× |
| cnae9 (O) | 1080 | 856 | 1.000 [0.999–1.000] | 2.37× |
| svhn (O) | 732 | 1024 | 1.007 [0.982–1.023] | 2.39× |
| cifar10 (O) | 3250 | 1024 | 1.000 [0.999–1.001] | 2.43× |
| fmd (O) | 997 | 1536 | 0.998 [0.998–0.999] | 2.47× |
| hiva (O) | 3076 | 1617 | 0.990 [0.981–1.001] | 3.10× |
| feynman_iii_15_12 (P) | 10000 | 3 | 1.000 [1.000–1.000] | 0.81× (slower) |
| feynman_iii_15_12 (P) | 50000 | 3 | 0.997 [0.997–1.000] | ≈0.89× |
| california_housing (P) | 10000 | 8 | 1.001 [0.997–1.002] | 0.85× (slower) |
| california_housing (P) | 20640 | 8 | 1.001 [1.000–1.004] | ≈0.94× |
| ag_news (P) | 10000 | 384 | 0.990 [0.978–0.991] | 2.27× |
| ag_news (P) | 50000 | 384 | 0.991 [0.991–0.994] | 4.46× |
| patchcamelyon (P) | 10000 | 512 | 0.998 [0.994–1.004] | 3.34× |
| tiny_imagenet (P) | 10000 | 512 | 0.981 [0.979–0.991] | 3.26× |
| patchcamelyon (P) | 50000 | 512 | 0.993 [0.990–0.999] | 4.21× |
| tiny_imagenet (P) | 50000 | 512 | 0.995 [0.995–0.997] | 4.21× |
| fashion_mnist_full (P) | 10000 | 784 | 0.998 [0.996–0.998] | 3.53× |
| mnist_digits (P) | 10000 | 784 | 0.964 [0.957–0.967] | 3.68× |
| fashion_mnist_full (P) | 50000 | 784 | 1.000 [0.999–1.000] | 3.92× |
| mnist_digits (P) | 50000 | 784 | 0.958 [0.954–0.969] | 3.73× |
| cifar10_raw (P) | 10000 | 3072 | 0.998 [0.997–0.999] | 3.32× |
| cifar10_raw (P) | 50000 | 3072 | 0.998 [0.997–0.999] | 3.22× |

- **Stress:** the median ratio is 0.999 (O) and 0.997 (P) over all k. Blocked
  is never more than 2.5% worse on any (dataset, N, k), and up to 6% better at
  low k (cifar10 at k=50: 0.938; mnist_digits: 0.96 at k=200).
- **Time:**
  - In O, blocked is faster by 1.08× at D=24, 1.3–1.6× at D=57–100, 1.7–1.9× at
    D=178–200, 2.2–2.5× at D=400–1,536, and 3.1× on hiva.
  - In P, the gain rises from 2.3–3.7× at 10k to 3.2–4.5× at 50k. The budgeted
    kernel turns memory-bound, while blocked reads each point once per block.
  - At D≤8, blocking costs 6–19% (0.81–0.94×). This is the random kernel's low-D
    advantage that the freeze kept on purpose (CLAUDE.md §4).
- **Against full at k=200 in P:** blocked gives a median 62× (4.8–1,179×) at
  1.017× stress; budgeted gives 15× at 1.020× (T6s).

### 3.2 SQuaD-MDS: chosen setting and its bound

"Last rung's gain" is the stress of the best setting one `n_iter` rung below the
top, divided by the stress at the top. Above 1 means the top rung still
improved.

**T10. Step 3, O: SQuaD-MDS's best setting**

| dataset | setting | stress ÷ full | on selection seeds | last rung's gain | time | authors' default ÷ full |
|---|---|--:|--:|--:|--:|--:|
| bank | i8000 lr550 pca cap1 | 1.041 [1.040–1.041] | 1.040 | 1.010× | 0.99 s | 1.321 |
| cifar10 | i8000 lr150 pca cap1 | 1.026 [1.026–1.027] | 1.027 | 1.004× | 6.57 s | 1.172 |
| cnae9 | i8000 lr1500 random cap1 | 1.018 [1.017–1.018] | 1.017 | 0.999× | 1.93 s | 2.783 |
| coil20 | i8000 lr150 pca cap1 | 1.054 [1.054–1.054] | 1.054 | 1.003× | 1.59 s | 1.249 |
| epileptic | i8000 lr550 pca capnone | 1.043 [1.042–1.043] | 1.043 | 1.033× | 3.14 s | 1.618 |
| fashion_mnist | i8000 lr550 pca cap4 | 1.051 [1.051–1.052] | 1.052 | 1.002× | 4.82 s | 1.442 |
| fmd | i8000 lr150 pca cap1 | 1.013 [1.013–1.014] | 1.013 | 1.008× | 2.8 s | 1.174 |
| har | i8000 lr550 pca cap4 | 1.050 [1.050–1.052] | 1.049 | 1.009× | 1 s | 1.576 |
| hatespeech | i8000 lr150 pca cap1 | 1.020 [1.020–1.020] | 1.020 | 1.004× | 1.73 s | 1.179 |
| hiva | i8000 lr150 pca cap1 | 1.040 [1.040–1.041] | 1.041 | 1.011× | 12 s | 1.336 |
| imdb | i8000 lr150 pca cap1 | 1.023 [1.023–1.023] | 1.023 | 1.005× | 4.79 s | 1.053 |
| secom | i8000 lr550 pca cap1 | 1.058 [1.058–1.059] | 1.059 | 1.032× | 2.15 s | 1.804 |
| seismic | i4000 lr150 pca cap1 | 1.045 [1.044–1.045] | 1.043 | 0.993× | 0.16 s | 1.807 |
| sentiment | i8000 lr150 pca cap1 | 1.010 [1.009–1.010] | 1.010 | 1.003× | 1.94 s | 1.143 |
| sms | i8000 lr550 pca cap1 | 1.014 [1.014–1.014] | 1.014 | 1.002× | 1.04 s | 1.578 |
| spambase | i8000 lr550 pca cap1 | 1.082 [1.081–1.082] | 1.080 | 1.036× | 2.14 s | 1.829 |
| svhn | i8000 lr150 pca cap1 | 0.979 [0.978–0.980] | 0.980 | 1.005× | 1.5 s | 1.495 |

**T11. Step 3, P: SQuaD-MDS's best setting (N=10k and each source's top N)**

| dataset | N | setting | stress ÷ full | last rung's gain | time | authors' default ÷ full |
|---|--:|---|--:|--:|--:|--:|
| ag_news | 10000 | i4000 lr550 exa pca cap4 | 1.054 [1.054–1.054] | 1.005× | 5.39 s | 1.064 |
| ag_news | 50000 | i4000 lr550 exa pca cap4 | 1.055 [1.055–1.055] | 1.005× | 47 s | 1.065 |
| california_housing | 10000 | i2000 lr550 pca cap4 | 1.199 [1.198–1.200] | 0.962× | 0.93 s | 1.899 |
| california_housing | 20640 | i4000 lr550 pca cap4 | 4.199 [4.195–4.205] | 1.035× | 3.82 s | 5.271 |
| cifar10_raw | 10000 | i4000 lr550 pca cap4 | 1.033 [1.033–1.033] | 1.007× | 35.8 s | 1.152 |
| cifar10_raw | 50000 | i4000 lr550 pca cap4 | 1.032 [1.032–1.032] | 1.008× | 177 s | 1.155 |
| fashion_mnist_full | 10000 | i4000 lr550 pca cap4 | 1.092 [1.092–1.092] | 1.018× | 12.3 s | 1.494 |
| fashion_mnist_full | 50000 | i4000 lr550 pca cap4 | 1.074 [1.074–1.075] | 1.012× | 67 s | 1.455 |
| feynman_iii_15_12 | 10000 | i4000 lr550 pca cap4 | 1.047 [1.045–1.048] | 1.017× | 1.77 s | 1.111 |
| feynman_iii_15_12 | 50000 | i4000 lr550 exa pca cap4 | 1.072 [1.068–1.073] | 1.026× | 8.83 s | 1.155 |
| mnist_digits | 10000 | i4000 lr550 pca cap4 | 1.470 [1.470–1.471] | 1.067× | 12.6 s | 1.994 |
| mnist_digits | 50000 | i4000 lr550 pca cap4 | 1.326 [1.326–1.326] | 1.065× | 63.9 s | 1.813 |
| patchcamelyon | 10000 | i4000 lr550 pca cap4 | 1.054 [1.054–1.054] | 1.011× | 9.14 s | 1.496 |
| patchcamelyon | 50000 | i4000 lr550 pca cap4 | 1.054 [1.054–1.054] | 1.011× | 53.6 s | 1.494 |
| tiny_imagenet | 10000 | i4000 lr550 pca cap4 | 1.058 [1.058–1.059] | 1.009× | 9.12 s | 1.076 |
| tiny_imagenet | 50000 | i4000 lr550 pca cap4 | 1.059 [1.059–1.059] | 1.008× | 60 s | 1.076 |

- **Chosen settings in O:** no exaggeration on all 17, PCA init on 16 (cnae9
  picks random), lr 150 on 9 datasets, 550 on 7 and 1,500 on 1, step cap 1 on
  14. In P the choice is no exaggeration, except ag_news (every N) and
  feynman_iii_15_12 at 50k.
- **At the top of the ladder** on 16 of 17 O datasets (seismic stops at 4,000)
  and 29 of 30 P rungs (california_housing at 10k stops at 2,000).
- **How loose the bound is:**
  - The last doubling gains a median 1.005× in O (at most 1.036×, spambase;
    secom and epileptic 1.032–1.033×) and 1.010× in P.
  - mnist_digits (1.065–1.067×) and california_housing at 20,640 (1.035×) are
    the loose cases in P.
  - So on most datasets a further doubling would likely gain under ~1%. On
    mnist_digits, spambase, secom and epileptic the bound is clearly loose.
- **Stress ÷ full on the reporting seeds:**
  - O: median 1.040 (0.979–1.082). SQuaD-MDS beats full SGDMDS on svhn (0.979).
  - P: median 1.057, or 1.027–1.092 apart from mnist_digits (1.33–1.47) and
    california_housing (1.20 and 4.20).
- **Selection effect:** the reporting seeds (47–51) and selection seeds (42–46)
  differ by at most 0.002. The split is cheap insurance, not a correction, and
  P's in-sample choice is harmless.
- **The authors' default** (i1000, lr550, exaggeration, PCA) gives a median
  1.442× full in O and 1.155× in P, as in M.

### 3.3 Time to reach 1.01× (and looser) full stress

Each cell: full run time ÷ time to reach the target, with the fastest setting
that reaches it. "—" means no setting reaches it on most seeds. The 1.02× and
1.10× targets are in `ttt_{o,p}.csv`.

**T12. Step 3, O: time to reach 1.01× and 1.05× full stress, as speedup over the full run**

| dataset | N | full run | full reaches 1.01× | 1.01× budget | 1.01× blocked | 1.01× squad | 1.05× budget | 1.05× blocked | 1.05× squad |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| bank | 2059 | 2.66 s | 1.52 s | — | — | — | 11.81× (k200) | 17.36× (k200) | 12.41× (i8000) |
| cifar10 | 3250 | 12.6 s | 7.07 s | 1.69× (k200) | 4.24× (k200) | — | 5.06× (k100) | 13.70× (k100) | 37.98× (i2000) |
| cnae9 | 1080 | 0.5 s | 0.3 s | 1.10× (k50) | 2.58× (k50) | — | 2.44× (k50) | 5.86× (k50) | 7.94× (i1000) |
| coil20 | 1440 | 1.73 s | 0.99 s | 0.78× (k400) | 1.65× (k400) | — | 3.70× (k100) | 8.07× (k100) | — |
| epileptic | 5750 | 55 s | 30.7 s | — | — | — | — | — | 36.50× (i8000) |
| fashion_mnist | 3000 | 8.74 s | 4.92 s | 1.80× (k200) | 2.79× (k400) | — | 4.94× (k100) | 8.53× (k100) | — |
| fmd | 997 | 0.46 s | 0.28 s | — | 0.22× (k400) | — | 0.84× (k50) | 2.40× (k50) | 2.91× (i8000) |
| har | 735 | 0.21 s | 0.13 s | — | — | — | 0.22× (k200) | — | — |
| hatespeech | 3221 | 11 s | 6.16 s | — | — | — | 73.42× (k50) | 120× (k50) | 207× (i1000) |
| hiva | 3076 | 10.2 s | 5.78 s | — | — | — | — | 1.61× (k400) | 3.94× (i8000) |
| imdb | 3250 | 11.9 s | 6.51 s | 9.51× (k50) | 27.43× (k50) | — | 32.77× (k50) | 89.04× (k50) | 52.44× (i2000) |
| secom | 1567 | 1.28 s | 0.73 s | — | — | — | 0.57× (k400) | 2.30× (k200) | — |
| seismic | 646 | 0.17 s | 0.1 s | 3.11× (k200) | 3.82× (k200) | — | 7.54× (k100) | 8.33× (k100) | 5.09× (i4000) |
| sentiment | 2748 | 6.77 s | 3.78 s | 12.39× (k100) | 24.06× (k100) | 6.92× (i8000) | 54.95× (k50) | 116× (k50) | 175× (i500) |
| sms | 835 | 0.29 s | 0.17 s | 1.47× (k50) | 3.23× (k50) | — | 3.26× (k50) | 8.07× (k50) | 22.08× (i500) |
| spambase | 4601 | 29 s | 16.2 s | — | — | — | 30.10× (k400) | 44.76× (k400) | — |
| svhn | 732 | 0.21 s | 0.13 s | — | — | 1.49× (i2000) | 0.20× (k200) | 0.44× (k200) | 3.43× (i1000) |

**T13. Step 3, P: time to reach 1.01× and 1.05× full stress (N=10k and each source's top N)**

| dataset | N | full run | full reaches 1.01× | 1.01× budget | 1.01× blocked | 1.01× squad | 1.05× budget | 1.05× blocked | 1.05× squad |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ag_news | 10000 | 168 s | 94.7 s | — | — | — | 11.98× (k400) | 34.75× (k400) | — |
| ag_news | 50000 | 4,431 s | 2,484 s | — | — | — | — | 144× (k400) | — |
| california_housing | 10000 | 175 s | 98.5 s | 179× (k400) | — | — | 791× (k100) | 764× (k100) | — |
| california_housing | 20640 | 767 s | 432 s | 345× (k400) | 389× (k400) | — | 1,021× (k200) | 1,538× (k100) | — |
| cifar10_raw | 10000 | 172 s | 97.5 s | 1.22× (k400) | 5.83× (k200) | — | 3.94× (k100) | 16.58× (k100) | 24.42× (i2000) |
| cifar10_raw | 50000 | 4,366 s | 2,491 s | 6.38× (k400) | 20.85× (k400) | — | 17.21× (k200) | 62.35× (k200) | 125× (i2000) |
| fashion_mnist_full | 10000 | 166 s | 93 s | 4.13× (k400) | 19.37× (k200) | — | 10.82× (k200) | 61.21× (k100) | — |
| fashion_mnist_full | 50000 | 4,351 s | 2,456 s | 18.94× (k400) | 75.10× (k400) | — | 50.38× (k200) | 210× (k200) | — |
| feynman_iii_15_12 | 10000 | 165 s | 92.3 s | 907× (k100) | 720× (k100) | — | 1,315× (k100) | 1,111× (k100) | 143× (i4000) |
| feynman_iii_15_12 | 50000 | 4,334 s | 2,428 s | 3,596× (k100) | 3,213× (k100) | — | 6,296× (k100) | 5,647× (k100) | — |
| mnist_digits | 10000 | 167 s | 93.5 s | — | — | — | 4.48× (k400) | 28.19× (k200) | — |
| mnist_digits | 50000 | 4,583 s | 2,601 s | — | — | — | 23.61× (k400) | 103× (k400) | — |
| patchcamelyon | 10000 | 166 s | 93.1 s | 5.94× (k400) | 20.25× (k400) | — | 15.84× (k200) | 58.33× (k200) | — |
| patchcamelyon | 50000 | 4,307 s | 2,427 s | 22.39× (k400) | 99.59× (k400) | — | 56.63× (k200) | 276× (k200) | — |
| tiny_imagenet | 10000 | 167 s | 93.9 s | — | — | — | 4.69× (k400) | 22.38× (k400) | — |
| tiny_imagenet | 50000 | 4,362 s | 2,449 s | — | — | — | — | 94.14× (k400) | — |

**T14. Step 3: which arm reaches the target first**

| suite | target | rungs | full | budget | blocked | squad | SQuaD-MDS reaches it |
|--:|--:|--:|--:|--:|--:|--:|--:|
| O | 1.01× | 17 | 10 | 0 | 7 | 0 | 2 |
| O | 1.02× | 17 | 6 | 0 | 9 | 2 | 6 |
| O | 1.05× | 17 | 2 | 0 | 6 | 9 | 12 |
| O | 1.10× | 17 | 0 | 0 | 2 | 15 | 17 |
| P | 1.01× | 30 | 12 | 5 | 13 | 0 | 0 |
| P | 1.02× | 30 | 9 | 5 | 16 | 0 | 0 |
| P | 1.05× | 30 | 0 | 5 | 21 | 4 | 7 |
| P | 1.10× | 30 | 0 | 4 | 9 | 17 | 24 |

- **Full SGDMDS's own crossing:** it reaches 1.01× its final stress at 55–62% of
  its run on every dataset. That doesn't mean a 17-epoch run would do as well:
  the schedule depends on `max_iter` (spec §7.4).
- **1.01× is beyond SQuaD-MDS almost everywhere.** Its chosen settings reach it
  only on svhn and sentiment in O, and never in P.
- **First at 1.01×:** the full sweep (10 of 17 in O, 12 of 30 in P) or blocked
  (7 and 13). Budgeted wins 5 P rungs, all at D≤8, where blocking costs time.
- **Looser targets go to SQuaD-MDS:** it is first at 1.05× on 9 of 17 O datasets
  and 4 of 30 P rungs (all cifar10_raw), and at 1.10× on 15 of 17 and 17 of 30.
- **Size of the gains in P:** where SGD reaches 1.01×, blocked gets there
  21–100× sooner than the full run at 50k for D≥512, and ~3,200× on feynman.

### 3.4 Blocked SGDMDS at k=200 vs SQuaD-MDS's best

**T15. Step 3: SQuaD-MDS's best vs blocked SGDMDS at k=200**

| dataset | N | SQuaD ÷ full | SQuaD time | blocked k200 ÷ full | blocked time | blocked ÷ SQuaD stress | blocked ÷ SQuaD time |
|---|--:|--:|--:|--:|--:|--:|--:|
| bank (O) | 2059 | 1.041 | 0.99 s | 1.028 | 0.29 s | 0.988 | 0.29× |
| cifar10 (O) | 3250 | 1.026 | 6.57 s | 1.008 | 3.84 s | 0.983 | 0.58× |
| cnae9 (O) | 1080 | 1.018 | 1.93 s | 1.004 | 1.07 s | 0.987 | 0.55× |
| coil20 (O) | 1440 | 1.054 | 1.59 s | 1.015 | 0.79 s | 0.962 | 0.50× |
| epileptic (O) | 5750 | 1.043 | 3.14 s | 1.090 | 1.58 s | 1.045 | 0.50× |
| fashion_mnist (O) | 3000 | 1.051 | 4.82 s | 1.011 | 2.73 s | 0.962 | 0.57× |
| fmd (O) | 997 | 1.013 | 2.8 s | 1.015 | 1.72 s | 1.002 | 0.61× |
| har (O) | 735 | 1.050 | 1 s | 1.045 | 0.5 s | 0.995 | 0.50× |
| hatespeech (O) | 3221 | 1.020 | 1.73 s | 1.035 | 0.57 s | 1.015 | 0.33× |
| hiva (O) | 3076 | 1.040 | 12 s | 1.081 | 5.75 s | 1.039 | 0.48× |
| imdb (O) | 3250 | 1.023 | 4.79 s | 1.003 | 2.67 s | 0.980 | 0.56× |
| secom (O) | 1567 | 1.058 | 2.15 s | 1.029 | 1.1 s | 0.973 | 0.51× |
| seismic (O) | 646 | 1.045 | 0.16 s | 1.003 | 0.075 s | 0.960 | 0.48× |
| sentiment (O) | 2748 | 1.010 | 1.94 s | 1.007 | 0.77 s | 0.998 | 0.40× |
| sms (O) | 835 | 1.014 | 1.04 s | 1.003 | 0.51 s | 0.989 | 0.49× |
| spambase (O) | 4601 | 1.082 | 2.14 s | 1.053 | 0.6 s | 0.973 | 0.28× |
| svhn (O) | 732 | 0.979 | 1.5 s | 1.032 | 0.86 s | 1.054 | 0.57× |
| ag_news (P) | 10000 | 1.054 | 5.39 s | 1.080 | 4.92 s | 1.024 | 0.91× |
| ag_news (P) | 50000 | 1.055 | 47 s | 1.091 | 27.9 s | 1.034 | 0.59× |
| california_housing (P) | 10000 | 1.199 | 0.93 s | 1.019 | 0.77 s | 0.850 | 0.83× |
| california_housing (P) | 20640 | 4.199 | 3.82 s | 1.021 | 1.58 s | 0.243 | 0.41× |
| cifar10_raw (P) | 10000 | 1.033 | 35.8 s | 1.009 | 35.8 s | 0.976 | 1.00× |
| cifar10_raw (P) | 50000 | 1.032 | 177 s | 1.011 | 178 s | 0.980 | 1.00× |
| fashion_mnist_full (P) | 10000 | 1.092 | 12.3 s | 1.009 | 9.84 s | 0.924 | 0.80× |
| fashion_mnist_full (P) | 50000 | 1.074 | 67 s | 1.012 | 50.4 s | 0.941 | 0.75× |
| feynman_iii_15_12 (P) | 10000 | 1.047 | 1.77 s | 1.002 | 0.74 s | 0.956 | 0.42× |
| feynman_iii_15_12 (P) | 50000 | 1.072 | 8.83 s | 1.005 | 3.68 s | 0.937 | 0.42× |
| mnist_digits (P) | 10000 | 1.470 | 12.6 s | 1.043 | 9.9 s | 0.710 | 0.79× |
| mnist_digits (P) | 50000 | 1.326 | 63.9 s | 1.069 | 49.7 s | 0.806 | 0.78× |
| patchcamelyon (P) | 10000 | 1.054 | 9.14 s | 1.016 | 6.64 s | 0.964 | 0.73× |
| patchcamelyon (P) | 50000 | 1.054 | 53.6 s | 1.018 | 35.5 s | 0.966 | 0.66× |
| tiny_imagenet (P) | 10000 | 1.058 | 9.12 s | 1.073 | 6.68 s | 1.014 | 0.73× |
| tiny_imagenet (P) | 50000 | 1.059 | 60 s | 1.095 | 36.3 s | 1.035 | 0.61× |

- **O:** blocked at k=200 has lower stress than SQuaD-MDS's best on 12 of 17
  datasets (median 0.988×), in a median 0.50× of SQuaD-MDS's time.
- **P:** lower on 22 of 30 rungs (median 0.964×), in a median 0.73× of the time.
- **Where SQuaD-MDS is better:** epileptic, hiva, hatespeech and svhn (fmd is a
  tie) in O; ag_news and tiny_imagenet by 2–5% in P, as in Experiment N.
- **Caveat:** this compares SQuaD-MDS at its most expensive setting. T14 is the
  fairer comparison across the whole trade-off.

---

## 4. Classical-init SMACOF (data for the later study)

**T16. Classical-init SMACOF, O (for the later study)**

| dataset | N | D | iters (at cap) | classical ÷ full stress | full's IQR ÷ classical | classical ÷ random-init SMACOF | classical time | full ÷ classical speed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| seismic | 646 | 24 | 131 (0/10) | 1.001 | 0.996–1.003 | 0.878 | 0.26 s | 1.54× |
| svhn | 732 | 1024 | 62 (0/10) | 0.952 | 1.046–1.064 | 0.858 | 0.21 s | ≈0.99× |
| har | 735 | 561 | 87 (0/10) | 0.976 | 1.022–1.035 | 0.894 | 0.25 s | 1.19× |
| sms | 835 | 500 | 87 (0/10) | 0.989 | 1.011–1.011 | 0.987 | 0.34 s | 1.19× |
| fmd | 997 | 1536 | 195 (0/10) | 0.979 | 1.019–1.026 | 0.970 | 1.18 s | 2.57× |
| cnae9 | 1080 | 856 | 149 (0/10) | 0.988 | 1.012–1.013 | 0.985 | 0.99 s | 1.98× |
| coil20 | 1440 | 400 | 105 (0/10) | 1.007 | 0.991–0.996 | 0.944 | 1.84 s | ≈1.07× |
| secom | 1567 | 590 | 223 (0/10) | 0.993 | 1.007–1.008 | 0.964 | 3.55 s | 2.77× |
| bank | 2059 | 63 | 213 (0/10) | 1.003 | 0.995–0.998 | 0.968 | 7.86 s | 2.95× |
| sentiment | 2748 | 200 | 186 (0/10) | 0.980 | 1.020–1.021 | 0.976 | 18.7 s | 2.76× |
| fashion_mnist | 3000 | 784 | 67 (0/10) | 1.002 | 0.997–0.998 | 0.913 | 6.53 s | 0.75× (slower) |
| hiva | 3076 | 1617 | 300 (10/10) | 1.001 | 0.998–1.000 | 0.927 | 24.7 s | 2.42× |
| hatespeech | 3221 | 100 | 263 (0/10) | 0.988 | 1.012–1.013 | 0.961 | 24.2 s | 2.21× |
| imdb | 3250 | 700 | 93 (0/10) | 0.988 | 1.012–1.012 | 0.988 | 10.3 s | ≈0.87× |
| cifar10 | 3250 | 1024 | 66 (0/10) | 1.000 | 1.000–1.001 | 0.921 | 8 s | 0.63× (slower) |
| spambase | 4601 | 57 | 182 (0/10) | 1.003 | 0.996–0.998 | 0.925 | 36.4 s | 1.26× |
| epileptic | 5750 | 178 | 184 (0/10) | 1.013 | 0.987–0.988 | 0.916 | 62.2 s | ≈1.13× |

- **Deterministic:** identical stress on all 10 seeds; only the times vary.
- **Against full SGDMDS:**
  - Lower than full SGDMDS's median on 10 of 17 datasets, and below all 10 full
    seeds on 9: svhn, har, sms, fmd, cnae9, secom, sentiment, hatespeech and
    imdb, at 0.95–0.99×.
  - Within 0.3% on cifar10, seismic, bank, fashion_mnist, hiva and spambase.
  - Worse on coil20 (+0.7%) and epileptic (+1.3%).
- **Against random-init SMACOF:** better everywhere (0.86–0.99×). It converges
  before the cap on 16 of 17 datasets; hiva runs all 300 iterations.
- **Time (including the classical-MDS init):** full SGDMDS is faster on 11,
  comparable on 4, and slower on fashion_mnist (0.75×) and cifar10 (0.63×). On
  those two, classical SMACOF converges in 66–67 iterations.
- **Use:** the later study compares classical (or PCA) init for SGDMDS with this
  arm (STATUS.md §0).

---

## 5. Consistency with earlier results

- **Reproduction:** P's budgeted stress reproduces K-optimized, and SQuaD-MDS
  reproduces M (spec §12).
- **Blocked vs budgeted** matches N's kernel measurements: the random kernel
  is the faster at D≤8, and blocked from D≈24 up (T8). At D=784 and N=50k,
  blocking gains 3.7–3.9× here, against 3.5× in N's kernel table.
- **Speed figures from before the freeze** (A, B, J, K, K-optimized, N) predate
  the `noexcept` fix. These results supersede them; don't mix the two
  (CLAUDE.md §6).

---

## 6. What the data supports for the paper

The paper is not edited here; this is what each step can claim as of now.

1. **Step 1:**
   - On all 17 real datasets, full SGDMDS reaches lower stress than SMACOF at its
     default budget. It is faster on 16 (median 2.3×) and reaches SMACOF's final
     stress a median 6.2× sooner.
   - The paper must also say that SMACOF matches SGDMDS's stress and is faster
     where it converges early (imdb; Q's swiss_roll and torus above N≈3,500),
     and that the margin shrinks with N.
2. **Step 2:**
   - On the large suite, budgeted SGDMDS at k=200 is 7.6–29× faster (medians at
     10k–50k) at 1.02× stress.
   - It is not faster at small N with high D. The cost model (§2.4) explains
     both, and every speedup appears with its overhead at the same k.
3. **Step 3:**
   - Blocking keeps stress and adds 2–4.5× at high D, so blocked SGDMDS is the
     fastest way to within 1–2% of the full sweep.
   - SQuaD-MDS, credited as the source of blocking, is fastest at looser targets
     and doesn't reach 1–2% within its ladder.
   - Its numbers are a bound, chosen from 252 settings.

---

## 7. Open issues and decisions

1. **The full sweep's shuffle (largest).**
   - ~95% of a large full fit is the per-epoch `shuffle`, by elimination (§1.4).
   - It decides how SGDMDS compares with SMACOF at large N. It also makes
     low-D budgeted speedups ~10× larger than the pair saving.
   - CLAUDE.md §2 counts implementation bugs as bugs, so a reviewer could read
     this as an implementation cost rather than a property of SGD over all pairs.
   - Options:
     - (a) Keep the code and present the cost decomposition in the paper.
     - (b) Make the per-epoch shuffle cheaper, for example in place in Cython.
       This breaks the freeze and means rerunning every full-sweep fit: 0.4 h
       in O, 15.3 h in P and 1.0 h in Q at today's speed, plus the step 1
       comparison built on them.
   - Measure first. This takes seconds and needs ~4 GB at N=10k; run it on COMA
     to match the campaign, or on the laptop for a first look:

   ```bash
   cd sklearn/manifold/benchmarks
   python - <<'EOF'
   import time
   import numpy as np
   from threadpoolctl import threadpool_limits
   from sklearn.utils import shuffle
   from sklearn.manifold._sgd_mds_cython import run_sgd_epoch

   for n in (1_000, 2_000, 5_000, 10_000):
       rows, cols = np.triu_indices(n, k=1)
       pairs = np.column_stack((rows, cols)).astype(np.int32)
       targets = np.random.RandomState(0).rand(pairs.shape[0])
       weights = np.ones(pairs.shape[0])
       embedding = np.random.RandomState(1).standard_normal((n, 2))
       rng = np.random.RandomState(2)
       with threadpool_limits(1):
           t0 = time.perf_counter()
           p, t, w = shuffle(pairs, targets, weights, random_state=rng)
           t1 = time.perf_counter()
           run_sgd_epoch(embedding, t, w, p, 0.01)
           t2 = time.perf_counter()
       m = pairs.shape[0]
       print(f"N={n:>6,}  shuffle {1e9 * (t1 - t0) / m:6.1f} ns/pair  "
             f"kernel {1e9 * (t2 - t1) / m:6.1f} ns/pair")
   EOF
   ```

2. **Scope step 1's claim** with Q, whatever happens with (1): SGDMDS's speed
   advantage holds when SMACOF needs most of its default 300 iterations.
3. **coil20's timing:** add a footnote, or leave coil20 out of the time claims.
   Its stress results stand, and a rerun was ruled out on 2026-11-17.
4. **SQuaD-MDS:**
   - Report its quality as a bound, with the last rung's gain.
   - The open seed-split decision (spec §13) can close: the split changes stress
     by at most 0.002. Keep it in O for rigour.
5. **Time-to-target:** a single 1.01× target leaves SQuaD-MDS out entirely.
   Report the frontier at 1.01×, 1.02×, 1.05× and 1.10× (T14) instead.
6. **P's single full seed:** caveat feynman_iii_15_12's P ratios, given the 4.4%
   seed spread of its full arm at 10k.

---

## 8. Proposed figures

Not made yet. Each figure reads a CSV in `results_paper/analysis/` or the
histories through `load_histories`.

| # | Step | Figure | Source |
|---|---|---|---|
| F1 | 1 | Convergence curves: raw stress against time, SMACOF vs full SGDMDS, median and IQR band. Four O datasets: fashion_mnist (SMACOF capped), epileptic (largest N), svhn (SMACOF stuck at a worse minimum) and imdb (SMACOF converges early). Mark SMACOF's final stress and SGDMDS's crossing. | histories, `step1_ttt.csv` |
| F2 | 1 | Per-dataset scatter: SMACOF ÷ full time (log) against full ÷ SMACOF stress, with IQR whiskers and the 15% band shaded; marker shows whether SMACOF capped or converged. | `step1.csv` |
| F3 | 1, 2 | Runtime against N, log-log, three panels (D=100/300/500): SMACOF, full, budgeted and blocked at k=200, with slopes annotated. Shows SMACOF's crossover and budgeting's break-even. Replaces B's figure. | `q.csv`, `q_slopes.csv` |
| F4 | 2 | Speedup (log) against stress overhead, one point per (dataset, N, k), coloured by k, O and P side by side, with the 15% band. Pairs every speedup with its overhead. | `vs_full_{o,p}.csv` |
| F5 | 2 | Speedup at k=200 against N, log-log, coloured by D: P sources along their ladder, Q's three datasets, O as points, with the cost model's curves. | `vs_full_*.csv`, `cost.csv` |
| F6 | 2, 3 | Cost per pair, two panels: (a) the full sweep against N, with SMACOF per iteration and the kernel's 3.6 ns line; (b) budgeted and blocked against D, cache-resident vs memory-bound. Explains F3–F5 and the shuffle issue. | `cost.csv`, `cost_fits.csv` |
| F7 | 3 | Stress–time trade-off, small multiples (e.g. fashion_mnist, hatespeech, epileptic; cifar10_raw, mnist_digits, ag_news at 50k): full, budgeted and blocked ladders over k, SQuaD-MDS's ladder over `n_iter`, and SMACOF in O. | `tradeoff_{o,p}.csv` |
| F8 | 3 | Convergence curves at one large rung (e.g. fashion_mnist_full at 50k): scaled stress against time for full, budgeted and blocked at k=200, and SQuaD-MDS's best, with lines at 1.01× and 1.05×. | histories (P, estimated) |
| F9 | 3 | Dataset × target grid coloured by the arm that arrives first, or T14 as stacked bars. | `fastest_{o,p}.csv` |
| F10 | 3, appendix | SQuaD-MDS's ladder: stress ÷ full against `n_iter`, per dataset, showing where it flattens and where it doesn't. | `squad_ladder_{o,p}.csv` |

Paper mapping (spec §9): F1–F3 and T1 for step 1; F4–F6 for step 2; F7–F9
for step 3, with F10 in an appendix.

---

## 9. Reproduce

```bash
cd sklearn/manifold/benchmarks
python analyze_experiment_o_p_q.py               # writes results_paper/analysis/
python analyze_experiment_o_p_q.py --out DIR     # elsewhere
```

In Python, `import analyze_experiment_o_p_q as A`, then `A.run()` returns every
table as a DataFrame. `A.load_histories("p")` gives the per-fit histories as
arrays.
