# Per-dataset tables: fashion_mnist (O) and fashion_mnist_full at 50k (P)

Every algorithm setting run on one small and one large dataset, with its stress,
time and quality metrics. Both datasets hold the same kind of images: O uses a
3,000-image set, P a fixed 50,000-image subsample of the 70,000-image set.
Section 3 asks which k to use, across all 17 O datasets.

Generated on 2026-11-21 from `benchmarks/results_paper/` by
[benchmarks/dataset_tables.py](benchmarks/dataset_tables.py):

```bash
cd sklearn/manifold/benchmarks
python dataset_tables.py o fashion_mnist             # table 1, ~3 s
python dataset_tables.py p fashion_mnist_full 50000  # table 2, ~2 min (kNN at 50k)
python dataset_tables.py k                           # table 4
```

Any other dataset or N works the same way. Findings across all datasets are in
[EXPERIMENT_O_P_Q_RESULTS.md](EXPERIMENT_O_P_Q_RESULTS.md).

## How to read the tables

**The base is full SGDMDS** (all pairs, 30 epochs, random init) on the same
dataset and N. Every "÷ full" and "speedup" uses its median stress and median
time over its seeds: 10 seeds in O, and the single seed 42 in P at 50k.

| column | meaning |
|---|---|
| fits | Seeds summarised. SGDMDS and SMACOF use all seeds; SQuaD-MDS in O uses its reporting seeds 47–51. |
| pairs per epoch | Share of the N(N−1)/2 pairs updated per epoch: 2k/(N−1) for budgeted and blocked. |
| stress | Median scale-optimal stress, Σ_{i<j}(α·d_ij − δ_ij)². Absolute, so compare it only within one table. |
| ÷ full [IQR] | Median and 25th–75th percentile over seeds of each fit's stress ÷ full's median. Full's own row shows its seed spread. |
| stress-1 | Kruskal's normalised stress, sqrt(stress / Σ_{i<j} δ_ij²), from the median stress. 0 is a perfect fit; comparable across datasets. |
| α | Median optimal scale. About 1 for SMACOF and SGDMDS. SQuaD-MDS ignores scale, so its embedding must be multiplied by α before its stress is compared. |
| time [IQR] | Wall time on one thread, from input features to the embedding. It includes the distance matrix and any classical-MDS or PCA init. It excludes stress scoring, except for SMACOF, whose stopping rule needs it. |
| speedup | Full's median time ÷ this median time. ≈ means within 15% (comparable); "(slower)" means more than 15% slower. |
| to 1.01× / 1.05× full | Median time at which a fit first reaches 1.01× or 1.05× full's median final stress, interpolated between snapshots, with the number of seeds that reach it. "—" if half the seeds or fewer reach it. P's come from its estimated histories. |
| iterations | SMACOF: median iterations run, and how many seeds hit the 300 cap. SGDMDS: epochs. SQuaD-MDS: iterations. |
| kNN@10 | Share of each point's 10 nearest input neighbours that are also among its 10 nearest in the embedding. Seed 42 only, the one embedding saved. A local-structure measure that none of these methods optimises. |

**SQuaD-MDS rows:**

- **O:** the best of the 36 settings at each `n_iter` rung, chosen on seeds
  42–46 by median stress and reported on 47–51, plus the authors' default.
- **P:** all 12 settings (lr 550, PCA init, step cap 4, exaggeration off or on).
- ★ marks the best setting overall.

## 1. fashion_mnist (O): N=3,000, D=784

Base: full SGDMDS over 10 seeds, median stress 4.634e+08 in 8.74 s.

**Table 1. fashion_mnist (O), every setting**

| algorithm | setting | fits | pairs per epoch | stress | ÷ full [IQR] | stress-1 | α | time [IQR] | speedup | to 1.01× full | to 1.05× full | iterations | kNN@10 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SMACOF | classical init | 10 | — | 4.645e+08 | 1.002 [1.002–1.002] | 0.2566 | 1.000 | 6.53 s [6.52–6.54 s] | 1.34× | 2.66 s (10/10) | 2 s (10/10) | 67 (0/10 at cap) | 0.116 |
| SMACOF | random init | 10 | — | 5.086e+08 | 1.097 [1.089–1.121] | 0.2685 | 1.000 | 21.6 s [21.4–21.6 s] | 0.41× (slower) | — (0/10) | — (0/10) | 300 (10/10 at cap) | 0.115 |
| SGDMDS full | all pairs | 10 | 100% | 4.634e+08 | 1.000 [1.000–1.000] | 0.2563 | 1.000 | 8.74 s [8.69–8.8 s] | ≈1.00× | 4.92 s (10/10) | 3.44 s (10/10) | 30 epochs | 0.118 |
| SGDMDS budgeted | k=50 | 10 | 3.33% | 5.131e+08 | 1.107 [1.092–1.151] | 0.2697 | 1.000 | 1.61 s [1.61–1.61 s] | 5.43× | — (0/10) | — (0/10) | 30 epochs | 0.092 |
| SGDMDS budgeted | k=100 | 10 | 6.67% | 4.79e+08 | 1.034 [1.032–1.035] | 0.2605 | 1.000 | 3.21 s [3.21–3.21 s] | 2.72× | — (0/10) | 1.77 s (8/10) | 30 epochs | 0.099 |
| SGDMDS budgeted | k=200 | 10 | 13.3% | 4.673e+08 | 1.008 [1.007–1.009] | 0.2573 | 1.000 | 6.42 s [6.41–6.42 s] | 1.36× | 4.86 s (9/10) | 2.71 s (10/10) | 30 epochs | 0.112 |
| SGDMDS budgeted | k=400 | 10 | 26.7% | 4.649e+08 | 1.003 [1.003–1.003] | 0.2567 | 1.000 | 12.8 s [12.8–12.8 s] | 0.68× (slower) | 7.42 s (10/10) | 5.17 s (10/10) | 30 epochs | 0.115 |
| SGDMDS blocked | k=50 | 10 | 3.33% | 5.274e+08 | 1.138 [1.097–1.165] | 0.2734 | 1.000 | 0.69 s [0.69–0.69 s] | 12.71× | — (0/10) | — (0/10) | 30 epochs | 0.085 |
| SGDMDS blocked | k=100 | 10 | 6.67% | 4.851e+08 | 1.047 [1.031–1.057] | 0.2622 | 1.000 | 1.37 s [1.37–1.37 s] | 6.39× | — (0/10) | 1.02 s (6/10) | 30 epochs | 0.104 |
| SGDMDS blocked | k=200 | 10 | 13.3% | 4.683e+08 | 1.011 [1.008–1.012] | 0.2576 | 1.000 | 2.73 s [2.73–2.73 s] | 3.20× | — (4/10) | 1.11 s (10/10) | 30 epochs | 0.110 |
| SGDMDS blocked | k=400 | 10 | 26.7% | 4.648e+08 | 1.003 [1.002–1.003] | 0.2567 | 1.000 | 5.45 s [5.45–5.45 s] | 1.60× | 3.13 s (10/10) | 2.01 s (10/10) | 30 epochs | 0.115 |
| SQuaD-MDS | i100 lr550 exa pca cap1 | 5 | — | 5.449e+08 | 1.176 [1.175–1.178] | 0.2779 | 1.872 | 0.094 s [0.094–0.094 s] | 93.30× | — (0/5) | — (0/5) | 100 | 0.091 |
| SQuaD-MDS | i250 lr1500 pca cap1 | 5 | — | 5.34e+08 | 1.152 [1.149–1.154] | 0.2751 | 1.503 | 0.18 s [0.18–0.18 s] | 47.51× | — (0/5) | — (0/5) | 250 | 0.062 |
| SQuaD-MDS | i500 lr550 pca cap4 | 5 | — | 5.097e+08 | 1.100 [1.100–1.101] | 0.2688 | 1.755 | 0.33 s [0.33–0.33 s] | 26.18× | — (0/5) | — (0/5) | 500 | 0.086 |
| SQuaD-MDS | i1000 lr550 pca cap1 | 5 | — | 4.965e+08 | 1.071 [1.071–1.072] | 0.2653 | 1.656 | 0.63 s [0.63–0.63 s] | 13.83× | — (0/5) | — (0/5) | 1000 | 0.103 |
| SQuaD-MDS | i1000 lr550 exa pca capnone (authors' default) | 5 | — | 6.672e+08 | 1.440 [1.439–1.444] | 0.3075 | 1.681 | 0.59 s [0.59–0.59 s] | 14.85× | — (0/5) | — (0/5) | 1000 | 0.119 |
| SQuaD-MDS | i2000 lr550 pca cap4 | 5 | — | 4.908e+08 | 1.059 [1.058–1.059] | 0.2637 | 1.517 | 1.23 s [1.23–1.23 s] | 7.09× | — (0/5) | — (0/5) | 2000 | 0.118 |
| SQuaD-MDS | i4000 lr550 pca cap4 | 5 | — | 4.883e+08 | 1.054 [1.053–1.055] | 0.2631 | 1.354 | 2.43 s [2.43–2.43 s] | 3.60× | — (0/5) | — (0/5) | 4000 | 0.123 |
| SQuaD-MDS | ★ i8000 lr550 pca cap4 | 5 | — | 4.87e+08 | 1.051 [1.051–1.052] | 0.2627 | 1.183 | 4.82 s [4.82–4.82 s] | 1.81× | — (0/5) | — (0/5) | 8000 | 0.131 |

- **SMACOF:**
  - Random init ends at 1.097× full, uses all 300 iterations on every seed, and
    takes 2.5× full's time.
  - Classical init ends at 1.002× in 6.5 s. It reaches 1.01× full in 2.7 s,
    sooner than full SGDMDS itself (4.9 s).
- **Budgeted and blocked:**
  - k=200 stays within about 1% of full (1.008 and 1.011), at 1.36× and 3.20×.
  - k=100 costs 3–5% (1.034 and 1.047), at 2.7× and 6.4×.
  - k=50 costs 11–14%.
  - Budgeted at k=400 is slower than full; blocked at k=400 is still 1.6×
    faster.
- **SQuaD-MDS:**
  - Its best setting, at the top of its ladder, ends at 1.051× in 4.8 s. So it
    reaches neither 1.01× nor 1.05× full.
  - The authors' default ends at 1.44×.
- **kNN@10:** 0.06–0.13 for every method. SQuaD-MDS's best setting is highest
  (0.131), then full (0.118) and SMACOF (0.115–0.116); k=50 gives 0.085–0.092.

## 2. fashion_mnist_full (P): N=50,000, D=784

Base: full SGDMDS, seed 42 only, stress 1.286e+11 in 4,351 s. One full sweep at
50k needs ~120 GB and 73 minutes, so P ran one seed above 10k. Every ratio here
rests on that single fit. At 10k, this source's 3 full seeds differ by 0.01%.

**Table 2. fashion_mnist_full at N=50,000 (P), every setting**

| algorithm | setting | fits | pairs per epoch | stress | ÷ full [IQR] | stress-1 | α | time [IQR] | speedup | to 1.01× full | to 1.05× full | iterations | kNN@10 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SGDMDS full | all pairs | 1 | 100% | 1.286e+11 | 1.000 [1.000–1.000] | 0.2558 | 1.000 | 4,351 s [4,351–4,351 s] | ≈1.00× | 2,456 s (1/1) | 1,738 s (1/1) | 30 epochs | 0.018 |
| SGDMDS budgeted | k=100 | 5 | 0.4% | 1.372e+11 | 1.067 [1.050–1.067] | 0.2642 | 1.000 | 98 s [98–98.9 s] | 44.38× | — (0/5) | — (1/5) | 30 epochs | 0.013 |
| SGDMDS budgeted | k=200 | 5 | 0.8% | 1.301e+11 | 1.012 [1.011–1.013] | 0.2573 | 1.000 | 197 s [195–198 s] | 22.05× | — (0/5) | 86.4 s (5/5) | 30 epochs | 0.016 |
| SGDMDS budgeted | k=400 | 5 | 1.6% | 1.29e+11 | 1.004 [1.003–1.004] | 0.2562 | 1.000 | 394 s [394–394 s] | 11.04× | 230 s (5/5) | 158 s (5/5) | 30 epochs | 0.017 |
| SGDMDS blocked | k=100 | 5 | 0.4% | 1.358e+11 | 1.056 [1.053–1.065] | 0.2629 | 1.000 | 25.2 s [25.2–25.2 s] | 173× | — (0/5) | — (0/5) | 30 epochs | 0.013 |
| SGDMDS blocked | k=200 | 5 | 0.8% | 1.301e+11 | 1.012 [1.011–1.013] | 0.2573 | 1.000 | 50.4 s [50.4–50.4 s] | 86.29× | — (0/5) | 20.7 s (5/5) | 30 epochs | 0.016 |
| SGDMDS blocked | k=400 | 5 | 1.6% | 1.29e+11 | 1.003 [1.003–1.003] | 0.2562 | 1.000 | 101 s [101–101 s] | 43.21× | 57.9 s (5/5) | 37.2 s (5/5) | 30 epochs | 0.017 |
| SQuaD-MDS | i100 lr550 pca cap4 | 5 | — | 1.769e+11 | 1.376 [1.376–1.377] | 0.3000 | 1.856 | 2.12 s [2.12–2.13 s] | 2,050× | — (0/5) | — (0/5) | 100 | 0.007 |
| SQuaD-MDS | i100 lr550 exa pca cap4 | 5 | — | 1.607e+11 | 1.250 [1.249–1.250] | 0.2859 | 1.869 | 2.13 s [2.11–2.13 s] | 2,045× | — (0/5) | — (0/5) | 100 | 0.011 |
| SQuaD-MDS | i250 lr550 pca cap4 | 5 | — | 1.572e+11 | 1.222 [1.222–1.223] | 0.2828 | 1.818 | 4.58 s [4.58–4.59 s] | 950× | — (0/5) | — (0/5) | 250 | 0.008 |
| SQuaD-MDS | i250 lr550 exa pca cap4 | 5 | — | 1.611e+11 | 1.253 [1.253–1.253] | 0.2863 | 1.831 | 4.57 s [4.57–4.57 s] | 952× | — (0/5) | — (0/5) | 250 | 0.013 |
| SQuaD-MDS | i500 lr550 pca cap4 | 5 | — | 1.473e+11 | 1.146 [1.145–1.147] | 0.2738 | 1.757 | 8.7 s [8.7–8.77 s] | 500× | — (0/5) | — (0/5) | 500 | 0.010 |
| SQuaD-MDS | i500 lr550 exa pca cap4 | 5 | — | 1.741e+11 | 1.354 [1.353–1.354] | 0.2976 | 1.773 | 8.71 s [8.69–8.74 s] | 500× | — (0/5) | — (0/5) | 500 | 0.016 |
| SQuaD-MDS | i1000 lr550 pca cap4 | 5 | — | 1.425e+11 | 1.108 [1.108–1.108] | 0.2693 | 1.657 | 17 s [16.9–17.1 s] | 256× | — (0/5) | — (0/5) | 1000 | 0.013 |
| SQuaD-MDS | i1000 lr550 exa pca cap4 (authors' default) | 5 | — | 1.871e+11 | 1.455 [1.455–1.456] | 0.3085 | 1.684 | 16.9 s [16.9–17 s] | 257× | — (0/5) | — (0/5) | 1000 | 0.017 |
| SQuaD-MDS | i2000 lr550 pca cap4 | 5 | — | 1.398e+11 | 1.087 [1.087–1.087] | 0.2667 | 1.520 | 33.6 s [33.5–33.9 s] | 130× | — (0/5) | — (0/5) | 2000 | 0.016 |
| SQuaD-MDS | i2000 lr550 exa pca cap4 | 5 | — | 1.969e+11 | 1.532 [1.532–1.532] | 0.3166 | 1.559 | 33.6 s [33.5–33.9 s] | 130× | — (0/5) | — (0/5) | 2000 | 0.018 |
| SQuaD-MDS | ★ i4000 lr550 pca cap4 | 5 | — | 1.382e+11 | 1.074 [1.074–1.075] | 0.2651 | 1.357 | 67 s [66.9–68.5 s] | 64.95× | — (0/5) | — (0/5) | 4000 | 0.018 |
| SQuaD-MDS | i4000 lr550 exa pca cap4 | 5 | — | 2.04e+11 | 1.587 [1.587–1.587] | 0.3222 | 1.403 | 67.5 s [66.6–67.7 s] | 64.50× | — (0/5) | — (0/5) | 4000 | 0.019 |

- **Budgeted and blocked:**
  - k=200 ends at 1.012× full, at 22× (budgeted) and 86× (blocked).
  - k=400 ends at 1.003–1.004×, at 11× and 43×.
  - Blocked at k=200 reaches 1.05× full in 21 s; full needs 1,738 s.
- **SQuaD-MDS:**
  - Its best setting ends at 1.074× full in 67 s, again at the top of its ladder.
  - Exaggeration helps only at 100 iterations. From 250 up it makes stress
    worse, and its stress rises with the iterations: 1.25× at 250, 1.59× at
    4,000.
  - The authors' default ends at 1.455×.
- **kNN@10:** 0.007–0.019 for every method. At N=50,000 a 2-D stress embedding
  keeps few exact 10-nearest neighbours, so this measure separates little here.

## 3. Which k? Quality depends on k, not on N

**Table 3. fashion: budgeted and blocked ÷ full (stress · speedup) at each N**

| arm | k | 3,000 (O) | 10k | 20k | 35k | 50k |
|---|--:|--:|--:|--:|--:|--:|
| budgeted | 50 | 1.107 · 5.4× | — | — | — | — |
| budgeted | 100 | 1.034 · 2.7× | 1.058 · 9.4× | 1.060 · 18.2× | 1.061 · 30.6× | 1.067 · 44.4× |
| budgeted | 200 | 1.008 · 1.4× | 1.013 · 4.8× | 1.011 · 8.8× | 1.011 · 15.2× | 1.012 · 22.1× |
| budgeted | 400 | 1.003 · 0.7× | 1.003 · 2.4× | 1.003 · 4.5× | 1.003 · 7.7× | 1.004 · 11.0× |
| blocked | 50 | 1.138 · 12.7× | — | — | — | — |
| blocked | 100 | 1.047 · 6.4× | 1.034 · 33.7× | 1.039 · 68.8× | 1.055 · 121× | 1.056 · 173× |
| blocked | 200 | 1.011 · 3.2× | 1.009 · 16.9× | 1.009 · 34.5× | 1.011 · 60.6× | 1.012 · 86.3× |
| blocked | 400 | 1.003 · 1.6× | 1.003 · 8.4× | 1.003 · 17.2× | 1.003 · 30.2× | 1.003 · 43.2× |

From `analysis/vs_full_{o,p}.csv`. P did not run k=50.

- **The overhead at a given k barely moves with N.** At k=200 it is 1.008–1.013
  from N=3,000 to 50,000, and at k=400 it is 1.003–1.004.
- **The speedup grows with N.** k counts updates per point, so a budgeted epoch
  costs O(kN) while a full epoch costs O(N²).
- **So k=200 isn't "too big" for small datasets in quality terms.** A given k
  costs no more stress at N=3,000 than at N=50,000 (at k=100 the overhead even
  grows with N, from 1.034 to 1.067 budgeted). Small N only saves less time.
- **What makes small datasets hard is cost per pair.** A budgeted pair computes
  its distance from X, over all D dimensions. A full-sweep pair reads it from
  the precomputed matrix. So at small N, budgeting pays off only at low D.

**Table 4. O: fastest k within each stress tolerance of full**

| dataset | N | D | k=50 pairs per epoch | k=200 pairs per epoch | budget break-even k | blocked break-even k | ≤1.01× budget | ≤1.01× blocked | ≤1.02× budget | ≤1.02× blocked | ≤1.05× budget | ≤1.05× blocked |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| seismic | 646 | 24 | 15.5% | 62% | ≈408 | ≈442 | k=200: 2.04× | k=200: 2.21× | k=200: 2.04× | k=200: 2.21× | k=100: 4.02× | k=100: 4.37× |
| svhn | 732 | 1024 | 13.7% | 54.7% | ≈20 | ≈49 | none | none | none | none | k=200: 0.10× (slower) | k=200: 0.24× (slower) |
| har | 735 | 561 | 13.6% | 54.5% | ≈37 | ≈84 | none | none | none | none | k=200: 0.19× (slower) | k=200: 0.42× (slower) |
| sms | 835 | 500 | 12% | 48% | ≈51 | ≈112 | k=50: ≈1.00× | k=50: 2.17× | k=50: ≈1.00× | k=50: 2.17× | k=50: ≈1.00× | k=50: 2.17× |
| fmd | 997 | 1536 | 10% | 40.2% | ≈22 | ≈53 | none | k=400: 0.13× (slower) | k=200: 0.11× (slower) | k=200: 0.27× (slower) | k=50: 0.43× (slower) | k=50: ≈1.06× |
| cnae9 | 1080 | 856 | 9.27% | 37.1% | ≈40 | ≈94 | k=50: 0.79× (slower) | k=50: 1.84× | k=50: 0.79× (slower) | k=50: 1.84× | k=50: 0.79× (slower) | k=50: 1.84× |
| coil20 | 1440 | 400 | 6.95% | 27.8% | ≈202 | ≈438 | k=400: 0.51× (slower) | k=400: ≈1.10× | k=200: ≈1.01× | k=200: 2.19× | k=100: 2.02× | k=100: 4.32× |
| secom | 1567 | 590 | 6.39% | 25.5% | ≈102 | ≈232 | none | none | k=400: 0.26× (slower) | k=400: 0.58× (slower) | k=400: 0.26× (slower) | k=200: 1.16× |
| bank | 2059 | 63 | 4.86% | 19.4% | ≈1,372 | ≈1,850 | none | none | k=400: 3.51× | k=400: 4.75× | k=200: 6.86× | k=200: 9.25× |
| sentiment | 2748 | 200 | 3.64% | 14.6% | ≈922 | ≈1,751 | k=100: 9.12× | k=100: 17.28× | k=50: 17.90× | k=50: 33.69× | k=50: 17.90× | k=50: 33.69× |
| fashion_mnist | 3000 | 784 | 3.33% | 13.3% | ≈272 | ≈641 | k=200: 1.36× | k=400: 1.60× | k=200: 1.36× | k=200: 3.20× | k=100: 2.72× | k=100: 6.39× |
| hiva | 3076 | 1617 | 3.25% | 13% | ≈115 | ≈356 | none | none | none | none | none | k=400: ≈0.89× |
| hatespeech | 3221 | 100 | 3.11% | 12.4% | ≈2,398 | ≈3,854 | none | none | none | none | k=50: 45.46× | k=50: 72.33× |
| imdb | 3250 | 700 | 3.08% | 12.3% | ≈380 | ≈889 | k=50: 7.58× | k=50: 17.63× | k=50: 7.58× | k=50: 17.63× | k=50: 7.58× | k=50: 17.63× |
| cifar10 | 3250 | 1024 | 3.08% | 12.3% | ≈270 | ≈656 | k=200: 1.35× | k=200: 3.28× | k=200: 1.35× | k=200: 3.28× | k=100: 2.70× | k=100: 6.54× |
| spambase | 4601 | 57 | 2.17% | 8.7% | ≈6,580 | ≈9,606 | none | none | none | none | k=400: 16.66× | k=400: 24.33× |
| epileptic | 5750 | 178 | 1.74% | 6.96% | ≈4,065 | ≈6,984 | none | none | none | none | none | none |

- **Columns:** "≤1.02× budget" is the fastest budgeted k whose median stress is
  within 2% of full's, with its speedup. "none" means no k from 50 to 400 gets
  that close.
- **Break-even k:** the k at which budgeted or blocked would take as long as
  full, extrapolated from k=200. Time is close to proportional to k: k=50 runs
  3.5–4.0× faster than k=200.

What the table says about lowering k:

- **N≤1,080 with D≥500 (svhn, har, fmd, cnae9, sms):** budgeted breaks even only
  below k≈20–51, and blocked below k≈49–112.
  - On svhn, har and fmd, stress at k=50 is already 3–13% above full, so no k
    gives good stress faster.
  - sms and cnae9 are the exception. Their overhead stays at 1.008 or less even
    at k=50, so blocked at k=50 is 2.2× and 1.8× faster within 1%.
- **seismic (D=24):** k=200 gives 2.0× within 1%; k=100 gives 4.0× within 3%.
- **N=1,400–3,250 at high D (coil20, fashion_mnist, cifar10, secom, hiva):**
  - fashion_mnist and cifar10: k=200 is within about 1% (1.008–1.011) at
    1.35× budgeted and 3.2–3.3× blocked. k=100 is within 5% at 2.7× and
    6.4–6.5×.
  - coil20 behaves the same way, but its full-sweep time is inflated (results
    doc §0), so its speedups are overstated.
  - secom and hiva don't reach 1% at any k. Within 5%, only blocked secom is
    faster, at 1.16×.
- **D≤178 (bank, hatespeech, spambase, epileptic):** 6.9–72× within 5%, except
  epileptic, which stays above 1.05× even at k=400. hatespeech and spambase
  stay above 1.02× at k=400.
  - hatespeech and epileptic are two of the three datasets whose overhead stops
    shrinking with k; hiva is the third.
  - spambase starts high (1.24 at k=50) and is still at 1.03 at k=400.
- **sms, cnae9, sentiment and imdb (almost no 2-D structure):** stress barely
  depends on k; k=50 is within 1.1%. Blocked at k=50 is the fastest choice on
  all four; budgeted is faster only on sentiment and imdb.
