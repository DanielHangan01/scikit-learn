# Experiment R: SGDMDS against Enhanced Force-Scheme

Per-dataset comparison of our SGDMDS with Enhanced Force-Scheme (EFS; Ros,
Arleo & Paulovich, Computers & Graphics 135, 2026), on the inputs and seeds of
Experiments O and P. Built by
[benchmarks/analyze_experiment_r.py](benchmarks/analyze_experiment_r.py),
which writes these tables to `results_paper/analysis_r/comparison.md`.

## Setup

- **EFS** is the authors' code (submodule `benchmarks/external/enhanced-force-scheme`,
  commit `3c84327`) with their defaults: up to 200 iterations, learning rate 1
  decaying by 0.9 per iteration, early stop when the mean absolute residual stops
  improving. It stopped after 80–130 iterations; no fit reached 200.
  - **Full EFS** updates every pair once per iteration, from a precomputed
    distance matrix.
  - **s-EFS** (`sample=True`, the authors' "EFS_sample") updates √N rounds of N/2
    pairs per iteration, about √N/2 updates per point, and computes distances on
    the fly.
- **Ours:** full SGDMDS is the golden configuration (30 epochs, every pair per
  epoch). Blocked SGDMDS draws k·N pairs per epoch, k=200, in blocks of 16 points.
- **Seeds:** 10 per dataset on the small suite. On the large suite, 5 for blocked
  SGDMDS and s-EFS; full SGDMDS and full EFS ran on 3 seeds, at N=10,000 only.
- **Threads:** our methods always run on one thread. EFS ran on 1 thread ("1t")
  and on 8 ("8t"). Its result doesn't depend on the thread count (the 1t and 8t
  embeddings are bit-identical), so each table has one EFS stress column.
- **Machine:** one COMA node type (AMD EPYC 9555), the same as Experiments O and P.

## Columns

| column | meaning |
|---|---|
| dataset, N, D | Dataset, number of points and number of features. |
| full SGDMDS stress, blocked stress | Our method's mean stress over seeds: raw stress Σ_{i<j}(d_ij − δ_ij)² at the stress-optimal scale. Absolute, so compare it only within a row. |
| EFS stress, s-EFS stress | The same for EFS. Its optimal scale is 1.000 (it optimises raw stress too). |
| ours / EFS stress | Our stress divided by EFS's. **Below 1, ours is lower**; above 1, EFS's is. |
| full SGDMDS time, blocked time | Our mean wall time on one thread, from input features to embedding. It includes the distance matrix where the method builds one, and excludes stress scoring. |
| EFS 1t time, EFS 8t time | EFS's mean wall time on 1 and on 8 threads, measured the same way. numba's compilation happens before the timed region. |
| ours / EFS 8t time | Our single-thread time divided by EFS's 8-thread time. **Below 1, ours is faster**, even against EFS on 8 threads. |
| ours / EFS 8t time to the higher stress | A comparison at a stress level both methods reach. Fits are paired by seed (both ran the same seeds), and each pair's level is the higher of its two final stresses. The time each fit first reaches it is read from its snapshots (after every epoch or iteration), interpolated, and the cell is the median over seeds of our time ÷ EFS's 8-thread time. **Below 1, ours gets there first.** On the large suite both histories are estimated from the same 10⁷ sampled pairs (within 0.5% of the exact stress). |

## 1. Full SGDMDS vs full EFS

**Table 1a. Small suite (Experiment O's 17 datasets)**

| dataset | N | D | full SGDMDS stress | EFS stress | full SGDMDS / EFS stress | full SGDMDS time | EFS 1t time | EFS 8t time | full SGDMDS / EFS 8t time | full SGDMDS / EFS 8t time to the higher stress |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| seismic | 646 | 24 | 4.053e+05 | 4.026e+05 | 1.007 | 0.17 s | 3.45 s | 0.81 s | 0.21 | 0.60 |
| svhn | 732 | 1,024 | 1.92e+07 | 1.865e+07 | 1.030 | 0.21 s | 5.20 s | 1.22 s | 0.17 | 0.70 |
| har | 735 | 561 | 1.279e+07 | 1.256e+07 | 1.018 | 0.21 s | 3.75 s | 0.83 s | 0.25 | 0.72 |
| sms | 835 | 500 | 4.106e+07 | 4.095e+07 | 1.003 | 0.29 s | 5.00 s | 1.11 s | 0.26 | 0.56 |
| fmd | 997 | 1,536 | 1.681e+08 | 1.674e+08 | 1.004 | 0.46 s | 7.36 s | 1.60 s | 0.29 | 0.62 |
| cnae9 | 1,080 | 856 | 7.286e+07 | 7.258e+07 | 1.004 | 0.50 s | 11.6 s | 2.47 s | 0.20 | 0.57 |
| coil20 | 1,440 | 400 | 7.131e+07 | 7.105e+07 | 1.004 | 1.73 s | 20.1 s | 4.06 s | 0.43 | 1.27 |
| secom | 1,567 | 590 | 1.132e+08 | 1.124e+08 | 1.007 | 1.28 s | 19.1 s | 3.77 s | 0.34 | 0.91 |
| bank | 2,059 | 63 | 2.809e+07 | 2.804e+07 | 1.002 | 2.69 s | 33.2 s | 5.81 s | 0.46 | 1.07 |
| sentiment | 2,748 | 200 | 1.899e+08 | 1.875e+08 | 1.012 | 6.90 s | 72.4 s | 12.6 s | 0.55 | 1.77 |
| fashion_mnist | 3,000 | 784 | 4.635e+08 | 4.633e+08 | 1.000 | 8.75 s | 93.3 s | 16.3 s | 0.54 | 1.17 |
| hiva | 3,076 | 1,617 | 1.706e+09 | 1.7e+09 | 1.003 | 10.2 s | 107 s | 19.4 s | 0.53 | 1.36 |
| hatespeech | 3,221 | 100 | 1.244e+08 | 1.229e+08 | 1.013 | 11.0 s | 98.6 s | 16.1 s | 0.68 | 2.03 |
| cifar10 | 3,250 | 1,024 | 7.801e+08 | 7.799e+08 | 1.000 | 12.7 s | 122 s | 20.6 s | 0.62 | 1.32 |
| imdb | 3,250 | 700 | 1.2e+09 | 1.199e+09 | 1.001 | 11.9 s | 125 s | 20.7 s | 0.57 | 1.50 |
| spambase | 4,601 | 57 | 6.785e+07 | 6.778e+07 | 1.001 | 29.1 s | 258 s | 38.7 s | 0.75 | 1.53 |
| epileptic | 5,750 | 178 | 2.025e+08 | 2.02e+08 | 1.002 | 55.1 s | 449 s | 68.3 s | 0.81 | 1.84 |

**Table 1b. Large suite at N=10,000**

| dataset | N | D | full SGDMDS stress | EFS stress | full SGDMDS / EFS stress | full SGDMDS time | EFS 1t time | EFS 8t time | full SGDMDS / EFS 8t time | full SGDMDS / EFS 8t time to the higher stress |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ag_news | 10,000 | 384 | 6.137e+09 | 6.13e+09 | 1.001 | 168 s | 1,655 s | 426 s | 0.39 | 1.05 |
| california_housing | 10,000 | 8 | 2.051e+07 | 2.05e+07 | 1.000 | 176 s | 1,248 s | 186 s | 0.94 | 1.27 |
| cifar10_raw | 10,000 | 3,072 | 2.397e+10 | 2.397e+10 | 1.000 | 172 s | 1,725 s | 303 s | 0.57 | 1.05 |
| fashion_mnist_full | 10,000 | 784 | 5.066e+09 | 5.065e+09 | 1.000 | 166 s | 1,651 s | 253 s | 0.66 | 1.24 |
| feynman_iii_15_12 | 10,000 | 3 | 1.694e+07 | 1.663e+07 | 1.018 | 165 s | 1,250 s | 184 s | 0.90 | 2.20 |
| mnist_digits | 10,000 | 784 | 5.721e+09 | 5.72e+09 | 1.000 | 167 s | 1,619 s | 253 s | 0.66 | 1.23 |
| patchcamelyon | 10,000 | 512 | 4.308e+09 | 4.307e+09 | 1.000 | 166 s | 1,609 s | 247 s | 0.67 | 1.29 |
| tiny_imagenet | 10,000 | 512 | 7.629e+09 | 7.621e+09 | 1.001 | 169 s | 1,674 s | 253 s | 0.67 | 1.56 |

- EFS ends with slightly lower stress everywhere: 0–1.3% on most datasets, at
  most 3% (svhn; 1.8% on har and feynman_iii_15_12). It runs 80–130 iterations to our 30 epochs.
- **Total time:** our single-thread full sweep finishes before EFS on 8 threads
  on every dataset, in 0.17–0.81 of its time on the small suite and 0.39–0.94 at
  10k. On 1 thread EFS takes 7.6–25× our time.
- **Time to a shared stress level:** our single-thread full sweep gets there
  first on the six smallest datasets (N ≤ 1,080) and secom (0.56–0.91). EFS on 8
  threads gets there first on the other 10 small datasets (1.07–2.03) and on all 8
  sources at 10k (1.05–2.20). On 1 thread EFS is far slower: a median 5.0× our
  time to reach our final stress on the small suite, and 4.4× at 10k.

## 2. Blocked SGDMDS (k=200) vs s-EFS

**Table 2a. Small suite**

| dataset | N | D | blocked stress | s-EFS stress | blocked / s-EFS stress | blocked time | s-EFS 1t time | s-EFS 8t time | blocked / s-EFS 8t time | blocked / s-EFS 8t time to the higher stress |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| seismic | 646 | 24 | 4.137e+05 | 5.029e+05 | 0.823 | 0.08 s | 0.05 s | 0.01 s | 5.13 | 0.63 |
| svhn | 732 | 1,024 | 1.977e+07 | 2.179e+07 | 0.907 | 0.86 s | 0.22 s | 0.04 s | 22.32 | 5.30 |
| har | 735 | 561 | 1.337e+07 | 1.469e+07 | 0.910 | 0.50 s | 0.15 s | 0.03 s | 17.74 | 4.14 |
| sms | 835 | 500 | 4.117e+07 | 4.185e+07 | 0.984 | 0.51 s | 0.14 s | 0.03 s | 17.82 | 8.81 |
| fmd | 997 | 1,536 | 1.705e+08 | 1.764e+08 | 0.967 | 1.72 s | 0.50 s | 0.08 s | 22.70 | 9.17 |
| cnae9 | 1,080 | 856 | 7.32e+07 | 7.42e+07 | 0.986 | 1.07 s | 0.32 s | 0.06 s | 18.70 | 10.10 |
| coil20 | 1,440 | 400 | 7.224e+07 | 7.833e+07 | 0.922 | 0.79 s | 0.37 s | 0.07 s | 10.94 | 2.93 |
| secom | 1,567 | 590 | 1.166e+08 | 1.246e+08 | 0.936 | 1.10 s | 0.49 s | 0.08 s | 13.07 | 3.86 |
| bank | 2,059 | 63 | 2.889e+07 | 3.295e+07 | 0.877 | 0.29 s | 0.39 s | 0.08 s | 3.42 | 0.63 |
| sentiment | 2,748 | 200 | 1.912e+08 | 1.922e+08 | 0.995 | 0.77 s | 0.79 s | 0.14 s | 5.48 | 3.16 |
| fashion_mnist | 3,000 | 784 | 4.687e+08 | 5.061e+08 | 0.926 | 2.73 s | 1.95 s | 0.29 s | 9.43 | 2.82 |
| hiva | 3,076 | 1,617 | 1.839e+09 | 1.877e+09 | 0.980 | 5.77 s | 4.31 s | 0.58 s | 10.02 | 4.95 |
| hatespeech | 3,221 | 100 | 1.288e+08 | 1.301e+08 | 0.990 | 0.57 s | 0.79 s | 0.15 s | 3.78 | 2.13 |
| cifar10 | 3,250 | 1,024 | 7.872e+08 | 8.614e+08 | 0.914 | 3.87 s | 2.75 s | 0.41 s | 9.54 | 2.50 |
| imdb | 3,250 | 700 | 1.203e+09 | 1.211e+09 | 0.994 | 2.67 s | 1.79 s | 0.27 s | 10.02 | 5.71 |
| spambase | 4,601 | 57 | 7.142e+07 | 7.988e+07 | 0.894 | 0.60 s | 1.41 s | 0.27 s | 2.22 | 0.54 |
| epileptic | 5,750 | 178 | 2.207e+08 | 2.269e+08 | 0.972 | 1.57 s | 2.41 s | 0.40 s | 3.95 | 1.80 |

**Table 2b. Large suite (Experiment P's 8 sources, N=10,000–50,000)**

| dataset | N | D | blocked stress | s-EFS stress | blocked / s-EFS stress | blocked time | s-EFS 1t time | s-EFS 8t time | blocked / s-EFS 8t time | blocked / s-EFS 8t time to the higher stress |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ag_news | 10,000 | 384 | 6.616e+09 | 6.76e+09 | 0.979 | 4.91 s | 7.84 s | 1.91 s | 2.57 | 1.06 |
| california_housing | 10,000 | 8 | 2.09e+07 | 2.109e+07 | 0.991 | 0.77 s | 4.30 s | 0.76 s | 1.01 | 0.59 |
| cifar10_raw | 10,000 | 3,072 | 2.418e+10 | 2.484e+10 | 0.973 | 35.8 s | 55.2 s | 12.8 s | 2.80 | 1.50 |
| fashion_mnist_full | 10,000 | 784 | 5.112e+09 | 5.175e+09 | 0.988 | 9.86 s | 21.4 s | 3.02 s | 3.26 | 1.83 |
| feynman_iii_15_12 | 10,000 | 3 | 1.688e+07 | 1.698e+07 | 0.994 | 0.74 s | 4.45 s | 0.76 s | 0.98 | 0.73 |
| mnist_digits | 10,000 | 784 | 5.977e+09 | 6.767e+09 | 0.883 | 9.89 s | 21.1 s | 2.98 s | 3.32 | 0.77 |
| patchcamelyon | 10,000 | 512 | 4.375e+09 | 4.592e+09 | 0.953 | 6.64 s | 16.9 s | 2.26 s | 2.93 | 1.13 |
| tiny_imagenet | 10,000 | 512 | 8.214e+09 | 8.406e+09 | 0.977 | 6.68 s | 15.1 s | 2.04 s | 3.28 | 1.55 |
| ag_news | 20,000 | 384 | 2.657e+10 | 2.698e+10 | 0.985 | 11.0 s | 46.1 s | 8.01 s | 1.37 | 0.72 |
| cifar10_raw | 20,000 | 3,072 | 9.661e+10 | 9.684e+10 | 0.998 | 71.5 s | 163 s | 40.6 s | 1.76 | 1.30 |
| fashion_mnist_full | 20,000 | 784 | 2.058e+10 | 2.068e+10 | 0.995 | 20.0 s | 66.1 s | 10.9 s | 1.83 | 1.23 |
| feynman_iii_15_12 | 20,000 | 3 | 6.727e+07 | 6.769e+07 | 0.994 | 1.47 s | 14.1 s | 2.33 s | 0.63 | 0.55 |
| mnist_digits | 20,000 | 784 | 2.425e+10 | 2.637e+10 | 0.920 | 20.1 s | 76.8 s | 12.6 s | 1.60 | 0.49 |
| patchcamelyon | 20,000 | 512 | 1.741e+10 | 1.764e+10 | 0.987 | 13.9 s | 57.0 s | 8.59 s | 1.62 | 0.90 |
| tiny_imagenet | 20,000 | 512 | 3.315e+10 | 3.36e+10 | 0.987 | 14.0 s | 53.6 s | 8.05 s | 1.73 | 0.94 |
| california_housing | 20,640 | 8 | 9.106e+07 | 9.025e+07 | 1.009 | 1.58 s | 14.3 s | 2.38 s | 0.66 | 0.81 |
| ag_news | 35,000 | 384 | 8.196e+10 | 8.248e+10 | 0.994 | 19.5 s | 114 s | 19.4 s | 1.00 | 0.58 |
| cifar10_raw | 35,000 | 3,072 | 2.962e+11 | 2.962e+11 | 1.000 | 124 s | 410 s | 103 s | 1.21 | 1.19 |
| fashion_mnist_full | 35,000 | 784 | 6.347e+10 | 6.326e+10 | 1.003 | 35.2 s | 167 s | 28.0 s | 1.26 | 1.61 |
| feynman_iii_15_12 | 35,000 | 3 | 2.071e+08 | 2.08e+08 | 0.996 | 2.58 s | 34.6 s | 5.59 s | 0.46 | 0.25 |
| mnist_digits | 35,000 | 784 | 7.492e+10 | 7.852e+10 | 0.954 | 35.2 s | 199 s | 33.7 s | 1.04 | 0.44 |
| patchcamelyon | 35,000 | 512 | 5.357e+10 | 5.363e+10 | 0.999 | 24.7 s | 151 s | 24.3 s | 1.02 | 0.94 |
| tiny_imagenet | 35,000 | 512 | 1.018e+11 | 1.028e+11 | 0.990 | 24.4 s | 140 s | 22.8 s | 1.07 | 0.61 |
| ag_news | 50,000 | 384 | 1.672e+11 | 1.68e+11 | 0.995 | 27.9 s | 210 s | 35.5 s | 0.79 | 0.49 |
| cifar10_raw | 50,000 | 3,072 | 6.064e+11 | 6.033e+11 | 1.005 | 178 s | 806 s | 193 s | 0.92 | 1.34 |
| fashion_mnist_full | 50,000 | 784 | 1.301e+11 | 1.292e+11 | 1.007 | 50.4 s | 348 s | 59.3 s | 0.85 | 1.35 |
| feynman_iii_15_12 | 50,000 | 3 | 4.239e+08 | 4.253e+08 | 0.997 | 3.68 s | 57.2 s | 9.19 s | 0.40 | 0.25 |
| mnist_digits | 50,000 | 784 | 1.525e+11 | 1.595e+11 | 0.956 | 49.7 s | 362 s | 63.1 s | 0.79 | 0.31 |
| patchcamelyon | 50,000 | 512 | 1.091e+11 | 1.085e+11 | 1.005 | 35.4 s | 263 s | 43.6 s | 0.81 | 1.26 |
| tiny_imagenet | 50,000 | 512 | 2.085e+11 | 2.095e+11 | 0.995 | 37.0 s | 257 s | 42.3 s | 0.87 | 0.70 |

- **Stress:** blocked SGDMDS is lower on all 17 small datasets (by 0.5–18%) and on
  25 of 30 large rungs. The margin shrinks with N, as s-EFS's √N/2 updates per
  point grow with it: 0.6–12% at 10k, mostly within ±0.7% at 50k (mnist_digits
  still 4.4%). Where s-EFS is lower, it is by at most 0.9%.
- **Time, thread for thread:** blocked SGDMDS is faster than s-EFS on 1 thread at
  every large rung (1.5–16×).
- **Time, against s-EFS on 8 threads:** s-EFS is faster on the small suite (our
  time is 2.2–23× its time) and at 10k–20k (1.4–3.3× on the high-D sources). It is
  about even at 35k (1.0–1.3) and slower at 50k on every source (0.40–0.92). On
  the low-D sources (feynman_iii_15_12, california_housing) it is slower from 20k
  on. The trend follows the work per iteration, which grows as N^1.5 for s-EFS and
  as k·N for blocked SGDMDS.
- **Time to a shared stress level:** on the small suite, s-EFS on 8 threads gets
  there first on 14 of 17 datasets (1.8–10×); blocked SGDMDS does on seismic, bank
  and spambase (0.54–0.63). On the large suite blocked SGDMDS gets there first on
  18 of 30 rungs, more often as N grows: 3 of 8 at 10k, 6 of 8 at 20k, 5 of 7 at
  35k and 4 of 7 at 50k. Always on feynman_iii_15_12 and mnist_digits (0.25–0.77),
  never on cifar10_raw and fashion_mnist_full (1.19–1.83).
