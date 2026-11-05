# Experiment N — results: point-blocked sampling

**Status:** complete (j18 2026-09-28, k8 2026-09-30). Design and protocol:
[EXPERIMENT_N_SPEC.md](EXPERIMENT_N_SPEC.md). The full report with
interactive charts is at <https://claude.ai/artifact/JkrYEGWyVSJtydnd798216>
(private; share it from the page).

Data: `benchmarks/results/experiment_n_{j18,k8}/` (8,925 + 720 fits) and
`benchmarks/results/bench_blocked_kernel/results.csv`. All timings are from
one machine (Intel Core Ultra 7 255HX), single-threaded.

## 0. Integrity

- **Original kernel reproduces J and K-optimized** on all 459 comparable fits,
  with relative error ≤ 8.6e-12. bank and hiva have no J reference at full N.
- **Blocked kernel at m=2** is bit-identical to the original kernel in 568 of
  568 fits.
- **Divergences:** 70, all in `m16-random-sum`.

## 1. Kernel cost

N = 50,000, ns per pair update:

| | D=3 | 8 | 64 | 384 | 784 | 3072 |
|---|--:|--:|--:|--:|--:|--:|
| original kernel | 50.7 | 51.9 | 120.2 | 502 | 748 | 2456 |
| blocked m=2 (GIL fix only) | 16.3 | 19.5 | 74.4 | 397 | 691 | 2378 |
| m=4 fused | 11.0 | 12.5 | 35.3 | 135 | 272 | 867 |
| m=8 | 10.8 | 11.6 | 28.3 | 120 | 219 | 798 |
| m=16 | 11.5 | 12.0 | 26.8 | 103 | 196 | 726 |

- **GIL overhead:** a fixed ~34 ns per update. The K-optimized cost model's
  ~52 ns intercept is mostly this.
- **Cost per extra dimension** (D = 784 → 3072):
  - m=2: 0.74 ns
  - m=4 fused: 0.26 ns, a ratio of 0.35 against the predicted 0.33. Both m=2
    and m=4 run at memory bandwidth, about 20 GB/s.
  - m=8 and m=16: 0.23–0.25 ns. That is an arithmetic floor (about one cycle
    per pair per dimension), so beyond m=4 the gains shrink.
- **Fused vs generic at m=4:** fused is 20–35% faster at high D.

## 2. Quality at equal pair budget

Stress ÷ original-kernel stress for the same dataset, k and seed. Median over
datasets.

- **Sequential updates:** j18 medians are 0.992–1.000 for m = 4, 8, 16 at
  k = 25–400; k8 medians are 0.9997 (m=4), 0.9987 (m=8) and 0.9981 (m=16).
  - m=4 is indistinguishable from m=2 (Wilcoxon p = 0.11).
  - m=8 and m=16 are slightly better (p ≈ 1e-6 to 1e-10), mostly at low k.
  - One consistent excess: m=4 on patchcamelyon at k=100 is +4–5% on all
    three seeds. It is gone by k=200, and m=8/16 don't show it.
- **Sum:** equal to sequential up to m=8. At m=16 it diverges (×8–11 stress,
  and all 70 divergences). This matches the worst-case overshoot condition
  m·μ < 4 with μ ≤ 0.5.
- **Mean:** 2–15% worse; it is too cautious to converge in 30 epochs.
- **Partition vs random:** stress ratio medians 0.9996–1.0006 at every m and
  k, and seed spread is no lower. Partition is useless here; its only use
  would be lock-free multithreading.

## 3. Full-fit speed

Original kernel time ÷ arm time, k=200, k8:

| | D | GIL fix (m=2) | m=4 | m=8 | m=16 | m=16 blocking only |
|---|--:|--:|--:|--:|--:|--:|
| feynman 50k | 3 | 3.14 | 4.63 | 4.74 | 4.44 | 1.42 |
| california 20,640 | 8 | 3.05 | 4.35 | 4.59 | 4.34 | 1.42 |
| ag_news 50k | 384 | 1.18 | 3.40 | 3.88 | 4.39 | 3.73 |
| patchcamelyon 50k | 512 | 1.21 | 3.18 | 3.59 | 4.01 | 3.31 |
| tiny_imagenet 50k | 512 | 1.20 | 3.13 | 3.53 | 3.95 | 3.28 |
| fashion_mnist_full 50k | 784 | 1.08 | 3.18 | 3.37 | 3.71 | 3.42 |
| mnist_digits 50k | 784 | 1.09 | 3.20 | 3.37 | 3.71 | 3.40 |
| cifar10_raw 50k | 3072 | 1.03 | 2.85 | 2.98 | 3.22 | 3.11 |

- **j18** (cache-resident, ≤ 40 MB): m = 4/8/16 all give about 2.2× over the
  original kernel, 1.8× from blocking alone.
- **Time to reach ≤ 1.05× the original kernel's k=200 stress** (j18, median):
  m=4 is 8.2× faster than the original kernel at k=200, against 4.6× for the
  GIL fix alone.

## 4. Against SQuaD-MDS

SQuaD-MDS data is from Experiment M on the same subsamples. Its best result per
dataset is the lowest median stress over its 8 tested settings; that was
n_iter = 1000 every time, the top of its tested range, so its stress is not its
floor. SQuaD's times come from M's run. Re-timing gave 0.84–1.30× of them, so
treat time ratios as ±25%. SGD below is blocked m=16, sequential updates,
random blocks, measured (median of 3 seeds; SQuaD 5 seeds).

**Blocked SGD at each budget** (N = 50k; california 20,640). Stress is × the
cycle stress.

| | D | k=100 | | k=200 | | k=400 | |
|---|--:|--:|--:|--:|--:|--:|--:|
| | | stress | time | stress | time | stress | time |
| feynman | 3 | 1.004 | 1.7 s | 1.001 | 3.4 s | 1.000 | 6.8 s |
| california | 8 | 1.046 | 0.7 s | 1.021 | 1.5 s | 1.010 | 2.9 s |
| ag_news | 384 | 1.102 | 15 s | 1.093 | 31 s | 1.041 | 62 s |
| patchcamelyon | 512 | 1.082 | 20 s | 1.020 | 40 s | 1.005 | 80 s |
| tiny_imagenet | 512 | 1.103 | 20 s | 1.094 | 40 s | 1.045 | 79 s |
| fashion_mnist_full | 784 | 1.056 | 30 s | 1.012 | 59 s | 1.003 | 118 s |
| mnist_digits | 784 | 1.183 | 30 s | 1.074 | 59 s | 1.016 | 118 s |
| cifar10_raw | 3072 | 1.046 | 109 s | 1.010 | 218 s | 1.004 | 438 s |

**SGD at k=200 vs SQuaD's best**

| | SQuaD stress | SQuaD time | SGD stress | SGD time | stress ratio | time ratio | original kernel time ratio |
|---|--:|--:|--:|--:|--:|--:|--:|
| feynman | 1.114 | 2.3 s | 1.001 | 3.4 s | 0.90 | 1.5× | 6.7× |
| california | 4.603 | 0.9 s | 1.021 | 1.5 s | 0.22 | 1.5× | 6.7× |
| ag_news | 1.065 (exa) | 12.8 s | 1.093 | 31 s | 1.03 | 2.4× | 10.6× |
| patchcamelyon | 1.084 | 13.3 s | 1.020 | 40 s | 0.94 | 3.0× | 12.0× |
| tiny_imagenet | 1.076 (exa) | 15.9 s | 1.094 | 40 s | 1.02 | 2.5× | 9.9× |
| fashion_mnist_full | 1.108 | 18.7 s | 1.012 | 59 s | 0.91 | 3.2× | 11.7× |
| mnist_digits | 1.519 | 16.2 s | 1.074 | 59 s | 0.71 | 3.6× | 13.5× |
| cifar10_raw | 1.052 | 44.0 s | 1.010 | 218 s | 0.96 | 5.0× | 16.0× |
| **median** | | | | | **0.93** | **2.7×** | **11.2×** |

- **Large datasets:** at k=200, SGD has lower stress than SQuaD on 6 of 8
  datasets (7% lower at the median). SQuaD is better only on ag_news and
  tiny_imagenet, by 2–3%; k=400 reverses those too.
- **Time:** SGD takes 2.7× SQuaD's time at the median, down from 11.2× with the
  original kernel.
- **N = 10k:** median stress ratio 0.93 and time ratio 3.2× (original kernel
  10.2×).
- **j18** (m16, k=200 vs SQuaD's best up to n_iter 2000): 2.3% lower stress,
  better on 14 of 17 datasets, at 2.3× the time.
- **Cost per pair processed:** blocked SGD costs 0.38–0.91× SQuaD's cost on 7
  of 8 datasets (cifar10_raw 1.24×). The time gap is pair count: k=200 is
  6,000·N pairs against SQuaD's 1,500·N.
- **Why SQuaD needs fewer pairs (hypothesis):** its PCA initialisation. In M's
  j18 sweep, SQuaD with a random init is 1.15–1.20× worse at every n_iter, and
  SGD always starts from a random init.

## 5. Recommendations

1. Use sequential updates with random blocks, at m = 8 or 16, for the fast
   path. Drop partition, sum and mean.
2. Decide on the `noexcept` fix for the original kernel. It changes no
   results, but all earlier budgeted timings include the GIL overhead.
3. Test PCA initialisation for SGD-MDS.
4. Run the extended Experiment M ladder, so SQuaD's best is no longer
   ceiling-bound.
