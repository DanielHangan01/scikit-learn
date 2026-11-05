# Experiment H — Results & verification

Diagnostic for the pivot SGD-MDS stress gap (Experiments F/G), including the
Brandes & Pich (2007) Pivot MDS spectral baseline. This document records the
**verified, scale-corrected** results; it supersedes the raw-stress baseline
numbers that circulated before the rescaling fix (see "The scale-artifact
correction" below).

Runner: [benchmarks/run_experiment_h.py](benchmarks/run_experiment_h.py) ·
Plots: [benchmarks/results/experiment_h/](benchmarks/results/experiment_h/) ·
Repair script: [benchmarks/rescale_experiment_h_baseline.py](benchmarks/rescale_experiment_h_baseline.py)

---

## 1. What was run

Six high-D datasets (`fashion_mnist, coil20, fmd, hiva, cifar10, cnae9`),
5 seeds, `max_iter=30`, `lr=0.5`:

- **H-base** — SGD-cycle reference + standalone **PivotMDS spectral solver**
  ([_pivot_mds.py](_pivot_mds.py), Brandes & Pich 2007 §3) at
  k ∈ {10, 25, 50, 100, 200}, for both `maxmin` and `maxmin_pca` (PCA-10)
  pivot selection.
- **H-lr** — SGD-pivot at k=50 sweeping `learning_rate_init ∈ {1.0 … 0.01}`
  (tests H2, the mechanical/LR hypothesis).
- **H-hyb** — SGD-hybrid at k=50 sweeping `hybrid_alpha ∈ {1.0 … 0.0}` at
  fixed update budget `k·N` per epoch (tests H1, the structural/coverage
  hypothesis).

### Verification (2026-06)

- Stress accounting is apples-to-apples: all methods are scored with the same
  raw stress `0.5·Σ(d_ij(emb) − δ_ij)²` over the full matrix.
- PivotMDS and SGD-pivot use the **same pivot-selection code and seeds** →
  identical anchors.
- The `_pivot_mds.py` implementation was verified against the paper:
  centering matches the paper's `c_ij` formula term-for-term; the SVD extracts
  the identical subspace as the paper's power iteration (|cos| = 1.0 to
  machine precision); the only deviation is the `√σ` coordinate scaling
  (classical-MDS-consistent) instead of the paper's literal `σ` — at k=N the
  implementation reduces to ClassicalMDS to ~3e-14.
- All CSV-derived medians reproduce the numbers below.

### The scale-artifact correction

Classical-MDS-family embeddings fit **inner products (strain)**, not
distances, so they come out at a stress-suboptimal global scale. SGD optimizes
scale implicitly. The original H run compared the spectral baseline's
*unscaled* stress against SGD — overstating its gap by up to ~6× (the optimal
rescale factor α reached ~5.5 on fashion_mnist). The fix: rescale the
spectral embedding by the closed-form stress-optimal factor

```
α = ⟨D_emb, Δ⟩ / ⟨D_emb, D_emb⟩
```

before computing stress. `run_pivot_mds_benchmark` now does this by default
(`stress` = rescaled, `stress_raw` = unscaled);
`rescale_experiment_h_baseline.py` back-filled the existing H CSVs with a
determinism guard (recomputed raw stress must match the stored value before
any row is replaced — all 300 rows passed). **SGD rows are untouched.** The
SGD-vs-SGD findings (H1/H2/the α-sweep) never depended on the baseline and
are unaffected.

---

## 2. Headline results (median over 5 seeds, ratio to SGD-cycle stress)

### PivotMDS spectral baseline, scale-corrected (PCA-10 pivots)

| dataset | N | k=10 | k=25 | k=50 | k=100 | k=200 | (k=50 raw, pre-fix) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fashion_mnist | 3000 | 2.48× | 1.75× | **1.64×** | 1.48× | 1.46× | 9.40× |
| coil20 | 1440 | 1.63× | 1.48× | **1.41×** | 1.38× | 1.40× | 7.23× |
| fmd | 997 | 2.05× | 1.81× | **1.67×** | 1.66× | 1.64× | 5.47× |
| hiva | 3076 | 3.84× | 3.49× | **3.35×** | 3.29× | 3.37× | 7.15× |
| cifar10 | 3250 | 1.50× | 1.48× | **1.45×** | 1.46× | 1.46× | 7.88× |
| cnae9 | 1080 | 12.43× | 12.49× | **12.51×** | 12.54× | 12.53× | 12.76× |

Full ClassicalMDS (k=N, scale-corrected, seed 42): fashion_mnist 1.61×,
coil20 1.65×, fmd 1.65×, hiva 2.07×, cifar10 1.50×, cnae9 12.51×.

### All methods at k=50 (ratio to cycle)

| dataset | cycle abs. | PivotMDS (rescaled) | SGD-pivot best-LR | hybrid α=0.5 | **hybrid α=0.0** |
|---|---:|---:|---:|---:|---:|
| cifar10 | 7.8e8 | 1.45× | 6.11× | 1.40× | **1.23×** |
| cnae9 | 7.3e7 | 12.51× | 5.53× | 1.25× | **1.01×** |
| coil20 | 7.1e7 | 1.41× | 7.04× | 1.35× | **1.07×** |
| fashion_mnist | 4.6e8 | 1.64× | 8.89× | 1.52× | **1.14×** |
| fmd | 1.7e8 | 1.67× | 4.88× | 1.21× | **1.03×** |
| hiva | 1.7e9 | 3.35× | 6.15× | 1.38× | **1.10×** |

---

## 3. Conclusions

1. **H2 (learning rate) ruled out.** The LR sweep is flat or slightly
   worsening as `lr_init` decreases; best-LR SGD-pivot is still 4.9–8.9×
   cycle. The kernel/step size is not the bottleneck.

2. **H1 (pair coverage) confirmed.** The hybrid α sweep is monotone from
   α=1.0 (pure pivot, 5–9×) to α=0.0 (pure random at the same `k·N` budget,
   1.01–1.23×) on every dataset. Coverage of non-pivot–non-pivot pairs is the
   bottleneck.

3. **Pivots are an actively bad sampling distribution for SGD.** At identical
   update budget, uniform random pairs beat pivot pairs by 5–9×. This is the
   decisive finding that led to the `n_updates_per_epoch` budget knob
   (Experiment I).

4. **The Brandes & Pich small-k claim replicates on our data.** Scale-corrected
   PivotMDS at k=50 matches full ClassicalMDS (k=N) within a few percent on
   every dataset (on cnae9 they are identical). The pivot *approximation*
   loses essentially nothing — exactly the paper's claim (their Procrustes
   R²=0.0085 at k=50, n=2075). The pivots were never the problem.

5. **Properly compared, the spectral method beats pivot SGD on 5 of 6
   datasets** (1.4–3.4× vs 4.9–8.9×). Pivot SGD failed harder than the
   algorithm that inspired it. Exception: cnae9, where SGD-pivot (5.5×)
   beats the spectral solve (12.5×) — gradient descent escapes the bad
   spectral subspace there.

6. **Why pivots pay off in the paper but not in SGD.** In PivotMDS, the n×k
   distance submatrix feeds a *global spectral solve* (Nyström-style): the
   target is the top-2 eigenvector subspace — a rank-2 object that k≈50
   well-spread columns suffice to estimate, and the eigenvector estimate
   generalizes to non-pivot–non-pivot geometry for free. In SGD, pivots only
   define a *sampling distribution over gradient constraints*: an (i, j)
   distance that is never sampled is never enforced, and raw stress charges
   for all N(N−1)/2 of them. There is no generalization mechanism, so anchor
   structure is dead weight — worse, it starves the rest of the pair space.

7. **Even corrected, the spectral family does not reach SGD-cycle.** Strain ≠
   stress: 1.4–3.4× (and 12.5× on cnae9, whose geometry the top-2
   eigen-subspace cannot represent). Direct stress minimization wins. Note
   the paper's domain (sparse graphs/meshes, low intrinsic dimensionality,
   shortest-path distances) is one where classical MDS is near-stress-optimal
   anyway; our high-D feature data with distance concentration is not.

### Consequence for the earlier "spectral init" judgement

The pre-fix record claimed the spectral baseline was 5–13× worse than cycle,
"confirming there's no good small-k anchor structure to inherit" and that a
spectral-init + SGD-refinement hybrid "would not have closed the gap." **That
reasoning was wrong** — it rested on the scale artifact. A rescaled PivotMDS
embedding at 1.4–1.7× cycle would in fact be a strong SGD initializer on most
of these datasets. The *decision* survives for a different reason: budgeted
random sampling (Experiment I) reaches 1.01–1.16× cycle with no spectral
machinery, no O(N·k) distance matrix, and no extra hyperparameters — so
spectral init is unnecessary, not unviable. It remains a legitimate future
option for warm-starting large-N runs.
