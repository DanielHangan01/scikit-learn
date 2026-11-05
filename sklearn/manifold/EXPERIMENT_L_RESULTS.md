# Experiment L — results & conclusions (Pivot MDS spectral baseline, all 26 datasets)

**Status: complete.** Both suites, 5 seeds each, 2,400 fits total. Runner:
[benchmarks/run_experiment_l.py](benchmarks/run_experiment_l.py).

A *spectral* baseline for the whole benchmark collection: standalone Pivot MDS
(Brandes & Pich, 2007) — select k pivots, build the N×k squared-distance
matrix, rectangular double-centre, one SVD. No epochs, no learning rate, no
iteration budget. Experiment H ran this on 6 datasets; this extends it to all
26 and to N=50,000, so the paper can state how a classical spectral
approximation compares with budgeted SGD across the full collection.

**Headline: budgeted SGD dominates on quality (26/26), Pivot MDS dominates on
speed (7–51×).** Neither is strictly better — it is a genuine Pareto
trade-off, and the interesting question is whether 1.2–18× worse stress is an
acceptable price for a one-shot method.

---

## 1. Conditions

| | Suite A (`j18`) | Suite B (`k8`) |
|---|---|---|
| datasets | the 18-dataset collection | the 8 large sources |
| N | each dataset's Experiment-J top rung (400–5,750) | K-optimized ladder {10k, 20k, 35k, 50k} |
| subsample | J's per-seed subsamples (seed 42+rep) | K-optimized's fixed subsample (seed 42) |
| SGD reference | Experiment J | Experiment K-optimized |
| seeds | 5 | 5 |
| hardware | local workstation (same as J) | cluster (same as K-optimized) |
| fits | 900 | 1,500 |

Pivot counts k ∈ {10, 25, 50, 100, 200} (filtered to 2 ≤ k < N) × selection
strategies {`maxmin`, `maxmin_pca10`}. Subsample conventions are matched to
each reference experiment, so Pivot-MDS stress is directly comparable to the
SGD cycle/budget numbers without re-running any SGD.

**Note on `k`.** Here k = **number of pivots** (cost ~O(k·N·D) + SVD). In the
budgeted-SGD experiments k = **updates-per-point multiplier** (cost
~O(epochs·k·N·D)). Same letter, different quantity.

### Two methodological requirements (both verified, not assumed)

1. **The embedding must be rescaled before its stress is comparable.**
   Classical-MDS-family methods fit inner products (strain), not distances, so
   they land at a stress-suboptimal global scale. We apply the closed-form
   optimum `α = ⟨D_emb,D⟩/⟨D_emb,D_emb⟩` before scoring; measured α for Pivot
   MDS is **4.4–8.0**, so this is a large correction, not a detail. Skipping
   it overstated the spectral gap by up to ~6× in Experiment H's first pass.
   SGD needs no such correction — its own α measures **0.99992–1.00034**
   (rescaling gain 0.00%), because minimising stress optimises scale
   implicitly. Applying the rescale to Pivot MDS only is therefore correct
   *and generous to Pivot MDS*.
2. **Timing must cover the same thing for both methods.** Both timers run from
   raw input array to finished embedding, with stress evaluation excluded.
   Notably this includes **pivot selection**, which dominates Pivot MDS's cost
   (33–95% of its total) — an earlier version of the wrapper started the timer
   after selection and under-reported by up to ~40×.

---

## 2. Quality — Pivot MDS loses on 26 of 26

"PMDS best" is the best over **all 10 configurations** (5 pivot counts × 2
strategies), i.e. generous; SGD k=200 is a single fixed setting. All ratios to
full-cycle SGD stress at the same (dataset, N, subsample).

### Suite A — the 18 datasets

| dataset | D | 2D-emb | N | **PMDS best** | SGD k=200 |
|---|--:|--:|--:|--:|--:|
| orl | 396 | 90.1% | 400 | 1.70× | — ¹ |
| svhn | 1024 | 63.1% | 732 | 1.56× | 1.01× |
| har | 561 | 57.3% | 735 | 1.90× | 1.03× |
| cifar10 | 1024 | 40.2% | 3,250 | 1.45× | 1.01× |
| fashion_mnist | 784 | 37.8% | 3,000 | 1.46× | 1.01× |
| seismic | 24 | 35.9% | 646 | 4.24× | 1.01× |
| coil20 | 400 | 34.2% | 1,440 | **1.38×** | 1.02× |
| fmd | 1536 | 22.1% | 997 | 1.57× | 1.01× |
| spambase | 57 | 20.8% | 4,601 | 7.73× | 1.06× |
| bank | 63 | 14.3% | 2,000 | 3.37× | 1.03× |
| epileptic | 178 | 13.9% | 5,750 | 7.44× | 1.10× |
| secom | 590 | 10.7% | 1,567 | 6.69× | 1.04× |
| hiva | 1617 | 8.5% | 3,000 | 3.32× | 1.09× |
| hatespeech | 100 | 4.5% | 3,221 | 3.80× | 1.04× |
| cnae9 | 856 | 3.6% | 1,080 | **12.42×** | 1.00× |
| sms | 500 | 2.6% | 835 | 5.73× | 1.00× |
| sentiment | 200 | 2.4% | 2,748 | 5.07× | 1.01× |
| imdb | 700 | 1.5% | 3,250 | 1.55× | 1.00× |

¹ orl has no k=200 SGD budget: at N=400, k=200 fails J's `k < (N−1)/2` filter.

**PMDS 1.38–12.42× (median 3.34×) vs SGD k=200 1.00–1.10× (median 1.01×).
0 of 17 wins** (18 excluding the orl gap).

### Suite B — the 8 large sources (top rung)

| dataset | D | 2D-emb | N | **PMDS best** | SGD k=200 | SGD k=400 |
|---|--:|--:|--:|--:|--:|--:|
| feynman_iii_15_12 | 3 | 68.7% | 50,000 | 1.35× | 1.00× | 1.00× |
| california_housing | 8 | 64.8% | 20,640 | **18.07×** ² | 1.02× | 1.01× |
| ag_news | 384 | 7.5% | 50,000 | **1.20×** | 1.10× | 1.07× |
| patchcamelyon | 512 | 31.4% | 50,000 | 2.01× | 1.02× | 1.01× |
| tiny_imagenet | 512 | 7.2% | 50,000 | 1.40× | 1.10× | 1.07× |
| fashion_mnist_full | 784 | 36.9% | 50,000 | 2.01× | 1.01× | 1.00× |
| mnist_digits | 784 | 11.5% | 50,000 | 4.04× | 1.10× | 1.02× |
| cifar10_raw | 3072 | 37.9% | 50,000 | 1.41× | 1.01× | 1.00× |

² Explained in §5.

**PMDS 1.20–18.07× (median 1.71×) vs SGD k=200 1.00–1.10× (median 1.02×).
0 of 8 wins**, including against k=400.

At large N the k-sweep has **saturated**: "PMDS best" and "PMDS at k=200" are
identical to two decimals on all 8, so more pivots would not help.

---

## 3. Speed — Pivot MDS is 7–51× faster, and selection is the reason

Both suites timed on the same hardware as their SGD reference. Times are the
full cost of producing an embedding.

### Suite B, top rung, k=200 (cluster)

| dataset | D | selection | spectral | **PMDS total** | SGD k=200 | cycle | vs SGD | vs cycle |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| feynman_iii_15_12 | 3 | 0.1 s | 0.20 s | **0.3 s** | 15.6 s | 4183 s | **51×** | 13,672× |
| california_housing | 8 | 0.1 s | 0.11 s | 0.2 s | 6.6 s | 687 s | 40× | 4,105× |
| ag_news | 384 | 3.5 s | 0.22 s | 3.8 s | 140.2 s | 4151 s | 37× | 1,102× |
| patchcamelyon | 512 | 4.6 s | 0.22 s | 4.8 s | 167.8 s | 4230 s | 35× | 883× |
| tiny_imagenet | 512 | 4.6 s | 1.02 s | 5.6 s | 167.2 s | 4155 s | 30× | 744× |
| fashion_mnist_full | 784 | 7.1 s | 0.25 s | 7.3 s | 215.9 s | 4233 s | 30× | 579× |
| mnist_digits | 784 | 7.1 s | 0.26 s | 7.3 s | 226.8 s | 4289 s | 31× | 585× |
| cifar10_raw | 3072 | 28.2 s | 1.40 s | **29.6 s** | 609.2 s | 4354 s | **21×** | 147× |

Suite A (local, smaller N) gives the same picture: **7.5–70.4× faster than
SGD k=200, median 19.1×**.

### The mechanism

**The spectral solve is nearly free** — 0.11–1.40 s regardless of D. It is one
N×k GEMM plus an SVD of an N×200 matrix: dense BLAS/LAPACK, measured at 14.4
cores.

**Pivot selection dominates and is strictly sequential.** Greedy maxmin is
O(k·N·D) and single-threaded (measured cpu/wall = 1.0); iteration i+1 cannot
start until i finishes. It grows from 33% of the total at D=3 to 95% at
D=3072, and it is **exactly linear in k** (fashion_mnist_full, N=50k: 0.69 s,
1.75 s, 3.51 s, 7.01 s, 14.06 s for k = 10, 25, 50, 100, 200).

So the honest statement is:

> Pivot MDS is not faster because it avoids per-pair distance work — it does
> the same O(k·N·D) work SGD does, but **once instead of once per epoch**.

That predicts the observed ratio: SGD runs 30 epochs, Pivot MDS ~one pass,
giving ≈30× at mid-D — which is what the table shows (30–37× for D=384–784).
The deviation at either end is the D-independent overheads: at D=3 SGD's
fixed per-epoch costs inflate the ratio to 51×, at D=3072 Pivot MDS's own
selection cost drags it to 21×.

Both algorithms are therefore in the **same complexity class per unit of
work**. The difference is iteration count, not vectorisation or a
fundamentally cheaper formulation — and neither benefits from the extra cores
in its dominant phase.

---

## 4. The regime theory does not transfer to the spectral method

`strain_top2` predicts budgeted-SGD behaviour with ρ = +0.75 (J) and −0.94
(K-optimized). It has **no such power for Pivot MDS**:

- `imdb` (strain 1.5%, least embeddable in Suite A) → 1.55×
- `cnae9` (strain 3.6%) → **12.42×**, the worst in Suite A
- `ag_news` (strain 7.5%) → **1.20×**, the *best* in Suite B
- `california_housing` (strain 64.8%, near the most embeddable) → **18.07×**,
  the worst overall

This is mechanistically expected. Budgeted SGD's gap is a *sampling* question
— can a random subset of pair constraints pin the layout? Pivot MDS's gap is a
*strain-versus-stress* question — how close is the strain-optimal embedding to
the stress-optimal one? Those are different properties of a dataset, so a
predictor for one carries no information about the other.

---

## 5. The `california_housing` outlier, explained

18.07× is 4.5× worse than the next-worst Suite B dataset and looks
anomalous. Three diagnostics resolve it, and none is a bug:

- **The k-sweep is flat** (10.8–12.5× at N=10k; 18.1–18.6× at N=20,640 across
  k=10…200). A pivot-*selection* pathology would vary with k; this does not.
- **Both strategies give bit-identical results** — because D=8, so the PCA-10
  projection is a no-op. Not an error, a degenerate case.
- **Exact ClassicalMDS is itself ~10× worse than SGD** on this data (measured
  at N=5,000: ClassicalMDS 9.89× vs PivotMDS k=200 17.36×). So most of the gap
  is not the pivot approximation at all — it is strain ≠ stress.

The cause is extreme outliers: per-feature excess kurtosis reaches **10,648**,
and the farthest point sits **59× the median distance** from the centroid.
Strain is dominated by those few extreme points, so the spectral embedding
spends its two dimensions separating them, while the stress optimum (which SGD
finds) fits the bulk.

This also exposes a limitation of `strain_top2` as a diagnostic: it reads
64.8% here — nominally "highly 2D-embeddable" — precisely *because* the
outliers create two dominant eigen-directions. High strain_top2 for a bad
reason. Worth noting wherever that metric is presented.

---

## 6. Conclusions

1. **Budgeted SGD dominates Pivot MDS on quality on all 26 datasets**, usually
   by a wide margin (median 3.34× on Suite A, 1.71× on Suite B, against SGD's
   1.01–1.02×). This holds even though Pivot MDS is given its best k, its best
   selection strategy, and its stress-optimal rescale.
2. **Pivot MDS is 7–51× faster** than budgeted SGD at k=200, and 147–13,672×
   faster than full cycle. This is a real Pareto trade-off, not a knockout.
3. **The speed advantage is "one pass instead of thirty"**, not a cheaper
   formulation: both do O(k·N·D) distance work, and Pivot MDS's dominant phase
   (selection) is as sequential and single-threaded as SGD's kernel.
4. **The regime theory is specific to subsampling.** `strain_top2` predicts
   budgeted-SGD quality but carries no information about spectral quality.
5. **Where Pivot MDS is competitive** (1.2–2.0×: ag_news, feynman,
   tiny_imagenet, cifar10_raw, fashion_mnist_full, patchcamelyon, coil20), it
   is a defensible choice when a ~1.5× stress penalty buys a 20–50× speedup —
   e.g. for interactive previews or initialisation. Where it is not (4–18×:
   cnae9, california_housing, spambase, epileptic, secom), it is not usable as
   a substitute at any budget.

---

## 7. Caveats

- **Suite B's cycle baseline is a single run** per (dataset, N), inherited
  from the K-optimized protocol. Every ratio there — for both methods — shares
  a single-sample denominator. Suite A is 5-seed paired throughout.
- **Suite A and Suite B ran on different machines** (local / cluster), but each
  suite is internally consistent with its own SGD reference. Do not compare
  absolute times across suites: the two machines differ by ~1.07× on the SGD
  workload but ~2.10× on Pivot MDS's selection loop, so no single scaling
  factor relates them.
- **`maxmin` vs `maxmin_pca10` are identical whenever D ≤ 10** (the projection
  is a no-op) — affects `california_housing` (D=8) and `seismic` (D=24 uses
  a real projection, but is close).
- Non-monotonicity in k appears on several small Suite A datasets (orl goes
  1.81× at k=10 → 8.52× at k=200 under maxmin, while exact ClassicalMDS is
  1.89×). Plausibly maxmin selection becoming outlier-dominated as k grows
  toward N, but this is an observation with a hypothesis, not a verified
  mechanism. It does not appear at large N, where the sweep saturates.
