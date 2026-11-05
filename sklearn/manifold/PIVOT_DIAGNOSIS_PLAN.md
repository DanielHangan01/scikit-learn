# Pivot SGD-MDS: diagnose the gap, then fix it

## Context

`SGDMDS` with `sampling_strategy='pivot'` plateaus at **5–11× the stress of full-cycle SGD** at small constant k (10–50) across six high-D datasets (Experiments F + G — see [CLAUDE.md](../../CLAUDE.md) §"Pivot SGD-MDS — the open question"). PCA-based pivot selection helps modestly (10–35% improvement over raw MaxMin) but does not close the structural gap. The current "Open question" lists four directions, none of which addresses the root cause because the root cause itself has not been isolated.

Two competing explanations are consistent with the empirical pattern and have very different fixes. Committing to a fix before isolating the cause risks weeks of work on the wrong mechanism. This plan executes the cheapest diagnostic first, then commits to a fix based on what the diagnostic shows.

### The two competing hypotheses

**H1 — Structural (coverage):** the SGD inner loop ([_sgd_mds_cython.pyx:298-308](_sgd_mds_cython.pyx)) only ever updates `(i, pivot_j)` pairs. For two non-pivot points `i, j`, there is no direct gradient enforcing `d(X_i, X_j) ≈ delta_ij` — only indirect coupling through pivots. The relative position of any two non-pivot points is under-determined by the objective. Pivot MDS (Brandes & Pich 2007) avoids this through a global spectral solve in which every point's position is a linear projection of its k-vector of pivot distances; the rank-d structure prevents drift. SGD has no analogous global constraint.

**H2 — Mechanical (LR + per-endpoint update count):** with `weighting='uniform'`, `_estimate_weight_bounds` returns `(1.0, 1.0)` ([_sgd_mds.py:301-302](_sgd_mds.py)), so `lr_init='auto'` resolves to `1/w_min = 1.0`. The kernel caps `step_val = min(w·lr, 1.0) = 1.0`, so every early update applies `(dist - target)/2` — a half-move-to-target per pair ([_sgd_mds_cython.pyx:114-145](_sgd_mds_cython.pyx)). Per epoch, each pivot endpoint is touched ~N times; each non-pivot endpoint is touched ~k times. The asymmetry is `N/k` (e.g. 60× at N=3000, k=50): pivots are hammered to near-convergence; non-pivots see a fraction of the gradient signal. The visible symptom — non-pivot positions look "off" — is identical to H1, but the fix is a per-endpoint LR rescue rather than a different algorithm.

Both must be tested before declaring victory for either, and neither has been ruled out by F or G.

---

## Phase 1 — Standalone Pivot MDS (Brandes & Pich, 2007) baseline

**Goal:** establish a no-SGD reference for "what stress can a purely spectral method achieve at this k?" Without it, no later claim about "SGD refines spectral" can be properly attributed.

**Implementation:** add `pivot_mds(X, pivot_indices, n_components, *, metric='euclidean')` as a module-level function in a new file [_pivot_mds.py](_pivot_mds.py). Algorithm:

1. Compute N×k pivot-distance matrix `D_pivot` (Euclidean distances from each point to each pivot).
2. Square element-wise: `D2 = D_pivot ** 2`.
3. Double-center: subtract row mean of `D2` (length N), subtract column mean (length k), add grand mean, multiply by `-0.5`. This is the Brandes & Pich §3 rectangular variant of Torgerson double-centering.
4. SVD: `U, S, Vt = scipy.linalg.svd(B, full_matrices=False)`. Take top-d singular vectors.
5. Embedding: `X_emb = U[:, :d] * sqrt(S[:d])`.
6. Apply `svd_flip` for sign determinism (consistent with [_classical_mds.py:193](_classical_mds.py)).

Reuse the existing pivot selection helpers (`_resolve_n_pivots`, `_maybe_pca_project`, `_select_pivots_maxmin` in `_sgd_mds.py`) — same MaxMin / PCA-MaxMin strategies. Do **not** make this a full `BaseEstimator` subclass yet; a module-level function is enough for benchmarking. Class-ification can wait until we know we want to keep it.

**Benchmark wrapper:** add `run_pivot_mds_benchmark(X, n_pivots, pivot_strategy, ...)` to [bench_utils.py](benchmarks/bench_utils.py). Mirrors the signature of `run_pivot_benchmark` and returns the same dict shape (`embedding`, `stress`, `time`, `n_iter=0`, `history=[(time, stress)]` — single-point history since spectral is one-shot).

**Cost & verification:**
- Implementation: ~50 lines (function + benchmark wrapper).
- Sanity check: on a 200-sample swiss_roll, with `n_pivots = N`, Pivot MDS should equal `ClassicalMDS` to within numerical noise. With `n_pivots = 20`, it should produce a recognizably correct 2D unrolling.
- No Cython compilation needed; pure Python/NumPy/SciPy.

---

## Phase 2 — Sharp diagnostic experiments

Run two cheap experiments designed to discriminate H1 (structural) from H2 (mechanical). Both reuse the existing Experiment G dataset list (`fashion_mnist`, `coil20`, `fmd`, `hiva`, `cifar10`, `cnae9`) and seed scheme (5 seeds) for direct comparability.

### Phase 2a — LR rescue sweep (tests H2)

**What to vary:** keep `sampling_strategy='pivot'`, `pivot_strategy='maxmin_pca'`, `pivot_pca_dim=10` (G's winner). Sweep `learning_rate_init ∈ {1.0 (current 'auto'), 0.3, 0.1, 0.03, 0.01}` at `k ∈ {25, 50, 100}`.

**What it tells us:**
- If lowering `lr_init` to 0.1 closes ≥30% of the stress gap at k=50 across most datasets → H2 is *at least partially* the cause, and a per-endpoint LR normalization is the right fix.
- If LR is flat across the sweep → H2 is ruled out; H1 is the dominant cause.
- The "auto" rule of `1/w_min` for uniform weighting is clearly wrong for pivot mode (it was designed for inverse weighting on cycle); even if H2 isn't the *only* cause, this experiment likely produces a small free win.

**Cost:** 3 datasets × 5 LRs × 3 ks × 5 seeds = 225 runs. Reuse `run_pivot_benchmark` with the new `lr_init` parameter (already accepted).

### Phase 2b — Hybrid pair sampling (tests H1)

**The decisive experiment for H1.** Add a new sampling mode that draws `α·k·N` pivot pairs + `(1-α)·k·N` uniform random non-pivot pairs each epoch, at **fixed total update count** (same wall-time budget as pure pivot). Sweep `α ∈ {1.0 (pure pivot), 0.75, 0.5, 0.25}` at `k ∈ {25, 50, 100}`.

**What it tells us:**
- If α=0.5 closes ≥50% of the stress gap at k=50 → H1 is confirmed: the bottleneck is non-pivot coverage. Hybrid is itself a near-free fix.
- If α has flat effect → H1 is partially ruled out; the gap is mechanical (H2) or stems from something else.
- If hybrid is *worse* than pure pivot → the random pairs are noisy and the pivot updates were doing real work; H1 needs to be reframed (e.g. as "wrong pivot subset, not too few constraints").

**Implementation:** the existing `run_sgd_epoch_lazy_random_native` (random) and `run_sgd_epoch_pivot_lazy` (pivot) kernels can be invoked back-to-back within one epoch with split update counts — no new Cython kernel needed. Add a `sampling_strategy='hybrid'` + `hybrid_alpha` parameter, dispatched at [_sgd_mds.py:843-856](_sgd_mds.py). This is ~20 lines.

**Cost:** 3 datasets × 4 αs × 3 ks × 5 seeds = 180 runs. Should fit in well under a day.

### Phase 2 — decisive plot to add

For the best-performing variant in each phase, generate a **per-point stress histogram** on one dataset at k=50: the distribution of `Σ_j (d_ij(emb) − δ_ij)²` over i, with pivot vs non-pivot indices color-coded. If non-pivots have a heavy upper tail and pivots are tight, H1's "drift" mechanism is directly visualized. This single plot may be more diagnostic than either experiment alone. Add as `plot_experiment_h.py` or as a post-hoc utility in `bench_utils.py`.

---

## Phase 3 — Commit to a fix based on Phase 1+2 outcomes

The right next move depends entirely on what Phase 2 shows. Decision tree:

| Phase 2 outcome | What it means | Recommended Phase 3 fix |
|---|---|---|
| LR sweep flat, hybrid wins at α=0.5 | H1 dominant: coverage is the bottleneck | Ship `sampling_strategy='hybrid'` with `α=0.5` as the default for pivot use cases. Document `pivot` as "use only with hybrid". |
| LR sweep wins, hybrid flat | H2 dominant: mechanical asymmetry | Implement per-endpoint LR (each point gets `lr_eff = lr · count[endpoint]⁻¹`); revisit `_estimate_weight_bounds` for pivot mode. |
| Both win partially | Both effects real and additive | Combine: LR fix + hybrid + (optionally) Pivot MDS spectral init for warm start. |
| Neither closes the gap | Structural problem deeper than coverage | The pivot pair set is fundamentally too narrow regardless of LR/coverage. Pivot MDS spectral init (option below) becomes the primary candidate; consider abandoning pivot SGD in favor of "Pivot MDS init + cycle SGD refinement". |
| Pivot MDS standalone (from Phase 1) is already competitive with SGD-cycle at small k | The whole "SGD pivot variant" premise is wrong | Recommend Brandes-Pich Pivot MDS as the lightweight fast mode; relegate SGD to full-cycle / random / refinement. |

**Pivot MDS spectral init as a candidate Phase 3 fix:** if Phase 2 shows the gap is genuinely structural (H1 strongly confirmed), the most principled fix is to **initialize the embedding from Phase 1's Pivot MDS solution** and then refine with `SGD-cycle` (not pivot — pivot has the defect we're escaping). This is added as a new `init` parameter on `SGDMDS` (`{'random' (default), 'spectral_pivot', ndarray}`), dispatched at [_sgd_mds.py:677](_sgd_mds.py) where the random init currently lives. Critical implementation detail: the spectral init's coordinate scale will not match the LR schedule (which assumes ~unit-variance init), so either rescale the spectral embedding to unit variance per dimension, or recompute `w_min/w_max` from the embedding's own initial distances. Without this rescaling, the first SGD epoch with `lr=1.0` will overshoot and destroy the spectral structure — a real failure mode flagged during plan review.

Do **not** plan Phase 3 in detail now. The branching is wide; planning all branches in advance wastes effort, and the diagnostic results may well surface options we haven't considered (e.g. a specific subset of datasets benefits from one fix and a different subset from another).

---

## Critical files

| File | What changes / is added |
|---|---|
| [_pivot_mds.py](_pivot_mds.py) | **NEW**: `pivot_mds(X, pivot_indices, n_components, ...)` function (Phase 1). |
| [_sgd_mds.py](_sgd_mds.py) | Add `sampling_strategy='hybrid'` + `hybrid_alpha` parameter and dispatcher in `_run_lazy` (Phase 2b). Reuse `_resolve_n_pivots`, `_select_pivots_maxmin`, `_maybe_pca_project` (no changes to those). |
| [benchmarks/bench_utils.py](benchmarks/bench_utils.py) | Add `run_pivot_mds_benchmark` (Phase 1). Extend `run_pivot_benchmark` for `lr_init` parameter (Phase 2a — likely already supported). |
| [benchmarks/run_experiment_h.py](benchmarks/run_experiment_h.py) | **NEW**: orchestrates Phase 1 (Pivot MDS baseline) + Phase 2a (LR sweep) + Phase 2b (hybrid). Mirrors `run_experiment_g.py` structure. |
| [benchmarks/plot_experiment_h.py](benchmarks/plot_experiment_h.py) | **NEW**: per-dataset stress-vs-k plots for each algorithm/variant; per-point stress histogram for the decisive plot. |
| [_sgd_mds_cython.pyx](_sgd_mds_cython.pyx) | **No changes needed** for Phases 1–2. Hybrid mode reuses existing `run_sgd_epoch_pivot_lazy` and `run_sgd_epoch_lazy_random_native` back-to-back. |
| [../../CLAUDE.md](../../CLAUDE.md) | Update "Pivot SGD-MDS — the open question" with Phase 1+2 results once they land. |

## Reusable existing pieces

- [_classical_mds.py:178-198](_classical_mds.py) — exact pattern for double-centering + eigendecomposition. Phase 1's `pivot_mds` is structurally identical but operates on a rectangular N×k matrix (use SVD instead of `eigh`).
- [_sgd_mds.py:204-241](_sgd_mds.py) — `_select_pivots_maxmin`, reusable verbatim for Pivot MDS standalone.
- [_sgd_mds.py:177-201](_sgd_mds.py) — `_maybe_pca_project`, reusable for Pivot MDS standalone if we want a `pivot_strategy='maxmin_pca'` variant of the baseline.
- [benchmarks/bench_utils.py](benchmarks/bench_utils.py) — copy the dataset-loading / seed-loop / results.csv structure from existing experiments.

---

## Verification

**Phase 1:**
- `python -c "from sklearn.manifold._pivot_mds import pivot_mds; ..."` — basic API smoke test.
- On `swiss_roll` (N=200), check that `pivot_mds(X, all_indices, 2)` matches `ClassicalMDS(2).fit_transform(X)` to within 1e-6.
- On `coil20` (N=1440), Pivot MDS at k=50 should produce a recognizable 2D layout (no random scatter); visualize.

**Phase 2a:**
- Reproduce `run_experiment_g.py`'s `pivot-lazy-pca10` numbers at `lr_init=1.0` (sanity that the sweep harness is consistent with G).
- Plot stress vs `lr_init` per dataset; expect either a U-shape (optimum below 1.0) or flat (LR ruled out).

**Phase 2b:**
- For α=1.0 (pure pivot), reproduce G's `pivot-lazy-pca10` stress within seed noise. This validates that hybrid dispatch reduces correctly to pure pivot.
- For α=0.0 (pure random), reproduce existing `sampling_strategy='random'` stress. Validates the random kernel is called correctly.
- Per-point stress histogram on one dataset at k=50 for: `pivot-pca10` vs `hybrid α=0.5` vs `cycle`. The pivot histogram should have heavy non-pivot tail; if hybrid flattens it, H1 is confirmed.

**End-to-end test (each phase):**
- Run all of [tests/](tests/) after each phase's code changes.
- New test file `test_pivot_mds.py`: at least `test_pivot_mds_reduces_to_classical_at_full_k`, `test_pivot_mds_output_shape`, `test_pivot_mds_deterministic_under_seed`.
- New test in `test_pivot_sgdmds.py`: `test_hybrid_sampling_dispatch` — at α=1.0 matches pure pivot, at α=0.0 matches pure random.

## Out of scope for this plan

- Implementing Pivot MDS as a full `BaseEstimator` subclass (defer until Phase 3 decides whether to ship it).
- `_sgd_mds_cython.pyx` changes (only needed if Phase 3 leads to per-endpoint LR or a fused hybrid kernel).
- API stabilization, changelog entry, `sklearn/manifold/__init__.py` export — all deferred per the existing PR readiness checklist.
- Multi-resolution / hierarchical approaches (CLAUDE.md option 2) — not addressed; can be picked up after Phase 2 if neither H1 nor H2 fully explains the gap.
