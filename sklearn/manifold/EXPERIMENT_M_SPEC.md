# Experiment M — SQuaD-MDS as a stress baseline

**Status:** specification, not yet run.
**Prerequisite reading:** `EXPERIMENT_L_RESULTS.md` (the closest precedent — same
design, different baseline), `RESEARCH_NOTES.md` §1 and `CLAUDE.md` §4, §6.

---

## 1. Why

The paper's abstract and §1 currently claim that the stochastic MDS methods that
*do* scale "optimize a relaxation of stress rather than stress itself", naming
SQuaD-MDS as the example. That claim is presently supported only by reading the
method's definition. Experiment M measures it: **what raw stress does SQuaD-MDS
actually reach, against a full SGD sweep on the same data?**

This is the same move Experiment L made for Pivot MDS, and it should slot into
§4.3 beside it. Two outcomes are both publishable:

- SQuaD-MDS lands well above a full sweep on stress → the relaxation claim is
  quantified rather than asserted, and the paper gains a second scaling
  baseline alongside Pivot MDS.
- SQuaD-MDS lands close to a full sweep → the claim in the abstract and §1 is
  too strong and must be softened. **Report this outcome plainly if it
  happens.** The experiment is worth running precisely because it can go
  either way.

---

## 2. The method

> Pierre Lambert, Cyril de Bodt, Michel Verleysen, John A. Lee.
> **"SQuadMDS: A lean Stochastic Quartet MDS improving global structure
> preservation in neighbor embedding like t-SNE and UMAP."**
> *Neurocomputing* 503 (2022) 17–27. DOI `10.1016/j.neucom.2022.06.099`
> Already in `SGD for MDS/pacificvis/references.bib` as `lambert2022squadmds`.

Reference implementation: the authors' public repository (search
"SQuaD-MDS Lambert"). Prefer porting from the paper and cross-checking against
the reference code over trusting either alone.

**What it does.** Points are partitioned into random *quartets* each iteration;
each quartet contributes a loss over its six pairwise distances, and points move
by the gradient of that loss. Time and memory are O(N) per iteration.

**The critical detail — verify this yourself before building anything.** The
optimized loss is **not** stress over the quartet. The high-dimensional
distances within each quartet are normalized by their own sum before the loss
is taken; in the reference implementation this appears as a hard-coded

```python
Dhd_quartet /= np.sum(Dhd_quartet)
```

which makes the objective **scale-free per quartet**. The paper's §2.1 sets up
notation that looks like plain stress; the loss actually optimized is in §2.2.
An earlier draft of our own paper got this wrong by reading §2.1 alone, so
**read §2.2 and the repository, and write down in the results doc exactly which
loss you found.** If it turns out the released code optimizes something other
than what is described here, that finding supersedes this spec.

---

## 3. The trap that will dominate this experiment

Because the objective is scale-free, the output embedding has **no meaningful
global scale**. Scoring it against raw stress without correcting for that
measures the arbitrary scale, not the method.

Apply the closed-form stress-optimal rescale before scoring:

$$\alpha = \frac{\langle D_{\text{emb}}, \Delta\rangle}{\langle D_{\text{emb}}, D_{\text{emb}}\rangle}$$

`bench_utils.stress_and_optimal_scale_chunked(X, embedding)` returns
`(stress_scaled, stress_raw, alpha)` from one blocked O(N·block) pass; use
`..._multi(X, embeddings)` to batch many embeddings of the same `X`.

This is not hypothetical. Skipping it for Pivot MDS once overstated that
baseline's gap by up to ~6× (`EXPERIMENT_H_RESULTS.md`, "scale-artifact
correction"). **Report `stress_scaled` as the headline and always publish
`alpha` alongside it.**

A second, subtler point worth measuring: SQuaD-MDS normalizes *per quartet*, not
globally, so a single global α may not be able to repair it — different distance
ranges may be shrunk by different amounts. If `alpha` varies wildly across seeds
or datasets, or if `stress_scaled` stays far above the sweep while the embedding
looks structurally reasonable, say so. That would be a more interesting result
than the headline number.

---

## 4. Protocol

Follow `run_experiment_l.py` closely — it solves the same problem for a
different baseline, including subsample reuse and result-joining.

1. **Implement** `sklearn/manifold/_squad_mds.py`. Research-only, like
   `_pivot_mds.py`; not for the upstream PR, not exported from `__init__.py`.
   Pure Python/NumPy is fine — no Cython, so no rebuild step.
2. **Reuse the existing subsamples** so stresses join without re-running SGD:
   - `--suite j18` → the 18-dataset breadth collection, joins `experiment_j`
   - `--suite k8` → the 8 large sources, joins `experiment_k_optimized`
   Read the subsample seed from each dataset's `meta.json`; do not draw fresh
   subsamples.
3. **Grid.** 5 seeds per dataset. Sweep whatever hyperparameters the paper
   exposes (iteration count, learning-rate schedule, and the optional
   "exaggeration"/local-refinement phase if present). Give SQuaD-MDS its **best
   quality configuration**, as Experiment L did for Pivot MDS — the comparison
   should be maximally favourable to the baseline.
4. **Score** every embedding with `stress_and_optimal_scale_chunked`. Record
   `stress`, `stress_raw`, `scale_alpha`.
5. **Time** the fit only. Set `compute_stress=False` on any SGD arm and score
   outside the timed region — in-fit O(N²) scoring consumed up to 97% of a
   budgeted run's wall time in plain Experiment K and made every speedup
   meaningless. This is the single biggest benchmarking trap in this repo.
6. **Output** `results/experiment_m_{j18,k8}/<dataset>/results.csv` with columns
   matching Experiment L's where they overlap: `run_id, dataset, suite,
   n_samples, n_features, algo, seed, time_solver, time_scoring, stress,
   stress_raw, scale_alpha`, plus a column per swept hyperparameter.

---

## 5. Validation before trusting any number

Experiment L's Pivot MDS was believable because it reduced to ClassicalMDS at
k=N to ~3e-14. Establish something equivalent here, and report it:

- **Reproduce a published result.** The paper evaluates with RNX curves, not
  stress. Implement RNX (or AUC of it) *for validation only* and check that the
  port reaches published RNX values on a dataset both use. If it cannot, the
  port is wrong and its stress numbers mean nothing.
- **Sanity-check the loss.** Confirm the quartet loss decreases monotonically in
  expectation, and that the implementation is genuinely O(N) per iteration.
- **Degenerate check.** On a dataset that is well described in 2D (feynman,
  california_housing), SQuaD-MDS should produce a visibly sensible embedding.

---

## 6. Deliverable

`EXPERIMENT_M_RESULTS.md`, mirroring `EXPERIMENT_L_RESULTS.md`:

- which loss the reference implementation actually optimizes (§2 above);
- the validation evidence from §5;
- per-dataset table: SQuaD-MDS `stress/stress_sweep`, its `alpha`, its fit time,
  against the full sweep and budgeted SGD at k=200;
- range and median of the stress ratio on each suite;
- an explicit statement of whether the paper's "relaxation" claim is supported.

### Framing, once the numbers exist

SQuaD-MDS is designed for neighbourhood preservation and is evaluated in its own
paper on RNX. Scoring it on stress measures it against an objective it does not
optimize — exactly the caveat §5 of the paper already records for Pivot MDS, and
it applies here **more strongly**, because a per-group normalization is a larger
departure from stress than a global scale error. The result to claim is "this is
the stress SQuaD-MDS reaches", not "SQuaD-MDS is worse". A reader should come
away understanding the two methods target different things.

---

## 7. Constraints

- All work confined to `sklearn/manifold/` (`CLAUDE.md` §1).
- Datasets are ~29 GB, untracked, already present in `benchmarks/datasets/`;
  they are not regenerable from any script in the repo for the bulk collections.
- Raw stress is unnormalized and scales with N² — comparable within a dataset,
  never across. Report ratios to each dataset's own full sweep.
- Every speedup is against full-cycle SGD-MDS, never against SMACOF, which
  cannot be run at the scale-study sizes at all.
