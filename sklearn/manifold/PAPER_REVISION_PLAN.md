# PacificVis journal-track revision plan

> **Status 2026-09-06.** Phases 0, 1, 2, 4, 5 are **applied** to
> `pacificvis/main.tex` (build clean: 0 errors, no undefined refs/citations,
> no dropped floats, no uncited bib entries, 9 pages). Outstanding:
> **Phase 3** (SMACOF rigour — needs re-runs), **Phase 6** (figures),
> **Phase 7** (Sparse SGD arm — supervisor decision), the node spec, and a
> teaser decision.
>
> **F13 — RESOLVED 2026-09-06.** The "14 out of the 18" count was never wrong:
> it is a **margin** claim. SGD attains lower mean stress on **18/18**
> (significant by paired CI on every one); the margin exceeds $0.5\%$ on
> exactly **14**, and falls under it on exactly **4** — IMDB (0.01 %),
> SMS (0.13 %), CNAE-9 (0.14 %), Sentiment (0.39 %). Only the *direction* of
> the IMDB sentence was wrong: SGD is marginally lower there and 3.1× faster,
> not 2.5 % worse. §4.1 now states this correctly, and the four exceptions are
> explained by the paper's own diagnostic — they are the four least
> 2D-embeddable datasets ($\mathrm{strain}_2 \leq 0.036$), and across the 18 the
> margin tracks $\mathrm{strain}_2$ at $\rho = +0.71$ ($p < 0.001$), FMD aside.
> This replaces the norm-concentration story and links §4.1 to §3.4. All three
> `\todo` markers are cleared.
>
> <details><summary>Original description of the conflict</summary>
> `results/experiment_a/sgd_30_epochs/experiment_a_means_per_dataset.csv` — the
> 30-epoch configuration §4.1 describes — shows SGD attaining lower mean stress
> on **18/18** datasets, significant by CI on every one. IMDB is ratio 0.99992
> (SGD marginally *lower*, $p = 2.4\times10^{-4}$) and SGD is 3.1× faster on it.
> §4.1 claims "14 out of the 18" and that "SMACOF achieved a 2.5% lower stress
> in less time" on IMDB. Both are unsupported by that file. Three `\todo`
> markers are live in the text (abstract, §4.1 count, §4.1 IMDB sentence);
> no number is asserted until this is resolved. This is Phase 3's first task,
> and it is more urgent than anything else remaining — under the stress-only
> scope §4.1 *is* the result.
> </details>

Consolidates three rounds of external review of `SGD for MDS/pacificvis/main.tex`
plus independent verification of every factual claim those reviews rested on.
Line numbers refer to the file as of 2026-09-06 (= the Overleaf state).

---

## 0. Scope, fixed by the supervisor

**The paper claims one thing: the algorithm optimises stress better.** Not
embedding quality, not neighbourhood preservation, not "better projections".

Consequences that follow directly:

- Stress is the sufficient metric. RNX / trustworthiness / continuity are
  **out of scope** and stay out.
- Every sentence that claims *quality* rather than *stress* breaks the
  defence and must go (two such sentences exist — §5.1 below).
- Comparisons against methods that do not optimise stress (Pivot MDS,
  SQuaD-MDS) are context, never headline, and must carry the rescaling
  caveat.
- The SMACOF comparison becomes the load-bearing experiment: if the whole
  claim is "lower stress than SMACOF", SMACOF's configuration *is* the result.

---

## 1. Verified facts the plan rests on

Each of these was checked against the primary source or the code, not taken
from a review.

| # | Fact | Source | Effect |
|---|---|---|---|
| F1 | Zheng et al. §IV-C, Algorithm 2 "Sparse SGD", applies SGD inside Ortmann's sparse stress model with `w'_ip = s_ip · w_ip`, `s_ip = |{j ∈ R(p): d_jp ≤ d_ip/2}|`, to 114,599 vertices with 200 pivots | arXiv 1710.04626 full text | §3.3's "a stochastic solver has no equivalent step" is **false about our primary reference** |
| F2 | Our Experiment H tested *unweighted* pivot restriction only: `weighting ∈ {uniform, inverse}`, `_pivot_pairs` attaches no region weight | `_sgd_mds.py:409`, `_pivot_pairs` | §3.3's *measurement* is valid; its *explanation* is wrong |
| F3 | SQuaD-MDS optimises a **per-quartet relative** loss (each distance ÷ within-quartet sum), hard-coded in `SQuaD_MDS.py` (`Dhd_quartet /= np.sum(Dhd_quartet)`), no absolute variant; never reports stress; evaluates on RNX only; SMACOF capped at N=5,000 | arXiv 2202.12087 §2.2, §3.2; reference repo | Not a stress competitor. But its on-the-fly distances + O(N) space (2022) kill the **Lazy Mode novelty claim** and the intro's "not adopted in DR" sentence |
| F4 | Chalmers 1996: constant per-point sample budget → O(N) iteration | title/abstract; already in our `.bib` | §3.2's budget invariant is prior art; claim its *validation*, not its *introduction* |
| F5 | 10-NN pairs carry **0.16–0.40 %** of Σδ² on 6 datasets, not separating regimes | measured, 1,200-pt subsamples | Raw stress is a far-pair objective → uniform sampling is well-matched to it. **Support**, not a threat, under this scope |
| F6 | Per-point 10-NN coverage under uniform sampling at k=200 collapses 42× from N=1,200 (≈100) to N=50,000 (≈2.4) while stress overhead stays flat | arithmetic + K-opt data | Confirms F5; explains why concentration-CV predicts nothing; why local-exact terms (Chalmers/Ortmann) are less necessary for *stress* |
| F7 | ρ = −0.94 for strain₂ → k\* is computed on **n = 6**; the two "never reach" sources are excluded and they are the low-strain₂ tail | `k_star()` on K-opt CSVs | Paper must state n=6 and treat the two as right-censored (k\* > 400) |
| F8 | Speedup-vs-D ρ = −1.00 is largely implied by Eq. 6 | — | Demote to consistency check |
| F9 | `strain₂` is the classical-MDS eigenvalue ratio (Torgerson scree) | textbook | Cite as classical; claim only the *use* as a fit-time gate |
| F10 | `chalmers1996linear`, `ingram2009glimmer`, `ortmann2017sparse` are in `.bib` and cited **0** times, because `\todo{Reducing number of pairs iterated family.}` (l.248) replaced the paragraph that cited them | grep | One deletion orphaned the entire DR-side sampling family |
| F11 | §5 still says "scalability on larger datasets becomes a bottleneck" and "stress approximation schemes used in graph drawing are future work" | l.≈660 | Contradicts §3.2/§4.2 and hands the reviewer the Ortmann citation |
| F12 | `fig:results` (Fig. 3) is silently dropped from the compiled PDF (`Float(s) lost`) | build log | Pre-existing; fixed locally, fix not yet on Overleaf |

---

## 2. Phase 0 — Hygiene (first; ~1–2 h; scope-independent)

Everything a reviewer sees before reaching the science.

| Item | Where | Fix |
|---|---|---|
| Teaser placeholder `Caption.` | l.101 | Delete the `\teaser{}` block, or supply a real figure (candidate: §6.4 at-scale embedding) |
| `NODE-SPEC-HERE` | l.528 footnote | Fill in the cluster node spec for the scale study |
| GitHub URL under `Anonymous Author(s)` | l.168, l.271 | Remove both for double-blind; restore at camera-ready |
| Stale `.bbl` → `[?]` citations in the uploaded PDF | Overleaf | Recompile; check the *uploaded* PDF, not the local build |
| Three uncited `.bib` entries | F10 | Resolved by Phase 1.3 |
| Add `checkcites` to the pre-submission checklist | — | `checkcites main.aux` — would have caught F10 mechanically |
| Float loss (Fig. 3 missing) | F12 | Carry over the local fixes: `\usepackage{float}`; `fig:results` `[h]→[tp]` (`h` is illegal for `figure*`); `[H]` on `tab:sources`, `tab:speedup`, `fig:kstar`, `fig:pareto`; `fig:budget-N` → single-column `figure[H]` using the **stacked** `budget_summary.pdf`; `tab:speedup` `table*`→`table`, 5 columns |
| Duplicate `combined_runtime_vs_n` figure with duplicate `\label{fig:runtime}` | was l.≈375 & l.≈453 | Check it did not reappear in the merge |

---

## 3. Phase 1 — Positioning: abstract, §1, §2

Unblocked (SQuaD-MDS is not a prerequisite, per F3).

### 1.1 Abstract

- **l.107** `matches or exceeds SMACOF's embedding quality` → a stress claim, with the
  real number: *"attains lower stress than SMACOF on 14 of the 18 benchmark
  datasets while reducing runtime."* The correct wording already sits
  commented out on l.110.
- `publicly available implementations` → `standard DR libraries`
  (SQuaD-MDS has a public repo; the narrower claim is the true one).
- Re-centre the last two sentences on what is actually ours: the tunable
  budget, `strain₂` as a fit-time gate, and k\* ∝ N^p to N = 50,000.

### 1.2 §1 Introduction

- **l.146** replace *"these stochastic advances have not been widely adopted
  in DR for MDS"* with the differentiator (defensible from SQuaD-MDS's own
  §2.2 and our §4.3):

  > Prior stochastic approaches to MDS in DR relax the objective: SQuaD-MDS
  > optimises a scale-free per-quartet surrogate, and Landmark and Pivot MDS
  > optimise strain. We instead optimise an **unbiased stochastic estimate of
  > the stress objective itself**, at scale.

  Do **not** write "the exact stress objective" — the budgeted mode samples
  it; §3.2 already draws exactly this unbiased/biased line.
- **l.163** Lazy Mode: demote from contribution to implementation detail
  (on-the-fly distances at O(N) space is SQuaD-MDS 2022, F3).
- **l.163** `This enables exact MDS` → `This enables MDS over the full stress
  objective` (full-cycle SGD is not "exact"; calling it so implies the
  budgeted mode approximates the objective rather than its coverage).
- Re-add the two budget paragraphs (previously drafted; absent from the file):
  the quadratic-*work* motivation and the three-result summary. Reframe result
  one per F4: *"a fixed per-point budget — the invariant Chalmers assumed —
  suffices as data grows, and we characterise when."*
- Add a **contributions list** (journal reviewers score against one).
  Candidate items: (i) tunable per-epoch budget with a measured
  quality/cost curve; (ii) `strain₂` as a fit-time predictor of the required
  budget, with the regime taxonomy; (iii) constant-k validated to N = 50,000;
  (iv) a spectral reference point across 26 datasets; (v) the unweighted-pivot
  negative result, correctly scoped (Phase 2).

### 1.3 §2 — restore the sampling-family paragraph, now differentiated

Replace `\todo{Reducing number of pairs iterated family.}` (l.248) with a
paragraph that does three things:

1. Names the family: Chalmers 1996 (neighbour cache + random set),
   Morrison/Ross/Chalmers 2003 (O(N√N) sample-and-interpolate), Glimmer
   (multilevel), Ortmann 2017 (sparse stress, local-exact + weighted pivots).
2. States the shared design: **local-exact + long-range-approximate**, and
   that we carry only the second half.
3. Justifies that with F5/F6: for a stress objective, near pairs carry
   < 0.4 % of the mass, so uniform sampling allocates budget in proportion
   to objective mass; the local term matters for neighbourhood metrics,
   which are outside this paper's claim.

Then the Chalmers sentence is honest *and* answered: *"we are closer to the
random half of Chalmers than to Chalmers entire, and for a stress objective
that is the half that matters."*

Also add SQuaD-MDS to the spectral/relaxed-objective paragraph as a third
example of relaxing the objective (F3).

---

## 4. Phase 2 — §3.3 correction (no new experiment required under this scope)

Keep the measurement (5–8×, the α-interpolation, Fig. 4). Replace the
explanation. Draft:

> At an equal budget, restricting updates to (i, pivot) pairs **without
> reweighting** yields 5–8× higher stress than uniform sampling. This is the
> expected consequence of estimator bias: uniform sampling with replacement is
> an unbiased sample of the full stress gradient (§3.2), whereas an unweighted
> pivot restriction drops every non-pivot pair and does not reweight the
> survivors, so it descends a biased surrogate. The sparse stress model of
> Ortmann et al. corrects exactly this by weighting each pivot term by the
> size of the region it stands in for, and Zheng et al. show SGD operates
> within that model (their Sparse SGD, §IV-C). Our result therefore
> characterises the unweighted case, and shows that the pivot *pattern* alone,
> without its weighting, is not a usable sampling distribution for a
> stochastic solver.

Delete: *"A stochastic solver has no equivalent step"* and everything built
on it. Keep the α-sweep as the evidence and the *"anchors are not at fault"*
sentence (Pivot MDS proper still works, §4.3).

---

## 5. Phase 3 — SMACOF rigour (the load-bearing experiment under this scope)

> **Partly done 2026-09-06.** The parameters were recovered from
> `run_experiment_a.py` / `bench_utils.py:278` and are now stated in a
> **Configuration** paragraph in §4.1, together with two robustness checks that
> both came out in our favour:
>
> - SMACOF: `n_components=2, metric=True, n_init=1, max_iter=300, eps=1e-12,
>   normalized_stress=False, n_jobs=1`, `dissimilarity='precomputed'`,
>   scikit-learn's own estimator unmodified.
> - SGD: `n_components=2, metric=True, n_init=1, max_iter=30`, $\eta_0=0.5$,
>   hybrid schedule `switch_ratio=0.5`, `epsilon=0.001`, `n_jobs=1`.
> - Both consume the **same precomputed matrix and the same seed**;
>   20 seeds (42–61); same unnormalized raw stress scorer.
> - `eps=1e-12` is 6 orders tighter than the library default (1e-6), so SMACOF
>   never stopped early: **all 380 runs used the full 300 iterations**.
> - The cap is not binding on the conclusions: SMACOF's improvement over its
>   final 10 iterations is $\leq 0.32\%$ (median 0.06 %), **10–426× smaller**
>   than the margin against it on the same dataset.
> - `n_init=1` is below sklearn's default of 4. Resampling the 20 runs to
>   simulate best-of-4 restarts flips **no** dataset; SGD's mean stays at or
>   below SMACOF's best-of-4 everywhere, exactly level only on IMDB.
>
> Still open in this phase: the classical-MDS-init arm (deferred by request),
> the matched-budget discussion, moving the 18-dataset table in-paper, and
> normalised stress in Figures 2–3.

Under "we optimise stress better than SMACOF", the SMACOF configuration is the
result. Currently unstated.

- **Report every parameter**: scikit-learn version, `n_init`, `max_iter`,
  `eps`, `normalized_stress`, `metric`, initialisation, `n_jobs`.
- **Add a classical-MDS-initialised SMACOF arm.** The paper cites SMACOF's
  init-sensitivity as its weakness and then tests only random init; classical
  init is the textbook remedy. Without this arm the comparison reads as a
  strawman.
- **Matched budget statement**: SMACOF's "hundreds of iterations" vs our 30
  epochs is not matched work. State per-iteration cost of each and compare at
  matched wall time *and* at convergence.
- **Put the 18-dataset table in the paper**, not supplemental. §4.1 carries
  the title claim in one column; it needs the results.
- Fig. 2/3: report **normalised** stress (ratios to a shared reference), not
  raw 1e7/1e8. §4.2/§4.3 already do this; §4.1 should match.
- Fig. 3 shows 3 of 18 with no stated criterion; Fig. 4 shows 6 of 18 (the
  Experiment H set). State the criterion for each or show all.

---

## 6. Phase 4 — Structure

- **Move §3.3, §3.4, §3.5 out of Methods.** They report measurements and
  forward-reference §4.2 **four** times. Either a new "§4.x Analysis of the
  sampling scheme" or split into Method / Analysis / Evaluation.
- **§5 rewrite** (drafted earlier, never merged — F11). Delete the three
  now-false limitations; add the real ones (N ≤ 50,000 bounded by the
  baseline's 84 GB; hard-middle floor; thin regime boundary; non-metric
  untested; two machines across §4.1 and §4.2/4.3; L's caveats). Close each
  ("X is out of scope because Y"), never open-ended. Add the two conclusion
  paragraphs (selective transfer; scale).
- **Re-weight or re-title.** §4.1 (title claim) is one column; §4.2/4.3 are
  three. Either give §4.1 real results (Phase 3) or re-title around budgeted
  sampling + the diagnostic. Recommend the former under this scope.
- State once that timings in §4.1 and §4.2/4.3 are from different machines
  and are not comparable across sections.

---

## 7. Phase 5 — Statistics and wording

| Item | Fix |
|---|---|
| ρ = −0.94 (F7) | State n = 6; report AG News and Tiny-ImageNet as **right-censored** (k\* > 400) rather than dropped; note they are the low-strain₂ tail and confine the claim to the ordering |
| ρ = −1.00 (F8) | Present as a consistency check on Eq. 6, not a finding |
| `strain₂` (F9) | Cite as the classical-MDS eigenvalue ratio; claim the *use*, not the quantity |
| Regime thresholds 0.22 / 0.05 | State they are descriptive on 26 datasets; no held-out validation claimed |
| California housing | Keep the §3.4 caveat; add one sentence on a robust variant (trimmed eigenvalues) as future work so it reads as scoped, not open |
| **l.257** `same embedding quality expected of an established DR algorithm` | Out of scope — rewrite as a stress statement |
| η₀ grid | Note it is inherited; one sentence is enough |

---

## 8. Phase 6 — Figures

| Fig | Fix |
|---|---|
| 1 | Replace or delete (Phase 0) |
| 2 | Add class legend; normalised stress; consider a second dataset |
| 3 | Unreadable at print; 3 of 18 unexplained → table of all 18 in §4.1, keep one exemplar |
| 4 | Fine; state why these 6 |
| 6 | Now stacked single-column; drop the per-source colour legend duplicated by Table 1, keep the regime linestyle key |
| 7 | Declutter labels; place censored points at an explicit "> 400" position and say so in the caption |
| 8 | **Label the points** by source — which dataset sits where is the interesting part |
| **new** | **At-scale embedding**: full sweep / budgeted k=200 / Pivot MDS on Fashion-MNIST at N=50,000, side by side. The one figure a vis venue will ask for; also a teaser candidate |

---

## 9. Phase 7 — Sparse SGD baseline arm (strongest remaining scientific gap)

Not required to *correct* §3.3 (Phase 2 does that by narrowing the claim).
But under a stress-only scope it is the one method that is (a) a stochastic
stress optimiser, (b) from our primary reference, (c) scalable — and the most
likely "why not compare against X" review comment. **Decide with the
supervisor.**

Cost, honestly priced:
- Region assignment `R(p)`: nearest-pivot Voronoi cell, one O(k·N·D) pass —
  the budgeted epoch already pays this.
- `s_ip`: sort each region's pivot-distances once (O(N log N)), binary-search
  `d_ip/2` per (i,p) pair (O(kN log N)). Cheap.
- **Plus** Ortmann's short-range exact term, which in vector data means a
  k-NN structure and a neighbourhood-size choice. This is the part the
  reviews under-priced. Still engineering, but budget a day, not an hour.
- Reuse the `hybrid` kernel (`hybrid_alpha`, `_sgd_mds.py:940`) for the
  budget split; new code is the pair generator + weights.

If built: compare at equal budget k·N against uniform, on stress, at the
K-opt rungs. Either outcome is a result.

---

## 10. Optional — SQuaD-MDS as context

Not a competitor under this scope (F3). If included: a third point in Fig. 8
(fast corner), scored **after optimal rescaling** and with an explicit
statement that it optimises a per-quartet relative surrogate — and that, unlike
Pivot MDS, per-quartet normalisation means **no** global rescale recovers the
stress optimum, so it is weaker evidence than the Pivot MDS comparison. Never
as "we beat SQuaD-MDS on stress" — that is the Pivot MDS fairness error again.

Its budget is the same order as ours (3 pair-terms/point/iteration × 3,000
default iterations ≈ 9,000 vs our ≈ 12,000), so "leaner" is not a claim it can
make against us either.

---

## 11. Dropped

- RNX / trustworthiness / continuity — out of scope.
- The k-NN/α experiment — its purpose was local-structure quality; F5/F6 now
  serve as the justification instead.
- "Why not UMAP/t-SNE" paragraph — reduced to one sentence: they are not
  stress optimisers.

---

## 12. Carry-over: local edits that never reached Overleaf

Drafted in earlier sessions, verified absent from the current file, all still
needed:

- §1 budget paragraphs and the `exact MDS` fix (Phase 1.2)
- §2 sampling-family paragraph (Phase 1.3 — rewrite, don't restore verbatim)
- §5 rewrite (Phase 4)
- Float fixes (Phase 0)
- `\usepackage[svgnames]{xcolor}` before `todonotes`, `\usepackage{paralist}`
  — check both survived the merge

---

## 13. Verification before upload

```bash
cd "SGD for MDS/pacificvis"
latexmk -C && latexmk -pdf main.tex
grep -E "^! |Citation .* undefined|Reference .* undefined|multiply defined|Float\(s\) lost" main.log
checkcites main.aux                          # zero uncited entries
pdftotext main.pdf - | grep -cE "^(Figure|Table) [0-9]+:"   # = number of floats defined
grep -nE "embedding quality|Anonymous|DanielHangan01|NODE-SPEC|Caption\.|todo\{" main.tex   # all zero for a blind submission
```

Then read the **uploaded** PDF end to end, not the local one.
