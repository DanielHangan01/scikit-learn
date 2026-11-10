# Project status and open work

## 0. Current state (2026-11-08) — read this first

Everything below §0 is the queue as of 2026-09-27 and is **superseded**: the plan
is now Experiments O, P and Q, in [EXPERIMENT_O_P_SPEC.md](EXPERIMENT_O_P_SPEC.md).
Sections 1–6 are kept as history; their queued runs (K-optimized at 70k/100k,
extended M, J/L for bank and hiva, re-timing J) are replaced by O, P and Q.

**Done**

- `research/sgd-mds` rebased on upstream `main` (1.10.dev0); old history kept under
  the tag `pre-upstream-sync`.
- Algorithm freeze (tag `algorithm-freeze`): `SGDMDS` aligned with `MDS`, GIL bug
  fixed, research options removed, snapshot hooks added. Spec §10.
- Harness and runners for O, P and Q, smoke-tested. Spec §10–§12.

**Pending**

- [x] Rewritten history (no Claude trailers) force-pushed with both tags, 2026-11-11.
- [x] Cluster details and job scripts: COMA, spec §13 and `benchmarks/cluster/`.
- [ ] Before the campaign: tag the commit to run, rsync data, `setup.sh`, smoke,
  calibrate.
- [ ] Paper: add COMA's acknowledgement sentence (spec §13).
- [ ] Open decision: SQuaD-MDS seed split (spec §13, §14).
- [ ] Still open from the old queue: the paper's wording on SQuaD-MDS (§2 below).
- [ ] Plotters for the new results tree, after the data is in.
- [ ] Off-machine copy of the paper, raw results and datasets (only a same-disk
  backup exists: `~/research-backups/sgd-mds/backup.sh`). COMA's home directory is
  backed up off-site, so copying the paper and `results/` there would cover it.
- [ ] Later: PCA init for SGDMDS (spec §3); reset `main` to upstream (ask first);
  the PR (RESEARCH_NOTES.md §6).

---

**As of 2026-09-27.** Working tree clean; last commit `7c4301d Increased grid
of experiment M`. Supervisor meeting on Experiment M is done.

This file is the work queue across experiments. Results live in the
`EXPERIMENT_*_RESULTS.md` files; paper edits are planned in
[PAPER_REVISION_PLAN.md](PAPER_REVISION_PLAN.md). Update this file when an item
moves.

---

## 1. Done

- **SQuaD-MDS implemented and validated** — [_squad_mds.py](_squad_mds.py)
  (NumPy reference) + [_squad_mds_cython.pyx](_squad_mds_cython.pyx) (fused
  kernel, 5–9× faster, agrees to 1.9e-15). Matches the authors' released code
  end-to-end to 5e-15; reproduces the paper's RNX claim against SMACOF (0.921
  ratio). 27 tests in [tests/test_squad_mds.py](tests/test_squad_mds.py).
- **Experiment M run in full on the original ladder** — 17,800 fits (16,200
  j18, 1,600 k8). Written up in
  [EXPERIMENT_M_RESULTS.md](EXPERIMENT_M_RESULTS.md).
- **Supervisor report** published as an artifact
  (<https://claude.ai/artifact/Pw3Ti8L9kMvSw6hbJao969>); meeting held.
- **Cycle-kernel overflow fixed** — `run_sgd_epoch` counted pairs in a 32-bit
  `int`, which silently turned every epoch into a no-op above N = 65,536. Now
  `Py_ssize_t` throughout `_sgd_mds_cython.pyx`.
- **j18 ladder fixed to always end at full N** — `bank` and `hiva` had been
  silently truncated to 2,000 / 3,000 of 2,059 / 3,076.
- **Point-blocked sampling implemented** (2026-09-27): `block_size`,
  `block_sampling`, `block_update` on `SGDMDS`, and the new kernel
  `run_sgd_epoch_lazy_blocked`. Every earlier run still reproduces bitwise:
  there are golden fixtures from 7c4301d, and the original kernels are
  untouched and still the default. 95 tests in
  [tests/test_blocked_sgdmds.py](tests/test_blocked_sgdmds.py). Experiment N
  is designed around it: [EXPERIMENT_N_SPEC.md](EXPERIMENT_N_SPEC.md).

### Headline findings from Experiment M

- SQuaD-MDS reaches **1.04× cycle stress** (j18 median) / **1.10×** (k8), at
  **7–14× less wall time** than SGD k=200 (k8, both sides timed cleanly).
- The authors' shipped default is **1.53×** — almost entirely one code-only
  flag, `exaggerate_d`, tuned for RNX.
- It **beats budgeted SGD on ag_news and tiny_imagenet**, the low-`strain_top2`
  datasets where our fast mode plateaus.
- It **fails on heavy-tailed data** (mnist_digits 1.52×, california_housing
  4.60×). Predictable before fitting: `p99.9 / median` pairwise distance,
  ρ = +0.53, p = 0.0085; ≤ 4 guaranteed a gap under 1.11× on 17 of 17.
- Its speed is **cache blocking** (4 rows → 6 pairs), separable from its
  objective — see §5.

---

## 2. Decision needed: the paper's SQuaD-MDS claims

Experiment M was specified to test the relaxation claim and to report plainly
if SQuaD-MDS landed close to a full sweep. It did. Two places are affected.

**The applied paper text** (`SGD for MDS/pacificvis/main.tex`):

| line | text | status |
|---|---|---|
| l.117 (abstract) | "the stochastic alternatives that do scale optimize a relaxation of stress rather than stress itself" | true of the *objective*; implies a stress gap M did not find (1.04×) |
| l.156 (§1) | "…SQuaD-MDS optimizes a scale-free per-quartet surrogate … Stress itself is left to the batch solver." | same |
| l.688 (conclusion) | "…it estimates the stress objective without bias, where prior approaches substitute a relaxation of it." | true, but a reviewer who runs SQuaD-MDS finds near-equal stress at ~1/11 the time |

Under the supervisor's scope (§0 of the plan: stress-only claim, relaxed
methods are context), these sentences are *defensible as objective-level
statements*. The question is whether they survive a reviewer who measures.

**[PAPER_REVISION_PLAN.md](PAPER_REVISION_PLAN.md) §10**, two claims now
contradicted by measurement:

- *"per-quartet normalisation means **no** global rescale recovers the stress
  optimum, so it is weaker evidence than the Pivot MDS comparison"* — measured
  the opposite: α is tightly clustered per dataset, a single global rescale
  works, and SQuaD-MDS (1.04×) is far closer to the optimum than Pivot MDS
  (3.34×).
- *"Its budget is the same order as ours (… 3,000 default iterations ≈ 9,000
  vs our ≈ 12,000), so 'leaner' is not a claim it can make against us"* — the
  reference default is **1,000** iterations, not 3,000, and measured wall time
  is **7–14× lower** than k=200.

- [ ] Decide wording for l.117 / l.156 / l.688 (possibly with supervisor).
- [ ] Correct PAPER_REVISION_PLAN.md §10.

---

## 3. Queued runs — configured and committed, not yet run

Order matters: K-optimized before M's k8 70k/100k rows (they need its SGD
numbers); J and L before M's bank/hiva ratios.

**On any machine other than this one** (e.g. the cluster), rebuild first — the
`Py_ssize_t` fix and the new SQuaD kernel are both `.pyx` changes:

```bash
pip install -e . --no-build-isolation
python -c "from sklearn.manifold._squad_mds_cython import run_squad_iteration; print('ok')"
```

### 3.1 K-optimized at 70k and 100k — budget arm only

- [ ] **Fix `RESULTS_DIR`** in `benchmarks/run_experiment_k_optimized.py`: it
  points at `experiment_k_optimized_studentvm` (doesn't exist), while L and M
  read `experiment_k_optimized` (in `_reference_cycle_stress`). Run as-is and
  it recomputes all 8 datasets from scratch into a folder nothing reads.
- [ ] `rm results/experiment_k_optimized/*/meta.json` (keep `results.csv`).
- [ ] `python run_experiment_k_optimized.py`

Cycle is skipped above `CYCLE_MAX_N = 50_000` (needs 164 GB at 70k, 335 GB at
100k), so those rungs have no overhead ratio or k\* — timings and cross-method
comparison only. On a node with ≥ 164 GB, set `CYCLE_MAX_N = 70_000` to get a
real full-N baseline for fashion_mnist_full and mnist_digits.

### 3.2 Experiment M — extended `n_iter` ladder

Every best config sat at the old ceiling (2000 j18, 1000 k8) with quality still
improving, so current quality numbers are **lower bounds** and speedups
**upper bounds**. Now j18 → 8000, k8 → 4000.

- [ ] `rm results/experiment_m_j18/*/meta.json results/experiment_m_k8/*/meta.json`
- [ ] `python run_experiment_m.py --suite j18` (~ +3.8 h)
- [ ] `python run_experiment_m.py --suite k8` (~ +10 h)

Resume is per fit (verified): only the new `n_iter` values run.

### 3.3 Experiment N — DONE (2026-09-30)

Microbenchmark, j18 and k8 (lean 5-arm: random, sequential, m in {2,4,8,16}) all
run. Results: [EXPERIMENT_N_RESULTS.md](EXPERIMENT_N_RESULTS.md). Follow-ups:

- [ ] Decide the `noexcept` fix for the original kernel (§4)
- [ ] Test PCA initialisation for SGD-MDS — the likely source of SQuaD-MDS's
  per-pair quality edge (SQuaD random-init is 1.15–1.20× worse)

### 3.4 J and L for bank / hiva at full N

- [ ] `rm -rf results/experiment_j/{bank,hiva}` then
  `python run_experiment_j.py bank hiva` — J has **no per-fit resume**, so this
  redoes both datasets' whole ladder; hiva (D = 1617) is slow.
- [ ] `rm -rf results/experiment_l_j18/{bank,hiva}` then
  `python run_experiment_l.py --suite j18 bank hiva` — cheap.

---

## 4. Known gaps

- [ ] **Experiment J timings are contaminated.** It never passes
  `compute_stress=False`, so SGD `time_solver` includes in-fit O(N²) scoring —
  up to ~74 % of the time on low-D / high-N data. Every Suite A speedup in L and
  M is inflated, including EXPERIMENT_L_RESULTS.md §3's "7.5–70.4×, median
  19.1×". Fix: re-time J's cycle and budget arms with `compute_stress=False`
  into a sidecar directory (stress unchanged — only the clock). Minutes of
  compute, repairs both experiments. Check whether the paper quotes any L Suite
  A speedup.
- [ ] **The original lazy kernels pay a GIL round-trip per pair update.**
  `_lazy_sgd_step` is not `noexcept`, so Cython checks for an exception,
  under the GIL, after every call. It costs ~35 ns per update: 3.5× at D=3
  in the microbenchmark, and 3.3× on a full k=200 fit (N=3,000, D=8), with
  a bit-identical embedding. Consequences:
  - The K-optimized cost model's ~52 ns intercept is mostly this overhead.
  - Experiment M compared SQuaD-MDS (no overhead) against SGD (overhead),
    which tilts it against SGD at low D.
  - The fix is one keyword and changes no results, but it changes the
    timing of every past budget run, so it is **left for you to decide**.
    Experiment N measures it directly (the `fix` column); decide after that.
- [ ] **`orl` is a defective dataset** — 360 of its 400 rows are identical. It
  wrecks SQuaD-MDS (§6 of M's results) and likely distorts orl's numbers in J,
  K and L too. Decide whether to drop or footnote it everywhere.
- [ ] **lr = 1500 may be a ceiling, not an optimum** — all five lr-sensitive
  j18 datasets (cnae9, sms, secom, epileptic, spambase) picked the highest value
  tested. Check lr ∈ {3000, 5000} on those five at `n_iter = 2000`.
- [ ] **CLAUDE.md does not mention Experiment M** — add it to the document map
  (§2) and benchmark table (§7), plus a pointer to this file.
- [ ] **Update EXPERIMENT_M_RESULTS.md** after §3.2 lands: §2, §3, §5.1 and the
  ladder-limited caveat in §7 all quote pre-extension numbers.

---

## 5. Research leads

### Point-blocked sampling for our own SGD (highest value) — now Experiment N (§3.3)

SQuaD-MDS is faster because four loaded rows yield six pairs. Our sampler draws
`i, j` independently — two rows, one pair. Drawing *m* points and using all
m(m−1)/2 pairs costs `2D/(m−1)` loads per pair:

| m | loads / pair | vs current |
|--:|--:|--:|
| 2 (ours) | 2.00 D | 1× |
| 4 (SQuaD) | 0.67 D | 3× |
| 8 | 0.29 D | 7× |
| 16 | 0.13 D | 15× (register spill) |

It stays SGD-MDS: same stress objective, estimator still unbiased (uniform
point draws give every pair equal inclusion probability), same sequential
update. **Open question:** do correlated in-block pairs cost statistical
efficiency? Test: fix `n_updates_per_epoch`, sweep m ∈ {2, 4, 8, 16}, compare
stress at equal pair count on a handful of j18 datasets. Either outcome is
publishable. Lazy mode only — cycle's O(1) matrix lookup has nothing to
amortise.

### Minor

- `pca_participation_ratio` → best lr (ρ = +0.54, p = 0.038) fails multiple-
  comparison correction; worth a targeted test only.
- Two RNG integer modulos per update in `run_sgd_epoch_lazy_random_native`
  plausibly cost 20–40 % of an update at low D. Multiply-shift would remove it
  (it would change the random stream, so it is not bit-compatible).
- **Look-ahead δ for plain random SGD.** The feature-space distance never
  depends on the embedding. So a kernel can draw the next 4 pairs, compute
  their 4 δ with independent running sums, then apply the 4 updates in
  order. The random draws, the order and the arithmetic are all unchanged,
  so it is **bit-identical to today's m=2**, but it hides add latency and
  overlaps row fetches. It is the pure-implementation half of blocking, with
  no correlated pairs; compare it against m=4 in Experiment N.

---

## 6. Suggested order

1. Fix K-optimized `RESULTS_DIR` (§3.1)
2. Decide the paper wording (§2) — independent of all compute
3. Experiment N (§3.3): microbenchmark, then j18. Cheap, and it decides the
   GIL-fix question (§4) before more budget runs are timed on the old kernel.
4. K-optimized 70k/100k → M extended ladder → J/L bank + hiva (§3)
5. Re-time J (§4) — cheap, can run any time
6. Update EXPERIMENT_M_RESULTS.md and CLAUDE.md
