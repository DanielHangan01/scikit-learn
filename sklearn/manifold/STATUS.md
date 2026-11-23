# Project status and open work

## 0. Current state (2026-11-08) — read this first

The plan is Experiments O, P and Q, in [EXPERIMENT_O_P_SPEC.md](EXPERIMENT_O_P_SPEC.md).
The earlier queue (§1–6, as of 2026-09-27) and the docs and code of Experiments A–N
are on the branch [`research/history`](https://github.com/DanielHangan01/scikit-learn/blob/research/history/sklearn/manifold/STATUS.md); O, P and Q replaced its
queued runs.

**Done**

- `research/sgd-mds` rebased on upstream `main` (1.10.dev0); old history kept under
  the tag `pre-upstream-sync`.
- Algorithm freeze (tag `algorithm-freeze`): `SGDMDS` aligned with `MDS`, GIL bug
  fixed, research options removed, snapshot hooks added. Spec §10.
- Harness and runners for O, P and Q, smoke-tested. Spec §10–§12.

**Pending**

- [x] Rewritten history (no Claude trailers) force-pushed with both tags, 2026-11-11.
- [x] Cluster details and job scripts: COMA, spec §13 and `benchmarks/cluster/`.
- [x] Cluster setup, smoke test and calibration done (2026-11-11; spec §12–§13).
- [x] Pilot and campaign: O, P and Q complete (2026-11-17), reproduction checked
  (spec §12). Timing outliers accepted with a 15% "comparable" rule.
- [x] Results copied to `benchmarks/results_paper/`; analysis in
  [EXPERIMENT_O_P_Q_RESULTS.md](EXPERIMENT_O_P_Q_RESULTS.md) (2026-11-18), from
  `benchmarks/analyze_experiment_o_p_q.py`. Figures still to do (proposed in its §8).
- [ ] Decisions from the analysis (EXPERIMENT_O_P_Q_RESULTS.md §7): the full
  sweep's per-epoch shuffle (~95% of its time at large N), scoping step 1 against
  Q (SMACOF faster above N≈3,500 where it converges early), coil20's timing.
- [ ] Open issues from the first look: SQuaD-MDS's best setting sits at the top of
  its `n_iter` ladder almost everywhere (its quality is a bound); choose its setting
  on seeds 42-46 and report on 47-51 (spec §14); classical-init SMACOF beats full
  SGDMDS on stress on 9 of 17 small datasets (later study).
- [ ] Paper: add COMA's acknowledgement sentence (spec §13).
- [ ] Open decision: SQuaD-MDS seed split (spec §13, §14).
- [ ] Still open from the old queue: the paper's wording on SQuaD-MDS (§2 of the
  old queue, on `research/history`).
- [ ] Plotters for the new results tree, after the data is in.
- [ ] Off-machine copy of the paper, raw results and datasets (only a same-disk
  backup exists: `~/research-backups/sgd-mds/backup.sh`). COMA's home directory is
  backed up off-site, so copying the paper and `results/` there would cover it.
- [ ] Later: classical (or PCA) init for SGDMDS, compared with classical-init
  SMACOF, whose data the campaign already collects. The main comparison uses
  random init for both (spec §13). Pilot hint: classical SMACOF reaches
  1.007–1.023x the full sweep's stress, against 1.05–1.11x with random init.
- [ ] Later: PCA init for SGDMDS (spec §3); reset `main` to upstream (ask first);
  the PR (RESEARCH_NOTES.md §6).

