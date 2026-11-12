"""
Experiment P: the large suite, no SMACOF sweep (EXPERIMENT_O_P_SPEC.md §5).

The 8 K-optimized sources on the ladder {10k, 20k, 35k, 50k} (california_housing
stops at its full 20,640), with K-optimized's fixed subsample per (dataset, N),
so the budget arm reproduces K-optimized's stress. Arms per rung:

  full              SGDMDS(), seed 42 only (42-44 at N=10k); ~50*N^2 bytes
  budget-k{k}       k in {100, 200, 400}, block_size=2, seeds 42-46
  blocked-k{k}-m16  the same budget in blocks of 16 points, seeds 42-46
  squad-...         Experiment M's k8 grid (12 configurations), seeds 42-46
  smacof-...        default and classical_mds init, N=10k only, seeds 42-44

Histories are estimated from a fixed sample of 10^7 pairs per (dataset, N);
every final embedding is also scored exactly, in batches outside the timed
region, and the estimate's error at the last snapshot is recorded. A fit is
written once its exact score is in, so batches are scored at least every
SCORE_EVERY_S seconds of fitting.

Set SGDMDS_MAX_MEM_GB to skip full-sweep fits whose estimated peak memory
exceeds the node's RAM; the dataset then stays incomplete, so a rerun on a
bigger node resumes it.

Usage:
    python run_experiment_p.py                       # all 8 datasets
    python run_experiment_p.py cifar10_raw           # one dataset (one job each)
    python run_experiment_p.py feynman_iii_15_12 --ladder 2000 --seeds 1 --quick
    python run_experiment_p.py --list
"""

from __future__ import annotations

import os
import sys

import numpy as np

import paper_harness as H
import run_experiment_k_optimized as KO
from bench_utils import load_local_dataset
from run_experiment_m import GRID_K8

EXPERIMENT = "experiment_p"
DATASETS = KO.DATASETS
MAX_N = 50_000
SEEDS = [42, 43, 44, 45, 46]
FULL_SEEDS = {10_000: [42, 43, 44]}
FULL_SEEDS_DEFAULT = [42]
K_VALUES = [100, 200, 400]
SQUAD_GRID = GRID_K8
SMACOF_N = 10_000
SMACOF_INITS = ["random", "classical_mds"]
SMACOF_SEEDS = [42, 43, 44]
N_SCORE_PAIRS = 10**7
SCORE_CHUNK = 16
SCORE_EVERY_S = 900
MAX_MEM_GB = (float(os.environ["SGDMDS_MAX_MEM_GB"])
              if "SGDMDS_MAX_MEM_GB" in os.environ else None)

QUICK = dict(k=[200], smacof=["random"],
             squad=dict(n_iter=[100], lr=[550.0], exaggerate_d=[False],
                        init=["pca"], max_step_frac=[4.0]))


def cycle_peak_gb(n: int) -> float:
    """Peak memory of a full-sweep fit: 20.9 GB measured at N=20,640 on COMA."""
    return 50.0 * n * n / 2**30


def load_input(name, n_samples):
    """K-optimized's fixed subsample at this N, and its subsample seed."""
    X_full, _ = load_local_dataset(name)
    X_full = np.ascontiguousarray(X_full, dtype=np.float64)
    if n_samples >= X_full.shape[0]:
        return X_full, None
    idx = np.random.RandomState(KO.SUBSAMPLE_SEED).choice(
        X_full.shape[0], size=n_samples, replace=False)
    return np.ascontiguousarray(X_full[idx]), KO.SUBSAMPLE_SEED


def _subset(wanted, seeds):
    return [s for s in wanted if s in seeds] or wanted[:1]


def arms_for(n_samples, seeds=SEEDS, ks=K_VALUES, smacof_inits=SMACOF_INITS,
             squad_grid=SQUAD_GRID):
    """(config id, algo, parameters, seeds) for every arm at this N."""
    cid, params = H.sgd_config("full", n_samples)
    arms = [(cid, "sgdmds", params,
             _subset(FULL_SEEDS.get(n_samples, FULL_SEEDS_DEFAULT), seeds))]
    for kind in ("budget", "blocked"):
        for k in H.valid_k(n_samples, ks):
            cid, params = H.sgd_config(kind, n_samples, k)
            arms.append((cid, "sgdmds", dict(params, k=k), seeds))
    arms += [(cid, "squad", p, seeds) for cid, p in H.squad_configs(squad_grid)]
    if n_samples == SMACOF_N:
        arms += [(f"smacof-{init}", "smacof", dict(init=init),
                  _subset(SMACOF_SEEDS, seeds)) for init in smacof_inits]
    return arms


def run_rung(name, X, N, D, sub_seed, arms, store, canary, top_rung,
             n_score_pairs):
    """All arms at one N. Returns False if a full-sweep fit was memory-skipped."""
    complete = True
    print(f"\n  --- N={N:,}: building the {n_score_pairs:,}-pair scorer ---")
    scorer = H.PairSampleScorer(X, n_pairs=n_score_pairs, seed=0)
    if MAX_MEM_GB is not None and cycle_peak_gb(N) > MAX_MEM_GB:
        print(f"      full sweep SKIPPED: ~{cycle_peak_gb(N):.0f} GB > "
              f"SGDMDS_MAX_MEM_GB={MAX_MEM_GB:g}")
        arms = [a for a in arms if a[0] != "full"]
        complete = False

    base = dict(dataset=name, n_samples=N, n_features=D, subsample_seed=sub_seed)
    pending = []

    def flush():
        if not pending:
            return
        finals = H.exact_final_scores(X, [p["fit"]["embedding"] for p in pending])
        for p, final in zip(pending, finals):
            row = H.make_row(p["fit"], p["history"], final, **p["fields"])
            save = top_rung and p["fields"]["seed"] == 42
            store.add(row, p["history"], p["fit"]["embedding"] if save else None)
        pending.clear()

    for cid, algo, params, seed in H.seed_major(arms):
        if store.done(N, cid, seed):
            continue
        timing = canary.maybe_run()
        fit, fields = H.run_fit(algo, X, params, seed)
        history = H.score_history(fit, scorer)
        fit["snapshots"] = []
        pending.append(dict(fit=fit, history=history, fields=dict(
            base, algo=algo, config=cid, seed=seed, time_scoring=None,
            **fields, **timing)))
        print(f"    {cid:<36} seed={seed}  {fit['time_fit']:9.2f}s")
        if (len(pending) >= SCORE_CHUNK
                or sum(p["fit"]["time_fit"] for p in pending) >= SCORE_EVERY_S):
            flush()
    flush()
    return complete


def run_dataset(name, seeds, ks, smacof_inits, squad_grid, ladder_override,
                n_score_pairs, out_root):
    out_dir = out_root / EXPERIMENT / name
    if H.is_complete(out_dir):
        print(f"Skipping {name} (complete).")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    H.check_provenance(out_dir)
    X_full, _ = load_input(name, np.inf)
    N_full, D = X_full.shape
    del X_full
    ladder = ladder_override or [n for n in KO.make_ladder(N_full) if n <= MAX_N]
    store = H.ResultStore(out_dir)
    H.write_meta(out_dir, dict(
        experiment="P", dataset=name, n_samples=N_full, n_features=D,
        ladder=ladder, seeds=seeds, k_values=ks, smacof_inits=smacof_inits,
        squad_grid=squad_grid, n_score_pairs=n_score_pairs,
        subsample_seed=KO.SUBSAMPLE_SEED, max_mem_gb=MAX_MEM_GB,
        dataset_sha256=H.dataset_checksum(name), complete=False))
    print(f"\n=== Experiment P: {name}  N_full={N_full:,}  D={D}  ladder={ladder}"
          f"  ({len(store)} fits on disk) ===")
    canary = H.Canary(out_dir)

    complete = True
    for N in ladder:
        X, sub_seed = load_input(name, N)
        arms = arms_for(N, seeds, ks, smacof_inits, squad_grid)
        complete &= run_rung(name, X, N, D, sub_seed, arms, store, canary,
                             top_rung=(N == ladder[-1]),
                             n_score_pairs=n_score_pairs)
    H.write_meta(out_dir, dict(complete=complete and not ladder_override))
    print(f"Saved: {store.csv}")


def main():
    args, opts = H.parse_args(sys.argv[1:], {
        "--seeds": int, "--quick": bool, "--list": bool,
        "--ladder": H.int_list, "--score-pairs": int})
    if opts.get("--list"):
        print("Experiment P datasets:", ", ".join(DATASETS))
        print(f"max N={MAX_N:,}  seeds={SEEDS}  k={K_VALUES}  "
              f"squad configs={len(H.squad_configs(SQUAD_GRID))}  "
              f"SMACOF at N={SMACOF_N:,}")
        for n in (10_000, 20_000, 35_000, 50_000):  # measured ~49 bytes per N^2
            print(f"  full sweep at N={n:,}: ~{cycle_peak_gb(n):.0f} GB")
        return
    quick = opts.get("--quick", False)
    seeds = SEEDS[: opts.get("--seeds", len(SEEDS))]
    print(f"Results root: {H.RESULTS_ROOT}")
    for name in args or DATASETS:
        run_dataset(name, seeds,
                    QUICK["k"] if quick else K_VALUES,
                    QUICK["smacof"] if quick else SMACOF_INITS,
                    QUICK["squad"] if quick else SQUAD_GRID,
                    opts.get("--ladder"),
                    opts.get("--score-pairs", N_SCORE_PAIRS),
                    H.RESULTS_ROOT)


if __name__ == "__main__":
    main()
