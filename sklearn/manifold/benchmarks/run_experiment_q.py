"""
Experiment Q: runtime vs N (EXPERIMENT_O_P_SPEC.md §5a). Replaces Experiment B.

The same data as B: s_curve, swiss_roll and torus lifted to D = 100, 300 and 500
with noise dimensions (load_local_dataset(..., random_state=85, noise_scale=0.1)),
at N = 500, 1000, ..., 5000, with 10 repeats (seeds 85-94). Arms:

  smacof-random     SMACOF's default
  full              SGDMDS(), all pairs per epoch
  random-auto-m2    random pairs at the full pair budget (B's lazy arm)
  budget-k200       200*N random pairs per epoch, block_size=2
  blocked-k200-m16  the same budget in blocks of 16 points

Same measurement protocol as Experiment O: exact stress histories, single
thread, per-fit resume, results tied to the code of their first run.

Usage:
    python run_experiment_q.py                         # all three datasets
    python run_experiment_q.py torus --n 500,1000 --seeds 1   # smoke test
"""

from __future__ import annotations

import sys

import numpy as np

import paper_harness as H
from bench_utils import load_local_dataset

EXPERIMENT = "experiment_q"
DATASETS = {"s_curve": 100, "swiss_roll": 300, "torus": 500}
N_VALUES = list(range(500, 5001, 500))
SEED_BASE = 85
N_REPEATS = 10
SEEDS = [SEED_BASE + r for r in range(N_REPEATS)]
K = 200


def load_input(name, n_samples):
    """B's generated data at this N, and its subsample seed (none here)."""
    X, _ = load_local_dataset(name, n_samples=n_samples, random_state=SEED_BASE,
                              target_dim=DATASETS[name], noise_scale=0.1)
    return np.ascontiguousarray(X, dtype=np.float64), None


def arms_for(n_samples, seeds=SEEDS):
    """(config id, algo, parameters, seeds) for every arm at this N."""
    arms = [("smacof-random", "smacof", dict(init="random"), seeds)]
    for kind, k in (("full", None), ("random-auto", None), ("budget", K),
                    ("blocked", K)):
        cid, params = H.sgd_config(kind, n_samples, k)
        arms.append((cid, "sgdmds", dict(params, k=k), seeds))
    return arms


def run_dataset(name, n_values, seeds, out_root):
    out_dir = out_root / EXPERIMENT / name
    if H.is_complete(out_dir):
        print(f"Skipping {name} (complete).")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    H.check_provenance(out_dir)
    D = DATASETS[name]
    store = H.ResultStore(out_dir)
    H.write_meta(out_dir, dict(
        experiment="Q", dataset=name, n_features=D, n_values=n_values,
        seeds=seeds, k=K, generator=dict(random_state=SEED_BASE, target_dim=D,
                                         noise_scale=0.1), complete=False))
    print(f"\n=== Experiment Q: {name}  D={D}  ({len(store)} fits on disk) ===")
    canary = H.Canary(out_dir)

    for N in n_values:
        X, _ = load_input(name, N)
        scorer = H.ExactScorer(X)
        base = dict(dataset=name, n_samples=N, n_features=D, subsample_seed=None)
        for cid, algo, params, arm_seeds in arms_for(N, seeds):
            for seed in arm_seeds:
                if store.done(N, cid, seed):
                    continue
                timing = canary.maybe_run()
                fit, fields = H.run_fit(algo, X, params, seed)
                history = H.score_history(fit, scorer)
                final = scorer(fit["embedding"])
                store.add(H.make_row(fit, history, final, algo=algo, config=cid,
                                     seed=seed, time_scoring=None, **base,
                                     **fields, **timing), history)
            print(f"  N={N:<5} {cid:<18} done")
    H.write_meta(out_dir, dict(complete=n_values == N_VALUES
                               and len(seeds) == N_REPEATS))
    print(f"Saved: {store.csv}")


def main():
    args, opts = H.parse_args(sys.argv[1:], {"--seeds": int, "--n": H.int_list})
    seeds = SEEDS[: opts.get("--seeds", N_REPEATS)]
    n_values = opts.get("--n", N_VALUES)
    print(f"Results root: {H.RESULTS_ROOT}")
    for name in args or list(DATASETS):
        run_dataset(name, n_values, seeds, H.RESULTS_ROOT)


if __name__ == "__main__":
    main()
