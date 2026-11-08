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
thread, per-fit resume.

Usage:
    python run_experiment_q.py                         # all three datasets
    python run_experiment_q.py torus --n 500,1000 --seeds 1   # smoke test
"""

from __future__ import annotations

import sys

import numpy as np

import paper_harness as H
from bench_utils import load_local_dataset

DATASETS = {"s_curve": 100, "swiss_roll": 300, "torus": 500}
N_VALUES = list(range(500, 5001, 500))
SEED_BASE = 85
N_REPEATS = 10
K = 200


def run_dataset(name, n_values, seeds, out_root):
    out_dir = out_root / "experiment_q" / name
    if H.is_complete(out_dir):
        print(f"Skipping {name} (complete).")
        return
    D = DATASETS[name]
    store = H.ResultStore(out_dir)
    H.write_meta(out_dir, dict(
        experiment="Q", dataset=name, n_features=D, n_values=n_values,
        seeds=seeds, k=K, generator=dict(random_state=SEED_BASE, target_dim=D,
                                         noise_scale=0.1), complete=False))
    print(f"\n=== Experiment Q: {name}  D={D}  ({len(store)} fits on disk) ===")

    for N in n_values:
        X, _ = load_local_dataset(name, n_samples=N, random_state=SEED_BASE,
                                  target_dim=D, noise_scale=0.1)
        X = np.ascontiguousarray(X, dtype=np.float64)
        scorer = H.ExactScorer(X)
        arms = [("smacof-random", "smacof", dict(init="random"))]
        for kind, k in (("full", None), ("random-auto", None), ("budget", K),
                        ("blocked", K)):
            cid, params = H.sgd_config(kind, N, k)
            arms.append((cid, "sgdmds", dict(params, k=k)))
        base = dict(dataset=name, n_samples=N, n_features=D, subsample_seed=None)
        for cid, algo, params in arms:
            for seed in seeds:
                if store.done(N, cid, seed):
                    continue
                if algo == "smacof":
                    fit = H.fit_smacof(X, "random", seed)
                    fields = dict(init="random", n_iter=300)
                else:
                    sgd_params = {k: v for k, v in params.items() if k != "k"}
                    fit = H.fit_sgd(X, sgd_params, seed)
                    fields = dict(k=params.get("k"),
                                  block_size=sgd_params.get("block_size"),
                                  n_updates_per_epoch=sgd_params.get(
                                      "n_updates_per_epoch"))
                history = H.score_history(fit, scorer)
                final = scorer(fit["embedding"])
                store.add(H.make_row(fit, history, final, algo=algo, config=cid,
                                     seed=seed, time_scoring=None, **base,
                                     **fields), history)
            print(f"  N={N:<5} {cid:<18} done")
    H.write_meta(out_dir, dict(complete=n_values == N_VALUES
                               and len(seeds) == N_REPEATS))
    print(f"Saved: {store.csv}")


def main():
    args, opts = H.parse_args(sys.argv[1:], {"--seeds": int, "--n": H.int_list})
    seeds = [SEED_BASE + r for r in range(opts.get("--seeds", N_REPEATS))]
    n_values = opts.get("--n", N_VALUES)
    print(f"Results root: {H.RESULTS_ROOT}")
    for name in args or list(DATASETS):
        run_dataset(name, n_values, seeds, H.RESULTS_ROOT)


if __name__ == "__main__":
    main()
