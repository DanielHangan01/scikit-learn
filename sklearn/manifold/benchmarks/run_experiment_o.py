"""
Experiment O: the small suite, all five algorithms (EXPERIMENT_O_P_SPEC.md §4).

17 datasets (J's 18 minus orl) at full N, one fixed X per dataset shared by
every method and seed. Arms:

  SMACOF            init in {random, classical_mds}, max_iter=300, eps=1e-6
  full              SGDMDS() -- the golden configuration, all pairs per epoch
  budget-k{k}       k*N random pairs per epoch, block_size=2, k in {50..400}
  blocked-k{k}-m16  the same budget in blocks of 16 points
  squad-...         Experiment M's j18 grid (252 configurations)

10 seeds (42-51). Stress is exact throughout: SGD and SQuaD-MDS histories are
scored from snapshots after each fit, SMACOF's comes from its own loop. Final
embeddings are saved for seed 42.

Per-fit resume; a dataset is skipped once its meta.json says complete.

Usage:
    python run_experiment_o.py                     # all 17 datasets
    python run_experiment_o.py coil20 hiva         # subset
    python run_experiment_o.py coil20 --seeds 1 --quick   # smoke test
    python run_experiment_o.py --list
"""

from __future__ import annotations

import sys

import numpy as np

import paper_harness as H
from bench_utils import load_local_dataset
from run_experiment_j import ALL_DATASETS as J_DATASETS
from run_experiment_m import GRID_J18

DATASETS = [d for d in J_DATASETS if d != "orl"]
SEEDS = list(range(42, 52))
K_VALUES = [50, 100, 200, 400]
SMACOF_INITS = ["random", "classical_mds"]
SQUAD_GRID = GRID_J18
SAVE_SEED = 42

QUICK = dict(k=[200], smacof=["random"],
             squad=dict(n_iter=[100], lr=[550.0], exaggerate_d=[False],
                        init=["pca"], max_step_frac=[4.0]))


def run_dataset(name: str, seeds, ks, smacof_inits, squad_grid, out_root):
    out_dir = out_root / "experiment_o" / name
    if H.is_complete(out_dir):
        print(f"Skipping {name} (complete).")
        return
    X, _ = load_local_dataset(name)
    X = np.ascontiguousarray(X, dtype=np.float64)
    N, D = X.shape
    store = H.ResultStore(out_dir)
    H.write_meta(out_dir, dict(
        experiment="O", dataset=name, n_samples=N, n_features=D,
        seeds=seeds, k_values=ks, smacof_inits=smacof_inits,
        squad_grid=squad_grid, dataset_sha256=H.dataset_checksum(name),
        complete=False))
    print(f"\n=== Experiment O: {name}  N={N}  D={D}  "
          f"({len(store)} fits on disk) ===")
    scorer = H.ExactScorer(X)

    arms = [(f"smacof-{init}", "smacof", dict(init=init)) for init in smacof_inits]
    cid, params = H.sgd_config("full", N)
    arms.append((cid, "sgdmds", params))
    for kind in ("budget", "blocked"):
        for k in H.valid_k(N, ks):
            cid, params = H.sgd_config(kind, N, k)
            arms.append((cid, "sgdmds", dict(params, k=k)))
    arms += [(cid, "squad", params) for cid, params in H.squad_configs(squad_grid)]

    base = dict(dataset=name, n_samples=N, n_features=D, subsample_seed=None)
    for cid, algo, params in arms:
        todo = [s for s in seeds if not store.done(N, cid, s)]
        if not todo:
            continue
        for seed in todo:
            if algo == "smacof":
                fit = H.fit_smacof(X, params["init"], seed)
                fields = dict(init=params["init"], n_iter=300)
            elif algo == "sgdmds":
                sgd_params = {k: v for k, v in params.items() if k != "k"}
                fit = H.fit_sgd(X, sgd_params, seed)
                fields = dict(k=params.get("k"),
                              block_size=sgd_params.get("block_size"),
                              n_updates_per_epoch=sgd_params.get(
                                  "n_updates_per_epoch"))
            else:
                fit = H.fit_squad(X, params, seed)
                fields = dict(params)
            history = H.score_history(fit, scorer)
            final = scorer(fit["embedding"])
            row = H.make_row(fit, history, final, algo=algo, config=cid,
                             seed=seed, time_scoring=None, **base, **fields)
            store.add(row, history,
                      fit["embedding"] if seed == SAVE_SEED else None)
            print(f"  {cid:<40} seed={seed}  {fit['time_fit']:8.2f}s  "
                  f"stress={final[0]:.6g}")

    H.write_meta(out_dir, dict(complete=True))
    print(f"Saved: {store.csv}")


def main():
    args, opts = H.parse_args(sys.argv[1:], {"--seeds": int, "--quick": bool,
                                             "--list": bool})
    if opts.get("--list"):
        print("Experiment O datasets:", ", ".join(DATASETS))
        print(f"seeds={SEEDS}  k={K_VALUES}  smacof={SMACOF_INITS}  "
              f"squad configs={len(H.squad_configs(SQUAD_GRID))}")
        return
    quick = opts.get("--quick", False)
    seeds = SEEDS[: opts.get("--seeds", len(SEEDS))]
    ks = QUICK["k"] if quick else K_VALUES
    smacof_inits = QUICK["smacof"] if quick else SMACOF_INITS
    grid = QUICK["squad"] if quick else SQUAD_GRID
    print(f"Results root: {H.RESULTS_ROOT}")
    for name in args or DATASETS:
        run_dataset(name, seeds, ks, smacof_inits, grid, H.RESULTS_ROOT)


if __name__ == "__main__":
    main()
