"""
Experiment D: Pivot Sampling vs SGD Baselines (Convergence + Quality Study).

For each dataset, runs four SGD variants over N_INITS seeds and records
stress history + final stress:

  1. SGD-cycle      sampling_strategy='cycle',  dissimilarity='euclidean'
  2. SGD-lazy       sampling_strategy='random',  dissimilarity='lazy'
  3. SGD-pivot-lazy sampling_strategy='pivot',   dissimilarity='lazy',   n_pivots=N_PIVOTS
  4. SGD-pivot-euc  sampling_strategy='pivot',   dissimilarity='euclidean', n_pivots=N_PIVOTS

No SMACOF. Saves results.json + best embedding per algorithm (.npy).

Usage:
    python run_experiment_d.py                   # all datasets
    python run_experiment_d.py coil20 s_curve    # specific datasets
"""

import sys
import json
import numpy as np
from time import time
from pathlib import Path

import bench_utils
from bench_utils import load_local_dataset, run_sgd_benchmark, run_pivot_benchmark

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_d"

N_INITS = 10
MAX_ITER = 30
N_PIVOTS = 50           # fixed k for both pivot variants

SGD_LR = 0.5
SGD_SWITCH = 0.5
SGD_EPSILON = 0.001

DEFAULT_DATASETS = ["coil20", "epileptic", "fashion_mnist", "s_curve"]


def get_all_datasets():
    data_root = bench_utils.DATASET_ROOT
    found = [d.name for d in data_root.iterdir() if d.is_dir()] if data_root.exists() else []
    for synthetic in ("s_curve", "swiss_roll", "torus"):
        if synthetic not in found:
            found.append(synthetic)
    return found


def _clean_history(hist):
    if not hist:
        return []
    return [[float(t), float(s)] for t, s in hist]


def run_dataset(dataset_name):
    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "results.json").exists():
        print(f"Skipping {dataset_name} (results.json exists).")
        return

    print(f"\n=== Experiment D: {dataset_name} ===")
    X, y = load_local_dataset(dataset_name)
    N = X.shape[0]

    # Resolve actual k (same logic as _resolve_n_pivots for comparability)
    actual_k = min(N_PIVOTS, max(3, N // 20))

    results_store = {
        "dataset": dataset_name,
        "n_samples": N,
        "n_features": X.shape[1],
        "n_pivots": actual_k,
        "runs": [],
    }

    best = {algo: {"stress": float("inf"), "emb": None}
            for algo in ("SGD-cycle", "SGD-lazy", "SGD-pivot-lazy", "SGD-pivot-euc")}

    for seed_i in range(N_INITS):
        seed = 42 + seed_i
        print(f"  [Seed {seed}]")

        common = dict(random_state=seed, max_iter=MAX_ITER,
                      switch_ratio=SGD_SWITCH, lr=SGD_LR, epsilon=SGD_EPSILON)

        runs = [
            ("SGD-cycle",      run_sgd_benchmark(X, dissimilarity="euclidean", **common)),
            ("SGD-lazy",       run_sgd_benchmark(X, dissimilarity="lazy",      **common)),
            ("SGD-pivot-lazy", run_pivot_benchmark(X, dissimilarity="lazy",      n_pivots=N_PIVOTS, **common)),
            ("SGD-pivot-euc",  run_pivot_benchmark(X, dissimilarity="euclidean", n_pivots=N_PIVOTS, **common)),
        ]

        for label, r in runs:
            print(f"    {label:20s} stress={r['stress']:.2f}  time={r['time']:.2f}s")

            if r["stress"] < best[label]["stress"]:
                best[label]["stress"] = r["stress"]
                best[label]["emb"] = r["embedding"]

            entry = {
                "algo": label,
                "seed": seed,
                "stress": float(r["stress"]),
                "time": float(r["time"]),
                "history": _clean_history(r["history"]),
            }
            if "n_pivots" in r:
                entry["n_pivots"] = r["n_pivots"]
            results_store["runs"].append(entry)

    print(f"\n  Saving to {out_dir}")
    with open(out_dir / "results.json", "w") as f:
        json.dump(results_store, f)

    for label, bst in best.items():
        if bst["emb"] is not None:
            safe = label.replace("/", "_").replace(" ", "_")
            np.save(out_dir / f"best_embedding_{safe}.npy", bst["emb"])

    if y is not None:
        np.save(out_dir / "y_labels.npy", y)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
    else:
        datasets = DEFAULT_DATASETS

    for ds in datasets:
        run_dataset(ds)
