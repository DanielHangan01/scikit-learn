"""
Experiment D: Pivot-euclidean vs SGD-cycle (Convergence + Quality Study).

For each dataset, runs SGD-cycle once and SGD-pivot-euclidean at several
pivot counts (fractions of N), recording stress history + final stress:

  1. SGD-cycle     sampling_strategy='cycle',  dissimilarity='euclidean'
  2. SGD-pivot-euc sampling_strategy='pivot',  dissimilarity='euclidean', k in K_FRACTIONS * N

Saves results.json + best embedding per (algorithm, k) (.npy).

Usage:
    python run_experiment_d.py                   # all datasets
    python run_experiment_d.py coil20 s_curve    # specific datasets
"""

import sys
import json
import numpy as np
from pathlib import Path

import bench_utils
from bench_utils import load_local_dataset, run_sgd_benchmark, run_pivot_benchmark

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_d"

N_INITS = 10
MAX_ITER = 30

# Pivot counts as fractions of N; resolved per dataset at runtime
K_FRACTIONS = [0.10, 0.20, 0.50, 0.67, 0.80]

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

    k_values = sorted(set(
        max(3, min(int(round(f * N)), N - 1))
        for f in K_FRACTIONS
    ))
    print(f"  N={N}, D={X.shape[1]}, sweeping k={k_values}")

    results_store = {
        "dataset": dataset_name,
        "n_samples": N,
        "n_features": X.shape[1],
        "k_values": k_values,
        "runs": [],
    }

    best = {("SGD-cycle", None): {"stress": float("inf"), "emb": None}}
    for k in k_values:
        best[("SGD-pivot-euc", k)] = {"stress": float("inf"), "emb": None}

    for seed_i in range(N_INITS):
        seed = 42 + seed_i
        print(f"  [Seed {seed}]")

        common = dict(random_state=seed, max_iter=MAX_ITER,
                      switch_ratio=SGD_SWITCH, lr=SGD_LR, epsilon=SGD_EPSILON)

        # Baseline: SGD-cycle (once per seed, independent of k)
        r = run_sgd_benchmark(X, dissimilarity="euclidean", **common)
        print(f"    {'SGD-cycle':25s} stress={r['stress']:.2f}  time={r['time']:.2f}s")
        if r["stress"] < best[("SGD-cycle", None)]["stress"]:
            best[("SGD-cycle", None)] = {"stress": r["stress"], "emb": r["embedding"]}
        results_store["runs"].append({
            "algo": "SGD-cycle", "k": None, "seed": seed,
            "stress": float(r["stress"]), "time": float(r["time"]),
            "history": _clean_history(r["history"]),
        })

        # Pivot-euclidean at each k
        for k in k_values:
            r = run_pivot_benchmark(X, dissimilarity="euclidean", n_pivots=k, **common)
            print(f"    {'SGD-pivot-euc':25s} k={k:<5d} stress={r['stress']:.2f}  time={r['time']:.2f}s")
            if r["stress"] < best[("SGD-pivot-euc", k)]["stress"]:
                best[("SGD-pivot-euc", k)] = {"stress": r["stress"], "emb": r["embedding"]}
            results_store["runs"].append({
                "algo": "SGD-pivot-euc", "k": k, "seed": seed,
                "stress": float(r["stress"]), "time": float(r["time"]),
                "history": _clean_history(r["history"]),
                "n_pivots": r["n_pivots"],
            })

    print(f"\n  Saving to {out_dir}")
    with open(out_dir / "results.json", "w") as f:
        json.dump(results_store, f)

    for (label, k), bst in best.items():
        if bst["emb"] is not None:
            safe = label.replace("/", "_").replace(" ", "_")
            k_tag = f"_k{k}" if k is not None else ""
            np.save(out_dir / f"best_embedding_{safe}{k_tag}.npy", bst["emb"])

    if y is not None:
        np.save(out_dir / "y_labels.npy", y)


if __name__ == "__main__":
    datasets = sys.argv[1:] or get_all_datasets()
    for ds in datasets:
        run_dataset(ds)
