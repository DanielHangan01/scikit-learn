"""
Experiment E: Pivot Scaling Study on Synthetic Datasets.

For increasing N, compares:
  1. SGD-euclidean   sampling_strategy='cycle',  dissimilarity='euclidean'  [O(N^2) baseline]
  2. SGD-lazy        sampling_strategy='random',  dissimilarity='lazy'       [O(N^2) baseline]
  3. SGD-pivot-k25   sampling_strategy='pivot',   dissimilarity='lazy', k=25 [O(kN)]
  4. SGD-pivot-k50   sampling_strategy='pivot',   dissimilarity='lazy', k=50
  5. SGD-pivot-k100  sampling_strategy='pivot',   dissimilarity='lazy', k=100

Shows: where pivot-lazy becomes faster than cycle/random, and how k trades quality for speed.

Usage:
    python run_experiment_e.py                # all synthetic datasets
    python run_experiment_e.py s_curve torus  # specific datasets
"""

from __future__ import annotations

import sys
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

import bench_utils
from bench_utils import load_local_dataset, run_sgd_benchmark, run_pivot_benchmark

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_e" / "pivot_scaling"

SYNTHETIC_DATASETS = ["s_curve", "swiss_roll", "torus"]
TARGET_DIMS = {
    "s_curve":    100,
    "swiss_roll": 300,
    "torus":      500,
}

N_START = 500
N_STOP  = 5000
N_STEP  = 500
N_VALUES = list(range(N_START, N_STOP + 1, N_STEP))

PIVOT_K_VALUES = [25, 50, 100]

N_REPEATS  = 10
SEED_BASE  = 85
MAX_ITER   = 30

SGD_LR      = 0.5
SGD_SWITCH  = 0.5
SGD_EPSILON = 0.001

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _clean_history(hist: Any) -> List[List[float]]:
    if not hist:
        return []
    try:
        return [[float(t), float(s)] for t, s in hist]
    except Exception:
        return []


def _run_all_algos(X, seed: int) -> Dict[str, Dict[str, Any]]:
    common = dict(random_state=seed, max_iter=MAX_ITER,
                  switch_ratio=SGD_SWITCH, lr=SGD_LR, epsilon=SGD_EPSILON)
    results = {}

    results["SGD-euclidean"] = run_sgd_benchmark(X, dissimilarity="euclidean", **common)
    results["SGD-lazy"]      = run_sgd_benchmark(X, dissimilarity="lazy",      **common)

    for k in PIVOT_K_VALUES:
        label = f"SGD-pivot-k{k}"
        results[label] = run_pivot_benchmark(X, dissimilarity="lazy", n_pivots=k, **common)

    return results


def run_synthetic_dataset(dataset_name: str) -> None:
    out_dir  = RESULTS_DIR / dataset_name
    out_csv  = out_dir / "results.csv"
    out_hist = out_dir / "histories.json"

    if out_csv.exists() and out_hist.exists():
        print(f"Skipping {dataset_name} (results already exist).")
        return

    if dataset_name not in TARGET_DIMS:
        raise ValueError(f"Unknown synthetic dataset: {dataset_name}")

    target_dim = TARGET_DIMS[dataset_name]
    print(f"\n=== Experiment E (synthetic sweep): {dataset_name} (D={target_dim}) ===")

    meta = {
        "experiment": "E",
        "dataset": dataset_name,
        "target_dim": target_dim,
        "n_values": N_VALUES,
        "pivot_k_values": PIVOT_K_VALUES,
        "n_repeats": N_REPEATS,
        "seed_base": SEED_BASE,
        "max_iter": MAX_ITER,
        "sgd_config": {"lr": SGD_LR, "switch_ratio": SGD_SWITCH, "epsilon": SGD_EPSILON},
        "algos": ["SGD-euclidean", "SGD-lazy"] + [f"SGD-pivot-k{k}" for k in PIVOT_K_VALUES],
    }

    rows:      List[Dict[str, Any]] = []
    histories: Dict[str, Any]       = {}
    run_id = 0

    for N in N_VALUES:
        print(f"\n--- N={N}, D={target_dim} ---")

        X, _ = load_local_dataset(
            dataset_name,
            n_samples=N,
            random_state=SEED_BASE,
            target_dim=target_dim,
            noise_scale=0.1,
        )
        n_samples, n_features = X.shape

        for rep in range(N_REPEATS):
            seed = SEED_BASE + rep
            res  = _run_all_algos(X, seed)

            for algo, r in res.items():
                rid = f"run_{run_id:06d}"
                run_id += 1

                row: Dict[str, Any] = {
                    "run_id":      rid,
                    "dataset":     dataset_name,
                    "n_samples":   int(n_samples),
                    "n_features":  int(n_features),
                    "algo":        algo,
                    "rep":         int(rep),
                    "seed":        int(seed),
                    "max_iter":    int(MAX_ITER),
                    "time_solver": float(r["time"]),
                    "stress":      float(r["stress"]),
                }
                if "n_pivots" in r:
                    row["n_pivots"] = int(r["n_pivots"])
                rows.append(row)

                histories[rid] = {
                    "dataset":   dataset_name,
                    "algo":      algo,
                    "rep":       int(rep),
                    "seed":      int(seed),
                    "n_samples": int(n_samples),
                    "history":   _clean_history(r.get("history")),
                }

            timing = "  ".join(
                f"{a}={res[a]['time']:.2f}s" for a in res
            )
            print(f"  rep={rep:02d} | {timing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_hist.write_text(json.dumps(histories, indent=2))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_hist}")


if __name__ == "__main__":
    targets = [a.lower() for a in sys.argv[1:]] or SYNTHETIC_DATASETS
    for ds in targets:
        run_synthetic_dataset(ds)
