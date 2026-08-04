"""
Experiment G: PCA-based Pivot Selection vs Standard MaxMin (High-D Datasets).

Hypothesis: in high-dimensional feature spaces, Euclidean distances concentrate
(curse of dimensionality), so MaxMin selects pivots whose "spread" is
near-uniform and uninformative. Projecting features through PCA before MaxMin
should yield better-separated pivots — and therefore better stress at small k.

For each dataset, sweeps k over a few small/medium values and compares:
  - SGD-cycle                       (baseline; full N(N-1)/2 pair coverage)
  - SGD-pivot-lazy + maxmin         (current behaviour)
  - SGD-pivot-lazy + maxmin_pca-10  (PCA pivots, 10 components)
  - SGD-pivot-lazy + maxmin_pca-30  (PCA pivots, 30 components)

Saves results.csv + histories.json per dataset.

Usage:
    python run_experiment_g.py                    # all default datasets
    python run_experiment_g.py fashion_mnist hiva
"""

from __future__ import annotations

import sys
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

import bench_utils
from bench_utils import load_local_dataset, run_sgd_benchmark, run_pivot_benchmark

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_g"

# High-dimensional real datasets where the curse of dimensionality should
# affect pivot quality the most.
DEFAULT_DATASETS = ["fashion_mnist", "coil20", "fmd", "hiva", "cifar10", "cnae9"]

# Small/medium k values; the hypothesis is sharpest at small k.
K_VALUES = [10, 25, 50, 100, 200]

# PCA dims to compare. None means standard MaxMin (current behaviour).
PCA_DIMS: List[int | None] = [None, 10, 30]

N_REPEATS  = 5
SEED_BASE  = 42
MAX_ITER   = 30

SGD_LR      = 0.5
SGD_SWITCH  = 0.5
SGD_EPSILON = 0.001


def _algo_name(pca_dim):
    if pca_dim is None:
        return "SGD-pivot-lazy-maxmin"
    return f"SGD-pivot-lazy-pca{pca_dim}"


def _clean_history(hist):
    if not hist:
        return []
    try:
        return [[float(t), float(s)] for t, s in hist]
    except Exception:
        return []


def run_dataset(dataset_name: str) -> None:
    out_dir  = RESULTS_DIR / dataset_name
    out_csv  = out_dir / "results.csv"
    out_hist = out_dir / "histories.json"

    if out_csv.exists() and out_hist.exists():
        print(f"Skipping {dataset_name} (results already exist).")
        return

    print(f"\n=== Experiment G (PCA pivots): {dataset_name} ===")
    X, _ = load_local_dataset(dataset_name)
    N, D = X.shape

    valid_k = [k for k in K_VALUES if 2 < k < N]
    print(f"  N={N}, D={D}, sweeping k={valid_k}, pca_dims={PCA_DIMS}")

    rows:      List[Dict[str, Any]] = []
    histories: Dict[str, Any]       = {}
    run_id = 0

    common = dict(max_iter=MAX_ITER, switch_ratio=SGD_SWITCH,
                  lr=SGD_LR, epsilon=SGD_EPSILON)

    # --- Baseline: SGD-cycle ---
    print(f"\n  [Baseline: SGD-cycle]")
    for rep in range(N_REPEATS):
        seed = SEED_BASE + rep
        r = run_sgd_benchmark(X, dissimilarity="euclidean",
                              random_state=seed, **common)
        rid = f"run_{run_id:06d}"; run_id += 1
        rows.append({
            "run_id": rid, "dataset": dataset_name, "n_samples": N, "n_features": D,
            "algo": "SGD-cycle", "pca_dim": None, "k": N,
            "rep": rep, "seed": seed,
            "time_solver": float(r["time"]), "stress": float(r["stress"]),
        })
        histories[rid] = {"algo": "SGD-cycle", "pca_dim": None, "k": N, "rep": rep,
                          "history": _clean_history(r["history"])}
        print(f"    rep={rep} stress={r['stress']:.2f}  time={r['time']:.2f}s")

    # --- Pivot variants ---
    for k in valid_k:
        for pca_dim in PCA_DIMS:
            algo = _algo_name(pca_dim)
            print(f"\n  [{algo}  k={k}]")
            pivot_strategy = "maxmin" if pca_dim is None else "maxmin_pca"
            kwargs = dict(
                pivot_strategy=pivot_strategy,
                pivot_pca_dim=(pca_dim if pca_dim is not None else 30),
            )
            for rep in range(N_REPEATS):
                seed = SEED_BASE + rep
                r = run_pivot_benchmark(
                    X, dissimilarity="lazy", n_pivots=k,
                    random_state=seed, **common, **kwargs,
                )
                rid = f"run_{run_id:06d}"; run_id += 1
                rows.append({
                    "run_id": rid, "dataset": dataset_name, "n_samples": N, "n_features": D,
                    "algo": algo, "pca_dim": pca_dim, "k": k,
                    "rep": rep, "seed": seed,
                    "time_solver": float(r["time"]), "stress": float(r["stress"]),
                })
                histories[rid] = {"algo": algo, "pca_dim": pca_dim, "k": k,
                                  "rep": rep, "history": _clean_history(r["history"])}
                print(f"    rep={rep} stress={r['stress']:.2f}  time={r['time']:.2f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_hist.write_text(json.dumps(histories, indent=2))

    df = pd.DataFrame(rows)
    print(f"\n  === Summary for {dataset_name} (mean over {N_REPEATS} seeds) ===")
    print(f"  {'algo':30s}  {'k':>6}  {'stress (mean)':>15}  {'time (mean)':>12}")
    print(f"  {'-'*70}")
    for (algo, k), grp in df.groupby(["algo", "k"], sort=False):
        print(f"  {algo:30s}  {int(k):6d}  {grp['stress'].mean():15.2f}  "
              f"{grp['time_solver'].mean():12.3f}s")
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    targets = sys.argv[1:] or DEFAULT_DATASETS
    for ds in targets:
        run_dataset(ds)
