"""
Experiment F: Pivot k-Sensitivity Sweep.

For a fixed dataset, sweeps n_pivots over a range and compares pivot-lazy and
pivot-euclidean against the SGD-cycle (euclidean) baseline. Answers:

  - How many pivots are needed to close the stress gap vs full-cycle SGD?
  - At what k does pivot mode stop being faster than cycle?
  - What is the quality/speed tradeoff curve?

Saves results.csv + histories.json per dataset.

Usage:
    python run_experiment_f.py                    # all datasets (skips existing results)
    python run_experiment_f.py coil20 fashion_mnist
"""

from __future__ import annotations

import sys
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

import bench_utils
from bench_utils import load_local_dataset, run_sgd_benchmark, run_pivot_benchmark

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_f" / "k_sweep"
DATASET_ROOT = Path(__file__).parent / "datasets"

SYNTHETIC_DATASETS = ["s_curve", "swiss_roll", "torus"]


def get_all_datasets() -> List[str]:
    """Returns all local datasets + synthetic ones, sorted."""
    local = sorted(d.name for d in DATASET_ROOT.iterdir() if d.is_dir()) if DATASET_ROOT.exists() else []
    return local + [s for s in SYNTHETIC_DATASETS if s not in local]

# k values to sweep; will be filtered to k < N at runtime
K_VALUES = [10, 25, 50, 100, 200, 300, 500, 750, 1000, 1440]

N_REPEATS  = 10
SEED_BASE  = 42
MAX_ITER   = 30

SGD_LR      = 0.5
SGD_SWITCH  = 0.5
SGD_EPSILON = 0.001


def _clean_history(hist: Any) -> List[List[float]]:
    if not hist:
        return []
    try:
        return [[float(t), float(s)] for t, s in hist]
    except Exception:
        return []


def run_k_sweep(dataset_name: str) -> None:
    out_dir  = RESULTS_DIR / dataset_name
    out_csv  = out_dir / "results.csv"
    out_hist = out_dir / "histories.json"

    if out_csv.exists() and out_hist.exists():
        print(f"Skipping {dataset_name} (results already exist).")
        return

    print(f"\n=== Experiment F (k-sweep): {dataset_name} ===")
    X, _ = load_local_dataset(dataset_name)
    N = X.shape[0]

    valid_k = [k for k in K_VALUES if k > 2 and k < N]
    print(f"  N={N}, D={X.shape[1]}, sweeping k={valid_k}")

    rows:      List[Dict[str, Any]] = []
    histories: Dict[str, Any]       = {}
    run_id = 0

    common = dict(max_iter=MAX_ITER, switch_ratio=SGD_SWITCH,
                  lr=SGD_LR, epsilon=SGD_EPSILON)

    # --- Baseline: SGD-cycle (full pair coverage) ---
    print(f"\n  [Baseline: SGD-cycle]")
    for rep in range(N_REPEATS):
        seed = SEED_BASE + rep
        r = run_sgd_benchmark(X, dissimilarity="euclidean", random_state=seed, **common)
        rid = f"run_{run_id:06d}"; run_id += 1
        rows.append({
            "run_id": rid, "dataset": dataset_name, "n_samples": N,
            "algo": "SGD-cycle", "k": N,  # k=N means all pairs
            "rep": rep, "seed": seed,
            "time_solver": float(r["time"]), "stress": float(r["stress"]),
        })
        histories[rid] = {"algo": "SGD-cycle", "k": N, "rep": rep,
                          "history": _clean_history(r["history"])}
        print(f"    rep={rep} stress={r['stress']:.2f}  time={r['time']:.2f}s")

    # --- Pivot sweep ---
    for k in valid_k:
        print(f"\n  [k={k}]")
        for rep in range(N_REPEATS):
            seed = SEED_BASE + rep

            for diss in ("lazy", "euclidean"):
                r = run_pivot_benchmark(X, dissimilarity=diss, n_pivots=k,
                                        random_state=seed, **common)
                rid = f"run_{run_id:06d}"; run_id += 1
                algo = f"SGD-pivot-{diss}"
                rows.append({
                    "run_id": rid, "dataset": dataset_name, "n_samples": N,
                    "algo": algo, "k": k,
                    "rep": rep, "seed": seed,
                    "time_solver": float(r["time"]), "stress": float(r["stress"]),
                })
                histories[rid] = {"algo": algo, "k": k, "rep": rep,
                                  "history": _clean_history(r["history"])}

            row_lazy = rows[-2]
            row_euc  = rows[-1]
            print(f"    rep={rep} "
                  f"pivot-lazy stress={row_lazy['stress']:.2f} t={row_lazy['time_solver']:.2f}s  |  "
                  f"pivot-euc  stress={row_euc['stress']:.2f} t={row_euc['time_solver']:.2f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_hist.write_text(json.dumps(histories, indent=2))

    # Print summary table: mean stress and time per (algo, k)
    df = pd.DataFrame(rows)
    print(f"\n  === Summary for {dataset_name} (mean over {N_REPEATS} seeds) ===")
    print(f"  {'algo':25s}  {'k':>6}  {'stress (mean)':>15}  {'time (mean)':>12}")
    print(f"  {'-'*65}")
    for (algo, k), grp in df.groupby(["algo", "k"], sort=False):
        print(f"  {algo:25s}  {k:6d}  {grp['stress'].mean():15.2f}  {grp['time_solver'].mean():12.3f}s")

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    targets = sys.argv[1:] or get_all_datasets()
    for ds in targets:
        run_k_sweep(ds)
