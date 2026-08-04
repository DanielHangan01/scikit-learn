"""
Experiment I: budget knob on sampling_strategy='random'.

Sweeps the per-epoch update budget for random pair sampling across the six
H datasets plus `epileptic` (N=5750), at two iteration budgets
(`max_iter in {15, 30}`).

For each dataset the runner produces two sub-experiments:

  I-base   ─ SGD-cycle baseline at each max_iter.
  I-budget ─ SGD-random at each (n_updates_per_epoch, max_iter) combo for
             k in {10, 25, 50, 100, 200} (budget = k * N) plus the full-cycle
             sanity point (budget = N*(N-1)//2).

Saves results.csv + histories.json per dataset under
``results/experiment_i/<dataset>/``.

Usage:
    python run_experiment_i.py                  # all default datasets
    python run_experiment_i.py coil20 hiva      # subset

Skips a dataset if both results.csv and histories.json already exist
(matches conventions from experiments F, G, H).
"""

from __future__ import annotations

import sys
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

from bench_utils import (
    load_local_dataset,
    run_sgd_benchmark,
    run_random_budget_benchmark,
)

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_i"

DEFAULT_DATASETS = [
    "fashion_mnist", "coil20", "fmd", "hiva", "cifar10", "cnae9", "epileptic",
]

# Budget k-grid (n_updates_per_epoch = k * N). Plus the full-cycle sanity point
# computed per-dataset as N*(N-1)//2.
K_VALUES = [10, 25, 50, 100, 200]

# Two iteration budgets.
MAX_ITER_VALUES = [15, 30]

N_REPEATS = 5
SEED_BASE = 42

SGD_SWITCH  = 0.5
SGD_EPSILON = 0.001
SGD_LR      = 0.5


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

    print(f"\n=== Experiment I: {dataset_name} ===")
    X, _ = load_local_dataset(dataset_name)
    N, D = X.shape

    full_cycle_budget = (N * (N - 1)) // 2
    valid_k = [k for k in K_VALUES if k * N < full_cycle_budget]
    budgets = [k * N for k in valid_k] + [full_cycle_budget]
    budget_labels = [(k, k * N) for k in valid_k] + [("full", full_cycle_budget)]

    print(f"  N={N}, D={D}")
    print(f"  I-base:   SGD-cycle at max_iter={MAX_ITER_VALUES}")
    print(f"  I-budget: budgets={budgets} x max_iter={MAX_ITER_VALUES}")

    rows:      List[Dict[str, Any]] = []
    histories: Dict[str, Any]       = {}
    run_id = 0

    def add_row(algo, *, phase, k, n_updates, max_iter, rep, seed,
                time_solver, stress, history=None):
        nonlocal run_id
        rid = f"run_{run_id:06d}"; run_id += 1
        rows.append({
            "run_id": rid, "dataset": dataset_name,
            "n_samples": N, "n_features": D,
            "phase": phase, "algo": algo,
            "k": k, "n_updates_per_epoch": n_updates, "max_iter": max_iter,
            "rep": rep, "seed": seed,
            "time_solver": float(time_solver),
            "stress": float(stress),
        })
        histories[rid] = {
            "phase": phase, "algo": algo,
            "k": k, "n_updates_per_epoch": n_updates, "max_iter": max_iter,
            "rep": rep, "history": _clean_history(history),
        }

    sgd_common = dict(switch_ratio=SGD_SWITCH, epsilon=SGD_EPSILON, lr=SGD_LR)

    # -------------------- I-base: SGD-cycle baseline --------------------

    for max_iter in MAX_ITER_VALUES:
        print(f"\n  [I-base / SGD-cycle  max_iter={max_iter}]")
        for rep in range(N_REPEATS):
            seed = SEED_BASE + rep
            r = run_sgd_benchmark(
                X, dissimilarity="euclidean", random_state=seed,
                max_iter=max_iter, **sgd_common,
            )
            add_row("SGD-cycle", phase="base",
                    k=None, n_updates=full_cycle_budget, max_iter=max_iter,
                    rep=rep, seed=seed,
                    time_solver=r["time"], stress=r["stress"],
                    history=r["history"])
            print(f"    rep={rep} stress={r['stress']:.2f} time={r['time']:.2f}s")

    # -------------------- I-budget: budgeted random sweep --------------------

    for max_iter in MAX_ITER_VALUES:
        for k_label, budget in budget_labels:
            tag = f"k{k_label}" if k_label != "full" else "full"
            algo_tag = f"SGD-random-{tag}"
            print(f"\n  [I-budget / {algo_tag}  budget={budget}  max_iter={max_iter}]")
            for rep in range(N_REPEATS):
                seed = SEED_BASE + rep
                r = run_random_budget_benchmark(
                    X, n_updates_per_epoch=budget,
                    random_state=seed, max_iter=max_iter, **sgd_common,
                )
                add_row(algo_tag, phase="budget",
                        k=(k_label if k_label != "full" else None),
                        n_updates=budget, max_iter=max_iter,
                        rep=rep, seed=seed,
                        time_solver=r["time"], stress=r["stress"],
                        history=r["history"])
                print(f"    rep={rep} stress={r['stress']:.2f} time={r['time']:.2f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_hist.write_text(json.dumps(histories, indent=2))

    # ---- Per-dataset summary ----
    df = pd.DataFrame(rows)
    print(f"\n  === Summary for {dataset_name} (median over {N_REPEATS} seeds) ===")

    for max_iter in MAX_ITER_VALUES:
        cycle_med = df[
            (df["phase"] == "base") & (df["max_iter"] == max_iter)
        ]["stress"].median()
        print(f"\n  -- max_iter={max_iter}  (cycle stress = {cycle_med:.2f}) --")
        sub = df[(df["phase"] == "budget") & (df["max_iter"] == max_iter)]
        for budget, grp in sub.groupby("n_updates_per_epoch", sort=True):
            med_s = grp["stress"].median()
            med_t = grp["time_solver"].median()
            ratio = med_s / max(cycle_med, 1e-12)
            tag = grp["algo"].iloc[0]
            print(f"    {tag:22s}  budget={int(budget):>10d}  "
                  f"stress={med_s:12.2f}  time={med_t:7.2f}s  "
                  f"ratio_to_cycle={ratio:6.2f}x")

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    targets = sys.argv[1:] or DEFAULT_DATASETS
    for ds in targets:
        run_dataset(ds)
