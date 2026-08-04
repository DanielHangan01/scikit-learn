"""
Experiment H: Diagnose the pivot SGD-MDS stress gap (H1 structural vs H2
mechanical) and provide the PivotMDS spectral baseline.

For each dataset in DEFAULT_DATASETS the runner produces three sub-experiments:

  H-base ─ baselines
      - SGD-cycle                      (the cycle reference)
      - PivotMDS-maxmin                (Brandes & Pich 2007 spectral baseline)
      - PivotMDS-pca10                 (PCA-10 pivots, same Pivot MDS solve)

  H-lr   ─ Phase 2a: LR rescue sweep (tests H2)
      - SGD-pivot-lazy-pca10 at lr in LR_VALUES, fixed k from K_FOR_LR_AND_HYBRID

  H-hyb  ─ Phase 2b: hybrid pair sampling (tests H1)
      - SGD-hybrid-pca10 at alpha in ALPHA_VALUES, same fixed k

Saves results.csv + histories.json per dataset under results/experiment_h/.

Usage:
    python run_experiment_h.py                    # all default datasets
    python run_experiment_h.py fashion_mnist hiva # subset

The script skips a dataset if both results.csv and histories.json already
exist (matching the convention of experiments F and G).
"""

from __future__ import annotations

import sys
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

import bench_utils
from bench_utils import (
    load_local_dataset,
    run_sgd_benchmark,
    run_pivot_benchmark,
    run_pivot_mds_benchmark,
    run_hybrid_benchmark,
)

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_h"

# Reuse the Experiment G dataset list for direct comparability.
DEFAULT_DATASETS = ["fashion_mnist", "coil20", "fmd", "hiva", "cifar10", "cnae9"]

# Baseline & PivotMDS sweep — same k grid as G for a side-by-side comparison.
K_VALUES_BASELINE = [10, 25, 50, 100, 200]

# For the LR and hybrid sweeps we hold k fixed at the regime where pivot
# falls behind the most. K=50 was the worst gap in G.
K_FOR_LR_AND_HYBRID = 50

# Phase 2a — LR sweep. 1.0 is the current 'auto' for uniform weighting.
LR_VALUES = [1.0, 0.3, 0.1, 0.03, 0.01]

# Phase 2b — hybrid mixing fractions. 1.0 = pure pivot; 0.0 = pure random.
ALPHA_VALUES = [1.0, 0.75, 0.5, 0.25, 0.0]

N_REPEATS  = 5
SEED_BASE  = 42
MAX_ITER   = 30

SGD_SWITCH  = 0.5
SGD_EPSILON = 0.001

# Default LR for the baseline / hybrid runs — match Experiment G's 0.5
# so the SGD-cycle and pivot-baseline numbers are directly comparable.
SGD_LR = 0.5


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

    print(f"\n=== Experiment H: {dataset_name} ===")
    X, _ = load_local_dataset(dataset_name)
    N, D = X.shape

    valid_k = [k for k in K_VALUES_BASELINE if 2 < k < N]
    print(f"  N={N}, D={D}")
    print(f"  H-base: cycle + PivotMDS over k={valid_k}")
    print(f"  H-lr:   pivot {{maxmin, pca10}} sweep over lr={LR_VALUES} at k={K_FOR_LR_AND_HYBRID}")
    print(f"  H-hyb:  hybrid {{maxmin, pca10}} sweep over alpha={ALPHA_VALUES} at k={K_FOR_LR_AND_HYBRID}")

    # Pivot strategies used inside the LR + hybrid sweeps.
    SWEEP_PIVOT_STRATEGIES = [
        # (strategy, pca_dim, algo_tag) — same tags as PivotMDS baselines.
        ("maxmin",     None, "maxmin"),
        ("maxmin_pca", 10,   "pca10"),
    ]

    rows:      List[Dict[str, Any]] = []
    histories: Dict[str, Any]       = {}
    run_id = 0

    def add_row(algo, *, phase, k, rep, seed, time_solver, stress,
                pivot_strategy=None, pca_dim=None, lr=None, alpha=None,
                history=None, stress_raw=None):
        nonlocal run_id
        rid = f"run_{run_id:06d}"; run_id += 1
        rows.append({
            "run_id": rid, "dataset": dataset_name,
            "n_samples": N, "n_features": D,
            "phase": phase, "algo": algo,
            "pivot_strategy": pivot_strategy,
            "pca_dim": pca_dim, "k": k, "lr": lr, "alpha": alpha,
            "rep": rep, "seed": seed,
            "time_solver": float(time_solver),
            "stress": float(stress),
            # PivotMDS-spectral rows only: stress before the optimal
            # uniform rescale (see run_pivot_mds_benchmark). NaN for SGD
            # rows, which optimize scale implicitly.
            "stress_raw": (float(stress_raw) if stress_raw is not None
                           else float("nan")),
        })
        histories[rid] = {
            "phase": phase, "algo": algo,
            "pivot_strategy": pivot_strategy, "pca_dim": pca_dim,
            "k": k, "lr": lr, "alpha": alpha, "rep": rep,
            "history": _clean_history(history),
        }

    sgd_common = dict(max_iter=MAX_ITER, switch_ratio=SGD_SWITCH,
                      epsilon=SGD_EPSILON)

    # -------------------- H-base: baselines --------------------

    print(f"\n  [H-base / SGD-cycle]")
    for rep in range(N_REPEATS):
        seed = SEED_BASE + rep
        r = run_sgd_benchmark(X, dissimilarity="euclidean", random_state=seed,
                              lr=SGD_LR, **sgd_common)
        add_row("SGD-cycle", phase="base", k=N, rep=rep, seed=seed,
                time_solver=r["time"], stress=r["stress"], history=r["history"])
        print(f"    rep={rep} stress={r['stress']:.2f} time={r['time']:.2f}s")

    for k in valid_k:
        for pivot_strategy, pca_dim, tag in SWEEP_PIVOT_STRATEGIES:
            algo_tag = f"PivotMDS-{tag}"
            print(f"\n  [H-base / {algo_tag}  k={k}]")
            for rep in range(N_REPEATS):
                seed = SEED_BASE + rep
                r = run_pivot_mds_benchmark(
                    X, n_pivots=k, random_state=seed,
                    pivot_strategy=pivot_strategy,
                    pivot_pca_dim=(pca_dim if pca_dim is not None else 30),
                )
                add_row(algo_tag, phase="base", k=k, rep=rep, seed=seed,
                        time_solver=r["time"], stress=r["stress"],
                        pivot_strategy=tag, pca_dim=pca_dim,
                        history=r["history"],
                        stress_raw=r.get("stress_raw"))
                print(f"    rep={rep} stress={r['stress']:.2f} time={r['time']*1000:.1f}ms")

    # -------------------- H-lr: LR rescue sweep --------------------

    k_lr = K_FOR_LR_AND_HYBRID
    if 2 < k_lr < N:
        for pivot_strategy, pca_dim, tag in SWEEP_PIVOT_STRATEGIES:
            algo_tag = f"SGD-pivot-{tag}"
            for lr in LR_VALUES:
                print(f"\n  [H-lr / {algo_tag}-lr{lr}  k={k_lr}]")
                for rep in range(N_REPEATS):
                    seed = SEED_BASE + rep
                    r = run_pivot_benchmark(
                        X, dissimilarity="lazy", n_pivots=k_lr,
                        random_state=seed, lr=lr, **sgd_common,
                        pivot_strategy=pivot_strategy,
                        pivot_pca_dim=(pca_dim if pca_dim is not None else 30),
                    )
                    add_row(algo_tag, phase="lr", k=k_lr,
                            rep=rep, seed=seed, time_solver=r["time"],
                            stress=r["stress"], pivot_strategy=tag,
                            pca_dim=pca_dim, lr=lr, history=r["history"])
                    print(f"    rep={rep} stress={r['stress']:.2f} time={r['time']:.2f}s")

    # -------------------- H-hyb: hybrid alpha sweep --------------------

    k_hy = K_FOR_LR_AND_HYBRID
    if 2 < k_hy < N:
        for pivot_strategy, pca_dim, tag in SWEEP_PIVOT_STRATEGIES:
            algo_tag = f"SGD-hybrid-{tag}"
            for alpha in ALPHA_VALUES:
                print(f"\n  [H-hyb / {algo_tag}-a{alpha:.2f}  k={k_hy}]")
                for rep in range(N_REPEATS):
                    seed = SEED_BASE + rep
                    r = run_hybrid_benchmark(
                        X, n_pivots=k_hy, hybrid_alpha=alpha,
                        random_state=seed, lr=SGD_LR, **sgd_common,
                        pivot_strategy=pivot_strategy,
                        pivot_pca_dim=(pca_dim if pca_dim is not None else 30),
                    )
                    add_row(algo_tag, phase="hyb", k=k_hy,
                            rep=rep, seed=seed, time_solver=r["time"],
                            stress=r["stress"], pivot_strategy=tag,
                            pca_dim=pca_dim, alpha=alpha,
                            history=r["history"])
                    print(f"    rep={rep} stress={r['stress']:.2f} time={r['time']:.2f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_hist.write_text(json.dumps(histories, indent=2))

    # ---- Per-dataset summary ----
    df = pd.DataFrame(rows)
    print(f"\n  === Summary for {dataset_name} (median over {N_REPEATS} seeds) ===")

    print(f"\n  -- Baseline --")
    for (algo, k), grp in df[df["phase"] == "base"].groupby(["algo", "k"], sort=False):
        print(f"    {algo:22s} k={int(k):5d}  "
              f"stress={grp['stress'].median():12.2f}  "
              f"time={grp['time_solver'].median():7.3f}s")

    cycle_stress = df[df["algo"] == "SGD-cycle"]["stress"].median()

    print(f"\n  -- LR sweep (k={k_lr}) --")
    for (strat, lr), grp in df[df["phase"] == "lr"].groupby(
        ["pivot_strategy", "lr"]
    ):
        ratio = grp["stress"].median() / max(cycle_stress, 1e-12)
        print(f"    {strat:6s}  lr={lr:6}  stress={grp['stress'].median():12.2f}  "
              f"ratio_to_cycle={ratio:6.2f}x")

    print(f"\n  -- Hybrid alpha sweep (k={k_hy}) --")
    for (strat, alpha), grp in df[df["phase"] == "hyb"].groupby(
        ["pivot_strategy", "alpha"]
    ):
        ratio = grp["stress"].median() / max(cycle_stress, 1e-12)
        print(f"    {strat:6s}  alpha={alpha:.2f}  "
              f"stress={grp['stress'].median():12.2f}  "
              f"ratio_to_cycle={ratio:6.2f}x")

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    targets = sys.argv[1:] or DEFAULT_DATASETS
    for ds in targets:
        run_dataset(ds)