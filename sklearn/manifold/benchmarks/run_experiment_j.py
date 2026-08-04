"""
Experiment J: budget-scaling behaviour of k across all 18 datasets.

Generalises Experiment I-scaling (run_experiment_i_scaling.py) from two
sources to the full dataset collection, and adds the mechanistic piece:
an intrinsic-dimensionality measurement per dataset, so the scaling
exponent p can be *explained*, not just reported.

For each dataset:
  1. Measure intrinsic-dimensionality scalars on a fixed seeded subsample
     (so the estimate is comparable across datasets of different N):
       - pca_participation_ratio  (Σλ)²/Σλ²  — smooth effective dim
       - pca_n90 / pca_n95        # PCA comps for 90% / 95% variance
       - strain_top2_frac         # fraction of classical-MDS (strain)
                                   #   eigen-energy in the top 2 dims —
                                   #   i.e. "how 2D-embeddable" the data is.
                                   #   THE quantity hypothesised to drive p.
  2. Subsample to a geometric N-ladder (shared absolute rungs across
     datasets so cross-dataset comparison at a common N is possible), and
     at each (N, seed): SGD-cycle baseline + SGD-random at k·N for each k.

Overhead(N, k) = median_seed(random_stress / cycle_stress).
The analyzer (plot_experiment_j.py) derives k*(N), the exponent p, the
overhead-vs-k steepness s, the saturation floor, and the cross-dataset
correlations of p / s against intrinsic dimensionality.

Outputs per dataset under results/experiment_j/<dataset>/:
    results.csv   — raw stress rows
    meta.json     — N, D, intrinsic-dim scalars, ladder, k grid

Skip-if-exists per dataset (resumable: rerun to continue where it stopped).

Usage:
    python run_experiment_j.py                 # all 18 (LONG — see note)
    python run_experiment_j.py coil20 hiva     # subset
    python run_experiment_j.py --list          # print datasets + cost hint

NOTE ON COST: the slow runs are SGD-cycle at the largest N rung for the
high-D datasets (hiva D=1617, fmd D=1536, cifar10/svhn D=1024). The full
sweep is many hours. Run high-D datasets individually / overnight. Reduce
N_REPEATS or trim BASE_LADDER for a faster first pass.
"""

from __future__ import annotations

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

from sklearn.metrics import euclidean_distances

from bench_utils import (
    load_local_dataset,
    run_sgd_benchmark,
    run_random_budget_benchmark,
)

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_j"

ALL_DATASETS = [
    "bank", "cifar10", "cnae9", "coil20", "epileptic", "fashion_mnist",
    "fmd", "har", "hatespeech", "hiva", "imdb", "orl", "secom", "seismic",
    "sentiment", "sms", "spambase", "svhn",
]

# Shared absolute N rungs so any common rung can be compared across datasets.
# Each dataset uses the rungs <= its full N, plus its full N as the top rung.
BASE_LADDER = [400, 600, 900, 1350, 2000, 3000, 4000, 5000]

# Budget grid (n_updates_per_epoch = k * N), filtered to k < (N-1)/2 per rung.
K_VALUES = [10, 25, 50, 100, 200]

# max_iter=30 is the scaling reference; 15 also probes the budget x iteration
# interaction (whether the mid-k 15-vs-30 penalty correlates with regime).
MAX_ITER_VALUES = [15, 30]

N_REPEATS = 5
SEED_BASE = 42

SGD_SWITCH = 0.5
SGD_EPSILON = 0.001
SGD_LR = 0.5

# Intrinsic-dimensionality estimate: fixed subsample size for cross-dataset
# comparability (removes the N-dependence of the estimate itself).
ID_SUBSAMPLE = 800
ID_SEED = 0


def make_ladder(N_full: int) -> List[int]:
    rungs = [n for n in BASE_LADDER if n <= N_full]
    if not rungs:
        rungs = [N_full]
    if rungs[-1] < N_full * 0.95:
        rungs.append(N_full)
    return rungs


def valid_k(N: int) -> List[int]:
    cap = (N - 1) // 2
    return [k for k in K_VALUES if k < cap]


def intrinsic_dim_measures(X: np.ndarray) -> Dict[str, float]:
    """PCA participation ratio + variance-threshold dims + strain top-2 frac.

    Computed on a fixed seeded subsample of ID_SUBSAMPLE points so the
    estimate does not depend on the dataset's N.
    """
    rng = np.random.RandomState(ID_SEED)
    n = X.shape[0]
    if n > ID_SUBSAMPLE:
        idx = rng.choice(n, ID_SUBSAMPLE, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    Xs = np.ascontiguousarray(Xs, dtype=np.float64)

    # --- PCA spectrum (features) ---
    Xc = Xs - Xs.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    var = sv ** 2
    tot = float(var.sum())
    pr = float(tot ** 2 / np.sum(var ** 2)) if tot > 0 else float("nan")
    cum = np.cumsum(var) / tot if tot > 0 else np.array([1.0])
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n95 = int(np.searchsorted(cum, 0.95) + 1)

    # --- Strain spectrum (classical-MDS double-centered distance matrix) ---
    D2 = euclidean_distances(Xs) ** 2
    rm = D2.mean(axis=1, keepdims=True)
    cm = D2.mean(axis=0, keepdims=True)
    B = -0.5 * (D2 - rm - cm + D2.mean())
    evals = np.linalg.eigvalsh(B)
    pos = np.sort(evals[evals > 0])[::-1]
    strain_top2 = float(pos[:2].sum() / pos.sum()) if pos.sum() > 0 else float("nan")
    # Participation ratio of the strain spectrum (effective MDS dim).
    strain_pr = float(pos.sum() ** 2 / np.sum(pos ** 2)) if pos.size else float("nan")

    return {
        "id_n": int(Xs.shape[0]),
        "pca_participation_ratio": pr,
        "pca_n90": n90,
        "pca_n95": n95,
        "strain_top2_frac": strain_top2,
        "strain_participation_ratio": strain_pr,
    }


def run_dataset(dataset: str) -> None:
    out_dir = RESULTS_DIR / dataset
    out_csv = out_dir / "results.csv"
    out_meta = out_dir / "meta.json"
    if out_csv.exists() and out_meta.exists():
        print(f"Skipping {dataset} (results already exist).")
        return

    print(f"\n=== Experiment J: {dataset} ===")
    X_full, _ = load_local_dataset(dataset)
    X_full = np.ascontiguousarray(X_full, dtype=np.float64)
    N_full, D = X_full.shape
    ladder = make_ladder(N_full)
    print(f"  full N={N_full}, D={D}")
    print(f"  N ladder: {ladder}")

    idm = intrinsic_dim_measures(X_full)
    print(f"  intrinsic dim: PR={idm['pca_participation_ratio']:.1f}  "
          f"n90={idm['pca_n90']}  strain_top2={idm['strain_top2_frac']:.3f}")

    rows: List[Dict[str, Any]] = []
    run_id = 0

    def add_row(phase, N, k, n_updates, max_iter, seed, time_solver, stress):
        nonlocal run_id
        rid = f"run_{run_id:06d}"; run_id += 1
        rows.append({
            "run_id": rid, "dataset": dataset,
            "n_samples": N, "n_features": D,
            "phase": phase, "k": k,
            "n_updates_per_epoch": n_updates, "max_iter": max_iter,
            "seed": seed, "time_solver": float(time_solver),
            "stress": float(stress),
        })

    common = dict(switch_ratio=SGD_SWITCH, epsilon=SGD_EPSILON, lr=SGD_LR)

    for max_iter in MAX_ITER_VALUES:
        for N in ladder:
            ks = valid_k(N)
            print(f"\n  --- max_iter={max_iter}  N={N}  (k in {ks}) ---")
            for rep in range(N_REPEATS):
                seed = SEED_BASE + rep
                sub_rng = np.random.RandomState(seed)
                idx = sub_rng.choice(N_full, size=N, replace=False)
                X = np.ascontiguousarray(X_full[idx], dtype=np.float64)

                rc = run_sgd_benchmark(X, dissimilarity="euclidean",
                                       random_state=seed, max_iter=max_iter,
                                       **common)
                add_row("cycle", N, None, N * (N - 1) // 2, max_iter, seed,
                        rc["time"], rc["stress"])

                for k in ks:
                    rb = run_random_budget_benchmark(
                        X, n_updates_per_epoch=k * N,
                        random_state=seed, max_iter=max_iter, **common)
                    add_row("budget", N, k, k * N, max_iter, seed,
                            rb["time"], rb["stress"])

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    meta = {
        "dataset": dataset, "n_samples": N_full, "n_features": D,
        "ladder": ladder, "k_values": K_VALUES,
        "max_iter_values": MAX_ITER_VALUES, "n_repeats": N_REPEATS,
        **idm,
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    # ---- per-dataset console summary (overhead at max_iter=30) ----
    df = pd.DataFrame(rows)
    mi = 30 if 30 in MAX_ITER_VALUES else MAX_ITER_VALUES[0]
    print(f"\n  === Summary: {dataset} (median over {N_REPEATS} seeds, "
          f"max_iter={mi}) ===")
    for N in ladder:
        cyc = df[(df["phase"] == "cycle") & (df["n_samples"] == N)
                 & (df["max_iter"] == mi)]["stress"].median()
        bud = df[(df["phase"] == "budget") & (df["n_samples"] == N)
                 & (df["max_iter"] == mi)]
        cells = []
        for k, grp in bud.groupby("k"):
            cells.append(f"k{int(k)}={grp['stress'].median()/cyc:.3f}x")
        print(f"    N={N:5d} | " + "  ".join(cells))
    print(f"\nSaved: {out_csv}")


def main():
    args = sys.argv[1:]
    if args and args[0] == "--list":
        print("Datasets (N, D) and rough cost rank (cycle ~ N^2 * D):")
        for ds in ALL_DATASETS:
            print(f"  {ds}")
        return
    targets = args or ALL_DATASETS
    for ds in targets:
        run_dataset(ds)


if __name__ == "__main__":
    main()
