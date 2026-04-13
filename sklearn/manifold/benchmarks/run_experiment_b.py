"""
Experiment D: Scaling Study on Synthetic + Selected Real-World Datasets (Execution).

Synthetic protocol (per dataset, per N):
- Generate synthetic feature matrix X with n_samples=N
- Run:
  (1) SMACOF with dissimilarity="euclidean"
  (2) SGD-MDS with dissimilarity="euclidean"   (non-lazy, stores Δ internally)
  (3) SGD-MDS with dissimilarity="lazy"        (lazy, computes δ_ij on the fly)
- Repeat each algorithm over N_REPEATS seeds
- Save per-run results.csv AND histories.json

Real-world protocol (per dataset):
- Load full dataset at its native N (no scaling sweep)
- Run the same 3 algorithms
- Repeat each algorithm over N_REPEATS seeds
- Save per-run results.csv AND histories.json
"""

from __future__ import annotations

import sys
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

import bench_utils
from bench_utils import load_local_dataset, run_smacof_benchmark, run_sgd_benchmark

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_b" / "scaling_mixed_euclid_vs_lazy"

SYNTHETIC_DATASETS = ["s_curve", "swiss_roll", "torus"]
TARGET_DIMS = {
    "s_curve": 100,
    "swiss_roll": 300,
    "torus": 500,
}

N_START = 500
N_STOP = 5000
N_STEP = 500
N_VALUES = list(range(N_START, N_STOP + 1, N_STEP))

REALWORLD_DATASETS = ["fashion_mnist", "spambase", "seismic"]

N_REPEATS = 10
SEED_BASE = 85

MAX_ITER_SGD = 30
MAX_ITER_SMACOF = 300

SGD_LR = 0.5
SGD_SWITCH = 0.5
SGD_EPSILON = 0.001

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _clean_history(hist: Any) -> List[List[float]]:
    """
    Convert history into JSON-safe list-of-[time, stress] floats.
    Handles None/empty gracefully.
    """
    if hist is None:
        return []
    out: List[List[float]] = []
    try:
        for t, s in hist:
            out.append([float(t), float(s)])
    except Exception:
        return []
    return out


def _run_three_algos(
    X,
    seed: int,
    max_iter_smacof: int,
    max_iter_sgd: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Run SMACOF (euclidean), SGD-euclidean, SGD-lazy for a given X and seed.
    Returns dict keyed by algo name with outputs including history.
    """
    smacof_res = run_smacof_benchmark(
        X,
        dissimilarity="euclidean",
        random_state=seed,
        max_iter=max_iter_smacof,
    )
    sgd_euc_res = run_sgd_benchmark(
        X,
        dissimilarity="euclidean",
        random_state=seed,
        max_iter=max_iter_sgd,
        switch_ratio=SGD_SWITCH,
        lr=SGD_LR,
        epsilon=SGD_EPSILON,
    )
    sgd_lazy_res = run_sgd_benchmark(
        X,
        dissimilarity="lazy",
        random_state=seed,
        max_iter=max_iter_sgd,
        switch_ratio=SGD_SWITCH,
        lr=SGD_LR,
        epsilon=SGD_EPSILON,
    )

    return {
        "SMACOF-euclidean": smacof_res,
        "SGD-euclidean": sgd_euc_res,
        "SGD-lazy": sgd_lazy_res,
    }


def _write_outputs(out_dir: Path, rows: List[Dict[str, Any]], histories: Dict[str, Any], meta: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "results.csv"
    out_hist = out_dir / "histories.json"
    out_meta = out_dir / "meta.json"

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_hist.write_text(json.dumps(histories, indent=2))
    out_meta.write_text(json.dumps(meta, indent=2))

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_hist}")
    print(f"Saved: {out_meta}")


def run_realworld_dataset(dataset_name: str) -> None:
    out_dir = RESULTS_DIR / dataset_name
    out_csv = out_dir / "results.csv"
    out_hist = out_dir / "histories.json"

    if out_csv.exists() and out_hist.exists():
        print(f"Skipping {dataset_name} (results.csv + histories.json exist).")
        return

    print(f"\n=== Experiment B (real-world): {dataset_name} ===")

    X, _ = load_local_dataset(dataset_name)
    n_samples, n_features = X.shape
    print(f"Loaded {dataset_name}: N={n_samples}, D={n_features}")

    meta = {
        "experiment": "B",
        "dataset": dataset_name,
        "dataset_type": "real_world",
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_repeats": N_REPEATS,
        "seed_base": SEED_BASE,
        "max_iter_sgd": MAX_ITER_SGD,
        "max_iter_smacof": MAX_ITER_SMACOF,
        "sgd_config": {
            "lr": SGD_LR,
            "switch_ratio": SGD_SWITCH,
            "epsilon": SGD_EPSILON,
            "scheduler": "hybrid",
        },
        "modes": ["SMACOF-euclidean", "SGD-euclidean", "SGD-lazy"],
        "timing_policy": "time_solver is the fit_transform runtime for each estimator call.",
        "history_policy": "histories.json stores per-run stress trajectory as [time, stress] pairs.",
    }

    rows: List[Dict[str, Any]] = []
    histories: Dict[str, Any] = {}
    run_id = 0

    for rep in range(N_REPEATS):
        seed = SEED_BASE + rep
        print(f"  [rep={rep:02d} seed={seed}] running...")

        res = _run_three_algos(X, seed, MAX_ITER_SMACOF, MAX_ITER_SGD)

        for algo, r in res.items():
            rid = f"run_{run_id:06d}"
            run_id += 1

            rows.append({
                "run_id": rid,
                "dataset": dataset_name,
                "dataset_type": "real_world",
                "n_samples": int(n_samples),
                "n_features": int(n_features),
                "algo": algo,
                "rep": int(rep),
                "seed": int(seed),
                "max_iter": int(MAX_ITER_SMACOF if algo.startswith("SMACOF") else MAX_ITER_SGD),
                "time_solver": float(r["time"]),
                "stress": float(r["stress"]),
            })

            histories[rid] = {
                "dataset": dataset_name,
                "algo": algo,
                "rep": int(rep),
                "seed": int(seed),
                "n_samples": int(n_samples),
                "n_features": int(n_features),
                "history": _clean_history(r.get("history")),
            }

        print(
            f"    SMACOF t={res['SMACOF-euclidean']['time']:.3f}s | "
            f"SGD-euc t={res['SGD-euclidean']['time']:.3f}s | "
            f"SGD-lazy t={res['SGD-lazy']['time']:.3f}s"
        )

    _write_outputs(out_dir, rows, histories, meta)


def run_synthetic_dataset(dataset_name: str) -> None:
    out_dir = RESULTS_DIR / dataset_name
    out_csv = out_dir / "results.csv"
    out_hist = out_dir / "histories.json"

    if out_csv.exists() and out_hist.exists():
        print(f"Skipping {dataset_name} (results.csv + histories.json exist).")
        return

    if dataset_name not in TARGET_DIMS:
        raise ValueError(f"Missing target dim for synthetic dataset: {dataset_name}")

    target_dim = TARGET_DIMS[dataset_name]

    print(f"\n=== Experiment B (synthetic sweep): {dataset_name} (target D={target_dim}) ===")

    meta = {
        "experiment": "B",
        "dataset": dataset_name,
        "dataset_type": "synthetic",
        "target_dim": int(target_dim),
        "n_values": N_VALUES,
        "n_repeats": N_REPEATS,
        "seed_base": SEED_BASE,
        "max_iter_sgd": MAX_ITER_SGD,
        "max_iter_smacof": MAX_ITER_SMACOF,
        "sgd_config": {
            "lr": SGD_LR,
            "switch_ratio": SGD_SWITCH,
            "epsilon": SGD_EPSILON,
            "scheduler": "hybrid",
        },
        "modes": ["SMACOF-euclidean", "SGD-euclidean", "SGD-lazy"],
        "timing_policy": "time_solver is the fit_transform runtime for each estimator call.",
        "history_policy": "histories.json stores per-run stress trajectory as [time, stress] pairs.",
        "generator_kwargs": {
            "noise_scale": 0.1,
            "target_dim": int(target_dim),
        },
    }

    rows: List[Dict[str, Any]] = []
    histories: Dict[str, Any] = {}
    run_id = 0

    for N in N_VALUES:
        print(f"\n--- N = {N} (D = {target_dim}) ---")

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

            res = _run_three_algos(X, seed, MAX_ITER_SMACOF, MAX_ITER_SGD)

            for algo, r in res.items():
                rid = f"run_{run_id:06d}"
                run_id += 1

                rows.append({
                    "run_id": rid,
                    "dataset": dataset_name,
                    "dataset_type": "synthetic",
                    "n_samples": int(n_samples),
                    "n_features": int(n_features),
                    "algo": algo,
                    "rep": int(rep),
                    "seed": int(seed),
                    "max_iter": int(MAX_ITER_SMACOF if algo.startswith("SMACOF") else MAX_ITER_SGD),
                    "time_solver": float(r["time"]),
                    "stress": float(r["stress"]),
                })

                histories[rid] = {
                    "dataset": dataset_name,
                    "algo": algo,
                    "rep": int(rep),
                    "seed": int(seed),
                    "n_samples": int(n_samples),
                    "n_features": int(n_features),
                    "history": _clean_history(r.get("history")),
                }

            print(
                f"  rep={rep:02d} seed={seed} | "
                f"SMACOF t={res['SMACOF-euclidean']['time']:.3f}s | "
                f"SGD-euc t={res['SGD-euclidean']['time']:.3f}s | "
                f"SGD-lazy t={res['SGD-lazy']['time']:.3f}s"
            )

    _write_outputs(out_dir, rows, histories, meta)

def main():
    # Optional CLI usage:
    #   python run_experiment_b.py                -> run all (synthetic sweep + real-world)
    #   python run_experiment_b.py synthetic      -> run only synthetic datasets
    #   python run_experiment_b.py real           -> run only real-world datasets
    #   python run_experiment_b.py torus seismic  -> run only these datasets
    args = [a.lower() for a in sys.argv[1:]]

    if not args:
        for ds in REALWORLD_DATASETS:
            run_realworld_dataset(ds)
        for ds in SYNTHETIC_DATASETS:
            run_synthetic_dataset(ds)
        return

    if args == ["synthetic"]:
        for ds in SYNTHETIC_DATASETS:
            run_synthetic_dataset(ds)
        return

    if args == ["real"]:
        for ds in REALWORLD_DATASETS:
            run_realworld_dataset(ds)
        return

    for ds in args:
        if ds in REALWORLD_DATASETS:
            run_realworld_dataset(ds)
        elif ds in SYNTHETIC_DATASETS:
            run_synthetic_dataset(ds)
        else:
            raise ValueError(
                f"Unknown dataset '{ds}'. "
                f"Known real-world: {REALWORLD_DATASETS}; synthetic: {SYNTHETIC_DATASETS}"
            )


if __name__ == "__main__":
    main()
