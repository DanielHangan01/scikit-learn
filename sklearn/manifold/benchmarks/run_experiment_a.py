"""
Experiment A: Comparison with SMACOF (Execution).
Runs SGD (Golden Config) vs SMACOF using Precomputed Matrices.
Saves results.json AND the best embeddings (.npy).
"""

import sys
import os
import json
import numpy as np
from time import time
from pathlib import Path
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_symmetric

import bench_utils
from bench_utils import load_local_dataset, run_smacof_benchmark, run_sgd_benchmark

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_a" / "sgd_30_epochs"

N_INITS = 20
MAX_ITER_SGD = 30
MAX_ITER_SMACOF = 300 

# Golden Config for SGD
SGD_LR = 0.5
SGD_SWITCH = 0.5
SGD_EPSILON = 0.001

def get_all_datasets():
    data_root = bench_utils.DATASET_ROOT
    if not data_root.exists():
        return []
    found = [d.name for d in data_root.iterdir() if d.is_dir()]
    if "s_curve" not in found:
        found.append("s_curve")
    return found

def run_dataset_comparison(dataset_name):
    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "results.json").exists():
        print(f"Skipping {dataset_name} (results.json exists).")
        return
    
    print(f"\n=== Experiment A: {dataset_name} ===")
    
    X, y = load_local_dataset(dataset_name)
    
    print(f"  Computing dissimilarity matrix (N={X.shape[0]})...")
    D = euclidean_distances(X, X)
    
    try:
        check_symmetric(D, raise_exception=True)
    except ValueError:
        print("  [Info] Matrix not strictly symmetric (precision noise). Symmetrizing...")
        D = (D + D.T) / 2.0
    
    results_store = {
        "dataset": dataset_name,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "runs": []
    }
    
    best_smacof_stress = float('inf')
    best_smacof_emb = None
    
    best_sgd_stress = float('inf')
    best_sgd_emb = None
    
    for seed_i in range(N_INITS):
        seed = 42 + seed_i
        print(f"  [Seed {seed}] Processing...")
        
        smacof_res = run_smacof_benchmark(
            D, 
            dissimilarity="precomputed", 
            random_state=seed, 
            max_iter=MAX_ITER_SMACOF
        )
        print(f"    SMACOF Final Stress: {smacof_res['stress']:.2f}")

        if smacof_res["stress"] < best_smacof_stress:
            best_smacof_stress = smacof_res["stress"]
            best_smacof_emb = smacof_res["embedding"]
        
        results_store["runs"].append({
            "algo": "SMACOF",
            "seed": seed,
            "stress": smacof_res["stress"],
            "time": smacof_res["time"],
            "history": smacof_res["history"]
        })
        
        sgd_res = run_sgd_benchmark(
            D, 
            dissimilarity="precomputed", 
            random_state=seed, 
            max_iter=MAX_ITER_SGD, 
            switch_ratio=SGD_SWITCH,
            lr=SGD_LR,
            epsilon=SGD_EPSILON,
        )
        print(f"    SGD-MDS Final Stress: {sgd_res['stress']:.2f}")

        if sgd_res["stress"] < best_sgd_stress:
            best_sgd_stress = sgd_res["stress"]
            best_sgd_emb = sgd_res["embedding"]

        clean_history = []
        if sgd_res["history"]:
            clean_history = [[float(t), float(s)] for t, s in sgd_res["history"]]

        results_store["runs"].append({
            "algo": "SGD-MDS",
            "seed": seed,
            "stress": sgd_res["stress"],
            "time": sgd_res["time"],
            "history": clean_history
        })

    print(f"  Saving results.json and best .npy embeddings to {out_dir}")
    
    with open(out_dir / "results.json", "w") as f:
        json.dump(results_store, f)
        
    if best_smacof_emb is not None:
        np.save(out_dir / "best_embedding_smacof.npy", best_smacof_emb)
        
    if best_sgd_emb is not None:
        np.save(out_dir / "best_embedding_sgd.npy", best_sgd_emb)
        
    if y is not None:
        np.save(out_dir / "y_labels.npy", y)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_datasets = sys.argv[1:]
    else:
        target_datasets = get_all_datasets()
        
    for ds in target_datasets:
        run_dataset_comparison(ds)