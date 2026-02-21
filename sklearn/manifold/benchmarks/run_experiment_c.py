"""
Experiment C: Hyperparameter Tuning (Runner).
Executes the grid search and saves FULL history.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from time import time
from pathlib import Path

# --- Import Benchmarking Utilities ---
import bench_utils
from bench_utils import load_local_dataset
from sklearn.manifold._sgd_mds import SGDMDS
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_symmetric

RESULTS_DIR = Path(__file__).parent / "results" / "experiment_c"

LEARNING_RATES = [0.1, 0.5, 1.0]
SCHEDULERS = ["exponential", "harmonic", "hybrid"]
MAX_ITERS = [15, 30, 50]
SWITCH_RATIOS = [0.2, 0.3, 0.4, 0.5]
EPSILONS = [0.001, 0.01, 0.1]

N_INIT_TUNING = 5
DATASETS = ["fashion_mnist", "imdb", "s_curve"] 

SEED_BASE = 70

def run_dataset_tuning(dataset_name):
    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Starting Experiment C (Execution) for {dataset_name} ===")
    
    X, y = load_local_dataset(dataset_name)

    print(f"  Computing dissimilarity matrix (N={X.shape[0]})...")
    D = euclidean_distances(X, X)

    try:
        check_symmetric(D, raise_exception=True)
    except ValueError:
        print("  [Info] Matrix not strictly symmetric (precision noise). Symmetrizing...")
        D = (D + D.T) / 2.0
    
    results_list = []
    histories_dict = {}
    
    run_id_counter = 0
    config_id_counter = 0
    
    for max_iter in MAX_ITERS:
        print(f"\n--- Budget: {max_iter} Iterations ---")
        
        for scheduler in SCHEDULERS:
            ratios_to_test = SWITCH_RATIOS if scheduler == "hybrid" else [0.4]
            
            for ratio in ratios_to_test:
                sched_label = f"hybrid (r={ratio})" if scheduler == "hybrid" else scheduler

                for epsilon in EPSILONS:
                    for lr in LEARNING_RATES:
                        config_id = f"cfg_{config_id_counter:05d}"
                        config_id_counter += 1

                        for init_id in range(N_INIT_TUNING):
                            seed = SEED_BASE + init_id
                        
                            est = SGDMDS(
                                n_components=2,
                                metric=True,
                                n_init=1,
                                max_iter=max_iter,
                                learning_rate_init=lr,
                                scheduler=scheduler,
                                scheduler_switch_ratio=ratio,
                                epsilon=epsilon,
                                dissimilarity="precomputed", 
                                random_state=seed,
                                n_jobs=1
                            )
                            
                            t0 = time()
                            embedding = est.fit_transform(D)
                            t_total = time() - t0
                            
                            final_stress = float(est.stress_)
                            
                            run_id = f"run_{run_id_counter:04d}"
                            run_id_counter += 1
                            
                            res_entry = {
                                "run_id": run_id,
                                "config_id": config_id,
                                "dataset": dataset_name,
                                "max_iter": int(max_iter),
                                "scheduler": scheduler,
                                "switch_ratio": float(ratio),
                                "scheduler_label": sched_label,
                                "lr": float(lr),
                                "epsilon": float(epsilon),
                                "init_id": int(init_id),
                                "seed": int(seed),
                                "stress": final_stress,
                                "time": float(t_total)
                            }
                            results_list.append(res_entry)
                            
                            if est.stress_history_:
                                clean_history = [[float(t), float(s)] for t, s in est.stress_history_]
                            else:
                                clean_history = []
                                
                            histories_dict[run_id] = {
                                "params": res_entry,
                                "history": clean_history,
                            }

                            print(
                                f"[{dataset_name}] run={run_id} cfg={config_id} seed={seed} init={init_id} "
                                f"T={max_iter} sched={scheduler} r={ratio} eps={epsilon} lr={lr} "
                                f"stress={final_stress:.6g} time={t_total:.2f}s"
                            )

    print(f"\nSaving results for {dataset_name}...")
    
    df = pd.DataFrame(results_list)
    df.to_csv(out_dir / "results.csv", index=False)
    
    with open(out_dir / "histories.json", "w") as f:
        json.dump(histories_dict, f, indent=None) # Compact JSON
            
    print(f"Done. Saved to {out_dir}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_datasets = sys.argv[1:]
    else:
        target_datasets = DATASETS
        
    for ds in target_datasets:
        run_dataset_tuning(ds)