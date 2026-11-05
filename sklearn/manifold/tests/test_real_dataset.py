import time
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold._sgd_mds import SGDMDS
from sklearn.utils import check_random_state

# Configuration
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'datasets', 'fashion_mnist')
N_COMPONENTS = 2
RANDOM_STATE = 42

def load_data():
    x_path = os.path.join(DATASET_DIR, 'X.npy')
    y_path = os.path.join(DATASET_DIR, 'y.npy')
    
    if not os.path.exists(x_path):
        print("Dataset not found. Generating synthetic data...")
        rng = np.random.RandomState(42)
        return rng.rand(2000, 50), rng.randint(0, 10, 2000)
        
    X = np.load(x_path)
    y = np.load(y_path)
    return X, y

def plot_benchmark(results, n_pairs, batch_size):
    """
    Plots two graphs:
    1. Stress vs Effective Epochs (Algorithmic Efficiency)
    2. Stress vs Wall-Clock Time (Real-world Speed)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Settings ---
    # Plot SMACOF first (bottom layer), then TrueSGD, then MiniBatch (top layer)
    # We make TrueSGD thick and transparent, and MiniBatch thin and opaque to see overlap
    plot_styles = {
        'SMACOF':    {'color': 'red',   'ls': '-',  'marker': None, 'alpha': 0.8, 'lw': 1.5, 'zorder': 1},
        'TrueSGD':   {'color': 'blue',  'ls': '-',  'marker': 'o',  'alpha': 0.3, 'lw': 4.0, 'zorder': 2},
        'MiniBatch': {'color': 'green', 'ls': '--', 'marker': None, 'alpha': 1.0, 'lw': 1.5, 'zorder': 3}
    }

    # --- Find Global Minimum for Baseline ---
    all_final_stresses = [res['stress'] for res in results.values() if 'stress' in res]
    if all_final_stresses:
        global_min_stress = min(all_final_stresses)
        for ax in (ax1, ax2):
            ax.axhline(y=global_min_stress, color='gray', linestyle=':', linewidth=1.5, 
                       label=f"Lowest Stress ({global_min_stress:.2e})", zorder=0)

    # --- Plotting Loop ---
    for name, data in results.items():
        hist = data.get('history', [])
        if not hist:
            continue
            
        times, stresses = zip(*hist)
        times = np.array(times)
        stresses = np.array(stresses)
        
        # Calculate Epochs
        if name == 'SMACOF':
            # SMACOF is iterative, 1 iter is roughly 1 epoch of work (O(N^2))
            epochs = np.arange(len(stresses))
        elif name == 'TrueSGD':
            # TrueSGD records once per epoch
            epochs = np.arange(len(stresses))
        elif name == 'MiniBatch':
            max_iter = data.get('max_iter_used', len(stresses))
            # Safe calculation for recording interval
            if len(stresses) > 1:
                record_interval = max_iter / (len(stresses) - 1)
            else:
                record_interval = 1
            
            iters = np.arange(len(stresses)) * record_interval
            epochs = (iters * batch_size) / n_pairs

        # Get style
        style = plot_styles.get(name, {})
        
        # Plot
        ax1.plot(epochs, stresses, label=name, **style)
        ax2.plot(times, stresses, label=name, **style)

    # --- Formatting Ax1 (Epochs) ---
    ax1.set_title("Convergence vs. Epochs")
    ax1.set_xlabel("Effective Epochs")
    ax1.set_ylabel("Stress (Log Scale)")
    ax1.set_yscale('log')
    ax1.grid(True, which="both", alpha=0.2)
    ax1.legend()
    
    # --- Formatting Ax2 (Time) ---
    ax2.set_title("Convergence vs. Time")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_yscale('log')
    ax2.grid(True, which="both", alpha=0.2)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def run_benchmark():
    print(f"--- BENCHMARK SUITE ---")

    X, y = load_data()
    n_samples = X.shape[0]
    print(f"Data: {n_samples} samples")

    # Center data
    X = X - X.mean(axis=0)

    results = {}

    # -------------------------------------------------------
    # 1. SMACOF
    # -------------------------------------------------------
    print(f"\n[1] Running SMACOF...")
    mds_smacof = MDS(n_components=N_COMPONENTS, metric=True, max_iter=300, eps=1e-6,
                     random_state=RANDOM_STATE, normalized_stress=False, n_init=1)
    
    t0 = time.perf_counter()
    mds_smacof.fit(X)
    t1 = time.perf_counter()
    
    results['SMACOF'] = {
        'stress': mds_smacof.stress_,
        'time': t1 - t0,
        'iters': mds_smacof.n_iter_,
        'history': getattr(mds_smacof, 'stress_history_', [])
    }
    print(f" Done in {t1 - t0:.2f}s | Stress: {mds_smacof.stress_:.2e}")

    # -------------------------------------------------------
    # 2. SGD Mini-Batch
    # -------------------------------------------------------
    print(f"\n[2] Running SGDMDS (Mini-Batch)...")
    
    n_pairs = (n_samples * (n_samples - 1)) // 2
    batch_size = 5000
    n_epochs = 15
    max_iter_mb = int((n_pairs * n_epochs) / batch_size)
    
    mds_mb = SGDMDS(n_components=N_COMPONENTS, metric=True,
                    max_iter=max_iter_mb, tol=1e-4, 
                    learning_rate_init="auto", batch_size=batch_size,
                    algorithm="mini_batch", scheduler="exponential",
                    weighting="inverse",
                    random_state=RANDOM_STATE)
    
    t0 = time.perf_counter()
    mds_mb.fit(X)
    t1 = time.perf_counter()
    
    results['MiniBatch'] = {
        'stress': mds_mb.stress_, 
        'time': t1 - t0,
        'iters': mds_mb.n_iter_,
        'history': getattr(mds_mb, 'stress_history_', []),
        'max_iter_used': max_iter_mb
    }
    print(f" Done in {t1 - t0:.2f}s | Stress: {mds_mb.stress_:.2e}")

    # -------------------------------------------------------
    # 3. True SGD
    # -------------------------------------------------------
    print(f"\n[3] Running SGDMDS (True SGD)...")
    
    mds_true = SGDMDS(n_components=N_COMPONENTS, metric=True,
                      max_iter=30, tol=1e-4, 
                      learning_rate_init=1.0,
                      algorithm="true_sgd", scheduler="convergent",
                      weighting="inverse",
                      random_state=RANDOM_STATE)
    
    t0 = time.perf_counter()
    mds_true.fit(X)
    t1 = time.perf_counter()
    
    results['TrueSGD'] = {
        'stress': mds_true.stress_, 
        'time': t1 - t0,
        'iters': mds_true.n_iter_,
        'history': getattr(mds_true, 'stress_history_', [])
    }
    print(f" Done in {t1 - t0:.2f}s | Stress: {mds_true.stress_:.2e}")

    # -------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------
    print("\n" + "="*60)
    print(f"{'Algorithm':<15} | {'Stress':<15} | {'Time (s)':<10} | {'Iters':<5}")
    print("-" * 60)
    
    # Sort by stress to see winner easily
    sorted_results = sorted(results.items(), key=lambda x: x[1]['stress'])
    
    for name, res in sorted_results:
        print(f"{name:<15} | {res['stress']:<15.2e} | {res['time']:<10.4f} | {res['iters']:<5}")
    print("="*60 + "\n")

    # -------------------------------------------------------
    # Visualize
    # -------------------------------------------------------
    print("Plotting results...")
    plot_benchmark(results, n_pairs, batch_size)

if __name__ == "__main__":
    run_benchmark()