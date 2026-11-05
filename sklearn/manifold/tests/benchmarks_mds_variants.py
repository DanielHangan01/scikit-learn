import time
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import MDS
from sklearn.manifold._sgd_mds import SGDMDS
from sklearn.utils import check_random_state

# --- Configuration ---
# Root directory for datasets
DATASETS_ROOT = os.path.join(os.path.dirname(__file__), 'datasets')
# Directory to save results
BENCHMARKS_DIR = os.path.join(os.path.dirname(__file__), 'benchmarks', 'results')
# Name of the current run (used for folder name)
RUN_NAME = "true_sgd_mds_comparison_v3_improved_lazy" 

N_COMPONENTS = 2
RANDOM_STATE = 42

def load_dataset(dataset_name):
    """Loads X and y for a specific dataset name."""
    dataset_dir = os.path.join(DATASETS_ROOT, dataset_name)
    x_path = os.path.join(dataset_dir, 'X.npy')
    y_path = os.path.join(dataset_dir, 'y.npy')
    
    if not os.path.exists(x_path):
        print(f"Dataset '{dataset_name}' not found at {x_path}. Skipping.")
        return None, None
        
    X = np.load(x_path)
    # y is optional for MDS but might be useful if you add visualization later
    y = np.load(y_path) if os.path.exists(y_path) else None
    return X, y

def get_all_datasets():
    """Returns a list of available dataset names in the DATASETS_ROOT."""
    if not os.path.exists(DATASETS_ROOT):
        print(f"Datasets root directory '{DATASETS_ROOT}' does not exist.")
        return []
    return [d for d in os.listdir(DATASETS_ROOT) if os.path.isdir(os.path.join(DATASETS_ROOT, d))]

def run_smacof(X, random_state=42):
    """Run the exact SMACOF baseline."""
    print(f"  Running SMACOF (Baseline)...")
    t0 = time.time()
    # Increased max_iter for baseline to ensure convergence
    mds = MDS(n_components=N_COMPONENTS, metric=True, max_iter=300, eps=1e-6,
              random_state=random_state, normalized_stress=False, n_init=1)
    mds.fit(X)
    t1 = time.time()
    
    return {
        'stress': mds.stress_,
        'time': t1 - t0,
        'history': getattr(mds, 'stress_history_', []), # Depends on sklearn version/patch
        'n_samples': X.shape[0],
        'type': 'SMACOF'
    }

def run_sgd_variant(name, X, params):
    """
    Runs a single configuration of SGDMDS.
    
    params: dict of arguments for SGDMDS constructor
    """
    print(f"  Running {name}...")
    
    # Calculate sensible defaults if not provided
    n_samples = X.shape[0]
    n_pairs = (n_samples * (n_samples - 1)) // 2
    
    # Default batch size if not in params
    # Adjust default max batch size as needed
    batch_size = params.get('batch_size', min(5000, n_pairs))
    
    # Handle Epochs vs Iterations logic
    target_epochs = params.pop('n_epochs', 15) 
    
    if params.get('algorithm') == 'mini_batch':
        # Convert epochs to iterations for MiniBatch
        # 1 Epoch = n_pairs updates. 
        # Total updates = (n_pairs / batch_size) * epochs * batch_size = n_pairs * epochs
        total_updates = n_pairs * target_epochs
        # Ensure at least 1 iteration
        max_iter = max(1, int(total_updates / batch_size))
        params['max_iter'] = max_iter
        params['batch_size'] = batch_size
    else:
        # True SGD uses max_iter as epochs directly
        params['max_iter'] = target_epochs
        max_iter = target_epochs

    # Initialize and Fit
    mds = SGDMDS(n_components=N_COMPONENTS, metric=True, 
                 random_state=RANDOM_STATE, **params)
    
    t0 = time.time()
    mds.fit(X)
    t1 = time.time()
    
    # Retrieve stress history if available
    history = getattr(mds, 'stress_history_', [])
    
    return {
        'stress': mds.stress_,
        'time': t1 - t0,
        'history': history,
        'n_samples': n_samples,
        'n_pairs': n_pairs,
        'batch_size': batch_size if params.get('algorithm') == 'mini_batch' else None,
        'type': params.get('algorithm', 'unknown'),
        'max_iter_used': max_iter
    }

def save_results_to_csv(results, csv_path):
    """Saves the stress and time results to a CSV file."""
    # Prepare data for CSV
    rows = []
    for name, data in results.items():
        rows.append({
            'Algorithm': name,
            'Stress': data['stress'],
            'Time (s)': data['time'],
            'Type': data.get('type', ''),
            'Samples': data.get('n_samples', ''),
            'Batch Size': data.get('batch_size', '')
        })
    
    # Sort by Stress
    rows.sort(key=lambda x: x['Stress'])
    
    # Write to file
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Algorithm', 'Stress', 'Time (s)', 'Type', 'Samples', 'Batch Size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  Saved results table to {csv_path}")

def plot_benchmark(dataset_name, results, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # --- Find Global Min Stress for Baseline Line ---
    all_final_stresses = [r['stress'] for r in results.values()]
    global_min = min(all_final_stresses) if all_final_stresses else 0
    
    for ax in (ax1, ax2):
        ax.axhline(global_min, color='gray', ls=':', lw=1.5, alpha=0.8,
                   label=f"Best Stress ({global_min:.2e})", zorder=0)

    # --- Setup Colors ---
    # Use a colormap to handle many unique runs
    unique_names = [k for k in results.keys() if k != 'SMACOF']
    # Use tab20 for more distinctive colors if many variants are present
    colors = cm.tab20(np.linspace(0, 1, len(unique_names)))
    name_to_color = dict(zip(unique_names, colors))
    name_to_color['SMACOF'] = 'black' # Baseline always black

    for name, data in results.items():
        hist = data.get('history', [])
        if not hist:
            # If no history, skip plotting trajectory
            continue

        # Check structure of history (time, stress) or just (stress)
        if isinstance(hist[0], (tuple, list, np.ndarray)) and len(hist[0]) == 2:
             times, stresses = zip(*hist)
        else:
             stresses = hist
             # Estimate times linearly if not provided in history
             times = np.linspace(0, data['time'], len(stresses))

        times = np.array(times)
        stresses = np.array(stresses)
        
        # --- Calculate Effective Epochs ---
        if data['type'] == 'SMACOF' or data['type'] == 'true_sgd':
            # 1 history point = 1 epoch (roughly)
            epochs = np.arange(len(stresses))
        elif data['type'] == 'mini_batch':
            # Convert iterations to effective epochs
            max_iter = data['max_iter_used']
            n_points = len(stresses)
            # If logging happened every k iterations
            record_interval = max_iter / (n_points - 1) if n_points > 1 else 1
            
            iters = np.arange(n_points) * record_interval
            epochs = (iters * data['batch_size']) / data['n_pairs']
        else:
            epochs = np.arange(len(stresses))

        # --- Styling ---
        color = name_to_color.get(name, 'blue')
        
        # Line style based on algorithm type
        if data['type'] == 'SMACOF':
            ls, marker, lw, alpha = '-', None, 2.0, 0.6
        elif data['type'] == 'true_sgd':
            ls, marker, lw, alpha = '-', 'o', 1.5, 0.7 # Solid with dots
        else:
            ls, marker, lw, alpha = '--', None, 1.5, 0.9 # Dashed

        # Plot
        # Use markevery to avoid cluttering the plot with too many markers
        mark_step = max(1, len(epochs)//10)
        
        ax1.plot(epochs, stresses, label=name, color=color, ls=ls, marker=marker, lw=lw, alpha=alpha, markevery=mark_step)
        ax2.plot(times, stresses, label=name, color=color, ls=ls, marker=marker, lw=lw, alpha=alpha, markevery=mark_step)

    # --- Formatting ---
    fig.suptitle(f"Benchmark Results: {dataset_name}", fontsize=16)
    
    for ax, title, xlabel in [(ax1, "Convergence vs Epochs", "Effective Epochs"), 
                              (ax2, "Convergence vs Time", "Time (s)")]:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Stress (Log Scale)")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
    
    # Legend
    handles, labels = ax1.get_legend_handles_labels()
    # Remove duplicates from legend if any
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout for title and legend
    
    # Save
    print(f"  Saving plot to {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # Close figure to free memory for next run

def main():
    print(f"--- BENCHMARK SUITE: {RUN_NAME} ---")
    
    # Base directory for this run: benchmarks/results/{RUN_NAME}
    run_dir = os.path.join(BENCHMARKS_DIR, RUN_NAME)
    
    datasets = get_all_datasets()
    if not datasets:
        print("No datasets found. Exiting.")
        return

    print(f"Found datasets: {datasets}")

    # Define SGD Configurations to Compare
    # Note: Create a fresh list of configs for each dataset run to avoid dictionary mutation issues
    base_configs = [
        # --- True SGD Variants ---
        {
            'name': 'TrueSGD (Exponential) Uniform',
            'algorithm': 'true_sgd',
            'scheduler': 'exponential',
            'learning_rate_init': 1.0,
            'weighting': 'uniform',
            'n_epochs': 30,
            'tol': 1e-12
        },
        {
            'name': 'TrueSGD (Exponential) Uniform Lazy Random',
            'algorithm': 'true_sgd',
            'scheduler': 'exponential',
            'learning_rate_init': 1.0,
            'weighting': 'uniform',
            'n_epochs': 30,
            'tol': 1e-12,
            'dissimilarity': 'lazy',
            'sampling_strategy': 'random'
        },
        # {
        #    'name': 'TrueSGD (Exponential) Uniform Weights More Epochs',
        #    'algorithm': 'true_sgd',
        #    'scheduler': 'exponential',
        #    'learning_rate_init': 1.0,
        #    'weighting': 'uniform',
        #    'n_epochs': 50,
        #    'tol': 1e-12
        # },
        {
            'name': 'TrueSGD (Hybrid) Uniform',
            'algorithm': 'true_sgd',
            'scheduler': 'hybrid',
            'learning_rate_init': 1.0,
            'weighting': 'uniform',
            'n_epochs': 30,
            'tol': 1e-12,
        },
        {
            'name': 'TrueSGD (Hybrid) Uniform Lazy Random',
            'algorithm': 'true_sgd',
            'scheduler': 'hybrid',
            'learning_rate_init': 1.0,
            'weighting': 'uniform',
            'n_epochs': 30,
            'tol': 1e-12,
            'dissimilarity': 'lazy',
            'sampling_strategy': 'random'
        },
        {
            'name': 'TrueSGD (Hybrid) FEWER EPOCHS',
            'algorithm': 'true_sgd',
            'scheduler': 'hybrid',
            'learning_rate_init': 1.0,
            'weighting': 'inverse',
            'n_epochs': 15,
            'tol': 1e-12,
        },
        {
            'name': 'TrueSGD (Hybrid)',
            'algorithm': 'true_sgd',
            'scheduler': 'hybrid',
            'learning_rate_init': 1.0,
            'weighting': 'inverse',
            'n_epochs': 30,
            'tol': 1e-12,
        },
        {
            'name': 'TrueSGD (Hybrid) Lazy Random',
            'algorithm': 'true_sgd',
            'scheduler': 'hybrid',
            'learning_rate_init': 1.0,
            'weighting': 'inverse',
            'n_epochs': 30,
            'tol': 1e-12,
            'dissimilarity': 'lazy',
            'sampling_strategy': 'random'
        },
        # {
        #    'name': 'TrueSGD (Convergent)',
        #    'algorithm': 'true_sgd',
        #    'scheduler': 'convergent',
        #    'learning_rate_init': 1.0,
        #    'weighting': 'inverse',
        #    'n_epochs': 30,
        #    'tol': 1e-12
        # },
    ]

    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        X, y = load_dataset(dataset_name)
        
        if X is None:
            continue
            
        n_samples = X.shape[0]
        print(f"  Data: {n_samples} samples, {X.shape[1]} features")

        # Create output directory for this dataset: benchmarks/results/{RUN_NAME}/{dataset_name}
        dataset_output_dir = os.path.join(run_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        # Center data (Standard practice for MDS)
        X = X - X.mean(axis=0)

        dataset_results = {}

        # 1. Run SMACOF Baseline
        try:
            dataset_results['SMACOF'] = run_smacof(X)
            print(f"  SMACOF done. Stress: {dataset_results['SMACOF']['stress']:.2f} | Time: {dataset_results['SMACOF']['time']:.2f}s")
        except Exception as e:
            print(f"  SMACOF failed: {e}")

        # 2. Run SGD Variants
        # Deep copy configs to prevent 'pop' from affecting next iteration
        current_configs = [c.copy() for c in base_configs]
        
        for config in current_configs:
            name = config.pop('name')
            try:
                # Run benchmark
                res = run_sgd_variant(name, X, config)
                dataset_results[name] = res
                print(f"  Stress: {res['stress']:.2f} | Time: {res['time']:.2f}s")
            except Exception as e:
                print(f"  {name} failed: {e}")

        # 3. Print Summary Table to Console
        print("\n  " + "="*80)
        print(f"  {'Algorithm':<40} | {'Stress':<15} | {'Time (s)':<10}")
        print("  " + "-" * 80)
        
        sorted_res = sorted(dataset_results.items(), key=lambda x: x[1].get('stress', float('inf')))
        
        for name, res in sorted_res:
            stress = res.get('stress', float('inf'))
            time_taken = res.get('time', 0.0)
            print(f"  {name:<40} | {stress:<15.2f} | {time_taken:<10.4f}")
        print("  " + "="*80 + "\n")

        # 4. Save Results to File (CSV)
        csv_filename = "results.csv"
        csv_path = os.path.join(dataset_output_dir, csv_filename)
        save_results_to_csv(dataset_results, csv_path)

        # 5. Save Plot
        plot_filename = "benchmark_plot.png"
        plot_path = os.path.join(dataset_output_dir, plot_filename)
        plot_benchmark(dataset_name, dataset_results, plot_path)

if __name__ == "__main__":
    main()